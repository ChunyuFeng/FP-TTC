import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from utils.loss import get_loss_scale_map, get_loss_risk_score_map

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet
from fpttc.rvt.range_view_transformer import RangeViewTransformer

import torch.distributed as dist
import numpy as np
from utils.dist import is_main_process

import matplotlib.pyplot as plt
import time

def _safe_bilinear(x, *, size=None, scale_factor=None, align_corners=True):
    orig_dtype = x.dtype
    if orig_dtype == torch.bfloat16:
        x = x.float()  # or x.half()
    y = F.interpolate(x, size=size, scale_factor=scale_factor,
                      mode='bilinear', align_corners=align_corners)
    return y.to(orig_dtype)
class CorrEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CorrEncoder, self).__init__()
        self.convc1 = nn.Conv2d(dim_in, 256, 1)
        self.convc2 = nn.Conv2d(256, dim_out, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.convc1(x))
        x = F.relu(self.convc2(x))
        return x

class FpTTC(nn.Module):
    def __init__(self,
                 num_scales             = 2,
                 feature_channels       = 128,
                 upsample_factor        = 4,
                 num_head               = 1,
                 ffn_dim_expansion      = 4,
                 num_transformer_layers = 6,
                 reg_refine             = False
                 ):
        super(FpTTC, self).__init__()
        self.num_scales = num_scales

        self.camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

        self.cnet = CNNEncoder(output_dim        = feature_channels, 
                               num_output_scales = num_scales)
        
        self.featnet = FeatureNet(num_scales             = num_scales,
                                  feature_channels       = feature_channels,
                                  num_head               = num_head, 
                                  ffn_dim_expansion      = ffn_dim_expansion,
                                  num_transformer_layers = num_transformer_layers)   
         
        self.corrnet = FlowNet(num_scales       = num_scales,
                               feature_channels = feature_channels,
                               upsample_factor  = upsample_factor,
                               reg_refine       = reg_refine) 
        
        # self.conv_corr = CorrEncoder(dim_in  = 2,
        #                              dim_out = feature_channels+1)
        
        self.conv_corr_rvt_in = CorrEncoder(dim_in  = 2,
                                            dim_out = feature_channels)
        
        self.conv_corr_rvt_out = CorrEncoder(dim_in  = feature_channels,
                                             dim_out = feature_channels+1)
        self.scale_net = ScaleNet(num_scales      = num_scales,
                                 feature_channels = feature_channels,
                                 upsample_factor  = upsample_factor,
                                 num_head         = 4,
                                 scale_level      = num_scales, 
                                 reg_refine       = reg_refine, 
                                 head_type        = 'scale')

        # Risk 分支
        self.conv_corr_risk = CorrEncoder(dim_in  = 2,
                                          dim_out = feature_channels+1)
        self.risk_net = ScaleNet(num_scales       = num_scales,
                                 feature_channels = feature_channels,
                                 upsample_factor  = upsample_factor,
                                 num_head         = 4,
                                 scale_level      = num_scales, 
                                 reg_refine       = reg_refine, 
                                 head_type        = 'risk')
        
        K_bins = 8
        num_views = len(self.camera_channels)
        self.register_buffer(
            "depth_bins",
            torch.logspace(math.log10(1.0), math.log10(40.0), K_bins)
        )
        
        # 特征分支：按尺度各一个
        self.rvt_feat = nn.ModuleList([
            RangeViewTransformer(
                num_layers=2,                 # 与 ScaleEncoder 层数保持一致
                input_dim=feature_channels,   # 输入通道（每相机该尺度特征的 C）
                d_model=feature_channels,     # 输出/隐藏维，建议与 input_dim 对齐，便于无缝替换
                nhead=4,
                num_level=num_views,          # 相机数 = 6
                num_points=K_bins,            # 深度 bins 数
                fov_up=8.0,
                fov_down=-15.0
            )
            for _ in range(num_scales)
        ])

        # corr 分支：输入是 conv_corr(corr) 的输出通道 = feature_channels+1
        self.rvt_corr = RangeViewTransformer(
            num_layers=2,
            input_dim=feature_channels,
            d_model=feature_channels,
            nhead=4,
            num_level=num_views,
            num_points=K_bins,
            fov_up=8.0,
            fov_down=-15.0
        )
        

    def forward(
        self,
        img_prev, img_curr,
        depth_prev, depth_curr,
        proj_pix_prev, proj_pix_curr,
        sensor_metas,
        attn_type,
        attn_splits_list,
        corr_radius_list,
        prop_radius_list,
        num_reg_refine,
        scale_only
    ):
        
        # ----- 预处理 -----
        img0, img1 = normalize_img(img_prev, img_curr)         # [B,V,3,H,W] -> normed
        rgbd0 = torch.cat([img0, depth_prev], dim=2)           # [B,V,4,H,W]
        rgbd1 = torch.cat([img1, depth_curr], dim=2)           # [B,V,4,H,W]
        B, V, C, H_img, W_img = rgbd0.shape

        # 合并视角到 batch 维，一次性提特征
        x0 = rgbd0.view(B*V, C, H_img, W_img)
        x1 = rgbd1.view(B*V, C, H_img, W_img)
        # extract_feature 会在内部 cat([x0, x1], 0) -> cnet -> 多尺度 -> chunk 回来
        prev_lvls_flat, curr_lvls_flat = self.extract_feature(x0, x1, branch=None)  # list[T][B*V,C,Hs,Ws]

        # reshape 回 [B,V,C,Hs,Ws]
        prev_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in prev_lvls_flat]
        curr_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in curr_lvls_flat]

        # ----- 多尺度 Transformer + 相关性，仍然“只按尺度循环”，但每次处理 B*V -----
        corr = None
        multi_level_feats_prev = []
        multi_level_feats_curr = []

        for lvl in range(self.num_scales):
            # 取出该尺度的 [B,V,C,Hs,Ws]，合并为 [B*V,C,Hs,Ws] 一次性送入
            p = prev_lvls[lvl].reshape(B*V, -1, prev_lvls[lvl].shape[3], prev_lvls[lvl].shape[4])
            c = curr_lvls[lvl].reshape(B*V, -1, curr_lvls[lvl].shape[3], curr_lvls[lvl].shape[4])

            fused_prev, fused_curr = self.featnet(
                p, c, lvl, attn_type, attn_splits_list, corr
            )

            # 保存为 [B,V,C,Hs,Ws] 以便后续投影
            fp = fused_prev.view(B, V, fused_prev.shape[1], fused_prev.shape[2], fused_prev.shape[3])
            fc = fused_curr.view(B, V, fused_curr.shape[1], fused_curr.shape[2], fused_curr.shape[3])
            multi_level_feats_prev.append(fp)
            multi_level_feats_curr.append(fc)

            corr, _ = self.corrnet(
                fused_prev, fused_curr,
                lvl, corr_radius_list, prop_radius_list,
                num_reg_refine, False, corr
            )
            if lvl < self.num_scales - 1:
                corr = _safe_bilinear(corr, scale_factor=2, align_corners=True) * 2

        # corr 此时是 [B*V, 2, Hc, Wc]，变回 [B,V,2,Hc,Wc] 后，为每个视角投影到 range
        Cc, Hc, Wc = corr.shape[1], corr.shape[2], corr.shape[3]
        corr_bv = corr.view(B, V, Cc, Hc, Wc)
        corr_list = [corr_bv[:, v] for v in range(V)]          # list[V] of [B,2,Hc,Wc]

        # # ====== 传统投影（不使用 RVT） ======
        # # 1) 风险/尺度的“相关性特征”投影（当前帧）
        # corr_range_init = self.project_views_to_range(
        #     corr_list, proj_pix_curr, H_img=H_img, W_img=W_img
        # )  # -> [B, Cc, H_r, W_r]

        # # 2) 每个尺度的多视角特征投影
        # multi_level_ranges_prev_init, multi_level_ranges_curr_init = [], []
        # for lvl in range(self.num_scales):
        #     scale = 2 ** (self.num_scales - 1 - lvl)
        #     if scale > 1:
        #         proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
        #         proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
        #     else:
        #         proj_prev_lvl = proj_pix_prev
        #         proj_curr_lvl = proj_pix_curr

        #     # 组建 list[V] of [B,C,Hs,Ws]
        #     prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
        #     curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]

        #     range_prev = self.project_views_to_range(prev_feats_list, proj_prev_lvl, H_img=H_img, W_img=W_img)
        #     range_curr = self.project_views_to_range(curr_feats_list, proj_curr_lvl, H_img=H_img, W_img=W_img)

        #     multi_level_ranges_prev_init.append(range_prev)  # [B,C,Hr,Wr]
        #     multi_level_ranges_curr_init.append(range_curr)
        # # ====== 传统投影（不使用 RVT） ======

        # ====== 基于 RVT 将多视角特征图聚合到 Range View ======
        # corr -> range（当前帧）
        corr_encoded_list = [self.conv_corr_rvt_in(c) for c in corr_list]  # list[V] of [B, Ccorr, Hc, Wc]
        H_r = corr_encoded_list[0].shape[2]
        W_r = corr_encoded_list[0].shape[3] * V
        # 将每个视角的 corr_encoded_list 在 W 维度横向拼接，作为初始查询
        # ini_corr_query = torch.cat(corr_encoded_list, dim=3)  # [B, Ccorr, Hc, Wc*V]
        # corr_range_init = self.conv_corr_rvt_in(corr_range_init.detach())
        corr_range = self.rvt_corr(
            feats_by_cam=corr_encoded_list,
            cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
            cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
            cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
            affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
            Hr=H_r, Wr=W_r,
            depth_bins=self.depth_bins,
            ini_query=None
        )

        # 多尺度特征 -> range（前/当前帧）
        multi_level_ranges_prev, multi_level_ranges_curr = [], []
        for lvl in range(self.num_scales):
            prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
            curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]
            H_r = prev_feats_list[0].shape[2]
            W_r = prev_feats_list[0].shape[3] * V

            range_prev = self.rvt_feat[lvl](
                feats_by_cam=prev_feats_list,
                cam_K=[sensor_metas['prev'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r,
                depth_bins=self.depth_bins,
                ini_query=None
            )
            range_curr = self.rvt_feat[lvl](
                feats_by_cam=curr_feats_list,
                cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r,
                depth_bins=self.depth_bins,
                ini_query=None
            )
            multi_level_ranges_prev.append(range_prev)  # [B,C,Hr,Wr]
            multi_level_ranges_curr.append(range_curr)


        # ----- 预测（两分支）-----
        # scale 分支
        corr_encoded_s = self.conv_corr_rvt_out(corr_range)
        initial_scale = F.softplus(corr_encoded_s[:, :1]) + 1e-3
        corr_encoded_s = corr_encoded_s[:, 1:]

        scales = self.scale_net(
            corr_encoded_s, multi_level_ranges_prev, multi_level_ranges_curr, initial_scale
        )
        if scale_only:
            return scales, None

        # risk 分支
        corr_encoded = self.conv_corr_risk(corr_range)
        initial_risk = corr_encoded[:, :1]
        corr_encoded = corr_encoded[:, 1:]
        risk_score = self.risk_net(
            corr_encoded, multi_level_ranges_prev, multi_level_ranges_curr, initial_risk
        )

        return scales, risk_score


    def forward_with_loss( 
            self,
            img_prev,                      # Tensor[B, V, 3, H, W]
            img_curr,                      # Tensor[B, V, 3, H, W]
            depth_prev,                    # Tensor[B, V, 1, H, W]
            depth_curr,                    # Tensor[B, V, 1, H, W]
            proj_pix_prev,                 
            proj_pix_curr,                 
            gt_scale_map_with_mask,        # Tensor[B, 2, H_sph, W_sph]
            gt_risk_score_map_with_mask,   # Tensor[B, 2, H_sph, W_sph]
            sensor_metas,
            attn_type,                     # str
            attn_splits_list,              # List[int]
            corr_radius_list,              # List[int]
            prop_radius_list,              # List[int]
            num_reg_refine,                # int
            scale_only
        ):

        scales, risks = self.forward(
            img_prev         = img_prev,
            img_curr         = img_curr,
            depth_prev       = depth_prev,
            depth_curr       = depth_curr,
            proj_pix_prev    = proj_pix_prev,
            proj_pix_curr    = proj_pix_curr,
            sensor_metas     = sensor_metas,
            attn_type        = attn_type,
            attn_splits_list = attn_splits_list,
            corr_radius_list = corr_radius_list,
            prop_radius_list = prop_radius_list,
            num_reg_refine   = num_reg_refine,
            scale_only       = scale_only,
        )

        # # ===== teacher 蒸馏（训练期才启用） =====
        # if self.training and self.cfg.get('use_rvt_distill', True):
        #     with torch.no_grad():
        #         # 用现有 teacher 投影：特征与 corr -> Range
        #         # teacher_feats_prev/teacher_feats_curr 通过 project_views_to_range(..., proj_pix_prev/curr, ...) 得到
        #         t_range_prev_list, t_range_curr_list = [], []
        #         for lvl in range(self.num_scales):
        #             prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
        #             curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]
        #             t_prev = self.project_views_to_range(prev_feats_list, proj_pix_prev, H_img=H_img, W_img=W_img)  # [B,C,Hr,Wr]
        #             t_curr = self.project_views_to_range(curr_feats_list, proj_pix_curr, H_img=H_img, W_img=W_img)
        #             t_range_prev_list.append(t_prev); t_range_curr_list.append(t_curr)

        #         t_corr_range = self.project_views_to_range(
        #             [self.conv_corr(c) for c in corr_list], proj_pix_curr, H_img=H_img, W_img=W_img
        #         )

        #     # L_feat：student vs teacher（多尺度 Range 特征）
        #     L_feat = 0.0
        #     for l in range(self.num_scales):
        #         L_feat = L_feat + (multi_level_ranges_prev[l] - t_range_prev_list[l]).abs().mean()
        #         L_feat = L_feat + (multi_level_ranges_curr[l] - t_range_curr_list[l]).abs().mean()

        #     # L_corr：student vs teacher（Range corr）
        #     L_corr = (corr_range - t_corr_range).abs().mean()

        #     # 汇总
        #     lam_f = self.cfg.get('lambda_feat', 1.0)
        #     lam_c = self.cfg.get('lambda_corr', 1.0)
        #     if scale_only:
        #         loss = loss_s + lam_f * L_feat + lam_c * L_corr
        #         return scales, None, loss, None
        #     else:
        #         loss = loss_r + lam_f * L_feat + lam_c * L_corr
        #         return None, risks, None, loss
        
        if scale_only:
            # 仅计算尺度分支的损失
            loss_s = get_loss_scale_map(scales, gt_scale_map_with_mask)
            return scales, None, loss_s, None
        else:
            loss_r = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)
            return None, risks, None, loss_r

    def forward_rvt_distill(
        self,
        img_prev, img_curr,
        depth_prev, depth_curr,
        proj_pix_prev, proj_pix_curr,
        sensor_metas,
        attn_type, attn_splits_list,
        corr_radius_list, prop_radius_list,
        num_reg_refine,
        lam_feat=1.0, lam_corr=1.0,
        loss_type="smoothl1",
        ema_momentum=0.99,            # EMA动量（越大越稳）
        eps=1e-3,                      # 数值稳定项
        return_debug=False,
        return_student=False
    ):
        """
        阶段1：RVT蒸馏。使用“硬投影”作为Teacher，RVT输出作为Student。
        关键改动：
        1) 逐level用teacher的EMA标准差做标准化，等权平均  -> 解决 L_feat 内部不一致
        2) learnable log-var 的不确定性加权  -> 解决 L_corr 与 L_feat 间量级不一致
        """
        # =============== 小工具：惰性注册 ===============
        def _get_or_register_buffer(name, init_tensor):
            if not hasattr(self, name):
                self.register_buffer(name, init_tensor.detach().clone())
            return getattr(self, name)

        def _get_or_register_param(name, init_value=0.0, device=None):
            if not hasattr(self, name):
                p = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32, device=device))
                self.register_parameter(name, p)
            return getattr(self, name)

        def _pick_loss(residual):
            # 对“归一化后的残差 residual”与 0 做监督
            if loss_type == "mse":
                return F.mse_loss(residual, torch.zeros_like(residual), reduction="mean")
            elif loss_type == "l1":
                return F.l1_loss(residual, torch.zeros_like(residual), reduction="mean")
            else:
                return F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="mean")
        
        def _feat_to_gray(x):  # x: [B,C,H,W] -> [H,W] in [0,1]
            # 通道范数 + 鲁棒归一化（按分位数）
            mag = x.pow(2).sum(1).sqrt()              # [B,H,W]
            m = mag[0]
            p1  = torch.quantile(m, 0.01)
            p99 = torch.quantile(m, 0.99)
            g = (m - p1) / (p99 - p1 + 1e-6)
            return g.clamp(0, 1)

        def _cos_map(a, b):  # a,b: [B,C,H,W] -> [H,W] in [-1,1]
            a0, b0 = a[0], b[0]
            an = a0 / (a0.pow(2).sum(0, keepdim=True).sqrt() + 1e-6)
            bn = b0 / (b0.pow(2).sum(0, keepdim=True).sqrt() + 1e-6)
            return (an * bn).sum(0).clamp(-1, 1)      # [H,W]

        # =============== 前向到多尺度特征与 corr（与原版一致） ===============
        img0, img1 = normalize_img(img_prev, img_curr)
        rgbd0 = torch.cat([img0, depth_prev], dim=2)
        rgbd1 = torch.cat([img1, depth_curr], dim=2)
        B, V, C, H_img, W_img = rgbd0.shape

        x0 = rgbd0.view(B*V, C, H_img, W_img)
        x1 = rgbd1.view(B*V, C, H_img, W_img)
        prev_lvls_flat, curr_lvls_flat = self.extract_feature(x0, x1, branch=None)

        prev_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in prev_lvls_flat]
        curr_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in curr_lvls_flat]

        corr = None
        multi_level_feats_prev, multi_level_feats_curr = [], []
        for lvl in range(self.num_scales):
            p = prev_lvls[lvl].reshape(B*V, -1, prev_lvls[lvl].shape[3], prev_lvls[lvl].shape[4])
            c = curr_lvls[lvl].reshape(B*V, -1, curr_lvls[lvl].shape[3], curr_lvls[lvl].shape[4])

            fused_prev, fused_curr = self.featnet(p, c, lvl, attn_type, attn_splits_list, corr)
            fp = fused_prev.view(B, V, fused_prev.shape[1], fused_prev.shape[2], fused_prev.shape[3])
            fc = fused_curr.view(B, V, fused_curr.shape[1], fused_curr.shape[2], fused_curr.shape[3])
            multi_level_feats_prev.append(fp)
            multi_level_feats_curr.append(fc)

            corr, _ = self.corrnet(
                fused_prev, fused_curr, lvl,
                corr_radius_list, prop_radius_list, num_reg_refine, False, corr
            )
            if lvl < self.num_scales - 1:
                corr = _safe_bilinear(corr, scale_factor=2, align_corners=True) * 2

        # corr -> list[V] of [B,2,Hc,Wc]
        Cc, Hc, Wc = corr.shape[1], corr.shape[2], corr.shape[3]
        corr_bv = corr.view(B, V, Cc, Hc, Wc)
        corr_list = [corr_bv[:, v] for v in range(V)]

        # =============== Student：RVT 聚合（可学习0偏置） ===============
        # corr
        corr_encoded_list = [self.conv_corr_rvt_in(c) for c in corr_list]
        H_r_corr = corr_encoded_list[0].shape[2]
        W_r_corr = corr_encoded_list[0].shape[3] * V
        corr_range_student = self.rvt_corr(
            feats_by_cam=corr_encoded_list,
            cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
            cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
            cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
            affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
            Hr=H_r_corr, Wr=W_r_corr, depth_bins=self.depth_bins,
            ini_query=None
        )

        # feats each scale
        ranges_prev_student, ranges_curr_student = [], []
        for lvl in range(self.num_scales):
            prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
            curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]
            H_r = prev_feats_list[0].shape[2]
            W_r = prev_feats_list[0].shape[3] * V

            range_prev_s = self.rvt_feat[lvl](
                feats_by_cam=prev_feats_list,
                cam_K=[sensor_metas['prev'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r, depth_bins=self.depth_bins, ini_query=None
            )
            range_curr_s = self.rvt_feat[lvl](
                feats_by_cam=curr_feats_list,
                cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r, depth_bins=self.depth_bins, ini_query=None
            )
            ranges_prev_student.append(range_prev_s)
            ranges_curr_student.append(range_curr_s)

        # =============== Teacher：硬投影 ===============
        with torch.no_grad():
            # corr teacher
            corr_range_teacher = self.project_views_to_range(
                corr_list, proj_pix_curr, H_img=H_img, W_img=W_img
            )  # [B, 2, Hr, Wr]
            corr_range_teacher = self.conv_corr_rvt_in(corr_range_teacher)  # [B, C, Hr, Wr]

            # feats teacher per scale (+ 下采样 proj_pix_* 对齐)
            ranges_prev_teacher, ranges_curr_teacher = [], []
            for lvl in range(self.num_scales):
                scale = 2 ** (self.num_scales - 1 - lvl)
                proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :] if scale > 1 else proj_pix_prev
                proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :] if scale > 1 else proj_pix_curr

                prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
                curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]

                range_prev_t = self.project_views_to_range(prev_feats_list, proj_prev_lvl, H_img=H_img, W_img=W_img)
                range_curr_t = self.project_views_to_range(curr_feats_list, proj_curr_lvl, H_img=H_img, W_img=W_img)
                ranges_prev_teacher.append(range_prev_t)
                ranges_curr_teacher.append(range_curr_t)

        # =============== (1) 逐项标准化：用 Teacher 的 EMA std ===============
        device = img_prev.device

        # corr 的 EMA std（标量）
        with torch.no_grad():
            sigma_corr_cur = corr_range_teacher.float().std()
        buf_corr = _get_or_register_buffer("rvt_ema_sigma_corr", torch.tensor(1.0, dtype=torch.float32, device=device))
        buf_corr.mul_(ema_momentum).add_((1.0 - ema_momentum) * sigma_corr_cur)
        sigma_corr = buf_corr.view(1, 1, 1, 1)  # broadcast

        # 每个 level 的 EMA std（prev/curr 各一）
        ema_sig_prev = []
        ema_sig_curr = []
        for l in range(self.num_scales):
            with torch.no_grad():
                s_prev = ranges_prev_teacher[l].float().std()
                s_curr = ranges_curr_teacher[l].float().std()

            buf_name_p = f"rvt_ema_sigma_feat_prev_{l}"
            buf_name_c = f"rvt_ema_sigma_feat_curr_{l}"
            buf_p = _get_or_register_buffer(buf_name_p, torch.tensor(1.0, dtype=torch.float32, device=device))
            buf_c = _get_or_register_buffer(buf_name_c, torch.tensor(1.0, dtype=torch.float32, device=device))

            buf_p.mul_(ema_momentum).add_((1.0 - ema_momentum) * s_prev)
            buf_c.mul_(ema_momentum).add_((1.0 - ema_momentum) * s_curr)

            ema_sig_prev.append(buf_p.view(1, 1, 1, 1))
            ema_sig_curr.append(buf_c.view(1, 1, 1, 1))

        # =============== (2) 构造“归一化后的残差”并等权平均 ===============
        # corr
        R_corr = (corr_range_student - corr_range_teacher.detach()) / (sigma_corr + eps)
        L_corr_norm = _pick_loss(R_corr)

        # feat per level + prev/curr 两个时刻，共 2*num_scales 项，等权平均
        L_feat_terms = []
        for l in range(self.num_scales):
            R_prev = (ranges_prev_student[l] - ranges_prev_teacher[l].detach()) / (ema_sig_prev[l] + eps)
            R_curr = (ranges_curr_student[l] - ranges_curr_teacher[l].detach()) / (ema_sig_curr[l] + eps)
            L_feat_terms.append(_pick_loss(R_prev))
            L_feat_terms.append(_pick_loss(R_curr))
        L_feat_norm = sum(L_feat_terms) / len(L_feat_terms)

        # =============== (3) 不确定性加权（learnable log-variance） ===============
        # 两个可学习标量：rvt_log_var_feat, rvt_log_var_corr（log σ^2）
        logv_feat = _get_or_register_param("rvt_log_var_feat", 0.0, device=device)
        logv_corr = _get_or_register_param("rvt_log_var_corr", 0.0, device=device)
        # 经典形式：exp(-s) * L + s（其中 s=log σ^2 ）
        distill_loss = (
            torch.exp(-logv_corr) * (lam_corr * L_corr_norm) + logv_corr +
            torch.exp(-logv_feat) * (lam_feat * L_feat_norm) + logv_feat
        )

        # =============== VIS ===============
        vis = {}
        # corr: 灰度 & 余弦
        vis["corr_student_gray"]  = _feat_to_gray(corr_range_student).detach().cpu()
        vis["corr_teacher_gray"]  = _feat_to_gray(corr_range_teacher).detach().cpu()
        vis["corr_cos"]           = (_cos_map(corr_range_student, corr_range_teacher) * 0.5 + 0.5).detach().cpu()

        # 最高分辨率那个尺度（例如 lvl=0），也可以换成你想看的 lvl
        lvl0_prev_s = ranges_prev_student[0]
        lvl0_prev_t = ranges_prev_teacher[0]
        lvl0_curr_s = ranges_curr_student[0]
        lvl0_curr_t = ranges_curr_teacher[0]
        vis["lvl0_prev_cos"]      = (_cos_map(lvl0_prev_s, lvl0_prev_t) * 0.5 + 0.5).detach().cpu()
        vis["lvl0_curr_cos"]      = (_cos_map(lvl0_curr_s, lvl0_curr_t) * 0.5 + 0.5).detach().cpu()
        vis["lvl0_prev_s_gray"]   = _feat_to_gray(lvl0_prev_s).detach().cpu()
        vis["lvl0_prev_t_gray"]   = _feat_to_gray(lvl0_prev_t).detach().cpu()
        vis["lvl0_curr_s_gray"]   = _feat_to_gray(lvl0_curr_s).detach().cpu()
        vis["lvl0_curr_t_gray"]   = _feat_to_gray(lvl0_curr_t).detach().cpu()
        
        # =============== VIS ===============


        # 返回日志：用归一化后的两个分量，便于观察是否同量级
        log_dict = {
            "L_corr_norm": L_corr_norm.detach(),
            "L_feat_norm": L_feat_norm.detach(),
            "sigma_corr_ema": buf_corr.detach(),
            # 也可以选择性地把 per-level 的 ema 打印出来（避免日志过多，这里给均值）
            "sigma_feat_prev_mean_ema": torch.stack([s.squeeze() for s in ema_sig_prev]).mean().detach(),
            "sigma_feat_curr_mean_ema": torch.stack([s.squeeze() for s in ema_sig_curr]).mean().detach(),
            "logvar_corr": logv_corr.detach(),
            "logvar_feat": logv_feat.detach(),
        }
        if return_debug:
            log_dict["vis"] = vis
        if return_student:
            # 仅用于可视化预览，全部 detach，避免显存持有计算图
            log_dict["student"] = {
                "corr": corr_range_student.detach(),
                "prev": [t.detach() for t in ranges_prev_student],
                "curr": [t.detach() for t in ranges_curr_student],
            }
        return distill_loss, log_dict

    @torch.no_grad()
    def preview_scale_from_student(self,
                                corr_range_student,           # Tensor[B, C, Hr, Wr]
                                ranges_prev_student,          # list[num_scales] of Tensor[B, C, Hr, Wr]
                                ranges_curr_student):         # list[num_scales] of Tensor[B, C, Hr, Wr]
        """
        用“学生RVT输出”的range特征，走一次scale_head，得到一个 scale 预测（只用于可视化预览）。
        不会影响梯度；内部会临时把 scale_net 切到 eval 再还原。
        返回: Tensor[B, 1, Hr, Wr]
        """
        was_training = self.scale_net.training
        self.scale_net.eval()

        # corr 分支的入头保持与正式前向一致
        corr_encoded_s = self.conv_corr_rvt_out(corr_range_student)    # [B, C+1, Hr, Wr]
        initial_scale  = F.softplus(corr_encoded_s[:, :1]) + 1e-3      # [B, 1, Hr, Wr]
        corr_encoded_s = corr_encoded_s[:, 1:]                         # [B, C, Hr, Wr]

        pred = self.scale_net(corr_encoded_s,
                            ranges_prev_student,
                            ranges_curr_student,
                            initial_scale)
        if isinstance(pred, list):
            pred = pred[-1]  # 取最高分辨率/最后一层输出

        if was_training:
            self.scale_net.train()
        return pred  # [B,1,Hr,Wr]

    @staticmethod
    @torch.no_grad()
    def save_scale_preview(scale_tensor, out_path, cmap="magma"):
        """
        把 scale_tensor[B,1,H,W] 的第0张图做鲁棒归一化并存成PNG。
        """
        import os, matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        s = scale_tensor[0, 0].detach().cpu()
        p1  = torch.quantile(s, 0.01)
        p99 = torch.quantile(s, 0.99)
        s_norm = (s - p1) / (p99 - p1 + 1e-6)
        s_norm = s_norm.clamp(0, 1).numpy()
        plt.imsave(out_path, s_norm, cmap=cmap)

    # def forward_rvt_distill(
    #     self,
    #     img_prev, img_curr,
    #     depth_prev, depth_curr,
    #     proj_pix_prev, proj_pix_curr,
    #     sensor_metas,
    #     attn_type, attn_splits_list,
    #     corr_radius_list, prop_radius_list,
    #     num_reg_refine,
    #     lam_feat=1.0, lam_corr=1.0,
    #     loss_type="smoothl1"
    # ):
    #     """
    #     仅用于“阶段1：RVT 蒸馏”。输出一个 distill loss。
    #     Teacher: 用硬投影(project_views_to_range)得到 RV 特征
    #     Student: RVT 输出
    #     仅回传 RVT 参数的梯度（其余模块在阶段1会被冻结）
    #     """
    #     # ======= 前向到多尺度特征与 corr =======
    #     img0, img1 = normalize_img(img_prev, img_curr)
    #     rgbd0 = torch.cat([img0, depth_prev], dim=2)
    #     rgbd1 = torch.cat([img1, depth_curr], dim=2)
    #     B, V, C, H_img, W_img = rgbd0.shape

    #     x0 = rgbd0.view(B*V, C, H_img, W_img)
    #     x1 = rgbd1.view(B*V, C, H_img, W_img)
    #     prev_lvls_flat, curr_lvls_flat = self.extract_feature(x0, x1, branch=None)

    #     prev_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in prev_lvls_flat]
    #     curr_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in curr_lvls_flat]

    #     corr = None
    #     multi_level_feats_prev = []
    #     multi_level_feats_curr = []
    #     for lvl in range(self.num_scales):
    #         p = prev_lvls[lvl].reshape(B*V, -1, prev_lvls[lvl].shape[3], prev_lvls[lvl].shape[4])
    #         c = curr_lvls[lvl].reshape(B*V, -1, curr_lvls[lvl].shape[3], curr_lvls[lvl].shape[4])

    #         fused_prev, fused_curr = self.featnet(p, c, lvl, attn_type, attn_splits_list, corr)

    #         fp = fused_prev.view(B, V, fused_prev.shape[1], fused_prev.shape[2], fused_prev.shape[3])
    #         fc = fused_curr.view(B, V, fused_curr.shape[1], fused_curr.shape[2], fused_curr.shape[3])
    #         multi_level_feats_prev.append(fp)
    #         multi_level_feats_curr.append(fc)

    #         corr, _ = self.corrnet(
    #             fused_prev, fused_curr, lvl, corr_radius_list, prop_radius_list, num_reg_refine, False, corr
    #         )
    #         if lvl < self.num_scales - 1:
    #             corr = _safe_bilinear(corr, scale_factor=2, align_corners=True) * 2

    #     # corr -> list[V] of [B,2,Hc,Wc]
    #     Cc, Hc, Wc = corr.shape[1], corr.shape[2], corr.shape[3]
    #     corr_bv = corr.view(B, V, Cc, Hc, Wc)
    #     corr_list = [corr_bv[:, v] for v in range(V)]

    #     # ======= Student：RVT 聚合（可学习0偏置） =======
    #     # corr
    #     corr_encoded_list = [self.conv_corr_rvt_in(c) for c in corr_list]
    #     H_r_corr = corr_encoded_list[0].shape[2]
    #     W_r_corr = corr_encoded_list[0].shape[3] * V
    #     corr_range_student = self.rvt_corr(
    #         feats_by_cam=corr_encoded_list,
    #         cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
    #         cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
    #         cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
    #         affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
    #         Hr=H_r_corr, Wr=W_r_corr, depth_bins=self.depth_bins,
    #         ini_query=None
    #     )

    #     # feats each scale
    #     ranges_prev_student, ranges_curr_student = [], []
    #     for lvl in range(self.num_scales):
    #         prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
    #         curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]
    #         H_r = prev_feats_list[0].shape[2]
    #         W_r = prev_feats_list[0].shape[3] * V

    #         range_prev_s = self.rvt_feat[lvl](
    #             feats_by_cam=prev_feats_list,
    #             cam_K=[sensor_metas['prev'][cam]['K']     for cam in self.camera_channels],
    #             cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
    #             cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
    #             affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
    #             Hr=H_r, Wr=W_r, depth_bins=self.depth_bins, ini_query=None
    #         )
    #         range_curr_s = self.rvt_feat[lvl](
    #             feats_by_cam=curr_feats_list,
    #             cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
    #             cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
    #             cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
    #             affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
    #             Hr=H_r, Wr=W_r, depth_bins=self.depth_bins, ini_query=None
    #         )
    #         ranges_prev_student.append(range_prev_s)
    #         ranges_curr_student.append(range_curr_s)

    #     # ======= Teacher：硬投影 =======
    #     with torch.no_grad():
    #         # corr teacher
    #         corr_range_teacher = self.project_views_to_range(
    #             corr_list, proj_pix_curr, H_img=H_img, W_img=W_img
    #         )  # [B, 2, Hr, Wr]
    #         corr_range_teacher = self.conv_corr_rvt_in(corr_range_teacher)  # [B, C, Hr, Wr]

    #         # feats teacher per scale (+ 下采样 proj_pix_* 对齐)
    #         ranges_prev_teacher, ranges_curr_teacher = [], []
    #         for lvl in range(self.num_scales):
    #             scale = 2 ** (self.num_scales - 1 - lvl)
    #             proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :] if scale > 1 else proj_pix_prev
    #             proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :] if scale > 1 else proj_pix_curr

    #             prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
    #             curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]

    #             range_prev_t = self.project_views_to_range(prev_feats_list, proj_prev_lvl, H_img=H_img, W_img=W_img)
    #             range_curr_t = self.project_views_to_range(curr_feats_list, proj_curr_lvl, H_img=H_img, W_img=W_img)
    #             ranges_prev_teacher.append(range_prev_t)
    #             ranges_curr_teacher.append(range_curr_t)

    #     # ======= Loss =======
    #     if loss_type == "l1":
    #         loss_fn = F.l1_loss
    #     elif loss_type == "mse":
    #         loss_fn = F.mse_loss
    #     else:
    #         loss_fn = F.smooth_l1_loss

    #     L_corr = loss_fn(corr_range_student, corr_range_teacher.detach())
    #     L_feat = 0.0
    #     for l in range(self.num_scales):
    #         L_feat = L_feat + loss_fn(ranges_prev_student[l], ranges_prev_teacher[l].detach())
    #         L_feat = L_feat + loss_fn(ranges_curr_student[l], ranges_curr_teacher[l].detach())

    #     distill_loss = lam_corr * L_corr + lam_feat * L_feat
    #     return distill_loss, {"L_corr": L_corr.detach(), "L_feat": L_feat.detach()}


    def extract_feature(self, im0, im1, branch):
        x = torch.cat([im0, im1], dim=0)
        feats = self.cnet(x, branch=branch)[::-1]
        p0, p1 = [], []
        for f in feats:
            a, b = torch.chunk(f, 2, dim=0)
            p0.append(a); p1.append(b)
        return p0, p1
    
    def project_views_to_range(
        self,
        features_list,   # list of V tensors, each [B, C, H_feat, W_feat]
        proj_pix,        # LongTensor [B, H_r, W_r, 3] = (cam_idx, u_orig, v_orig)
        H_img=160, W_img=320
    ):
        """
        把 V 路视角、同一尺度的特征图，按 proj_pix 映射到 range-view [B, C, H_r, W_r]。
        """
        B, H_r, W_r, _ = proj_pix.shape
        V = len(features_list)
        C = features_list[0].shape[1]
        H_feat, W_feat = features_list[0].shape[2], features_list[0].shape[3]

        # 1) 下采样比例
        s_u = W_feat / W_img
        s_v = H_feat / H_img

        # 2) 合并成 [B, V, C, Hf, Wf] → [B*V, C, Hf, Wf]
        feats = torch.stack(features_list, dim=1)         # [B,V,C,Hf,Wf]
        feats = feats.view(B*V, C, H_feat, W_feat)
        feats_flat = feats.view(B*V, C, -1)               # [B*V, C, Hf*Wf]

        # 3) 拆出并缩放像素索引
        cam_idx = proj_pix[..., 0].reshape(B, -1)         # [B, N]
        u_orig  = proj_pix[..., 1].float().reshape(B, -1) # [B, N]
        v_orig  = proj_pix[..., 2].float().reshape(B, -1) # [B, N]

        u_feat = (u_orig * s_u).long().clamp(0, W_feat-1) # [B, N]
        v_feat = (v_orig * s_v).long().clamp(0, H_feat-1) # [B, N]

        # 4) 计算扁平化后的批次＋视角索引
        #    batch_idx ∈ [0..B) 重复 N 次，拼接 cam_idx → [B*N]
        batch_idx = torch.arange(B, device=cam_idx.device)\
                        .unsqueeze(1).repeat(1, H_r*W_r)\
                        .reshape(-1)
        view_idx  = batch_idx * V + cam_idx.reshape(-1)   # [B*N]

        # 5) 计算在 Hf*Wf 上的线性化像素
        pix_idx   = (v_feat * W_feat + u_feat).reshape(-1)  # [B*N]

        # 6) 一次性 gather
        #    feats_flat[view_idx, :, pix_idx] → [B*N, C]
        selected = feats_flat[view_idx, :, pix_idx]

        # 7) 重塑回 [B, C, H_r, W_r]
        range_feat = selected.view(B, H_r*W_r, C) \
                            .permute(0,2,1) \
                            .reshape(B, C, H_r, W_r)
        return range_feat
