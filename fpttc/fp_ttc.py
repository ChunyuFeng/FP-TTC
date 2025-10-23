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
        
        self.register_buffer("anneal_alpha", torch.tensor(1.0))  # [0,1]

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

        # # ====== 基于 RVT 将多视角特征图聚合到 Range View ======
        # # corr -> range（当前帧）
        # corr_encoded_list = [self.conv_corr_rvt_in(c) for c in corr_list]  # list[V] of [B, Ccorr, Hc, Wc]
        # H_r = corr_encoded_list[0].shape[2]
        # W_r = corr_encoded_list[0].shape[3] * V
        # # 将每个视角的 corr_encoded_list 在 W 维度横向拼接，作为初始查询
        # # ini_corr_query = torch.cat(corr_encoded_list, dim=3)  # [B, Ccorr, Hc, Wc*V]
        # corr_range_init = self.conv_corr_rvt_in(corr_range_init.detach())
        # corr_range = self.rvt_corr(
        #     feats_by_cam=corr_encoded_list,
        #     cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
        #     cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
        #     cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
        #     affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
        #     Hr=H_r, Wr=W_r,
        #     depth_bins=self.depth_bins,
        #     ini_query=corr_range_init
        # )

        # # 多尺度特征 -> range（前/当前帧）
        # multi_level_ranges_prev, multi_level_ranges_curr = [], []
        # for lvl in range(self.num_scales):
        #     prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
        #     curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]
        #     H_r = prev_feats_list[0].shape[2]
        #     W_r = prev_feats_list[0].shape[3] * V

        #     range_prev = self.rvt_feat[lvl](
        #         feats_by_cam=prev_feats_list,
        #         cam_K=[sensor_metas['prev'][cam]['K']     for cam in self.camera_channels],
        #         cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
        #         cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
        #         affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
        #         Hr=H_r, Wr=W_r,
        #         depth_bins=self.depth_bins,
        #         ini_query=multi_level_ranges_prev_init[lvl].detach()  # 传统投影结果作为初始查询
        #     )
        #     range_curr = self.rvt_feat[lvl](
        #         feats_by_cam=curr_feats_list,
        #         cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
        #         cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
        #         cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
        #         affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
        #         Hr=H_r, Wr=W_r,
        #         depth_bins=self.depth_bins,
        #         ini_query=multi_level_ranges_curr_init[lvl].detach()  # 传统投影结果作为初始查询
        #     )
        #     multi_level_ranges_prev.append(range_prev)  # [B,C,Hr,Wr]
        #     multi_level_ranges_curr.append(range_curr)
                # ====================== Teacher/Student 查询初始化（统一管理） ======================
        # 训练期：alpha>0 时启用 DA teacher 初始化；推理期/alpha=0 时只用 student
        alpha = float(self.anneal_alpha.item()) if self.training else 0.0
        use_teacher = (alpha > 0.0)

        # ---------- (A) corr 路：构造 student/teacher 的 ini_query ----------
        # Student：各相机的 corr（2ch -> 128ch）先编码，再按宽拼接
        corr_encoded_list = [self.conv_corr_rvt_in(c) for c in corr_list]  # list[V]: [B,C,Hc,Wc]
        H_r_corr = corr_encoded_list[0].shape[2]
        W_r_corr = corr_encoded_list[0].shape[3] * V

        # ★ 学生端：几何无缝聚合（geom_bootstrap），不再拼接、不再 0-init
        q_student_corr = self.rvt_corr.geom_bootstrap(
            feats_by_cam=corr_encoded_list,
            cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
            cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
            cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
            affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
            Hr=H_r_corr, Wr=W_r_corr, depth_bins=self.depth_bins
        )

        if use_teacher:
            with torch.no_grad():
                corr_range_teacher = self.project_views_to_range(
                    corr_list, proj_pix_curr, H_img=H_img, W_img=W_img
                )  # [B, 2, Hr, Wr]
                corr_range_teacher = self.conv_corr_rvt_in(corr_range_teacher)  # [B,C,Hr,Wr]
        else:
            corr_range_teacher = None

        # ★ 退火：从 DA（teacher） → 几何无缝（student）
        ini_corr_query = (alpha * corr_range_teacher.detach() + (1.0 - alpha) * q_student_corr) \
                        if (use_teacher and corr_range_teacher is not None) else q_student_corr

        # 进入 RVT（corr 聚合）
        corr_range = self.rvt_corr(
            feats_by_cam=corr_encoded_list,   # RVT 采样来自几何，相机级特征仍按列表输入
            cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
            cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
            cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
            affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
            Hr=H_r_corr, Wr=W_r_corr,
            depth_bins=self.depth_bins,
            ini_query=ini_corr_query
        )

        # ---------- (B) 多尺度图像特征路：同样用 teacher/student 统一逻辑 ----------
        multi_level_ranges_prev, multi_level_ranges_curr = [], []

        for lvl in range(self.num_scales):
            # per-camera 特征列表（保持给 RVT 的 feats_by_cam 接口）
            prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
            curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]
            H_r = prev_feats_list[0].shape[2]
            W_r = prev_feats_list[0].shape[3] * V
            C_ = prev_feats_list[0].shape[1]

            # ★ 学生端改为 0-init
            # ★ 学生端：几何无缝聚合（prev / curr）
            q_student_prev = self.rvt_feat[lvl].geom_bootstrap(
                feats_by_cam=prev_feats_list,
                cam_K=[sensor_metas['prev'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r, depth_bins=self.depth_bins
            )
            q_student_curr = self.rvt_feat[lvl].geom_bootstrap(
                feats_by_cam=curr_feats_list,
                cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r, depth_bins=self.depth_bins
            )
            if use_teacher:
                scale = 2 ** (self.num_scales - 1 - lvl)
                proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :] if scale > 1 else proj_pix_prev
                proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :] if scale > 1 else proj_pix_curr
                with torch.no_grad():
                    t_prev = self.project_views_to_range(prev_feats_list, proj_prev_lvl, H_img=H_img, W_img=W_img)
                    t_curr = self.project_views_to_range(curr_feats_list, proj_curr_lvl, H_img=H_img, W_img=W_img)
            else:
                t_prev = t_curr = None

            # ★ 退火：从 DA → 0-init
            ini_prev = (alpha * t_prev.detach() + (1.0 - alpha) * q_student_prev) if (use_teacher and t_prev is not None) else q_student_prev
            ini_curr = (alpha * t_curr.detach() + (1.0 - alpha) * q_student_curr) if (use_teacher and t_curr is not None) else q_student_curr
            # RVT 聚合（同一摄像头集合作为 levels；几何采样依赖 K/R/t/affine，不依赖 proj_pix_*）
            range_prev = self.rvt_feat[lvl](
                feats_by_cam=prev_feats_list,
                cam_K=[sensor_metas['prev'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r, depth_bins=self.depth_bins,
                ini_query=ini_prev
            )
            range_curr = self.rvt_feat[lvl](
                feats_by_cam=curr_feats_list,
                cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r, depth_bins=self.depth_bins,
                ini_query=ini_curr
            )
            multi_level_ranges_prev.append(range_prev)
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
    
    # === 便捷接口：设置/读取退火系数 ===
    @torch.no_grad()
    def set_anneal_alpha(self, alpha: float):
        alpha = float(alpha)
        self.anneal_alpha.fill_(max(0.0, min(1.0, alpha)))

    def get_anneal_alpha(self) -> float:
        return float(self.anneal_alpha.item())

