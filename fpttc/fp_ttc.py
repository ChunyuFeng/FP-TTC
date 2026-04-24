import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from utils.loss import get_loss_scale_map, get_loss_scale_gradient_map, get_loss_risk_score_map

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

RVT_QUERY_INIT_MODES = ('gvb', 'concatfeat', 'zero')


class FpTTC(nn.Module):
    def __init__(self,
                 num_scales             = 2,
                 feature_channels       = 128,
                 upsample_factor        = 4,
                 num_head               = 1,
                 ffn_dim_expansion      = 4,
                 num_transformer_layers = 6,
                 reg_refine             = False,
                 no_depth               = False,
                 rvt_query_init         = 'gvb',
                 ):
        super(FpTTC, self).__init__()
        self.num_scales = num_scales
        self.no_depth = no_depth
        if rvt_query_init not in RVT_QUERY_INIT_MODES:
            raise ValueError(
                f"Unsupported rvt_query_init={rvt_query_init!r}; "
                f"expected one of {RVT_QUERY_INIT_MODES}"
            )
        self.rvt_query_init = rvt_query_init

        self.camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

        backbone_in_ch = 3 if no_depth else 4
        self.cnet = CNNEncoder(output_dim        = feature_channels, 
                               num_output_scales = num_scales,
                               in_channels       = backbone_in_ch)
        
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
        
    def _concat_init_query(self, feats_by_cam, Hr, Wr):
        query = torch.cat(feats_by_cam, dim=-1).contiguous()
        if query.shape[-2:] != (Hr, Wr):
            query = _safe_bilinear(query, size=(Hr, Wr), align_corners=True)
        return query

    def _rvt_query_kwargs(self, feats_by_cam, Hr, Wr):
        if self.rvt_query_init == 'gvb':
            return {'ini_query': None, 'use_geom_bootstrap': True}
        if self.rvt_query_init == 'zero':
            return {'ini_query': None, 'use_geom_bootstrap': False}
        if self.rvt_query_init == 'concatfeat':
            return {
                'ini_query': self._concat_init_query(feats_by_cam, Hr, Wr),
                'use_geom_bootstrap': False,
            }
        raise RuntimeError(f'Unhandled RVT query init mode: {self.rvt_query_init}')


    def forward(
        self,
        img_prev, img_curr,
        depth_prev=None, depth_curr=None,
        proj_pix_prev=None, proj_pix_curr=None,
        sensor_metas=None,
        attn_type='swin',
        attn_splits_list=None,
        corr_radius_list=None,
        prop_radius_list=None,
        num_reg_refine=1,
        scale_only=True,
        no_depth=None,
        use_teacher_distill=False,
    ):
        if no_depth is None:
            no_depth = self.no_depth

        # ----- 预处理 -----
        img0, img1 = normalize_img(img_prev, img_curr)         # [B,V,3,H,W] -> normed
        if no_depth:
            # depth-free: backbone 直接吃 3ch RGB
            B, V, C, H_img, W_img = img0.shape
            x0 = img0.view(B*V, C, H_img, W_img)
            x1 = img1.view(B*V, C, H_img, W_img)
        else:
            rgbd0 = torch.cat([img0, depth_prev], dim=2)           # [B,V,4,H,W]
            rgbd1 = torch.cat([img1, depth_curr], dim=2)           # [B,V,4,H,W]
            B, V, C, H_img, W_img = rgbd0.shape
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

        # ====== Teacher 投影（仅训练蒸馏时使用，推理时跳过） ======
        teacher_corr_range = None
        teacher_ranges_prev = None
        teacher_ranges_curr = None
        has_teacher = (
            use_teacher_distill
            and self.training
            and (proj_pix_prev is not None)
            and (proj_pix_curr is not None)
        )

        if has_teacher:
            with torch.no_grad():
                corr_range_init_teacher = self.project_views_to_range(
                    corr_list, proj_pix_curr, H_img=H_img, W_img=W_img
                )
                teacher_corr_range = self.conv_corr_rvt_in(corr_range_init_teacher)

                teacher_ranges_prev, teacher_ranges_curr = [], []
                for lvl in range(self.num_scales):
                    scale = 2 ** (self.num_scales - 1 - lvl)
                    if scale > 1:
                        proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
                        proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
                    else:
                        proj_prev_lvl = proj_pix_prev
                        proj_curr_lvl = proj_pix_curr

                    prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
                    curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]

                    range_prev_t = self.project_views_to_range(prev_feats_list, proj_prev_lvl, H_img=H_img, W_img=W_img)
                    range_curr_t = self.project_views_to_range(curr_feats_list, proj_curr_lvl, H_img=H_img, W_img=W_img)

                    teacher_ranges_prev.append(range_prev_t)
                    teacher_ranges_curr.append(range_curr_t)

        # ====== 基于 RVT (student) 将多视角特征图聚合到 Range View ======
        # corr -> range（当前帧）
        corr_encoded_list = [self.conv_corr_rvt_in(c) for c in corr_list]  # list[V] of [B, Ccorr, Hc, Wc]
        H_r = corr_encoded_list[0].shape[2]
        W_r = corr_encoded_list[0].shape[3] * V
        corr_query_kwargs = self._rvt_query_kwargs(corr_encoded_list, H_r, W_r)

        corr_range = self.rvt_corr(
            feats_by_cam=corr_encoded_list,
            cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
            cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
            cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
            affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
            Hr=H_r, Wr=W_r,
            depth_bins=self.depth_bins,
            **corr_query_kwargs,
        )

        # 多尺度特征 -> range（前/当前帧）
        multi_level_ranges_prev, multi_level_ranges_curr = [], []
        for lvl in range(self.num_scales):
            prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
            curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]
            H_r = prev_feats_list[0].shape[2]
            W_r = prev_feats_list[0].shape[3] * V
            prev_query_kwargs = self._rvt_query_kwargs(prev_feats_list, H_r, W_r)
            curr_query_kwargs = self._rvt_query_kwargs(curr_feats_list, H_r, W_r)

            range_prev = self.rvt_feat[lvl](
                feats_by_cam=prev_feats_list,
                cam_K=[sensor_metas['prev'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r,
                depth_bins=self.depth_bins,
                **prev_query_kwargs,
            )
            range_curr = self.rvt_feat[lvl](
                feats_by_cam=curr_feats_list,
                cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r,
                depth_bins=self.depth_bins,
                **curr_query_kwargs,
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

        risk_score = None
        if not scale_only:
            corr_encoded = self.conv_corr_risk(corr_range)
            initial_risk = corr_encoded[:, :1]
            corr_encoded = corr_encoded[:, 1:]
            risk_score = self.risk_net(
                corr_encoded, multi_level_ranges_prev, multi_level_ranges_curr, initial_risk
            )

        # 打包 teacher targets 用于蒸馏 loss
        teacher_targets = None
        if has_teacher:
            teacher_targets = {
                'corr_range': teacher_corr_range,
                'ranges_prev': teacher_ranges_prev,
                'ranges_curr': teacher_ranges_curr,
            }

        return scales, risk_score, corr_range, multi_level_ranges_prev, multi_level_ranges_curr, teacher_targets


    def forward_with_loss( 
            self,
            img_prev,                      # Tensor[B, V, 3, H, W]
            img_curr,                      # Tensor[B, V, 3, H, W]
            depth_prev=None,               # Tensor[B, V, 1, H, W] or None
            depth_curr=None,               # Tensor[B, V, 1, H, W] or None
            proj_pix_prev=None,
            proj_pix_curr=None,
            gt_scale_map_with_mask=None,   # Tensor[B, 2, H_sph, W_sph]
            gt_risk_score_map_with_mask=None,
            sensor_metas=None,
            attn_type='swin',
            attn_splits_list=None,
            corr_radius_list=None,
            prop_radius_list=None,
            num_reg_refine=1,
            scale_only=True,
            no_depth=None,
            use_teacher_distill=False,
            lambda_feat_distill=1.0,
            lambda_corr_distill=1.0,
            loss_weight_alpha=0.0,
            edge_loss_weight=0.0,
        ):

        scales, risks, corr_range, ranges_prev, ranges_curr, teacher_targets = self.forward(
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
            no_depth         = no_depth,
            use_teacher_distill = use_teacher_distill,
        )

        # ===== task loss =====
        loss_s, loss_r = None, None
        if scale_only:
            loss_s = get_loss_scale_map(
                scales,
                gt_scale_map_with_mask,
                loss_weight_alpha=loss_weight_alpha,
            )
        else:
            loss_r = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)

        loss_edge = torch.tensor(0.0, device=img_prev.device)
        if scale_only and edge_loss_weight > 0:
            loss_edge = get_loss_scale_gradient_map(scales, gt_scale_map_with_mask)

        # ===== distillation loss (feat + corr, 分别加权) =====
        L_feat_distill = torch.tensor(0.0, device=img_prev.device)
        L_corr_distill = torch.tensor(0.0, device=img_prev.device)

        if teacher_targets is not None:
            t_corr = teacher_targets['corr_range']
            L_corr_distill = (corr_range - t_corr).abs().mean()

            for lvl in range(self.num_scales):
                L_feat_distill = L_feat_distill + (ranges_prev[lvl] - teacher_targets['ranges_prev'][lvl]).abs().mean()
                L_feat_distill = L_feat_distill + (ranges_curr[lvl] - teacher_targets['ranges_curr'][lvl]).abs().mean()

        total_distill = lambda_feat_distill * L_feat_distill + lambda_corr_distill * L_corr_distill

        loss_total = (loss_s if scale_only else loss_r) + total_distill + edge_loss_weight * loss_edge
        loss_task = loss_s if scale_only else loss_r
        zero = torch.tensor(0.0, device=img_prev.device)

        loss_dict = {
            'total': loss_total,
            'task': loss_task,
            'scale': loss_s if loss_s is not None else zero,
            'risk': loss_r if loss_r is not None else zero,
            'feat_distill': L_feat_distill,
            'corr_distill': L_corr_distill,
            'edge': loss_edge,
        }

        if scale_only:
            return scales, None, loss_total, None, loss_dict
        else:
            return None, risks, None, loss_total, loss_dict

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

        valid = cam_idx >= 0
        cam_idx_safe = cam_idx.clamp(0, V - 1)
        u_orig_safe = torch.where(valid, u_orig, torch.zeros_like(u_orig))
        v_orig_safe = torch.where(valid, v_orig, torch.zeros_like(v_orig))

        u_feat = (u_orig_safe * s_u).long().clamp(0, W_feat-1) # [B, N]
        v_feat = (v_orig_safe * s_v).long().clamp(0, H_feat-1) # [B, N]

        # 4) 计算扁平化后的批次＋视角索引
        #    batch_idx ∈ [0..B) 重复 N 次，拼接 cam_idx → [B*N]
        batch_idx = torch.arange(B, device=cam_idx.device)\
                        .unsqueeze(1).repeat(1, H_r*W_r)\
                        .reshape(-1)
        view_idx  = batch_idx * V + cam_idx_safe.reshape(-1)   # [B*N]

        # 5) 计算在 Hf*Wf 上的线性化像素
        pix_idx   = (v_feat * W_feat + u_feat).reshape(-1)  # [B*N]

        # 6) 一次性 gather
        #    feats_flat[view_idx, :, pix_idx] → [B*N, C]
        selected = feats_flat[view_idx, :, pix_idx]
        valid_flat = valid.reshape(-1)
        if not torch.all(valid_flat):
            selected = selected.clone()
            selected[~valid_flat] = 0

        # 7) 重塑回 [B, C, H_r, W_r]
        range_feat = selected.view(B, H_r*W_r, C) \
                            .permute(0,2,1) \
                            .reshape(B, C, H_r, W_r)
        return range_feat
