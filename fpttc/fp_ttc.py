import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from utils.loss import (
    get_loss_scale_map,
    get_loss_scale_gradient_map,
    get_loss_risk_score_map,
    get_loss_depth_distribution,
)

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet
from .da_backbone import DepthAnythingBackbone
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
                 reg_refine             = False,
                 no_depth               = False,
                 activation_checkpointing = False,
                 backbone_type          = 'cnn',
                 da_pretrained_ckpt     = 'pretrained/depth_anything_v2_metric_vkitti_vits.pth',
                 da_branch_mode         = 'off',
                 aggregation_mode       = 'rvt',
                 image_size             = (160, 320),
                 use_adapter_alignment_teacher = False,
                 ):
        super(FpTTC, self).__init__()
        self.num_scales = num_scales
        self.no_depth = no_depth
        self.backbone_type = backbone_type
        self.da_branch_mode = da_branch_mode
        self.aggregation_mode = aggregation_mode
        self.image_size = tuple(int(v) for v in image_size)
        self.use_adapter_alignment_teacher = bool(use_adapter_alignment_teacher)

        if self.aggregation_mode not in ('rvt', 'hardproj_cache'):
            raise ValueError(f"Unsupported aggregation_mode: {self.aggregation_mode}")
        if self.da_branch_mode not in ('off', 'log'):
            raise ValueError(f"Unsupported da_branch_mode: {self.da_branch_mode}")

        self.camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

        backbone_in_ch = 3 if no_depth else 4
        self.da_branch = None
        if backbone_type == 'da_vits':
            if backbone_in_ch != 3:
                raise ValueError("DepthAnything backbone only supports RGB input.")
            self.cnet = DepthAnythingBackbone(
                output_dim=feature_channels,
                num_output_scales=num_scales,
                pretrained_ckpt=da_pretrained_ckpt,
                image_size=self.image_size,
                metric_max_depth=80.0,
            )
        elif backbone_type == 'dual_backbone':
            if not no_depth:
                raise ValueError("dual_backbone only supports RGB-only main-path input.")
            if self.aggregation_mode != 'hardproj_cache':
                raise ValueError("dual_backbone currently only supports aggregation_mode='hardproj_cache'.")
            if self.use_adapter_alignment_teacher:
                raise ValueError("dual_backbone does not support use_adapter_alignment_teacher.")
            self.cnet = CNNEncoder(output_dim=feature_channels,
                                   num_output_scales=num_scales,
                                   in_channels=3)
            self.da_branch = DepthAnythingBackbone(
                output_dim=feature_channels,
                num_output_scales=num_scales,
                pretrained_ckpt=da_pretrained_ckpt,
                image_size=self.image_size,
                metric_max_depth=80.0,
            )
            for p in self.da_branch.parameters():
                p.requires_grad = False
            self.da_branch.eval()
        elif backbone_type == 'cnn':
            self.cnet = CNNEncoder(output_dim        = feature_channels,
                                   num_output_scales = num_scales,
                                   in_channels       = backbone_in_ch)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        if self.backbone_type == 'da_vits' and self.use_adapter_alignment_teacher:
            self.cnet_teacher = CNNEncoder(
                output_dim=feature_channels,
                num_output_scales=num_scales,
                in_channels=4,
            )
            for p in self.cnet_teacher.parameters():
                p.requires_grad = False
            self.cnet_teacher.eval()
        else:
            self.cnet_teacher = None
        
        self.featnet = FeatureNet(num_scales             = num_scales,
                                  feature_channels       = feature_channels,
                                  num_head               = num_head, 
                                  ffn_dim_expansion      = ffn_dim_expansion,
                                  num_transformer_layers = num_transformer_layers,
                                  activation_checkpointing = activation_checkpointing)   
         
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
        
        K_bins = 16
        num_views = len(self.camera_channels)
        self.register_buffer(
            "depth_bins",
            torch.logspace(math.log10(1.0), math.log10(40.0), K_bins)
        )
        self.depth_prior_alpha = 0.7
        if backbone_type == 'cnn':
            self.depth_head = nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1, bias=True),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(feature_channels // 2, K_bins, 1, bias=True),
            )
        else:
            self.depth_head = None
        
        if self.aggregation_mode == 'rvt':
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
                    fov_down=-15.0,
                    activation_checkpointing=activation_checkpointing,
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
                fov_down=-15.0,
                activation_checkpointing=activation_checkpointing,
            )
        else:
            self.rvt_feat = None
            self.rvt_corr = None
    
    def _predict_depth_logits(self, feat_bv):
        B, V, C, H, W = feat_bv.shape
        logits = self.depth_head(feat_bv.view(B * V, C, H, W))
        return logits.view(B, V, -1, H, W)

    def _extract_teacher_features(self, rgbd_prev, rgbd_curr):
        if self.cnet_teacher is None:
            return None, None

        if rgbd_prev.dim() != 5 or rgbd_curr.dim() != 5:
            raise ValueError("Teacher RGBD inputs must have shape [B,V,4,H,W].")

        B, V = rgbd_prev.shape[:2]
        x_prev = rgbd_prev.view(B * V, rgbd_prev.shape[2], rgbd_prev.shape[3], rgbd_prev.shape[4])
        x_curr = rgbd_curr.view(B * V, rgbd_curr.shape[2], rgbd_curr.shape[3], rgbd_curr.shape[4])
        x = torch.cat([x_prev, x_curr], dim=0)
        with torch.no_grad():
            feats = self.cnet_teacher(x, branch=None)[::-1]

        prev_lvls, curr_lvls = [], []
        for feat in feats:
            prev_feat, curr_feat = torch.chunk(feat, 2, dim=0)
            prev_lvls.append(prev_feat.view(B, V, prev_feat.shape[1], prev_feat.shape[2], prev_feat.shape[3]))
            curr_lvls.append(curr_feat.view(B, V, curr_feat.shape[1], curr_feat.shape[2], curr_feat.shape[3]))
        return prev_lvls, curr_lvls

    @staticmethod
    def _feature_align_term(student_feat, teacher_feat, eps=1e-6):
        smooth = F.smooth_l1_loss(student_feat, teacher_feat)
        student_mean = student_feat.mean(dim=(-2, -1))
        teacher_mean = teacher_feat.mean(dim=(-2, -1))
        student_std = student_feat.var(dim=(-2, -1), unbiased=False).add(eps).sqrt()
        teacher_std = teacher_feat.var(dim=(-2, -1), unbiased=False).add(eps).sqrt()
        mean_loss = (student_mean - teacher_mean).abs().mean()
        std_loss = (student_std - teacher_std).abs().mean()
        return smooth + 0.1 * mean_loss + 0.1 * std_loss

    def _compute_alignment_loss(self, student_prev_lvls, student_curr_lvls, teacher_prev_lvls, teacher_curr_lvls):
        if teacher_prev_lvls is None or teacher_curr_lvls is None:
            return student_prev_lvls[0].sum() * 0.0

        loss = student_prev_lvls[0].sum() * 0.0
        for lvl in range(self.num_scales):
            prev_loss = self._feature_align_term(student_prev_lvls[lvl], teacher_prev_lvls[lvl])
            curr_loss = self._feature_align_term(student_curr_lvls[lvl], teacher_curr_lvls[lvl])
            loss = loss + 0.5 * (prev_loss + curr_loss)
        return loss

    def _depth_prob_by_cam(self, depth_logits, target_hw):
        B, V, K, H, W = depth_logits.shape
        logits = depth_logits.view(B * V, K, H, W)
        if (H, W) != target_hw:
            logits = _safe_bilinear(logits, size=target_hw, align_corners=True)
        probs = F.softmax(logits, dim=1).view(B, V, K, target_hw[0], target_hw[1])
        return [probs[:, v] for v in range(V)]

    def _depth_map_to_probs(self, depth_map_bv, target_hw=None, eps=1e-6):
        if depth_map_bv.dim() != 5 or depth_map_bv.shape[2] != 1:
            raise ValueError(f"Expected depth map shape [B,V,1,H,W], got {tuple(depth_map_bv.shape)}")

        B, V, _, H0, W0 = depth_map_bv.shape
        depth = depth_map_bv.view(B * V, 1, H0, W0)
        if target_hw is not None and (H0, W0) != tuple(target_hw):
            depth = _safe_bilinear(depth, size=target_hw, align_corners=True)
        H, W = depth.shape[-2:]
        depth = depth.view(B, V, H, W)

        device = depth.device
        dtype = depth.dtype
        bins = self.depth_bins.to(device=device, dtype=dtype)
        log_bins = torch.log(bins.clamp_min(eps))
        depth = depth.clamp(min=float(bins[0].item()), max=float(bins[-1].item()))
        log_depth = torch.log(depth.clamp_min(eps))

        idx_hi = torch.searchsorted(log_bins, log_depth, right=True)
        idx_hi = idx_hi.clamp(1, bins.numel() - 1)
        idx_lo = idx_hi - 1

        log_lo = log_bins[idx_lo]
        log_hi = log_bins[idx_hi]
        denom = (log_hi - log_lo).clamp_min(eps)
        w_hi = ((log_depth - log_lo) / denom).clamp(0.0, 1.0)
        w_lo = 1.0 - w_hi

        probs = torch.zeros(B, V, bins.numel(), H, W, device=device, dtype=dtype)
        probs.scatter_(2, idx_lo.unsqueeze(2), w_lo.unsqueeze(2))
        probs.scatter_add_(2, idx_hi.unsqueeze(2), w_hi.unsqueeze(2))

        low_mask = log_depth <= log_bins[0]
        high_mask = log_depth >= log_bins[-1]
        if low_mask.any():
            probs = probs.masked_fill(low_mask.unsqueeze(2), 0.0)
            probs[:, :, 0:1] = torch.where(
                low_mask.unsqueeze(2),
                torch.ones_like(probs[:, :, 0:1]),
                probs[:, :, 0:1],
            )
        if high_mask.any():
            probs = probs.masked_fill(high_mask.unsqueeze(2), 0.0)
            probs[:, :, -1:] = torch.where(
                high_mask.unsqueeze(2),
                torch.ones_like(probs[:, :, -1:]),
                probs[:, :, -1:],
            )
        return probs

    def _depth_prob_by_cam_from_depth_map(self, depth_map_bv, target_hw):
        probs = self._depth_map_to_probs(depth_map_bv, target_hw=target_hw)
        return [probs[:, v] for v in range(probs.shape[1])]

    def _depth_stats_from_probs(self, depth_probs, depth_gt=None, eps=1e-6):
        B, V, K, H, W = depth_probs.shape
        device = depth_probs.device
        dtype = depth_probs.dtype
        probs = depth_probs.clamp_min(eps)

        if depth_gt is not None:
            gt = depth_gt.view(B * V, 1, depth_gt.shape[-2], depth_gt.shape[-1])
            gt = F.interpolate(gt, size=(H, W), mode='bilinear', align_corners=True)
            gt = gt.view(B, V, H, W)
            valid = gt > 0
        else:
            valid = torch.ones(B, V, H, W, device=device, dtype=torch.bool)

        expected = (depth_probs * self.depth_bins.view(1, 1, K, 1, 1).to(device=device, dtype=dtype)).sum(dim=2)
        valid_count = valid.sum()
        if valid_count == 0:
            zero = expected.sum() * 0.0
            return {
                'entropy': zero,
                'valid_ratio': zero,
                'top1_prob': zero,
                'top1_margin': zero,
                'expected': expected.detach(),
            }

        top2_probs = torch.topk(depth_probs, k=min(2, K), dim=2).values
        entropy = -(probs * probs.log()).sum(dim=2)
        top1_prob = top2_probs[:, :, 0]
        if K > 1:
            top1_margin = top2_probs[:, :, 0] - top2_probs[:, :, 1]
        else:
            top1_margin = top1_prob

        return {
            'entropy': entropy[valid].mean(),
            'valid_ratio': valid.float().mean(),
            'top1_prob': top1_prob[valid].mean(),
            'top1_margin': top1_margin[valid].mean(),
            'expected': expected.detach(),
        }

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
        use_internal_depth_guidance=False,
        use_adapter_alignment_teacher=False,
        depth_selection_mode='hard_topk',
        bootstrap_topk=4,
        attn_topk=8,
        bootstrap_prior_scale=2.0,
        attn_prior_scale=2.0,
        depth_prior_eps=1e-6,
    ):
        if no_depth is None:
            no_depth = self.no_depth

        # ----- 预处理 -----
        img0, img1 = normalize_img(img_prev, img_curr)         # [B,V,3,H,W] -> normed
        teacher_prev_lvls = None
        teacher_curr_lvls = None
        if no_depth:
            # depth-free: backbone 直接吃 3ch RGB
            B, V, C, H_img, W_img = img0.shape
            x0 = img0.view(B*V, C, H_img, W_img)
            x1 = img1.view(B*V, C, H_img, W_img)
            if self.cnet_teacher is not None and use_adapter_alignment_teacher:
                if depth_prev is None or depth_curr is None:
                    raise ValueError("Adapter alignment teacher requires depth_prev/depth_curr for RGBD teacher inputs.")
                rgbd0_teacher = torch.cat([img0, depth_prev], dim=2)
                rgbd1_teacher = torch.cat([img1, depth_curr], dim=2)
                teacher_prev_lvls, teacher_curr_lvls = self._extract_teacher_features(rgbd0_teacher, rgbd1_teacher)
        else:
            rgbd0 = torch.cat([img0, depth_prev], dim=2)           # [B,V,4,H,W]
            rgbd1 = torch.cat([img1, depth_curr], dim=2)           # [B,V,4,H,W]
            B, V, C, H_img, W_img = rgbd0.shape
            x0 = rgbd0.view(B*V, C, H_img, W_img)
            x1 = rgbd1.view(B*V, C, H_img, W_img)
        # extract_feature 会在内部 cat([x0, x1], 0) -> backbone -> 多尺度 -> chunk 回来
        prev_lvls_flat, curr_lvls_flat, depth_map_prev_flat, depth_map_curr_flat = self.extract_feature(
            x0, x1, branch=None, return_depth=True
        )  # list[T][B*V,C,Hs,Ws]

        # reshape 回 [B,V,C,Hs,Ws]
        prev_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in prev_lvls_flat]
        curr_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in curr_lvls_flat]
        depth_map_prev = None
        depth_map_curr = None
        if depth_map_prev_flat is not None:
            depth_map_prev = depth_map_prev_flat.view(B, V, 1, depth_map_prev_flat.shape[-2], depth_map_prev_flat.shape[-1])
        if depth_map_curr_flat is not None:
            depth_map_curr = depth_map_curr_flat.view(B, V, 1, depth_map_curr_flat.shape[-2], depth_map_curr_flat.shape[-1])
        depth_logits_prev = None
        depth_logits_curr = None
        if self.backbone_type == 'cnn' and use_internal_depth_guidance:
            depth_logits_prev = self._predict_depth_logits(prev_lvls[-1])
            depth_logits_curr = self._predict_depth_logits(curr_lvls[-1])

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
            self.aggregation_mode == 'rvt'
            and use_teacher_distill
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

        # ====== 聚合到 Range View ======
        if self.aggregation_mode == 'hardproj_cache':
            if proj_pix_prev is None or proj_pix_curr is None:
                raise ValueError("aggregation_mode='hardproj_cache' requires precomputed proj_pix_prev/proj_pix_curr.")

            corr_range_init = self.project_views_to_range(
                corr_list, proj_pix_curr, H_img=H_img, W_img=W_img
            )
            corr_range = self.conv_corr_rvt_in(corr_range_init)

            multi_level_ranges_prev, multi_level_ranges_curr = [], []
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

                range_prev = self.project_views_to_range(prev_feats_list, proj_prev_lvl, H_img=H_img, W_img=W_img)
                range_curr = self.project_views_to_range(curr_feats_list, proj_curr_lvl, H_img=H_img, W_img=W_img)
                multi_level_ranges_prev.append(range_prev)
                multi_level_ranges_curr.append(range_curr)

            selection_summary = {
                'bootstrap_active_candidates': torch.tensor(0.0, device=img_prev.device),
                'attn_active_candidates': torch.tensor(0.0, device=img_prev.device),
                'depth_selection_sparsity': torch.tensor(0.0, device=img_prev.device),
            }
        else:
            # ====== 基于 RVT (student) 将多视角特征图聚合到 Range View ======
            # corr -> range（当前帧）
            corr_encoded_list = [self.conv_corr_rvt_in(c) for c in corr_list]  # list[V] of [B, Ccorr, Hc, Wc]
            H_r = corr_encoded_list[0].shape[2]
            W_r = corr_encoded_list[0].shape[3] * V
            corr_depth_prob = None
            if self.backbone_type == 'da_vits' and depth_map_curr is not None:
                corr_depth_prob = self._depth_prob_by_cam_from_depth_map(
                    depth_map_curr, corr_encoded_list[0].shape[-2:]
                )
            elif depth_logits_curr is not None:
                corr_depth_prob = self._depth_prob_by_cam(depth_logits_curr, corr_encoded_list[0].shape[-2:])

            corr_range, corr_stats = self.rvt_corr(
                feats_by_cam=corr_encoded_list,
                cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
                cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r,
                depth_bins=self.depth_bins,
                ini_query=None,
                use_geom_bootstrap=True,
                depth_prob_by_cam=corr_depth_prob,
                depth_prior_alpha=self.depth_prior_alpha,
                depth_selection_mode=depth_selection_mode,
                bootstrap_topk=bootstrap_topk,
                attn_topk=attn_topk,
                bootstrap_prior_scale=bootstrap_prior_scale,
                attn_prior_scale=attn_prior_scale,
                depth_prior_eps=depth_prior_eps,
            )

            # 多尺度特征 -> range（前/当前帧）
            multi_level_ranges_prev, multi_level_ranges_curr = [], []
            selection_stats = [corr_stats]
            for lvl in range(self.num_scales):
                prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
                curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]
                H_r = prev_feats_list[0].shape[2]
                W_r = prev_feats_list[0].shape[3] * V
                prev_depth_prob = None
                curr_depth_prob = None
                if self.backbone_type == 'da_vits' and depth_map_prev is not None:
                    prev_depth_prob = self._depth_prob_by_cam_from_depth_map(
                        depth_map_prev, prev_feats_list[0].shape[-2:]
                    )
                elif depth_logits_prev is not None:
                    prev_depth_prob = self._depth_prob_by_cam(depth_logits_prev, prev_feats_list[0].shape[-2:])
                if self.backbone_type == 'da_vits' and depth_map_curr is not None:
                    curr_depth_prob = self._depth_prob_by_cam_from_depth_map(
                        depth_map_curr, curr_feats_list[0].shape[-2:]
                    )
                elif depth_logits_curr is not None:
                    curr_depth_prob = self._depth_prob_by_cam(depth_logits_curr, curr_feats_list[0].shape[-2:])

                range_prev, prev_stats = self.rvt_feat[lvl](
                    feats_by_cam=prev_feats_list,
                    cam_K=[sensor_metas['prev'][cam]['K']     for cam in self.camera_channels],
                    cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
                    cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
                    affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
                    Hr=H_r, Wr=W_r,
                    depth_bins=self.depth_bins,
                    ini_query=None,
                    use_geom_bootstrap=True,
                    depth_prob_by_cam=prev_depth_prob,
                    depth_prior_alpha=self.depth_prior_alpha,
                    depth_selection_mode=depth_selection_mode,
                    bootstrap_topk=bootstrap_topk,
                    attn_topk=attn_topk,
                    bootstrap_prior_scale=bootstrap_prior_scale,
                    attn_prior_scale=attn_prior_scale,
                    depth_prior_eps=depth_prior_eps,
                )
                range_curr, curr_stats = self.rvt_feat[lvl](
                    feats_by_cam=curr_feats_list,
                    cam_K=[sensor_metas['curr'][cam]['K']     for cam in self.camera_channels],
                    cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                    cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                    affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                    Hr=H_r, Wr=W_r,
                    depth_bins=self.depth_bins,
                    ini_query=None,
                    use_geom_bootstrap=True,
                    depth_prob_by_cam=curr_depth_prob,
                    depth_prior_alpha=self.depth_prior_alpha,
                    depth_selection_mode=depth_selection_mode,
                    bootstrap_topk=bootstrap_topk,
                    attn_topk=attn_topk,
                    bootstrap_prior_scale=bootstrap_prior_scale,
                    attn_prior_scale=attn_prior_scale,
                    depth_prior_eps=depth_prior_eps,
                )
                multi_level_ranges_prev.append(range_prev)
                multi_level_ranges_curr.append(range_curr)
                selection_stats.extend([prev_stats, curr_stats])

            def _mean_stat(key):
                values = [s[key] for s in selection_stats if key in s]
                if not values:
                    return torch.tensor(0.0, device=img_prev.device)
                return torch.stack([v if torch.is_tensor(v) else torch.tensor(v, device=img_prev.device) for v in values]).mean()

            selection_summary = {
                'bootstrap_active_candidates': _mean_stat('bootstrap_active_candidates'),
                'attn_active_candidates': _mean_stat('attn_active_candidates'),
                'depth_selection_sparsity': _mean_stat('depth_selection_sparsity'),
            }

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
        aux_outputs = {
            'depth_logits_prev': depth_logits_prev,
            'depth_logits_curr': depth_logits_curr,
            'depth_map_prev': depth_map_prev,
            'depth_map_curr': depth_map_curr,
            'student_prev_lvls': prev_lvls,
            'student_curr_lvls': curr_lvls,
            'teacher_prev_lvls': teacher_prev_lvls,
            'teacher_curr_lvls': teacher_curr_lvls,
            'feature_monitor_prev': prev_lvls[-1].detach() if (depth_logits_prev is not None or depth_map_prev is not None) else None,
            'feature_monitor_curr': curr_lvls[-1].detach() if (depth_logits_curr is not None or depth_map_curr is not None) else None,
            'selection_summary': selection_summary,
        }

        return scales, risk_score, corr_range, multi_level_ranges_prev, multi_level_ranges_curr, teacher_targets, aux_outputs


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
            use_internal_depth_guidance=False,
            use_adapter_alignment_teacher=False,
            adapter_align_weight=0.0,
            adapter_warmup_active=False,
            depth_loss_weight=0.0,
            depth_selection_mode='hard_topk',
            bootstrap_topk=4,
            attn_topk=8,
            bootstrap_prior_scale=2.0,
            attn_prior_scale=2.0,
            depth_prior_eps=1e-6,
            return_aux=False,
        ):

        scales, risks, corr_range, ranges_prev, ranges_curr, teacher_targets, aux_outputs = self.forward(
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
            use_internal_depth_guidance = use_internal_depth_guidance,
            use_adapter_alignment_teacher = use_adapter_alignment_teacher,
            depth_selection_mode = depth_selection_mode,
            bootstrap_topk = bootstrap_topk,
            attn_topk = attn_topk,
            bootstrap_prior_scale = bootstrap_prior_scale,
            attn_prior_scale = attn_prior_scale,
            depth_prior_eps = depth_prior_eps,
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
        loss_depth = torch.tensor(0.0, device=img_prev.device)
        depth_entropy = torch.tensor(0.0, device=img_prev.device)
        depth_valid_ratio = torch.tensor(0.0, device=img_prev.device)
        depth_top1_prob = torch.tensor(0.0, device=img_prev.device)
        depth_top1_margin = torch.tensor(0.0, device=img_prev.device)
        depth_expected_prev = None
        depth_expected_curr = None
        bootstrap_active_candidates = aux_outputs['selection_summary']['bootstrap_active_candidates']
        attn_active_candidates = aux_outputs['selection_summary']['attn_active_candidates']
        depth_selection_sparsity = aux_outputs['selection_summary']['depth_selection_sparsity']
        loss_align = torch.tensor(0.0, device=img_prev.device)

        if teacher_targets is not None:
            t_corr = teacher_targets['corr_range']
            L_corr_distill = (corr_range - t_corr).abs().mean()

            for lvl in range(self.num_scales):
                L_feat_distill = L_feat_distill + (ranges_prev[lvl] - teacher_targets['ranges_prev'][lvl]).abs().mean()
                L_feat_distill = L_feat_distill + (ranges_curr[lvl] - teacher_targets['ranges_curr'][lvl]).abs().mean()

        total_distill = lambda_feat_distill * L_feat_distill + lambda_corr_distill * L_corr_distill
        if (
            self.backbone_type == 'dual_backbone'
            and self.da_branch_mode == 'log'
            and aux_outputs['depth_map_prev'] is not None
        ):
            depth_expected_prev = aux_outputs['depth_map_prev'].squeeze(2).detach()
            depth_expected_curr = aux_outputs['depth_map_curr'].squeeze(2).detach()
        elif self.aggregation_mode == 'rvt' and self.backbone_type == 'da_vits' and aux_outputs['depth_map_prev'] is not None:
            depth_probs_prev = self._depth_map_to_probs(aux_outputs['depth_map_prev'])
            depth_probs_curr = self._depth_map_to_probs(aux_outputs['depth_map_curr'])
            gt_prev = depth_prev if (depth_prev is not None and torch.any(depth_prev > 0)) else None
            gt_curr = depth_curr if (depth_curr is not None and torch.any(depth_curr > 0)) else None
            depth_stats_prev = self._depth_stats_from_probs(depth_probs_prev, gt_prev)
            depth_stats_curr = self._depth_stats_from_probs(depth_probs_curr, gt_curr)
            depth_entropy = 0.5 * (depth_stats_prev['entropy'] + depth_stats_curr['entropy'])
            depth_valid_ratio = 0.5 * (depth_stats_prev['valid_ratio'] + depth_stats_curr['valid_ratio'])
            depth_top1_prob = 0.5 * (depth_stats_prev['top1_prob'] + depth_stats_curr['top1_prob'])
            depth_top1_margin = 0.5 * (depth_stats_prev['top1_margin'] + depth_stats_curr['top1_margin'])
            depth_expected_prev = depth_stats_prev['expected']
            depth_expected_curr = depth_stats_curr['expected']
            loss_depth = img_prev.sum() * 0.0
        elif self.aggregation_mode == 'rvt' and use_internal_depth_guidance and aux_outputs['depth_logits_prev'] is not None:
            loss_depth_prev, depth_stats_prev = get_loss_depth_distribution(
                aux_outputs['depth_logits_prev'], depth_prev, self.depth_bins
            )
            loss_depth_curr, depth_stats_curr = get_loss_depth_distribution(
                aux_outputs['depth_logits_curr'], depth_curr, self.depth_bins
            )
            loss_depth = 0.5 * (loss_depth_prev + loss_depth_curr)
            depth_entropy = 0.5 * (depth_stats_prev['entropy'] + depth_stats_curr['entropy'])
            depth_valid_ratio = 0.5 * (depth_stats_prev['valid_ratio'] + depth_stats_curr['valid_ratio'])
            depth_top1_prob = 0.5 * (depth_stats_prev['top1_prob'] + depth_stats_curr['top1_prob'])
            depth_top1_margin = 0.5 * (depth_stats_prev['top1_margin'] + depth_stats_curr['top1_margin'])
            depth_expected_prev = depth_stats_prev['expected']
            depth_expected_curr = depth_stats_curr['expected']

        if use_adapter_alignment_teacher:
            teacher_prev_lvls = aux_outputs['teacher_prev_lvls']
            teacher_curr_lvls = aux_outputs['teacher_curr_lvls']
            if teacher_prev_lvls is None or teacher_curr_lvls is None:
                raise RuntimeError("Adapter alignment teacher is enabled, but teacher features were not produced.")
            loss_align = self._compute_alignment_loss(
                student_prev_lvls=aux_outputs.get('student_prev_lvls', None),
                student_curr_lvls=aux_outputs.get('student_curr_lvls', None),
                teacher_prev_lvls=teacher_prev_lvls,
                teacher_curr_lvls=teacher_curr_lvls,
            )

        task_loss = loss_s if scale_only else loss_r
        if use_adapter_alignment_teacher and adapter_warmup_active:
            loss_total = loss_align
        else:
            loss_total = task_loss + total_distill + edge_loss_weight * loss_edge + depth_loss_weight * loss_depth + adapter_align_weight * loss_align
        loss_task = task_loss
        zero = torch.tensor(0.0, device=img_prev.device)

        loss_dict = {
            'total': loss_total,
            'task': loss_task,
            'scale': loss_s if loss_s is not None else zero,
            'risk': loss_r if loss_r is not None else zero,
            'feat_distill': L_feat_distill,
            'corr_distill': L_corr_distill,
            'edge': loss_edge,
            'depth': loss_depth,
            'align': loss_align,
            'depth_entropy': depth_entropy,
            'depth_valid_ratio': depth_valid_ratio,
            'depth_top1_prob': depth_top1_prob,
            'depth_top1_margin': depth_top1_margin,
            'bootstrap_active_candidates': bootstrap_active_candidates,
            'attn_active_candidates': attn_active_candidates,
            'depth_selection_sparsity': depth_selection_sparsity,
        }
        if return_aux:
            loss_dict['aux'] = {
                'depth_expected_prev': depth_expected_prev,
                'depth_expected_curr': depth_expected_curr,
                'feature_monitor_prev': aux_outputs['feature_monitor_prev'],
                'feature_monitor_curr': aux_outputs['feature_monitor_curr'],
            }

        if scale_only:
            return scales, None, loss_total, None, loss_dict
        else:
            return None, risks, None, loss_total, loss_dict

    def extract_feature(self, im0, im1, branch, return_depth=False):
        x = torch.cat([im0, im1], dim=0)
        depth = None
        if self.backbone_type == 'da_vits':
            if return_depth:
                feats, aux = self.cnet(x, branch=branch, return_depth=True)
                depth = aux['depth']
            else:
                feats = self.cnet(x, branch=branch)
        elif self.backbone_type == 'dual_backbone':
            feats = self.cnet(x, branch=branch)
            if return_depth and self.da_branch_mode == 'log':
                self.da_branch.eval()
                with torch.no_grad():
                    _, aux = self.da_branch(x, branch=branch, return_depth=True)
                depth = aux['depth']
        else:
            feats = self.cnet(x, branch=branch)

        feats = feats[::-1]
        p0, p1 = [], []
        for f in feats:
            a, b = torch.chunk(f, 2, dim=0)
            p0.append(a); p1.append(b)
        if return_depth:
            if depth is None:
                return p0, p1, None, None
            d0, d1 = torch.chunk(depth, 2, dim=0)
            return p0, p1, d0, d1
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
