import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from utils.loss import get_loss_scale_map, get_loss_risk_score_map

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet

import numpy as np


class CorrEncoder(nn.Module):
    """Lightweight 1x1+3x3 encoder with in-place ReLUs.
    Used as a shared corr feature pre-encoder in the optimized pipeline.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.convc1 = nn.Conv2d(dim_in, 256, kernel_size=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.convc2 = nn.Conv2d(256, dim_out, kernel_size=3, padding=1, bias=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.convc1(x))
        x = self.act2(self.convc2(x))
        return x


class FpTTC(nn.Module):
    """
    Optimized FpTTC with the following changes:
      - Batch all views through CNNEncoder/FeatureNet/FlowNet (treat BV as batch)
      - Share corr encoder for scale/risk, with separate 1x1 heads for init maps
      - Faster range-projection kernel using single-dim gather (avoid 2D advanced indexing)
      - In-place activations where safe
      - Keep public API & outputs compatible with original forward/forward_with_loss
    """

    def __init__(
        self,
        num_scales: int = 2,
        feature_channels: int = 128,
        upsample_factor: int = 4,
        num_head: int = 1,
        ffn_dim_expansion: int = 4,
        num_transformer_layers: int = 6,
        reg_refine: bool = False,
    ):
        super().__init__()
        self.num_scales = num_scales

        # (kept for compatibility / external references)
        self.camera_channels = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
        ]

        # Backbone
        self.cnet = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Feature fusion & correlation
        self.featnet = FeatureNet(
            num_scales=num_scales,
            feature_channels=feature_channels,
            num_head=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            num_transformer_layers=num_transformer_layers,
        )
        self.corrnet = FlowNet(
            num_scales=num_scales,
            feature_channels=feature_channels,
            upsample_factor=upsample_factor,
            reg_refine=reg_refine,
        )

        # Shared corr encoder + small heads
        self.corr_enc_shared = CorrEncoder(dim_in=2, dim_out=feature_channels)
        self.init_scale = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.init_risk = nn.Conv2d(feature_channels, 1, kernel_size=1)

        # Heads
        self.scale_net = ScaleNet(
            num_scales=num_scales,
            feature_channels=feature_channels,
            upsample_factor=upsample_factor,
            num_head=4,
            scale_level=num_scales,
            reg_refine=reg_refine,
            head_type='scale',
        )
        self.risk_net = ScaleNet(
            num_scales=num_scales,
            feature_channels=feature_channels,
            upsample_factor=upsample_factor,
            num_head=4,
            scale_level=num_scales,
            reg_refine=reg_refine,
            head_type='risk',
        )

    # ---------------------------------------------------------------------
    # Feature extraction (ALL views at once)
    # ---------------------------------------------------------------------
    def extract_feature_all(self, rgbd0: torch.Tensor, rgbd1: torch.Tensor):
        """
        Args:
            rgbd0: Tensor [B, V, 4, H, W]
            rgbd1: Tensor [B, V, 4, H, W]
        Returns:
            prev_lvls: list of length L, each [B, V, C_l, H_l, W_l]
            curr_lvls: list of length L, each [B, V, C_l, H_l, W_l]
        """
        B, V, C, H, W = rgbd0.shape
        x = torch.cat([rgbd0, rgbd1], dim=0)  # [2B, V, 4, H, W]
        x = x.view(2 * B * V, C, H, W).contiguous()  # [2BV, 4, H, W]

        feats = self.cnet(x, branch=None)[::-1]  # list of L: [2BV, C_l, H_l, W_l]
        prev_lvls, curr_lvls = [], []
        for f in feats:
            C_l, H_l, W_l = f.shape[1], f.shape[2], f.shape[3]
            f = f.view(2, B, V, C_l, H_l, W_l)  # [2, B, V, C, H, W]
            prev_lvls.append(f[0])  # [B, V, C, H, W]
            curr_lvls.append(f[1])  # [B, V, C, H, W]
        return prev_lvls, curr_lvls

    # (kept for backward-compat with any external call sites)
    def extract_feature(self, im0: torch.Tensor, im1: torch.Tensor, branch):
        x = torch.cat([im0, im1], dim=0)
        feats = self.cnet(x, branch=branch)[::-1]
        p0, p1 = [], []
        for f in feats:
            a, b = torch.chunk(f, 2, dim=0)
            p0.append(a)
            p1.append(b)
        return p0, p1

    # ---------------------------------------------------------------------
    # Fast range projection (single-dim gather)
    # ---------------------------------------------------------------------
    def project_views_to_range(self, features_list, proj_pix, H_img=160, W_img=320):
        """
        Project V-view, same-scale feature maps into a spherical range-view using index mapping.
        This version avoids materializing a huge [B*N, C, HW] tensor by using
        chunked advanced indexing. It is both memory-safe and fast.

        Args:
            features_list: list of V tensors, each [B, C, Hf, Wf]
            proj_pix: LongTensor [B, H_r, W_r, 3] = (cam_idx, u_orig, v_orig)
            H_img, W_img: base image size used to compute downscale to feature map
        Returns:
            range_feat: [B, C, H_r, W_r]
        """
        B, H_r, W_r, _ = proj_pix.shape
        V = len(features_list)
        assert V > 0, "features_list must be non-empty"
        device = features_list[0].device

        C = features_list[0].shape[1]
        Hf, Wf = features_list[0].shape[2], features_list[0].shape[3]

        # downsample ratios from original img grid -> feature grid
        s_u = Wf / float(W_img)
        s_v = Hf / float(H_img)

        # [B,V,C,Hf,Wf] -> [BV,C,HW]
        feats = torch.stack(features_list, dim=1).contiguous()  # [B,V,C,Hf,Wf]
        feats = feats.view(B * V, C, Hf * Wf).contiguous()      # [BV,C,HW]

        cam_idx = proj_pix[..., 0].reshape(B, -1)  # [B,N]
        u_orig = proj_pix[..., 1].float().reshape(B, -1)
        v_orig = proj_pix[..., 2].float().reshape(B, -1)

        u_feat = (u_orig * s_u).long().clamp_(0, Wf - 1)  # [B,N]
        v_feat = (v_orig * s_v).long().clamp_(0, Hf - 1)  # [B,N]
        pix_idx = (v_feat * Wf + u_feat).reshape(-1)      # [B*N]

        batch_idx = (
            torch.arange(B, device=device).unsqueeze(1).repeat(1, H_r * W_r).reshape(-1)
        )  # [B*N]
        view_idx = (batch_idx * V + cam_idx.reshape(-1)).long()  # [B*N]

        # ---- Chunked advanced indexing to avoid huge temporaries ----
        N = B * H_r * W_r
        picked = torch.empty((N, C), device=feats.device, dtype=feats.dtype)
        CHUNK = 65536  # tune if needed
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            vi = view_idx[start:end]
            pi = pix_idx[start:end]
            # Direct advanced indexing yields [chunk, C]
            picked[start:end] = feats[vi, :, pi]

        range_feat = (
            picked.view(B, H_r * W_r, C)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, C, H_r, W_r)
        )
        return range_feat

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        img_prev,
        img_curr,
        depth_prev,
        depth_curr,
        proj_pix_prev,
        proj_pix_curr,
        attn_type,
        attn_splits_list,
        corr_radius_list,
        prop_radius_list,
        num_reg_refine,
        scale_only,
    ):
        # 1) normalize & pack RGBD per view
        img0, img1 = normalize_img(img_prev, img_curr)        # [B,V,3,H,W]
        rgbd0 = torch.cat([img0, depth_prev], dim=2)          # [B,V,4,H,W]
        rgbd1 = torch.cat([img1, depth_curr], dim=2)          # [B,V,4,H,W]
        B, V = rgbd0.shape[:2]

        # 2) extract multi-level features for ALL views in one pass
        prev_lvls, curr_lvls = self.extract_feature_all(rgbd0, rgbd1)  # lists of [B,V,C,H,W]

        # Prepare containers per view
        multi_level_feats_prev = [[] for _ in range(V)]  # each item: list over levels of [B,C,H,W]
        multi_level_feats_curr = [[] for _ in range(V)]
        corr_features = []  # one corr map per view at the final scale: each [B,2,H,W]

        # 3) multi-scale feature fusion + correlation (run by level, flatten BV as batch)
        corr = None  # per-level recurrent state (kept per BV batch)
        for lvl in range(self.num_scales):
            prev_feat_lvl = prev_lvls[lvl]  # [B,V,C,H,W]
            curr_feat_lvl = curr_lvls[lvl]

            # (BV) as batch for better kernel utilization
            B_, V_, C_, H_, W_ = prev_feat_lvl.shape
            prev_feat_bv = prev_feat_lvl.view(B_ * V_, C_, H_, W_).contiguous()
            curr_feat_bv = curr_feat_lvl.view(B_ * V_, C_, H_, W_).contiguous()

            fused_prev, fused_curr = self.featnet(
                prev_feat_bv,
                curr_feat_bv,
                lvl,
                attn_type,
                attn_splits_list,
                corr,  # pass recurrent corr if your featnet uses it
            )

            corr, _ = self.corrnet(
                fused_prev,
                fused_curr,
                lvl,
                corr_radius_list,
                prop_radius_list,
                num_reg_refine,
                False,
                corr,
            )

            # upsample corr to next scale if needed
            if lvl < self.num_scales - 1:
                corr = F.interpolate(corr, scale_factor=2, mode='bilinear', align_corners=False) * 2

            # reshape fused features back to [B,V,C,H,W] and store per view
            C_f, H_f, W_f = fused_prev.shape[1], fused_prev.shape[2], fused_prev.shape[3]
            fused_prev_bv = fused_prev.view(B, V, C_f, H_f, W_f)
            fused_curr_bv = fused_curr.view(B, V, C_f, H_f, W_f)
            for v in range(V):
                multi_level_feats_prev[v].append(fused_prev_bv[:, v])  # [B,C,H,W]
                multi_level_feats_curr[v].append(fused_curr_bv[:, v])

            # collect final-level corr maps per view for projection
            if lvl == self.num_scales - 1:
                Cc, Hc, Wc = corr.shape[1], corr.shape[2], corr.shape[3]
                corr_bv = corr.view(B, V, Cc, Hc, Wc)
                for v in range(V):
                    corr_features.append(corr_bv[:, v])  # each [B,2,H,W]

        # 4) project corr to range-view (risk/scale shared encoder)
        #    NOTE: the base H_img/W_img are the *image* grid sizes used to create proj_pix.
        #    Keep them consistent with your data pipeline; 160x320 are your defaults.
        corr_range = self.project_views_to_range(corr_features, proj_pix_curr, H_img=160, W_img=320)

        # 5) project multi-level, multi-view features
        multi_level_ranges_prev = []
        multi_level_ranges_curr = []
        for lvl in range(self.num_scales):
            scale = 2 ** (self.num_scales - 1 - lvl)
            if scale > 1:
                proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
                proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
            else:
                proj_prev_lvl = proj_pix_prev
                proj_curr_lvl = proj_pix_curr

            # collect all views' fused features at this level
            prev_feats_lvl = [multi_level_feats_prev[v][lvl] for v in range(V)]  # list of [B,C,H,W]
            curr_feats_lvl = [multi_level_feats_curr[v][lvl] for v in range(V)]

            range_prev = self.project_views_to_range(prev_feats_lvl, proj_prev_lvl, H_img=160, W_img=320)
            range_curr = self.project_views_to_range(curr_feats_lvl, proj_curr_lvl, H_img=160, W_img=320)

            multi_level_ranges_prev.append(range_prev)
            multi_level_ranges_curr.append(range_curr)

        # 6) shared corr encoding + heads
        corr_feat_shared = self.corr_enc_shared(corr_range)

        initial_scale = F.softplus(self.init_scale(corr_feat_shared)) + 1e-3
        scales = self.scale_net(
            corr_feat_shared, multi_level_ranges_prev, multi_level_ranges_curr, initial_scale
        )

        if scale_only:
            return scales, None

        initial_risk = self.init_risk(corr_feat_shared)
        risk_score = self.risk_net(
            corr_feat_shared, multi_level_ranges_prev, multi_level_ranges_curr, initial_risk
        )

        return scales, risk_score

    # ---------------------------------------------------------------------
    # Training wrapper (unchanged public interface)
    # ---------------------------------------------------------------------
    def forward_with_loss(
        self,
        img_prev,
        img_curr,
        depth_prev,
        depth_curr,
        proj_pix_prev,
        proj_pix_curr,
        gt_scale_map_with_mask,
        gt_risk_score_map_with_mask,
        attn_type,
        attn_splits_list,
        corr_radius_list,
        prop_radius_list,
        num_reg_refine,
        scale_only,
    ):
        scales, risks = self.forward(
            img_prev=img_prev,
            img_curr=img_curr,
            depth_prev=depth_prev,
            depth_curr=depth_curr,
            proj_pix_prev=proj_pix_prev,
            proj_pix_curr=proj_pix_curr,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
            scale_only=scale_only,
        )

        if scale_only:
            loss_s = get_loss_scale_map(scales, gt_scale_map_with_mask)
            return scales, None, loss_s, None
        else:
            loss_r = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)
            return None, risks, None, loss_r
