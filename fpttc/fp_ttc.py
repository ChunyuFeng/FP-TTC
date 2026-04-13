import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.utils import normalize_img
from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet
from .rvt.range_view_transformer import RangeViewTransformer
from utils.loss import get_loss_risk_score_map, get_loss_scale_map


def _safe_bilinear(x, *, size=None, scale_factor=None, align_corners=True):
    orig_dtype = x.dtype
    if orig_dtype == torch.bfloat16:
        x = x.float()
    y = F.interpolate(
        x,
        size=size,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=align_corners,
    )
    return y.to(orig_dtype)


class CorrEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.convc1 = nn.Conv2d(dim_in, 256, 1)
        self.convc2 = nn.Conv2d(256, dim_out, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.convc1(x))
        x = F.relu(self.convc2(x))
        return x


class CorrResidualDecoder(nn.Module):
    """Decode RVT corr features into a residual 2-channel corr map."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = CorrEncoder(hidden_dim, hidden_dim)
        self.proj = nn.Conv2d(hidden_dim, 2, kernel_size=1)

    def forward(self, x):
        return self.proj(self.encoder(x))


class FpTTC(nn.Module):
    def __init__(
        self,
        num_scales=2,
        feature_channels=128,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=False,
        rvt_depth_guided_sampling=False,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.rvt_depth_guided_sampling = rvt_depth_guided_sampling

        self.camera_channels = [
            'CAM_FRONT_LEFT',
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
        ]

        self.cnet = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
        )
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

        # Existing heads keep their names so current hardproj checkpoints still load cleanly.
        self.conv_corr = CorrEncoder(dim_in=2, dim_out=feature_channels + 1)
        self.conv_corr_risk = CorrEncoder(dim_in=2, dim_out=feature_channels + 1)
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

        num_views = len(self.camera_channels)
        num_depth_bins = 8
        self.register_buffer(
            'depth_bins',
            torch.linspace(0.0, 48.0, num_depth_bins),
        )

        self.rvt_feat = nn.ModuleList(
            [
                RangeViewTransformer(
                    num_layers=2,
                    input_dim=feature_channels,
                    d_model=feature_channels,
                    nhead=4,
                    num_level=num_views,
                    num_points=num_depth_bins,
                    fov_up=8.0,
                    fov_down=-15.0,
                )
                for _ in range(num_scales)
            ]
        )
        self.rvt_corr = RangeViewTransformer(
            num_layers=2,
            input_dim=2,
            d_model=feature_channels,
            nhead=4,
            num_level=num_views,
            num_points=num_depth_bins,
            fov_up=8.0,
            fov_down=-15.0,
        )
        self.corr_residual_decoder = CorrResidualDecoder(feature_channels)

    def _camera_meta_list(self, sensor_metas, frame_key, meta_key):
        return [sensor_metas[frame_key][camera][meta_key] for camera in self.camera_channels]

    def _camera_meta_tensors(self, sensor_metas, frame_key):
        cam_K = torch.stack(self._camera_meta_list(sensor_metas, frame_key, 'K'), dim=1)
        cam_R_l2c = torch.stack(self._camera_meta_list(sensor_metas, frame_key, 'R_l2c'), dim=1)
        cam_t_l2c = torch.stack(self._camera_meta_list(sensor_metas, frame_key, 't_l2c'), dim=1)
        cam_affine = torch.stack(self._camera_meta_list(sensor_metas, frame_key, 'affine'), dim=1)

        cam_K_inv = torch.linalg.inv(cam_K)
        cam_affine_inv = torch.linalg.inv(cam_affine)
        cam_R_c2l = cam_R_l2c.transpose(-1, -2)
        cam_t_c2l = -(cam_R_c2l @ cam_t_l2c.unsqueeze(-1)).squeeze(-1)

        return {
            'K_inv': cam_K_inv,
            'affine_inv': cam_affine_inv,
            'R_c2l': cam_R_c2l,
            't_c2l': cam_t_c2l,
        }

    def _build_depth_guided_range(self, depth_maps, proj_pix, camera_meta):
        B, num_views, _, H_depth, W_depth = depth_maps.shape
        _, H_r, W_r, _ = proj_pix.shape
        Q = H_r * W_r
        device = depth_maps.device
        dtype = depth_maps.dtype

        cam_idx = proj_pix[..., 0].long()
        u_proc = proj_pix[..., 1].long()
        v_proc = proj_pix[..., 2].long()
        valid = (
            (cam_idx >= 0)
            & (cam_idx < num_views)
            & (u_proc >= 0)
            & (u_proc < W_depth)
            & (v_proc >= 0)
            & (v_proc < H_depth)
        )

        cam_idx_safe = cam_idx.clamp(0, num_views - 1)
        u_safe = u_proc.clamp(0, W_depth - 1)
        v_safe = v_proc.clamp(0, H_depth - 1)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1)
        sampled_depth = depth_maps[batch_idx, cam_idx_safe, 0, v_safe, u_safe]
        valid = valid & torch.isfinite(sampled_depth) & (sampled_depth > 0)

        batch_idx_flat = torch.arange(B, device=device).view(B, 1).expand(B, Q)
        cam_idx_flat = cam_idx_safe.view(B, Q)

        affine_inv = camera_meta['affine_inv'][batch_idx_flat, cam_idx_flat]
        K_inv = camera_meta['K_inv'][batch_idx_flat, cam_idx_flat]
        R_c2l = camera_meta['R_c2l'][batch_idx_flat, cam_idx_flat]
        t_c2l = camera_meta['t_c2l'][batch_idx_flat, cam_idx_flat]

        pix_proc = torch.stack(
            [
                u_safe.to(dtype),
                v_safe.to(dtype),
                torch.ones_like(sampled_depth, dtype=dtype),
            ],
            dim=-1,
        ).view(B, Q, 3, 1)
        pix_orig = (affine_inv @ pix_proc).squeeze(-1)
        denom = pix_orig[..., 2:3]
        pix_orig = pix_orig / torch.where(denom.abs() > 1e-6, denom, torch.ones_like(denom))

        cam_points = (K_inv @ pix_orig.unsqueeze(-1)).squeeze(-1)
        cam_points = cam_points * sampled_depth.view(B, Q, 1)

        lidar_points = (R_c2l @ cam_points.unsqueeze(-1)).squeeze(-1) + t_c2l
        guide_range = lidar_points.norm(dim=-1).view(B, 1, H_r, W_r)

        valid = valid & torch.isfinite(guide_range[:, 0]) & (guide_range[:, 0] > 0)
        return guide_range.masked_fill(~valid.unsqueeze(1), 0.0)

    def _extract_multi_view_features(
        self,
        img_prev,
        img_curr,
        attn_type,
        attn_splits_list,
        corr_radius_list,
        prop_radius_list,
        num_reg_refine,
    ):
        img0, img1 = normalize_img(img_prev, img_curr)
        B, V, C, H_img, W_img = img0.shape

        x0 = img0.view(B * V, C, H_img, W_img)
        x1 = img1.view(B * V, C, H_img, W_img)
        prev_lvls_flat, curr_lvls_flat = self.extract_feature(x0, x1, branch=None)

        prev_lvls = [feat.view(B, V, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in prev_lvls_flat]
        curr_lvls = [feat.view(B, V, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in curr_lvls_flat]

        corr = None
        multi_level_feats_prev = []
        multi_level_feats_curr = []

        for lvl in range(self.num_scales):
            prev_level = prev_lvls[lvl].reshape(B * V, -1, prev_lvls[lvl].shape[3], prev_lvls[lvl].shape[4])
            curr_level = curr_lvls[lvl].reshape(B * V, -1, curr_lvls[lvl].shape[3], curr_lvls[lvl].shape[4])

            fused_prev, fused_curr = self.featnet(
                prev_level,
                curr_level,
                lvl,
                attn_type,
                attn_splits_list,
                corr,
            )

            multi_level_feats_prev.append(
                fused_prev.view(B, V, fused_prev.shape[1], fused_prev.shape[2], fused_prev.shape[3])
            )
            multi_level_feats_curr.append(
                fused_curr.view(B, V, fused_curr.shape[1], fused_curr.shape[2], fused_curr.shape[3])
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
            if lvl < self.num_scales - 1:
                corr = _safe_bilinear(corr, scale_factor=2, align_corners=True) * 2

        corr = corr.view(B, V, corr.shape[1], corr.shape[2], corr.shape[3])
        corr_list = [corr[:, view_idx] for view_idx in range(V)]

        return {
            'H_img': H_img,
            'W_img': W_img,
            'multi_level_feats_prev': multi_level_feats_prev,
            'multi_level_feats_curr': multi_level_feats_curr,
            'corr_list': corr_list,
        }

    def _build_hardproj_initial_ranges(
        self,
        multi_level_feats_prev,
        multi_level_feats_curr,
        corr_list,
        proj_pix_prev,
        proj_pix_curr,
        H_img,
        W_img,
    ):
        hardproj_corr_range_init = self.project_views_to_range(
            corr_list,
            proj_pix_curr,
            H_img=H_img,
            W_img=W_img,
        )

        hardproj_ranges_prev_init = []
        hardproj_ranges_curr_init = []
        for lvl in range(self.num_scales):
            scale = 2 ** (self.num_scales - 1 - lvl)
            if scale > 1:
                proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
                proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
            else:
                proj_prev_lvl = proj_pix_prev
                proj_curr_lvl = proj_pix_curr

            prev_feats_list = [multi_level_feats_prev[lvl][:, view_idx] for view_idx in range(len(self.camera_channels))]
            curr_feats_list = [multi_level_feats_curr[lvl][:, view_idx] for view_idx in range(len(self.camera_channels))]

            hardproj_ranges_prev_init.append(
                self.project_views_to_range(prev_feats_list, proj_prev_lvl, H_img=H_img, W_img=W_img)
            )
            hardproj_ranges_curr_init.append(
                self.project_views_to_range(curr_feats_list, proj_curr_lvl, H_img=H_img, W_img=W_img)
            )

        return hardproj_corr_range_init, hardproj_ranges_prev_init, hardproj_ranges_curr_init

    def _refine_corr_range(
        self,
        corr_list,
        hardproj_corr_range_init,
        sensor_metas,
        input_hw,
        guide_range=None,
    ):
        Hr, Wr = hardproj_corr_range_init.shape[-2:]
        corr_delta = self.rvt_corr(
            feats_by_cam=corr_list,
            cam_K=self._camera_meta_list(sensor_metas, 'curr', 'K'),
            cam_R=self._camera_meta_list(sensor_metas, 'curr', 'R_l2c'),
            cam_t=self._camera_meta_list(sensor_metas, 'curr', 't_l2c'),
            affine_M=self._camera_meta_list(sensor_metas, 'curr', 'affine'),
            Hr=Hr,
            Wr=Wr,
            depth_bins=self.depth_bins,
            ini_query=hardproj_corr_range_init.detach(),
            input_hw=input_hw,
            guide_range=guide_range,
        )
        corr_delta = self.corr_residual_decoder(corr_delta)
        return hardproj_corr_range_init + corr_delta

    def _refine_feature_ranges(
        self,
        multi_level_feats_prev,
        multi_level_feats_curr,
        hardproj_ranges_prev_init,
        hardproj_ranges_curr_init,
        sensor_metas,
        input_hw,
        guide_ranges_prev=None,
        guide_ranges_curr=None,
    ):
        rvt_ranges_prev = []
        rvt_ranges_curr = []

        for lvl in range(self.num_scales):
            prev_feats_list = [multi_level_feats_prev[lvl][:, view_idx] for view_idx in range(len(self.camera_channels))]
            curr_feats_list = [multi_level_feats_curr[lvl][:, view_idx] for view_idx in range(len(self.camera_channels))]

            Hr, Wr = hardproj_ranges_prev_init[lvl].shape[-2:]
            prev_delta = self.rvt_feat[lvl](
                feats_by_cam=prev_feats_list,
                cam_K=self._camera_meta_list(sensor_metas, 'prev', 'K'),
                cam_R=self._camera_meta_list(sensor_metas, 'prev', 'R_l2c'),
                cam_t=self._camera_meta_list(sensor_metas, 'prev', 't_l2c'),
                affine_M=self._camera_meta_list(sensor_metas, 'prev', 'affine'),
                Hr=Hr,
                Wr=Wr,
                depth_bins=self.depth_bins,
                ini_query=hardproj_ranges_prev_init[lvl].detach(),
                input_hw=input_hw,
                guide_range=None if guide_ranges_prev is None else guide_ranges_prev[lvl],
            )
            curr_delta = self.rvt_feat[lvl](
                feats_by_cam=curr_feats_list,
                cam_K=self._camera_meta_list(sensor_metas, 'curr', 'K'),
                cam_R=self._camera_meta_list(sensor_metas, 'curr', 'R_l2c'),
                cam_t=self._camera_meta_list(sensor_metas, 'curr', 't_l2c'),
                affine_M=self._camera_meta_list(sensor_metas, 'curr', 'affine'),
                Hr=Hr,
                Wr=Wr,
                depth_bins=self.depth_bins,
                ini_query=hardproj_ranges_curr_init[lvl].detach(),
                input_hw=input_hw,
                guide_range=None if guide_ranges_curr is None else guide_ranges_curr[lvl],
            )

            rvt_ranges_prev.append(hardproj_ranges_prev_init[lvl] + prev_delta)
            rvt_ranges_curr.append(hardproj_ranges_curr_init[lvl] + curr_delta)

        return rvt_ranges_prev, rvt_ranges_curr

    def forward(
        self,
        img_prev,
        img_curr,
        depth_prev,
        depth_curr,
        proj_pix_prev,
        proj_pix_curr,
        sensor_metas,
        attn_type,
        attn_splits_list,
        corr_radius_list,
        prop_radius_list,
        num_reg_refine,
        scale_only,
        return_debug=False,
    ):
        if sensor_metas is None:
            raise ValueError('sensor_metas must be provided when RVT refinement is enabled.')

        feature_outputs = self._extract_multi_view_features(
            img_prev=img_prev,
            img_curr=img_curr,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
        )

        hardproj_corr_range_init, hardproj_ranges_prev_init, hardproj_ranges_curr_init = (
            self._build_hardproj_initial_ranges(
                multi_level_feats_prev=feature_outputs['multi_level_feats_prev'],
                multi_level_feats_curr=feature_outputs['multi_level_feats_curr'],
                corr_list=feature_outputs['corr_list'],
                proj_pix_prev=proj_pix_prev,
                proj_pix_curr=proj_pix_curr,
                H_img=feature_outputs['H_img'],
                W_img=feature_outputs['W_img'],
            )
        )

        corr_guide_range = None
        guide_ranges_prev = None
        guide_ranges_curr = None
        if self.rvt_depth_guided_sampling:
            camera_meta_prev = self._camera_meta_tensors(sensor_metas, 'prev')
            camera_meta_curr = self._camera_meta_tensors(sensor_metas, 'curr')

            corr_guide_range = self._build_depth_guided_range(depth_curr, proj_pix_curr, camera_meta_curr)
            guide_ranges_prev = []
            guide_ranges_curr = []
            for lvl in range(self.num_scales):
                scale = 2 ** (self.num_scales - 1 - lvl)
                if scale > 1:
                    proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
                    proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
                else:
                    proj_prev_lvl = proj_pix_prev
                    proj_curr_lvl = proj_pix_curr

                guide_ranges_prev.append(
                    self._build_depth_guided_range(depth_prev, proj_prev_lvl, camera_meta_prev)
                )
                guide_ranges_curr.append(
                    self._build_depth_guided_range(depth_curr, proj_curr_lvl, camera_meta_curr)
                )

        input_hw = (feature_outputs['H_img'], feature_outputs['W_img'])
        corr_range = self._refine_corr_range(
            corr_list=feature_outputs['corr_list'],
            hardproj_corr_range_init=hardproj_corr_range_init,
            sensor_metas=sensor_metas,
            input_hw=input_hw,
            guide_range=corr_guide_range,
        )
        multi_level_ranges_prev, multi_level_ranges_curr = self._refine_feature_ranges(
            multi_level_feats_prev=feature_outputs['multi_level_feats_prev'],
            multi_level_feats_curr=feature_outputs['multi_level_feats_curr'],
            hardproj_ranges_prev_init=hardproj_ranges_prev_init,
            hardproj_ranges_curr_init=hardproj_ranges_curr_init,
            sensor_metas=sensor_metas,
            input_hw=input_hw,
            guide_ranges_prev=guide_ranges_prev,
            guide_ranges_curr=guide_ranges_curr,
        )

        debug_dict = None
        if return_debug:
            debug_dict = {
                'multi_level_feats_prev': feature_outputs['multi_level_feats_prev'],
                'multi_level_feats_curr': feature_outputs['multi_level_feats_curr'],
                'multi_level_ranges_prev_init': hardproj_ranges_prev_init,
                'multi_level_ranges_curr_init': hardproj_ranges_curr_init,
                'multi_level_ranges_prev': multi_level_ranges_prev,
                'multi_level_ranges_curr': multi_level_ranges_curr,
                'corr_range_init': hardproj_corr_range_init,
                'corr_range': corr_range,
            }
            if self.rvt_depth_guided_sampling:
                debug_dict['corr_guide_range'] = corr_guide_range
                debug_dict['guide_ranges_prev'] = guide_ranges_prev
                debug_dict['guide_ranges_curr'] = guide_ranges_curr

        corr_encoded_scale = self.conv_corr(corr_range)
        initial_scale = F.softplus(corr_encoded_scale[:, :1]) + 1e-3
        corr_encoded_scale = corr_encoded_scale[:, 1:]
        scales = self.scale_net(
            corr_encoded_scale,
            multi_level_ranges_prev,
            multi_level_ranges_curr,
            initial_scale,
        )
        if scale_only:
            if return_debug:
                return scales, None, debug_dict
            return scales, None

        corr_encoded_risk = self.conv_corr_risk(corr_range)
        initial_risk = corr_encoded_risk[:, :1]
        corr_encoded_risk = corr_encoded_risk[:, 1:]
        risk_score = self.risk_net(
            corr_encoded_risk,
            multi_level_ranges_prev,
            multi_level_ranges_curr,
            initial_risk,
        )

        if return_debug:
            return scales, risk_score, debug_dict
        return scales, risk_score

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
        sensor_metas,
        attn_type,
        attn_splits_list,
        corr_radius_list,
        prop_radius_list,
        num_reg_refine,
        scale_only,
        loss_weight_alpha=0.0,
        return_debug=False,
    ):
        outputs = self.forward(
            img_prev=img_prev,
            img_curr=img_curr,
            depth_prev=depth_prev,
            depth_curr=depth_curr,
            proj_pix_prev=proj_pix_prev,
            proj_pix_curr=proj_pix_curr,
            sensor_metas=sensor_metas,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
            scale_only=scale_only,
            return_debug=return_debug,
        )
        if return_debug:
            scales, risks, debug_dict = outputs
        else:
            scales, risks = outputs

        if scale_only:
            loss_scale = get_loss_scale_map(
                scales,
                gt_scale_map_with_mask,
                loss_weight_alpha=loss_weight_alpha,
            )
            if return_debug:
                return scales, None, loss_scale, None, debug_dict
            return scales, None, loss_scale, None

        loss_risk = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)
        if return_debug:
            return scales, risks, None, loss_risk, debug_dict
        return scales, risks, None, loss_risk

    def extract_feature(self, im0, im1, branch):
        x = torch.cat([im0, im1], dim=0)
        feats = self.cnet(x, branch=branch)[::-1]
        p0, p1 = [], []
        for feat in feats:
            a, b = torch.chunk(feat, 2, dim=0)
            p0.append(a)
            p1.append(b)
        return p0, p1

    def project_views_to_range(self, features_list, proj_pix, H_img=160, W_img=320):
        """Project multi-view image features to range-view using cached hard projection indices."""
        B, H_r, W_r, _ = proj_pix.shape
        V = len(features_list)
        C = features_list[0].shape[1]
        H_feat, W_feat = features_list[0].shape[2], features_list[0].shape[3]

        scale_u = W_feat / W_img
        scale_v = H_feat / H_img

        feats = torch.stack(features_list, dim=1)
        feats = feats.view(B * V, C, H_feat, W_feat)
        feats_flat = feats.view(B * V, C, -1)

        cam_idx = proj_pix[..., 0].reshape(B, -1)
        u_orig = proj_pix[..., 1].float().reshape(B, -1)
        v_orig = proj_pix[..., 2].float().reshape(B, -1)

        u_feat = (u_orig * scale_u).long().clamp(0, W_feat - 1)
        v_feat = (v_orig * scale_v).long().clamp(0, H_feat - 1)

        batch_idx = torch.arange(B, device=cam_idx.device).unsqueeze(1).repeat(1, H_r * W_r).reshape(-1)
        view_idx = batch_idx * V + cam_idx.reshape(-1)
        pix_idx = (v_feat * W_feat + u_feat).reshape(-1)

        selected = feats_flat[view_idx, :, pix_idx]
        range_feat = selected.view(B, H_r * W_r, C).permute(0, 2, 1).reshape(B, C, H_r, W_r)
        return range_feat
