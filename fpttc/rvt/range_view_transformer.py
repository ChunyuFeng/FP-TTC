import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fpttc.modules.position import PositionEmbeddingSine
from fpttc.scale_net.utils.multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32 as MSDeformAttn,
)


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    padding = ((kernel_size - 1) * dilation) // 2
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.1, inplace=False),
    )


def make_range_rays(Hr, Wr, fov_up_deg=8.0, fov_down_deg=-15.0, device='cuda'):
    """Create one LiDAR ray direction for each range-view pixel."""
    fov_up = math.radians(fov_up_deg)
    fov_down = math.radians(fov_down_deg)
    fov = abs(fov_down) + abs(fov_up)

    x = torch.linspace(0, Wr - 1, Wr, device=device)
    y = torch.linspace(0, Hr - 1, Hr, device=device)
    gx, gy = torch.meshgrid(x, y, indexing='xy')
    gx = gx.t().contiguous()
    gy = gy.t().contiguous()

    yaw = ((gx + 0.5) / Wr) * (2 * math.pi) - math.pi
    pitch = (1.0 - (gy + 0.5) / Hr) * fov - abs(fov_down)

    rays = torch.stack(
        [
            torch.cos(pitch) * torch.cos(-yaw),
            torch.cos(pitch) * torch.sin(-yaw),
            torch.sin(pitch),
        ],
        dim=-1,
    )
    rays = rays / (rays.norm(dim=-1, keepdim=True) + 1e-8)
    return rays.unsqueeze(0)


@torch.no_grad()
def project_points_to_image(X_lidar, K, R_l2c, t_l2c, eps=1e-6):
    """Project LiDAR-frame 3D points to image pixels."""
    if t_l2c.ndim == 2:
        t_l2c = t_l2c.unsqueeze(-1)

    X = X_lidar.transpose(1, 2)
    X_cam = R_l2c @ X + t_l2c
    Z = X_cam[:, 2:3, :].clamp_min(eps)
    proj = K @ X_cam

    u = proj[:, 0, :] / Z[:, 0, :]
    v = proj[:, 1, :] / Z[:, 0, :]
    uv = torch.stack([u, v], dim=-1)
    valid = X_cam[:, 2, :] > 0
    return uv, valid


def apply_affine_to_uv(uv, affine_M):
    """Move pixels from original-image coordinates to augmented-image coordinates."""
    B, N, _ = uv.shape
    ones = torch.ones(B, N, 1, device=uv.device, dtype=uv.dtype)
    homo = torch.cat([uv, ones], dim=-1)
    transformed = (affine_M @ homo.transpose(1, 2)).transpose(1, 2)
    transformed = transformed[..., :2] / (transformed[..., 2:3] + 1e-8)
    return transformed


def uv_to_01(uv, W_in, H_in):
    x = uv[..., 0] / max(W_in - 1, 1)
    y = uv[..., 1] / max(H_in - 1, 1)
    return torch.stack([x, y], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model=128,
        num_head=4,
        num_points=8,
        num_level=6,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.num_points = num_points
        self.num_level = num_level

        self.sampling_offsets = nn.Linear(d_model, num_head * num_level * num_points * 2)
        self.attention_weights = nn.Linear(d_model, num_head * num_level * num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model * num_head, d_model)
        self.dropout = nn.Dropout(dropout)
        self.MultiScaleDeformableAttnFunction = MSDeformAttn
        self._init_weights()

    def _init_weights(self):
        from mmcv.cnn import constant_init, xavier_init

        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_head, dtype=torch.float32) * (2.0 * math.pi / self.num_head)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.num_head, 1, 1, 2).repeat(1, self.num_level, self.num_points, 1)
        for idx in range(self.num_points):
            grid_init[:, :, idx, :] *= idx + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)

        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution='uniform', bias=0.0)
        xavier_init(self.output_proj, distribution='uniform', bias=0.0)

    def forward(
        self,
        query,
        value,
        query_location,
        spatial_shapes,
        level_start_index,
        sampling_locations=None,
        attention_weights=None,
    ):
        B, Q, _ = query.shape

        if sampling_locations is None or attention_weights is None:
            sampling_offsets = self.sampling_offsets(query).view(
                B, Q, self.num_head, self.num_level, self.num_points, 2
            )
            attention_weights = self.attention_weights(query).view(
                B, Q, self.num_head, self.num_level * self.num_points
            )
            attention_weights = F.softmax(attention_weights, dim=-1).view(
                B, Q, self.num_head, self.num_level, self.num_points
            )

            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1
            )
            sampling_locations = (
                query_location[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )

        out = self.MultiScaleDeformableAttnFunction.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        out = self.output_proj(out)
        return self.dropout(out)


class RangeViewTransformer(nn.Module):
    """Aggregate multi-camera features into range-view features with geometry-guided sampling."""

    def __init__(
        self,
        num_layers=2,
        input_dim=128,
        d_model=128,
        nhead=4,
        num_level=6,
        num_points=8,
        fov_up=8.0,
        fov_down=-15.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_level = num_level
        self.num_points = num_points
        self.fov_up = fov_up
        self.fov_down = fov_down

        self.query_encoder = nn.Sequential(
            conv(input_dim, d_model * 2),
            conv(d_model * 2, d_model),
            conv(d_model, d_model),
        )
        self.value_proj = nn.Conv1d(input_dim, d_model, 1)
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_head=nhead,
                    num_points=num_points,
                    num_level=num_level,
                )
                for _ in range(num_layers)
            ]
        )
        self.level_embeds = nn.Parameter(torch.Tensor(num_level, input_dim))
        nn.init.xavier_uniform_(self.level_embeds)
        self.register_buffer(
            'local_depth_multipliers',
            torch.exp(torch.linspace(-0.7, 0.7, steps=num_points)),
        )
        self.local_depth_min = 1e-3
        self.local_depth_max = 48.0

    @staticmethod
    def _pack_levels(feats_by_cam):
        value_levels = []
        spatial_shapes = []
        feature_sizes = []

        for feat in feats_by_cam:
            B, C, H, W = feat.shape
            value_levels.append(feat.view(B, C, H * W).transpose(1, 2).contiguous())
            spatial_shapes.append([H, W])
            feature_sizes.append((H, W))

        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            device=feats_by_cam[0].device,
            dtype=torch.long,
        )
        level_start_index = torch.cat(
            [
                spatial_shapes.new_zeros(1),
                (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1],
            ]
        )
        return value_levels, spatial_shapes, level_start_index, feature_sizes

    def _make_geom_sampling(
        self,
        B,
        Hr,
        Wr,
        depth_bins,
        cam_K,
        cam_R,
        cam_t,
        affine_M,
        feature_sizes,
        input_hw,
        guide_range=None,
    ):
        input_h, input_w = input_hw
        Q = Hr * Wr
        rays = make_range_rays(Hr, Wr, self.fov_up, self.fov_down, device=cam_K[0].device)
        rays = rays.view(1, Q, 3).repeat(B, 1, 1)

        if depth_bins.numel() != self.num_points:
            raise ValueError(
                f'depth_bins length ({depth_bins.numel()}) must match num_points ({self.num_points}).'
            )

        depth_bins = depth_bins.to(device=rays.device, dtype=rays.dtype)
        if guide_range is None:
            depth_samples = depth_bins.view(1, 1, self.num_points).expand(B, Q, self.num_points)
        else:
            if guide_range.ndim == 4:
                if guide_range.shape[1] != 1:
                    raise ValueError('guide_range must have channel dimension 1 when provided as BCHW.')
                guide_range = guide_range[:, 0]
            if guide_range.shape != (B, Hr, Wr):
                raise ValueError(
                    f'guide_range must have shape {(B, Hr, Wr)}; got {tuple(guide_range.shape)}.'
                )

            guide_flat = guide_range.reshape(B, Q, 1).to(device=rays.device, dtype=rays.dtype)
            local_depth_bins = guide_flat * self.local_depth_multipliers.to(
                device=rays.device,
                dtype=rays.dtype,
            ).view(1, 1, self.num_points)
            local_depth_bins = local_depth_bins.clamp(self.local_depth_min, self.local_depth_max)

            valid_guide = torch.isfinite(guide_flat) & (guide_flat > 0)
            fallback_depth_bins = depth_bins.view(1, 1, self.num_points).expand(B, Q, self.num_points)
            depth_samples = torch.where(
                valid_guide.expand_as(local_depth_bins),
                local_depth_bins,
                fallback_depth_bins,
            )

        K_bins = depth_samples.shape[-1]
        points = rays.unsqueeze(2) * depth_samples.unsqueeze(-1)

        all_locations = []
        all_masks = []
        for level_idx, (feat_h, feat_w) in enumerate(feature_sizes):
            scale_x = feat_w / float(input_w)
            scale_y = feat_h / float(input_h)
            scale_m = torch.tensor(
                [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
                device=affine_M[level_idx].device,
                dtype=affine_M[level_idx].dtype,
            ).unsqueeze(0)
            affine_feat = scale_m @ affine_M[level_idx]

            uv, z_mask = project_points_to_image(
                points.view(B, -1, 3),
                cam_K[level_idx],
                cam_R[level_idx],
                cam_t[level_idx],
            )
            uv = apply_affine_to_uv(uv, affine_feat)

            u = uv[..., 0]
            v = uv[..., 1]
            in_bounds = (u >= 0) & (u <= (feat_w - 1)) & (v >= 0) & (v <= (feat_h - 1))
            valid = (z_mask & in_bounds).view(B, Q, K_bins)

            all_locations.append(uv_to_01(uv, feat_w, feat_h).view(B, Q, K_bins, 2))
            all_masks.append(valid)

        sampling_locations = torch.stack(all_locations, dim=2)
        visibility_mask = torch.stack(all_masks, dim=2)

        sampling_locations = sampling_locations.unsqueeze(2).repeat(1, 1, self.nhead, 1, 1, 1)
        visibility_mask = visibility_mask.unsqueeze(2).repeat(1, 1, self.nhead, 1, 1)
        return sampling_locations, visibility_mask

    def forward(
        self,
        feats_by_cam,
        cam_K,
        cam_R,
        cam_t,
        affine_M,
        Hr,
        Wr,
        depth_bins,
        ini_query=None,
        input_hw=None,
        guide_range=None,
    ):
        B, C_in, _, _ = feats_by_cam[0].shape
        if ini_query is None:
            ini_query = torch.zeros(B, C_in, Hr, Wr, device=feats_by_cam[0].device)
        if input_hw is None:
            raise ValueError('input_hw must be provided for geometry-guided RVT sampling.')

        query = self.query_encoder(ini_query)
        query = query + self.pos_enc(query)
        query = query.flatten(2).transpose(1, 2).contiguous()

        value_levels, spatial_shapes, level_start_index, feature_sizes = self._pack_levels(feats_by_cam)
        value_with_level_embeds = []
        for level_idx, level_value in enumerate(value_levels):
            level_embed = self.level_embeds[None, level_idx : level_idx + 1, :].to(level_value.dtype)
            value_with_level_embeds.append(level_value + level_embed)
        value = torch.cat(value_with_level_embeds, dim=1)
        value = self.value_proj(value.transpose(1, 2)).transpose(1, 2)
        value = value.unsqueeze(2).repeat(1, 1, self.nhead, 1)

        query_location = torch.zeros(
            B,
            Hr * Wr,
            self.num_level,
            2,
            device=value.device,
            dtype=value.dtype,
        )
        geom_locations, visibility_mask = self._make_geom_sampling(
            B,
            Hr,
            Wr,
            depth_bins,
            cam_K,
            cam_R,
            cam_t,
            affine_M,
            feature_sizes,
            input_hw,
            guide_range=guide_range,
        )

        out = query
        for layer in self.layers:
            attn = layer.attention_weights(out).view(
                B, Hr * Wr, self.nhead, self.num_level * self.num_points
            )
            attn = F.softmax(attn, dim=-1).view(
                B, Hr * Wr, self.nhead, self.num_level, self.num_points
            )
            attn = attn * visibility_mask.float()
            attn = attn / attn.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

            delta = layer.sampling_offsets(out).view(
                B, Hr * Wr, self.nhead, self.num_level, self.num_points, 2
            )
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]],
                dim=-1,
            ).to(delta.dtype)
            offset_normalizer = offset_normalizer.view(1, 1, 1, self.num_level, 1, 2)
            sampling_locations = (geom_locations + delta / offset_normalizer).clamp(0.0, 1.0)

            out = layer(
                out,
                value,
                query_location=query_location,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                sampling_locations=sampling_locations,
                attention_weights=attn,
            )

        out = out.view(B, Hr, Wr, self.d_model).permute(0, 3, 1, 2).contiguous()
        return out
