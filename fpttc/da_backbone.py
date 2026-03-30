import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


def _round_up_to_multiple(value, multiple):
    return int(math.ceil(float(value) / float(multiple)) * multiple)


def _compute_da_input_hw(image_hw, multiple=14):
    h, w = image_hw
    scale = max(
        float(_round_up_to_multiple(h, multiple)) / float(h),
        float(_round_up_to_multiple(w, multiple)) / float(w),
    )
    h_da = _round_up_to_multiple(int(math.ceil(h * scale)), multiple)
    w_da = _round_up_to_multiple(int(math.ceil(w * scale)), multiple)
    return h_da, w_da


class DAFeatureAdapter(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, num_res_blocks=3):
        super().__init__()
        num_groups = 8 if out_channels % 8 == 0 else 1

        def _residual_block():
            return nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(num_groups, out_channels),
            )

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList([
            _residual_block() for _ in range(int(num_res_blocks))
        ])
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.net(x)
        for block in self.res_blocks:
            x = self.act(x + block(x))
        return self.out_proj(x)


class DepthAnythingBackbone(nn.Module):
    def __init__(
        self,
        *,
        output_dim=128,
        num_output_scales=2,
        pretrained_ckpt="pretrained/depth_anything_v2_metric_vkitti_vits.pth",
        image_size=(160, 320),
        metric_max_depth=80.0,
    ):
        super().__init__()

        if num_output_scales not in (1, 2):
            raise ValueError(f"DepthAnythingBackbone only supports 1 or 2 scales, got {num_output_scales}")

        self.output_dim = int(output_dim)
        self.num_output_scales = int(num_output_scales)
        self.image_size = tuple(int(v) for v in image_size)
        self.depth_output_hw = self.image_size
        self.da_input_hw = _compute_da_input_hw(self.image_size, multiple=14)
        self.high_res_hw = (self.image_size[0] // 4, self.image_size[1] // 4)
        self.low_res_hw = (self.image_size[0] // 8, self.image_size[1] // 8)

        model_cfg = {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
            "max_depth": metric_max_depth,
        }
        self.backbone = DepthAnythingV2(**model_cfg)

        ckpt_path = Path(pretrained_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Metric DepthAnything checkpoint not found: {ckpt_path}. "
                "Expected a vits metric checkpoint compatible with current 4_depth_map / proj_cache."
            )
        state_dict = torch.load(str(ckpt_path), map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.adapter_hi = DAFeatureAdapter(in_channels=64, out_channels=self.output_dim)
        self.adapter_lo = DAFeatureAdapter(in_channels=64, out_channels=self.output_dim)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x, branch=None, return_depth=False):
        del branch  # kept for CNNEncoder compatibility

        x_da = F.interpolate(
            x,
            size=self.da_input_hw,
            mode="bilinear",
            align_corners=True,
        )

        with torch.no_grad():
            depth_da, intermediates = self.backbone(x_da, return_intermediates=True)

        feat_hi = self.adapter_hi(intermediates["path_2"])
        feat_hi = F.interpolate(
            feat_hi,
            size=self.high_res_hw,
            mode="bilinear",
            align_corners=True,
        )

        feats = [feat_hi]
        if self.num_output_scales > 1:
            feat_lo = self.adapter_lo(intermediates["path_3"])
            feat_lo = F.interpolate(
                feat_lo,
                size=self.low_res_hw,
                mode="bilinear",
                align_corners=True,
            )
            feats.append(feat_lo)

        depth = F.interpolate(
            depth_da.unsqueeze(1),
            size=self.depth_output_hw,
            mode="bilinear",
            align_corners=True,
        )

        if return_depth:
            return feats, {"depth": depth}
        return feats
