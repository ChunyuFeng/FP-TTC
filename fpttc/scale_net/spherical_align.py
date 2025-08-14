# fpttc/scale_net/spherical_align.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def grid_sample_spherical(x, dx, dy):
    """
    对 range-view 特征做可学习的小位移 (dx,dy)，单位：像素。
    - 水平方向采用 360° wrap（模 W）
    - 竖直方向 clamp 到 [0,H-1]
    Args:
      x : [B,C,H,W]
      dx: [B,1,H,W]  (+右移)
      dy: [B,1,H,W]  (+下移)
    """
    B, C, H, W = x.shape
    device = x.device
    # 基网格（像素）
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=x.dtype),
        torch.arange(W, device=device, dtype=x.dtype),
        indexing='ij'
    )
    xs = xs.unsqueeze(0).expand(B,-1,-1)   # [B,H,W]
    ys = ys.unsqueeze(0).expand(B,-1,-1)

    x_pix = xs + dx.squeeze(1)             # [B,H,W]
    y_pix = ys + dy.squeeze(1)

    # 水平 wrap
    x_pix = torch.remainder(x_pix, W)
    # 垂直 clamp
    y_pix = y_pix.clamp(0, H-1)

    # 归一化到 [-1,1]
    x_norm = 2.0 * x_pix / (W-1.0) - 1.0
    y_norm = 2.0 * y_pix / (H-1.0) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)  # [B,H,W,2]

    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

class SphericalAlignment(nn.Module):
    """
    把“软投影后的 prev/curr 球面特征”做一次可学习对齐：
      输入: prev_feat, curr_feat (可选额外 cond)
      输出: prev_feat (原样), curr_feat_warped, 以及位移场 (dx,dy)
    默认只把 curr 对齐到 prev（也最常用）。
    """
    def __init__(self, in_ch, hidden=128, max_shift=2.0):
        super().__init__()
        self.max_shift = float(max_shift)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch*2, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1)  # dx, dy, gate
        )

    def forward(self, prev_feat, curr_feat):
        B,C,H,W = prev_feat.shape
        x = torch.cat([prev_feat, curr_feat], dim=1)   # [B,2C,H,W]
        out = self.net(x)
        dx  = torch.tanh(out[:, :1]) * self.max_shift  # [-max,+max] 像素
        dy  = torch.tanh(out[:, 1:2]) * self.max_shift
        g   = torch.sigmoid(out[:, 2:3])               # [0,1] 门控

        curr_warp = grid_sample_spherical(curr_feat, dx, dy)       # 水平 wrap
        curr_aligned = g * curr_warp + (1.0 - g) * curr_feat       # 残差门
        return prev_feat, curr_aligned, (dx,dy,g)
