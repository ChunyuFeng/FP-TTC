# fpttc/rvt/range_view_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 与 ScaleEncoder 使用同一版 Deformable Attention（保证接口/行为一致）
from fpttc.scale_net.utils.multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32 as MSDeformAttn
)

# --------- 几何/网格/仿射工具 ---------

def make_range_rays(Hr, Wr, fov_up_deg=8.0, fov_down_deg=-15.0, device='cuda'):
    """
    生成 Range-View 中每个 (y,x) 像素对应的单位射线方向（LiDAR 坐标系）。
    返回:
      rays: [1, Hr, Wr, 3]  (单位向量)
      yaw:  [1, Hr, Wr]     (-pi, pi]
      pitch:[1, Hr, Wr]     [fov_down, fov_up]
    """
    fov_up = math.radians(fov_up_deg)
    fov_down = math.radians(fov_down_deg)
    fov = abs(fov_down) + abs(fov_up)

    x = torch.linspace(0, Wr - 1, Wr, device=device)
    y = torch.linspace(0, Hr - 1, Hr, device=device)
    gx, gy = torch.meshgrid(x, y, indexing='xy')
    gx = gx.t().contiguous()
    gy = gy.t().contiguous()

    yaw   = ((gx + 0.5) / Wr) * 2*math.pi - math.pi
    pitch = (1.0 - (gy + 0.5) / Hr) * fov - abs(fov_down)

    ray = torch.stack([
        torch.cos(pitch) * torch.cos(-yaw),
        torch.cos(pitch) * torch.sin(-yaw),
        torch.sin(pitch)
    ], dim=-1)  # [Hr,Wr,3]
    ray = ray / (ray.norm(dim=-1, keepdim=True) + 1e-8)
    return ray.unsqueeze(0), yaw.unsqueeze(0), pitch.unsqueeze(0)  # [1,Hr,Wr,3], ...

@torch.no_grad()
def project_points_to_image(X_lidar, K, R_l2c, t_l2c, eps=1e-6):
    """
    LiDAR 系 3D 点 -> 原始像素 uv（不裁边界，仅做 Zc>0）。
    输入:
      X_lidar: [B, N, 3], K/R:[B,3,3], t:[B,3,1] 或 [B,3]
    返回:
      uv:   [B, N, 2]
      mask: [B, N]  (Zc>0)
    """
    B, N, _ = X_lidar.shape
    R = R_l2c
    t = t_l2c
    if t.ndim == 2:  # [B,3] -> [B,3,1]
        t = t.unsqueeze(-1)

    X = X_lidar.transpose(1, 2)         # [B,3,N]
    X_cam = R @ X + t                   # [B,3,N]
    Z = X_cam[:, 2:3, :].clamp_min(eps) # [B,1,N]
    proj = K @ X_cam                    # [B,3,N]
    u = proj[:, 0, :] / Z[:, 0, :]
    v = proj[:, 1, :] / Z[:, 0, :]
    uv = torch.stack([u, v], dim=-1)    # [B,N,2]
    mask = (X_cam[:, 2, :] > 0)         # [B,N]
    return uv, mask

def apply_affine_to_uv(uv, affine_M):
    """
    把 uv (原像素系) 变换到“网络输入像素系”（与数据增强一致）。
    uv:       [B,N,2]
    affine_M: [B,3,3]
    return:   [B,N,2]
    """
    B, N, _ = uv.shape
    ones = torch.ones(B, N, 1, device=uv.device, dtype=uv.dtype)
    homo = torch.cat([uv, ones], dim=-1)                    # [B,N,3]
    out = (affine_M @ homo.transpose(1, 2)).transpose(1, 2) # [B,N,3]
    out = out[..., :2] / (out[..., 2:3] + 1e-8)
    return out

def uv_to_01(uv, W_in, H_in):
    """把像素 uv 归一化到 [0,1]（MSDeformAttn 期望的采样坐标系）。"""
    x = uv[..., 0] / (W_in - 1)
    y = uv[..., 1] / (H_in - 1)
    return torch.stack([x, y], dim=-1)

# --------- 完全复用自 ScaleEncoder 的 TransformerBlock（仅加两个可选入口） ---------

class TransformerBlock(nn.Module):
    """
    self attention + cross attention + FFN
    与 scale_net/scale_encoder.py 中一致，仅在 forward 增加：
      - ext_sampling_locations:  [B, Q, num_head, L, K, 2] 或 None
      - ext_attention_weights:   [B, Q, num_head, L, K]   或 None
    当两者提供时，**跳过** sampling_offsets/attention_weights 的线性回归，直接使用外部输入。
    """
    def __init__(self,
                 d_model=128,
                 num_head=4,
                 num_points=8,
                 num_level=2,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_points = num_points
        self.num_level = num_level
        self.num_head = num_head

        self.sampling_offsets  = nn.Linear(d_model, num_head*num_level*num_points*2)
        self.attention_weights = nn.Linear(d_model, num_head*num_level*num_points)
        self.value_proj        = nn.Linear(d_model, d_model)        # 与原版一致（可不显式使用）
        self.output_proj       = nn.Linear(d_model*num_head, d_model)
        self.dropout           = nn.Dropout(dropout)

        self.MultiScaleDeformableAttnFunction = MSDeformAttn
        self._init_weights()

    def _init_weights(self):
        from mmcv.cnn import xavier_init, constant_init
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_head, dtype=torch.float32) * (2.0*math.pi/self.num_head)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init/grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_head, 1, 1, 2
        ).repeat(1, self.num_level, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj,  distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self, query, value,
                height=None, width=None,
                query_location=None,         # [B, Q, L, 2] （未用 ext_* 时生效）
                spatial_shapes=None,         # [L,2]
                level_start_index=None,      # [L]
                ext_sampling_locations=None, # [B, Q, H, L, K, 2]
                ext_attention_weights=None   # [B, Q, H, L, K]
                ):
        """
        与原版相同的主体数据流；当 ext_* 提供时，直接喂给可变形注意力。
        输入:
          query: [B, Q, C]
          value: [B, S, H, C]   （与 ScaleEncoder 中一致：value 已经展开成 per-head 通道）
        返回:
          [B, Q, C]
        """
        B, Q, C = query.shape

        if (ext_sampling_locations is None) or (ext_attention_weights is None):
            # === 走“原始 ScaleEncoder 逻辑”：规则点 + 学习偏移 ===
            # generate sampling offsets / attn from query
            sampling_offsets = self.sampling_offsets(query).view(
                B, Q, self.num_head, self.num_level, self.num_points, 2
            )
            attention_weights = self.attention_weights(query).view(
                B, Q, self.num_head, self.num_level*self.num_points
            ).softmax(-1).view(B, Q, self.num_head, self.num_level, self.num_points)

            # 规则参考点（来自 query_location）+ 归一化
            # offset_normalizer: [..., (W_l, H_l)]
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)  # [L,2]
            sampling_locations = query_location[:, :, None, :, None, :] \
                               + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        else:
            # === 走“几何投影逻辑”：直接使用外部采样点与权重 ===
            sampling_locations = ext_sampling_locations   # [B, Q, H, L, K, 2] in [0,1]
            attention_weights  = ext_attention_weights    # [B, Q, H, L, K]

        # Deformable Attention
        out = self.MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights
        )  # 约定返回 [B, Q, H*C]

        out = self.output_proj(out)  # [B, Q, C]
        return self.dropout(out)

# --------- 与 ScaleEncoder 同结构的“外壳”，仅改“参考点生成”为几何投影 ---------

def conv(in_planes, out_planes, k=3, s=1, d=1, isReLU=True, padding_mode="zeros"):
    pad = ((k - 1) * d) // 2
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, k, s, padding=pad, dilation=d, bias=True, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=False)
        )
    else:
        return nn.Conv2d(in_planes, out_planes, k, s, padding=pad, dilation=d, bias=True, padding_mode=padding_mode)

class RangeViewTransformer(nn.Module):
    """
    复制自 ScaleEncoder 的主体骨架：
      - 保留: scale_conv / value_fs_conv / layers(ModuleList) / level_embeds / pos_enc 等
      - 仅在 forward 中，用“几何投影”生成 ext_sampling_locations / ext_attention_weights，
        并传给与 ScaleEncoder 相同的 TransformerBlock。
    用法：把“多相机同尺度特征”聚合到 Range-View 网格上。
    """
    def __init__(self,
                 num_layers=2,
                 input_dim=128,     # per-camera feature C_in
                 d_model=128,
                 nhead=4,
                 num_level=6,       # 相机数（把每个相机当作一个 level）
                 num_points=8,      # 深度 bins 数
                 fov_up=8.0,
                 fov_down=-15.0,
                 input_hw=(160, 320)):
        super().__init__()
        self.d_model = d_model
        self.nhead   = nhead
        self.num_level = num_level
        self.num_points = num_points
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.input_h, self.input_w = input_hw
        self.qbias = nn.Parameter(torch.zeros(1, input_dim, 1, 1))  # learnable zero-bias query seed

        # 与 ScaleEncoder 对齐的“query 编码器”（这里输入通常是 0，占位）
        self.scale_conv = nn.Sequential(
            conv(input_dim, d_model*2),
            conv(d_model*2, d_model),
            conv(d_model, d_model)
        )
        from fpttc.modules.position import PositionEmbeddingSine
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        # value 编码（与 ScaleEncoder 相同风格，但这里只有“单时刻多相机”，所以是 input_dim -> d_model）
        self.value_fs_conv = nn.Conv1d(input_dim, d_model, 1)

        # 层与嵌入
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_head=nhead,
                             num_points=num_points, num_level=num_level)
            for _ in range(num_layers)
        ])
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_level, input_dim))
        nn.init.xavier_uniform_(self.level_embeds)

        for m in self.layers:
            nn.init.constant_(m.sampling_offsets.weight, 0.0)
            nn.init.constant_(m.sampling_offsets.bias,   0.0)

    @staticmethod
    def _pack_levels(feats_by_cam):
        """
        feats_by_cam: list(L) of [B, C, H, W]
        return:
          value_lvls: list(L) of [B, H*W, C]
          spatial_shapes: LongTensor[L,2]
          level_start_index: LongTensor[L]
          in_sizes: list(L) of (H_in, W_in) —— 这里假设输入像素系与特征尺寸一致（若有下采样，请传入真实网络输入尺寸）
        """
        L = len(feats_by_cam)
        B, C, H, W = feats_by_cam[0].shape
        vlist, shapes, insizes = [], [], []
        for t in feats_by_cam:
            b, c, h, w = t.shape
            vlist.append(t.view(b, c, h*w).transpose(1, 2).contiguous())  # [B,HW,C]
            shapes.append([h, w])
            insizes.append((h, w))
        spatial_shapes = torch.as_tensor(shapes, device=feats_by_cam[0].device, dtype=torch.long)
        lvl_start = torch.cat([
            spatial_shapes.new_zeros(1),
            (spatial_shapes[:, 0]*spatial_shapes[:, 1]).cumsum(0)[:-1]
        ])
        return vlist, spatial_shapes, lvl_start, insizes

    def _make_geom_sampling(self, B, Hr, Wr, depth_bins, cam_K, cam_R, cam_t, affine_M, in_sizes):
        """
        生成几何版 sampling_locations 与 valid mask。
        返回:
          samp_loc: [B, Q, H, L, K, 2] in [0,1]
          attn_msk: [B, Q, H, L, K]（这里只返回可见性/边界 mask；后续会与由 query 预测的权重相乘并归一化）
        """
        device = cam_K[0].device
        H = Hr; W = Wr; Kbins = depth_bins.numel(); L = len(cam_K); HWr = H*W

        rays, _, _ = make_range_rays(H, W, self.fov_up, self.fov_down, device=device)  # [1,H,W,3]
        rays = rays.view(1, HWr, 3).repeat(B, 1, 1)                                     # [B,Q,3]
        depth = depth_bins.view(1, 1, Kbins, 1).to(device)                              # [1,1,K,1]
        pts = rays.unsqueeze(2) * depth                                                 # [B,Q,K,3]

        loc_lvls = []
        msk_lvls = []
        for l in range(L):
            Ki, Ri, ti, Ai = cam_K[l], cam_R[l], cam_t[l], affine_M[l]
            Hin, Win = in_sizes[l]

            sx = Win / self.input_w
            sy = Hin / self.input_h
            S = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], device=Ai.device, dtype=Ai.dtype)
            Affine_feat = S @ Ai

            X = pts.view(B, -1, 3)                                    # [B,Q*K,3]
            uv, zmask = project_points_to_image(X, Ki, Ri, ti)        # [B,QK,2], [B,QK]
            uv_aug = apply_affine_to_uv(uv, Affine_feat)                       # [B,QK,2]

            u = uv_aug[..., 0]; v = uv_aug[..., 1]
            inb = (u >= 0) & (u <= (Win - 1)) & (v >= 0) & (v <= (Hin - 1))  # [B,QK]
            valid = (zmask & inb).view(B, HWr, Kbins)                         # [B,Q,K]

            uv_01 = uv_to_01(uv_aug, Win, Hin).view(B, HWr, Kbins, 2)        # [B,Q,K,2]
            loc_lvls.append(uv_01)                                           # 列表长度 L
            msk_lvls.append(valid)

        # [B,Q,L,K,2] / [B,Q,L,K]
        samp = torch.stack(loc_lvls, dim=2)
        msk  = torch.stack(msk_lvls,  dim=2)

        # 扩到多头： [B,Q,H,L,K,2] / [B,Q,H,L,K]
        samp = samp.unsqueeze(2).repeat(1, 1, self.nhead, 1, 1, 1)
        msk  =  msk.unsqueeze(2).repeat(1, 1, self.nhead, 1, 1)
        return samp, msk

    def forward(self,
                feats_by_cam,         # list(L) of [B, C_in, Hf, Wf]
                cam_K, cam_R, cam_t,  # list(L) of [B,3,3],[B,3,3],[B,3]或[B,3,1]
                affine_M,             # list(L) of [B,3,3]
                Hr, Wr,
                depth_bins=None,      # Tensor[K] （若 None，则自动对数均匀采样）
                ini_query=None):
        """
        输出：Range-View 聚合特征 [B, d_model, Hr, Wr]
        """
        B, C_in, _, _ = feats_by_cam[0].shape
        device = feats_by_cam[0].device

        # 1) 构造“query特征”，保持与 ScaleEncoder 风格一致（有 pos_enc）
        if ini_query is None:
            # 这里没有天然的 Range-View 输入，给零特征也可以（scale_conv 的 bias/BN 不会引入问题）
            # ini_query = torch.zeros(B, C_in, Hr, Wr, device=device)
            ini_query = self.qbias.expand(B, C_in, Hr, Wr)
        query = self.scale_conv(ini_query)            # [B,d_model,Hr,Wr]
        query = query + self.pos_enc(query)           # 与 ScaleEncoder 一致
        query = query.flatten(2).transpose(1, 2).contiguous()  # [B,Q,C]

        # 2) value 组织（与 ScaleEncoder 一致：flatten + level_embed + 1×1 conv 到 d_model）
        value_lvls, spatial_shapes, lvl_start, in_sizes = self._pack_levels(feats_by_cam)  # list(L)[B,HW,C_in]
        value_with_le = []
        for l, v in enumerate(value_lvls):
            v = v + self.level_embeds[None, l:l+1, :].to(v.dtype)  # [B,HW,C_in]
            value_with_le.append(v)
        value_cat = torch.cat(value_with_le, dim=1)    # [B, sum(HW_l), C_in]
        # Conv1d 按通道把 C_in -> d_model（与 ScaleEncoder 相同做法）
        value = self.value_fs_conv(value_cat.transpose(1, 2)).transpose(1, 2)  # [B,S,d_model]
        # 展开到 per-head 通道（保持 ScaleEncoder 的接口）
        value = value.unsqueeze(2).repeat(1, 1, self.nhead, 1)                  # [B,S,H,C]

        # 3) 参考点：与 ScaleEncoder 一致需要 query_location（但我们走几何分支，用 ext_* 覆盖即可）
        #    仍构造占位的 query_location（全零）以匹配接口
        query_location = torch.zeros(B, Hr*Wr, self.num_level, 2, device=device, dtype=value.dtype)

        # 4) 深度 bins
        if depth_bins is None:
            d_min, d_max = 1.0, 40.0
            K = self.num_points
            depth_bins = torch.logspace(math.log10(d_min), math.log10(d_max), K, device=device)
        else:
            K = depth_bins.numel()
            assert K == self.num_points, f"num_points({self.num_points}) != len(depth_bins)({K})"

        # 5) 生成“几何采样点 + 可见性 mask”，并据此构造 ext_attention_weights
        ext_loc, vis_mask = self._make_geom_sampling(
            B, Hr, Wr, depth_bins, cam_K, cam_R, cam_t, affine_M, in_sizes
        )  # [B,Q,H,L,K,2], [B,Q,H,L,K] bool

        # # 用 query 回归的注意力（与 ScaleEncoder 一致），再乘 mask、按 K 归一化
        # # 先建立一个“哑层”来复用 attention_weights 的线性头（不想改 Block 的话也可复制参数）
        # attn_head = getattr(self, "_attn_head", None)
        # if attn_head is None:
        #     self._attn_head = nn.Linear(self.d_model, self.nhead*self.num_level*self.num_points).to(device)
        #     # 初始化为与 TransformerBlock.attention_weights 相同风格（零+softmax），也可拷贝参数
        #     nn.init.constant_(self._attn_head.weight, 0.0)
        #     nn.init.constant_(self._attn_head.bias,   0.0)
        #     attn_head = self._attn_head

        # attn_raw = attn_head(query).view(B, Hr*Wr, self.nhead, self.num_level, self.num_points)  # [B,Q,H,L,K]
        # attn_raw = F.softmax(attn_raw, dim=-1)
        # ext_attn = attn_raw * vis_mask.float()
        # denom = ext_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # ext_attn = ext_attn / denom  # [B,Q,H,L,K]

        # 6) 逐层 DeformableAttn
        out = query
        for layer in self.layers:
            # === (A) 每层重算注意力，并与可见性 mask 融合 + 在 L×K 上归一化 ===
            attn_raw = layer.attention_weights(out)                           # [B, Q, H*L*K]
            attn_raw = attn_raw.view(B, Hr*Wr, self.nhead, self.num_level*self.num_points)
            attn_raw = F.softmax(attn_raw, dim=-1)                            # over (L*K)
            attn_raw = attn_raw.view(B, Hr*Wr, self.nhead, self.num_level, self.num_points)
            ext_attn = attn_raw * vis_mask.float()                            # [B,Q,H,L,K]
            # 在 L×K 维度上重归一化，确保每个 head 的权重和为 1
            ext_attn = ext_attn / ext_attn.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

            # === (B) 几何采样位置 + 学习偏移 ===
            # 生成未归一化的像素偏移 Δ，形状与 ext_loc 对齐
            delta = layer.sampling_offsets(out)                                # [B, Q, H*L*K*2]
            delta = delta.view(B, Hr*Wr, self.nhead, self.num_level, self.num_points, 2)

            # 以 (W_l, H_l) 为归一化因子，把像素偏移变成 [0,1] 空间的增量
            # spatial_shapes: [L, 2] = (H_l, W_l)，构造 [L,2] = (W_l, H_l)
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1      # [L,2] = (W, H)
            ).to(delta.dtype).to(delta.device)
            offset_normalizer = offset_normalizer.view(1, 1, 1, self.num_level, 1, 2)  # [1,1,1,L,1,2]

            delta01 = delta / offset_normalizer                                # 像素 → 归一化
            sampling_locations = (ext_loc + delta01).clamp(0.0, 1.0)           # 叠加并裁剪

            # === (C) 本层 Deformable Attention（位置=几何+学习偏移，权重=融合 vis_mask 后的注意力）
            out = layer(
                out, value,
                height=Hr, width=Wr,
                query_location=query_location,          # 占位，无实际用处
                spatial_shapes=spatial_shapes,
                level_start_index=lvl_start,
                ext_sampling_locations=sampling_locations,
                ext_attention_weights=ext_attn
            )

        # 7) reshape 回 Range-View 张量
        out = out.view(B, Hr, Wr, self.d_model).permute(0, 3, 1, 2).contiguous()  # [B,C,Hr,Wr]
        return out
