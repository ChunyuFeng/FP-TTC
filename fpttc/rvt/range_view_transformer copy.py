# fpttc/rvt/range_view_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用已有的 FP32 版可变形注意力
from fpttc.scale_net.utils.multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32 as MSDeformAttn
)

# ========== 工具函数（几何/网格/仿射） ==========

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

    # yaw ∈ [-pi, pi]
    x = torch.linspace(0, Wr - 1, Wr, device=device)
    y = torch.linspace(0, Hr - 1, Hr, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')  # [Wr,Hr] -> we want [Hr,Wr]
    grid_x = grid_x.t().contiguous()  # [Hr,Wr]
    grid_y = grid_y.t().contiguous()

    yaw   = ((grid_x + 0.5) / Wr) * 2*math.pi - math.pi
    pitch = (1.0 - (grid_y + 0.5) / Hr) * fov - abs(fov_down)

    # 单位射线
    # NuScenes 坐标系：x右y前z上
    # 这里采用：x前y左z上 将车的正前方调整到range image的左1/4处 与真值对齐
    ray = torch.stack([
        torch.cos(pitch) * torch.cos(-yaw),
        torch.cos(pitch) * torch.sin(-yaw),
        torch.sin(pitch)
    ], dim=-1)  # [Hr,Wr,3]
    ray = ray / (ray.norm(dim=-1, keepdim=True) + 1e-8)

    return ray.unsqueeze(0), yaw.unsqueeze(0), pitch.unsqueeze(0)


def project_points_to_image(X_lidar, K, R_l2c, t_l2c, eps=1e-6):
    """
    LiDAR 坐标系 3D 点 -> 原始图像像素 (u,v)
    返回:
      uv:   [B, N, 2]  （原始图像坐标系）
      mask: [B, N]     （仅 Zc>0；不做边界裁剪——边界裁剪在 apply_affine_to_uv 之后进行）
    """
    B, N, _ = X_lidar.shape
    R, t = R_l2c, t_l2c.unsqueeze(-1)   # [B,3,3], [B,3,1]

    X = X_lidar.transpose(1, 2)           # [B,3,N]
    X_cam = R @ X + t                     # [B,3,N]
    Z = X_cam[:, 2:3, :].clamp_min(eps)   # [B,1,N]
    proj = K @ X_cam                      # [B,3,N]
    u = proj[:, 0, :] / Z[:, 0, :]
    v = proj[:, 1, :] / Z[:, 0, :]
    uv = torch.stack([u, v], dim=-1)      # [B,N,2]
    mask = (X_cam[:, 2, :] > 0)           # [B,N] 仅前向可见性
    return uv, mask

def apply_affine_to_uv(uv, affine_M):
    """
    对像素坐标应用与你的数据增强一致的仿射: p_new = M @ p_old.
    uv:       [B, N, 2] in original pixel
    affine_M: [B, 3, 3]
    return:   [B, N, 2] in network input pixel
    """
    B, N, _ = uv.shape
    ones = torch.ones(B, N, 1, device=uv.device, dtype=uv.dtype)
    homo = torch.cat([uv, ones], dim=-1)                 # [B,N,3]
    out = (affine_M @ homo.transpose(1, 2)).transpose(1, 2)  # [B,N,3]
    out = out[..., :2] / (out[..., 2:3] + 1e-8)
    return out


def norm_uv_to_grid(uv, W_in, H_in):
    """
    把像素坐标 uv (in network input image space) 归一化到 [-1,1] 作为 grid_sample / deformable attn 的采样坐标
    返回: [B, N, 2] in [-1,1]
    """
    # 注意 MSDeformAttn 的 sampling_locations 期望的是 [0,1] 归一化; 
    # 我们会在主层里做 [0,1] 归一化，这里给出通用 [-1,1] 或 [0,1] 的两种接口
    x = uv[..., 0]
    y = uv[..., 1]
    # 这里返回 [0,1] 坐标，便于直接喂给 MSDeformAttn
    xn = x / (W_in - 1)
    yn = y / (H_in - 1)
    return torch.stack([xn, yn], dim=-1)  # [B,N,2]


# ========== RVT Layer（一次聚合） ==========

class RVTLayer(nn.Module):
    """
    一个可变形注意力层：以 Range-View 的 query 为基，跨 camera-level 对 value 进行采样聚合。
    - 采样点：按照 Range 射线 + K 个深度 bins 投影到每个相机（每个 level）。
    - attention_weights：由 query 预测；(可选) 学习到对不同 level 的权重。
    - (可选) delta_offsets：对几何参考点做微小偏移，提升鲁棒性。
    """
    def __init__(self, d_model=128, num_heads=4, num_points=8, use_delta=False, cam_embed=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_points = num_points  # == K
        self.use_delta = use_delta
        self.cam_embed = cam_embed

        # 将 value features 投到 d_model（若输入已对齐可去掉）
        self.value_proj = nn.Linear(d_model, d_model)
        # attn 权重头: [B, Q, H*L*K]
        self.attn_proj = nn.Linear(d_model, num_heads * 1 * num_points)  # L 维度由外层在 concat 后通过 repeat/expand 处理
        # (可选) 对参考点做微调
        if use_delta:
            self.delta_proj = nn.Linear(d_model, num_heads * 1 * num_points * 2)

        # 输出合并
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query_feat,           # [B, Hr, Wr, C]
        rays_lidar,           # [1, Hr, Wr, 3]
        depth_bins,           # [K]  (正数，单位米)
        value_feats_lvls,     # list of L tensors, each [B, H_l*W_l, C]
        spatial_shapes,       # [L, 2]  (H_l, W_l)
        level_start_index,    # [L]
        cam_K, cam_R, cam_t,  # each list of L tensors: K:[B,3,3], R:[B,3,3], t:[B,3,1]
        affine_M,             # list(L) of [B,3,3]  (图像增强仿射)
        in_sizes,             # list(L) of (H_in, W_in)  输入特征对应的网络输入大小(像素空间)
        cam_embed_table=None  # [L, C] 或 None
    ):
        """
        输出：
         - range_feat: [B, Hr, Wr, C]
        说明：
         - 这里我们直接构造 sampling_locations 给 MSDeformAttn，用几何参考点+可选 delta 而非 'query_location + offsets'
        """
        B, Hr, Wr, C = query_feat.shape
        L = len(value_feats_lvls)
        HWr = Hr * Wr
        device = query_feat.device
        K = self.num_points

        # 1) query 构造
        q = query_feat.view(B, HWr, C)  # [B, Q, C]
        if cam_embed_table is not None and self.cam_embed:
            # 简单做法：把所有 level 的 embed 求均值加到 q 上；更复杂可以在 attn 权重上做 bias
            q = q + cam_embed_table.mean(dim=0, keepdim=True)  # [L,C] -> broadcast

        q_proj = q  # 也可以做线性映射

        # 2) 由 q 预测 attn 权重（先按 1 个 level 预测，再在 L 维复制/归一化）
        attn = self.attn_proj(q_proj)  # [B, Q, H*K]
        attn = attn.view(B, HWr, self.num_heads, 1, K)  # [B,Q,H,1,K]
        attn = attn.repeat(1, 1, 1, L, 1)               # [B,Q,H,L,K]
        attn = F.softmax(attn, dim=-1)

        # 3) 构造几何参考点（每个 level 一个相机）
        #   对每个 RV 像素 q：ray * depth_bins -> N=K 3D 点，投影到各相机 -> uv -> 仿射 -> 归一化到 [0,1]
        #   组合得到 sampling_locations: [B, Q, H, L, K, 2]
        rays = rays_lidar.view(1, HWr, 3).repeat(B, 1, 1)  # [B,Q,3]
        depth = depth_bins.view(1, 1, K, 1).to(device)     # [1,1,K,1]
        pts = rays.unsqueeze(2) * depth                    # [B,Q,K,3]

        sampling_locations = []
        valid_masks = []
        for l in range(L):
            # 相机参数
            Kb = cam_K[l]       # [B,3,3]
            Rb = cam_R[l]       # [B,3,3]
            tb = cam_t[l]       # [B,3,1]
            Ab = affine_M[l]    # [B,3,3]
            H_in, W_in = in_sizes[l]

            # 展开 3D 点并投影
            X = pts.view(B, -1, 3)                               # [B, Q*K, 3]
            uv, zmask = project_points_to_image(X, Kb, Rb, tb)   # uv:[B,Q*K,2], zmask:[B,Q*K]
            # 应用图像增强仿射
            uv_aug = apply_affine_to_uv(uv, Ab)                  # [B,Q*K,2]

            # —— 边界 mask（在预处理后坐标系上做）——
            u = uv_aug[..., 0]
            v = uv_aug[..., 1]
            inb = (u >= 0) & (u <= (W_in - 1)) & (v >= 0) & (v <= (H_in - 1))  # [B,Q*K]
            valid = zmask & inb                                                # [B,Q*K]

            # 归一化到 [0,1]
            uv_01 = norm_uv_to_grid(uv_aug, W_in, H_in)          # [B,Q*K,2]

            # 失效点 mask（超出 FOV 的点会在采样时被丢弃/attention→0）

            # reshape 成 [B,Q,K,2]
            loc = uv_01.view(B, HWr, K, 2)
            sampling_locations.append(loc)

            valid_masks.append(valid.view(B, HWr, K))

        # 拼到 [B,Q,L,K,2] 再广播到 heads 维
        sampling_locations = torch.stack(sampling_locations, dim=2)  # [B,Q,L,K,2]
        sampling_locations = sampling_locations.unsqueeze(2).repeat(1, 1, self.num_heads, 1, 1, 1)
        # -> [B,Q,H,L,K,2]

        valid_masks = torch.stack(valid_masks, dim=2)      # [B,Q,L,K]
        valid_masks = valid_masks.unsqueeze(2).float()     # [B,Q,1,L,K] 以便与 attn [B,Q,H,L,K] 广播

        attn = attn * valid_masks  # 让无效点的 attn 权重为 0
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6) # 在 K 维归一化，防止全 0
        attn = attn / denom


        # (可选) delta 偏移
        if self.use_delta:
            delta = self.delta_proj(q_proj)  # [B,Q,H*K*2]
            delta = delta.view(B, HWr, self.num_heads, 1, K, 2).repeat(1, 1, 1, L, 1, 1)
            # 归一化：以相机分辨率为尺度的相对偏移；也可直接学习 [0,1] 偏移
            sampling_locations = sampling_locations + delta.tanh() * 0.01  # 小范围扰动

        # 4) value 准备：拼接各 level 并投影到 d_model
        value = torch.cat(value_feats_lvls, dim=1)        # [B, S, C]
        value = self.value_proj(value)                    # [B, S, d_model]

        # 将 value 拆成 [B, S, n_heads, C_per_head]
        Bv, Sv, Cv = value.shape
        assert Cv % self.num_heads == 0, f"value dim {Cv} not divisible by num_heads {self.num_heads}"
        CpH = Cv // self.num_heads
        value = value.view(Bv, Sv, self.num_heads, CpH).contiguous()   # [B, S, H, CpH]

        # 5) MSDeformAttn 前向
        #  sampling_locations: [B, Q, H, L, K, 2]
        #  attn:               [B, Q, H, L, K]
        out = MSDeformAttn.apply(
            value,                      # [B, S, H, CpH]
            spatial_shapes,             # [L, 2] (long/int32)
            level_start_index,          # [L]    (long/int32)
            sampling_locations,         # [B, Q, H, L, K, 2]  in [0,1]
            attn                        # [B, Q, H, L, K]
        )                               # -> [B, Q, H, CpH]

        # 展平成 [B, Q, d_model] 再线性
        out = out.reshape(B, Hr*Wr, Cv).contiguous()      # [B, Q, d_model]
        out = self.output_proj(out)                       # [B, Q, d_model]


        return out.view(B, Hr, Wr, C)


# ========== 顶层聚合器（每个尺度独立调用） ==========

class RangeViewAggregator(nn.Module):
    """
    入口模块：对“同一尺度”的多相机特征或 corr 进行聚合。
    - 实例化两份：一份聚合“特征”，一份聚合“corr”（通道不同）。
    """
    def __init__(
        self,
        d_model=128,
        num_heads=4,
        num_points=8,
        use_delta=False,
        cam_embed=True,
        fov_up=8.0,
        fov_down=-15.0,
    ):
        super().__init__()
        self.layer = RVTLayer(d_model=d_model, num_heads=num_heads, num_points=num_points,
                              use_delta=use_delta, cam_embed=cam_embed)
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.register_parameter("cam_level_embed", nn.Parameter(torch.zeros(6, d_model)))  # 简化示例，实际可做 [L,C]

    @staticmethod
    def _pack_levels(feats_by_cam):
        """
        feats_by_cam: list of V tensors, each [B, C, Hf, Wf]
        返回：
          value_lvls: [ [B, Hf*Wf, C] for each level ]
          spatial_shapes: [L,2]
          level_start_index: [L]
          in_sizes: [(H_in, W_in) for each level]  (这里 H_in,W_in = 特征对应的输入图像像素大小；若等于 Hf,Wf 也可直接填)
        """
        L = len(feats_by_cam)
        B, C, Hf, Wf = feats_by_cam[0].shape
        value_lvls = []
        spatial_shapes = []
        in_sizes = []
        for t in feats_by_cam:
            b, c, h, w = t.shape
            value_lvls.append(t.view(b, c, h * w).transpose(1, 2).contiguous())  # [B,HW,C]
            spatial_shapes.append([h, w])
            in_sizes.append((h, w))  # 若有下采样/输入像素->特征像素比例，这里应填“网络输入分辨率”
        spatial_shapes = torch.as_tensor(spatial_shapes, device=feats_by_cam[0].device, dtype=torch.long)
        lvl_start_idx = torch.cat([spatial_shapes.new_zeros(1), (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]])
        return value_lvls, spatial_shapes, lvl_start_idx, in_sizes

    def forward(
        self,
        query_feat,              # [B,C,Hr,Wr] 或 None（可用零特征 + 位置编码）
        feats_by_cam,            # list(V) of [B,C,Hf,Wf]
        cam_K, cam_R, cam_t,     # list(V) of [B,3,3 / 3,3 / 3,1]
        affine_M,                # list(V) of [B,3,3]
        Hr, Wr
    ):
        """
        返回: [B,C,Hr,Wr] 的 Range-View 聚合结果
        """
        B, C, Hf, Wf = feats_by_cam[0].shape
        device = feats_by_cam[0].device

        # 1) 构造 Range-View query 特征（若外部无，则零初始化 + 位置编码）
        if query_feat is None:
            query_feat = torch.zeros(B, C, Hr, Wr, device=device)
        query_feat = query_feat.permute(0, 2, 3, 1).contiguous()  # [B,Hr,Wr,C]

        # 2) 射线
        rays, _, _ = make_range_rays(Hr, Wr, self.fov_up, self.fov_down, device=device)  # [1,Hr,Wr,3]

        # 3) pack levels（将 V 个相机当作 L 个 level）
        value_lvls, spatial_shapes, lvl_start_idx, in_sizes = self._pack_levels(feats_by_cam)

        # 4) 构造深度 bins
        #    经验：对数均匀更稳；范围视数据而定，可配置
        K = self.layer.num_points
        d_min, d_max = 1.0, 40.0
        depth_bins = torch.logspace(math.log10(d_min), math.log10(d_max), K, device=device)

        # 5) 进入单层 RVT
        out = self.layer(
            query_feat=query_feat,                 # [B,Hr,Wr,C]
            rays_lidar=rays,                       # [1,Hr,Wr,3]
            depth_bins=depth_bins,                 # [K]
            value_feats_lvls=value_lvls,           # list(L)[B,HW,C]
            spatial_shapes=spatial_shapes,         # [L,2]
            level_start_index=lvl_start_idx,       # [L]
            cam_K=cam_K, cam_R=cam_R, cam_t=cam_t, # list(L)[B,3,3/3,3/3,1]
            affine_M=affine_M,                     # list(L)[B,3,3]
            in_sizes=in_sizes,                     # list(L)[(H_in, W_in)]
            cam_embed_table=None                   # 需要时换成 [L,C] 的表
        )
        return out.permute(0, 3, 1, 2).contiguous()  # [B,C,Hr,Wr]
