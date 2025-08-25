import torch
import torch.nn.functional as F
import math
from torch_scatter import scatter_min
import open3d as o3d
import numpy as np

# ----------------------------
#  像素<->相机<->自车 的几何工具
# ----------------------------

# 在线 proj_pix 用 z-buffer + 近似回写 uv（快且稳定）。
# 要追求完全严谨的 (cam_idx,u,v)，可以把 输入像素网格 uv 保存并一路跟随 gather；
# 实现略长，这里给你先跑通版本。

def _make_homog_torch(R: torch.Tensor, t: torch.Tensor):
    """R:[...,3,3], t:[...,3] -> T:[...,4,4]"""
    *bs, _, _ = R.shape
    T = torch.eye(4, device=R.device, dtype=R.dtype).expand(*bs, 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3]  = t
    return T

def _expand_K_to_BV(K, B, V, device, dtype):
    """
    把输入 K 扩展为 [B, V, 3, 3]
    允许的输入形状：
      [3,3] 或 [V,3,3] 或 [B,3,3] 或 [B,V,3,3]
    """
    if K.ndim == 2:                       # [3,3]
        K = K.view(1, 1, 3, 3).to(device=device, dtype=dtype).expand(B, V, 3, 3)
    elif K.ndim == 3:
        if K.shape[0] == V:               # [V,3,3]
            K = K.to(device=device, dtype=dtype).unsqueeze(0).expand(B, V, 3, 3)
        elif K.shape[0] == B:             # [B,3,3]
            K = K.to(device=device, dtype=dtype).unsqueeze(1).expand(B, V, 3, 3)
        else:
            raise ValueError(f"K shape {tuple(K.shape)} cannot broadcast to (B,V,3,3) with B={B}, V={V}")
    elif K.ndim == 4:                     # [B,V,3,3]
        assert K.shape[:2] == (B, V), f"K shape {tuple(K.shape)} mismatches B={B}, V={V}"
        K = K.to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported K ndim={K.ndim}")
    return K

def backproject_depth_to_cam(depth, K):
    """
    depth: [B,V,1,H,W]（米）
    K    : [3,3] or [V,3,3] or [B,3,3] or [B,V,3,3]
    return:
      Xc  : [B,V,3,H,W]，相机坐标系下的 3D 点 (x,y,z)
    """
    assert depth.ndim == 5 and depth.size(2) == 1, f"depth shape {tuple(depth.shape)} must be [B,V,1,H,W]"
    B, V, _, H, W = depth.shape
    device = depth.device
    dtype  = depth.dtype

    # 构造 (u,v,1) 网格：[3,H,W]，再扩展到 [B,V,3,H,W]
    u = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)   # 每行从 0..W-1
    v = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)   # 每列从 0..H-1
    ones = torch.ones((H, W), device=device, dtype=dtype)
    pix = torch.stack([u, v, ones], dim=0)                  # [3,H,W]
    pix = pix.view(1, 1, 3, H, W).expand(B, V, 3, H, W)     # [B,V,3,H,W]

    # 扩展 K -> [B,V,3,3] 并求逆
    K = _expand_K_to_BV(K, B, V, device, dtype)
    Kinv = torch.inverse(K)                                 # [B,V,3,3]

    # batched matmul：([B,V,3,3] @ [B,V,3,N]) -> [B,V,3,N]
    N = H * W
    pix_flat  = pix.view(B, V, 3, N)                        # [B,V,3,N]
    rays_flat = torch.matmul(Kinv, pix_flat)                # [B,V,3,N]
    rays = rays_flat.view(B, V, 3, H, W)                    # [B,V,3,H,W]

    # 乘以深度，得到相机坐标
    Xc = rays * depth                                       # 广播到 [B,V,3,H,W]
    return Xc


def cam_to_cam(Xc_src, T_E_from_C_src, T_E_from_C_tgt):
    """
    Xc_src           : [B,V,3,H,W]  in src camera frame
    T_E_from_C_src   : [B,V,4,4]
    T_E_from_C_tgt   : [B,V,4,4]
    return Xc_tgt    : [B,V,3,H,W]
    """
    B,V,_,H,W = Xc_src.shape
    device = Xc_src.device
    ones = torch.ones((B,V,1,H,W), device=device, dtype=Xc_src.dtype)
    Xh = torch.cat([Xc_src, ones], dim=2).view(B,V,4,-1)             # [B,V,4,N]
    Xe = (T_E_from_C_src @ Xh).view(B,V,4,-1)                        # Ego
    T_C_from_E_tgt = torch.inverse(T_E_from_C_tgt)                   # [B,V,4,4]
    Xc_tgt_h = (T_C_from_E_tgt @ Xe).view(B,V,4,-1)
    Xc_tgt   = Xc_tgt_h[:, :, :3, :].view(B,V,3,H,W)
    return Xc_tgt

def project_cam_to_pixels(Xc, K, H_img, W_img, clamp=True):
    """
    Xc   : [B,V,3,H,W]
    K    : [B,V,3,3]
    return norm_grid: [B*V, H, W, 2] in [-1,1], valid:[B,V,1,H,W], z:[B,V,1,H,W], uv:[B,V,2,H,W]
    """
    B,V,_,H,W = Xc.shape
    X = Xc[:, :, 0]; Y = Xc[:, :, 1]; Z = Xc[:, :, 2].clamp(min=1e-6)
    fx = K[:, :, 0, 0].view(B,V,1,1);  fy = K[:, :, 1, 1].view(B,V,1,1)
    cx = K[:, :, 0, 2].view(B,V,1,1);  cy = K[:, :, 1, 2].view(B,V,1,1)
    u = fx*X/Z + cx
    v = fy*Y/Z + cy

    # valid inside image
    valid = (Z > 0) & (u >= 0) & (u <= W_img-1) & (v >= 0) & (v <= H_img-1)
    # grid_sample normalized grid
    u_n = 2.0 * (u/(W_img-1.0)) - 1.0
    v_n = 2.0 * (v/(H_img-1.0)) - 1.0
    grid = torch.stack([u_n, v_n], dim=-1)  # [B,V,H,W,2]
    if clamp:
        grid = grid.clamp(-1, 1)

    grid = grid.view(B*V, H, W, 2)
    return grid, valid.float(), Z.unsqueeze(2), torch.stack([u, v], dim=2)

def warp_image(src_img, depth_src, K_src, T_E_from_C_src, K_tgt, T_E_from_C_tgt, H_img, W_img):
    """
    src_img  : [B,V,3,H,W] (0..1)
    depth_src: [B,V,1,H,W]
    return: recon_on_tgt [B,V,3,H,W], valid_mask [B,V,1,H,W]
    """
    Xc_src = backproject_depth_to_cam(depth_src, K_src)
    Xc_tgt = cam_to_cam(Xc_src, T_E_from_C_src, T_E_from_C_tgt)
    grid, valid, _, _ = project_cam_to_pixels(Xc_tgt, K_tgt, H_img, W_img)
    src = src_img.view(src_img.shape[0]*src_img.shape[1], src_img.shape[2], H_img, W_img)
    recon = F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    recon = recon.view_as(src_img)
    return recon, valid

def warp_depth(depth_src, K_src, T_E_from_C_src, K_tgt, T_E_from_C_tgt, H_img, W_img):
    """
    把 src 的深度投到 tgt 像素坐标系：
    返回 (z_pred_in_tgt, uv_in_tgt, valid_mask)
    """
    Xc_src = backproject_depth_to_cam(depth_src, K_src)
    Xc_tgt = cam_to_cam(Xc_src, T_E_from_C_src, T_E_from_C_tgt)
    grid, valid, z_tgt, uv_tgt = project_cam_to_pixels(Xc_tgt, K_tgt, H_img, W_img)
    return z_tgt, uv_tgt, valid

@torch.no_grad()
def build_proj_pix_from_depth_torch(
    depth,               # [B,V,1,H,W] —— “已做数据增强后的”深度图（>=0，无效为0）
    K,                   # [B,V,3,3]   —— 原始相机内参（未叠加仿射）
    T_E_from_C,          # [B,V,4,4]   —— Camera -> Ego 外参
    H_r: int, W_r: int,  # 目标 range-view 尺寸
    fov_up: float, fov_down: float,  # 激光雷达垂直视场（度）
    affine_matrix=None   # None 或 [B,3,3] 或 [B,V,3,3]（把增强后的像素折回原图坐标再乘 K^-1）
):
    """
    返回:
      proj_pix:   [B,H_r,W_r,3] long  = (cam_idx, u, v)
                  - cam_idx ∈ {-1, 0..V-1}，-1 表示该像素未被任何点覆盖
                  - (u,v) 为“增强后图像”的像素坐标（整数，未覆盖为 -1）
      proj_mask:  [B,1,H_r,W_r] float (0/1)
      proj_range: [B,1,H_r,W_r] float  最近点的 3D 距离（未覆盖为 -1）
    说明:
      - 水平展开为 yaw∈[-pi,pi] 的线性映射；垂直由 pitch 与 (fov_up, fov_down) 映射。
      - 对每个 range 像素用 scatter_min 选择“最近深度”(最小 r) 的来源像素。
      - 不对 proj_pix 做任何插值或平滑（严禁 bilinear），仅整数索引。
    """
    assert depth.dim() == 5 and depth.size(2) == 1, "depth must be [B,V,1,H,W]"
    B, V, _, H, W = depth.shape
    device = depth.device
    dtype  = depth.dtype

    # ---------- 构建增强后的像素网格 (u,v,1) ----------
    uu, vv = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing='xy'
    )  # [H,W]
    uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=0).view(3, H*W)  # [3,N]
    N = H * W

    # ---------- 处理 affine：把增强后像素折回原图坐标 ----------
    if affine_matrix is None:
        invA = None
    else:
        if affine_matrix.dim() == 3:        # [B,3,3]
            invA = torch.inverse(affine_matrix).unsqueeze(1).expand(B, V, 3, 3)
        elif affine_matrix.dim() == 4:      # [B,V,3,3]
            invA = torch.inverse(affine_matrix)
        else:
            raise ValueError("affine_matrix must be None, [B,3,3] or [B,V,3,3]")

    # ---------- K^-1 ----------
    Kinv = torch.inverse(K)                 # [B,V,3,3]

    # ---------- 反投影到相机系 ----------
    uv1_bvn = uv1.view(1, 1, 3, N).expand(B, V, 3, N)      # [B,V,3,N]
    if invA is not None:
        pix_orig = torch.matmul(invA, uv1_bvn)             # [B,V,3,N]
    else:
        pix_orig = uv1_bvn

    rays_c = torch.matmul(Kinv, pix_orig)                  # [B,V,3,N]
    d = depth.view(B, V, 1, N)                             # [B,V,1,N]
    Xc = rays_c * d                                        # [B,V,3,N]

    # ---------- Camera -> Ego ----------
    R = T_E_from_C[:, :, :3, :3]                           # [B,V,3,3]
    t = T_E_from_C[:, :, :3,  3:4]                         # [B,V,3,1]
    Xe = torch.matmul(R, Xc) + t                           # [B,V,3,N]

    x, y, z = Xe[:, :, 0], Xe[:, :, 1], Xe[:, :, 2]        # [B,V,N]

    # # visualize first batch, all cameras
    # pcds = []
    # # define distinct colors for up to 6 cameras
    # colors = [
    #     [1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, 1.0],
    #     [1.0, 1.0, 0.0],
    #     [1.0, 0.0, 1.0],
    #     [0.0, 1.0, 1.0]
    # ]
    # B, V, _ = x.shape[:3]
    # for cam_idx in range(V):
    #     pts = torch.stack([
    #         x[0, cam_idx, :],
    #         y[0, cam_idx, :],
    #         z[0, cam_idx, :]
    #     ], dim=1).cpu().numpy()
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pts)
    #     color = colors[cam_idx % len(colors)]
    #     pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (pts.shape[0], 1)))
    #     pcds.append(pcd)

    # o3d.visualization.draw_geometries(pcds)
    r = torch.sqrt(x*x + y*y + z*z).clamp_min(1e-6)        # [B,V,N]
    yaw   = -torch.atan2(y, x)                             # [-pi,pi]
    pitch = torch.asin((z / r).clamp(-1, 1))               # [-pi/2,pi/2]

    fov_up_rad   = torch.tensor(fov_up,   device=device, dtype=dtype) * math.pi / 180.0
    fov_down_rad = torch.tensor(fov_down, device=device, dtype=dtype) * math.pi / 180.0
    fov = torch.abs(fov_down_rad) + torch.abs(fov_up_rad)

    # ---------- 角度 -> range-view 像素 (floor+clip，与 numpy 版一致) ----------
    # 注意这里乘 W_r/H_r 后 floor，再 clip 到 [0,W_r-1]/[0,H_r-1]
    px = (0.5 * (yaw / math.pi + 1.0) * W_r).floor().long().clamp(0, W_r - 1)  # [B,V,N]
    py = ((1.0 - (pitch + torch.abs(fov_down_rad)) / fov) * H_r).floor().long().clamp(0, H_r - 1)

    # ---------- Z-buffer: 选最近 r ----------
    lin = (py * W_r + px)                                      # [B,V,N]  range 像素线性坐标
    b_id = torch.arange(B, device=device).view(B, 1, 1).expand(B, V, N)
    lin_global = b_id * (H_r * W_r) + lin                       # [B,V,N]  跨 batch 的全局桶索引

    valid = (d.squeeze(2) > 0)                                  # [B,V,N]  深度>0
    r_src = torch.where(valid, r, torch.full_like(r, float('inf')))  # [B,V,N]

    # 扁平化到 1D，按 lin_global 聚合
    r_src_f = r_src.reshape(-1)                                 # [B*V*N]
    lin_g_f = lin_global.reshape(-1)                            # [B*V*N]
    out_size = B * H_r * W_r

    rmin, argmin = scatter_min(r_src_f, lin_g_f, dim=0, dim_size=out_size)  # 各桶的最小 r 及其来源下标
    rmin = rmin.view(B, H_r, W_r)                               # [B,H_r,W_r]
    argmin = argmin.view(B, H_r, W_r)                           # [B,H_r,W_r]
    mask = torch.isfinite(rmin)                                 # [B,H_r,W_r]

    # ---------- 从全局来源下标反解 cam_idx 与像素 (u,v) ----------
    # r_src_f 的全局下标结构: global_id = b*(V*N) + v*N + n
    argmin_safe = torch.where(mask, argmin, torch.zeros_like(argmin))  # 无效先占位 0
    cam_sel = ((argmin_safe // N) % V).long()                  # [B,H_r,W_r] ∈ [0, V-1]
    pix_sel = (argmin_safe % N).long()                         # [B,H_r,W_r] ∈ [0, N-1]

    # 还原 (u,v)（增强后图像像素）
    u_flat = uu.reshape(-1).to(torch.long)                     # [N]
    v_flat = vv.reshape(-1).to(torch.long)                     # [N]
    u_sel  = u_flat[pix_sel]                                   # [B,H_r,W_r]
    v_sel  = v_flat[pix_sel]                                   # [B,H_r,W_r]

    # ---------- 组装输出 ----------
    proj_pix = torch.full((B, H_r, W_r, 3), -1, dtype=torch.long, device=device)
    proj_pix[..., 0] = torch.where(mask, cam_sel, torch.full_like(cam_sel, -1))
    proj_pix[..., 1] = torch.where(mask, u_sel,   torch.full_like(u_sel,   -1))
    proj_pix[..., 2] = torch.where(mask, v_sel,   torch.full_like(v_sel,   -1))

    proj_range = torch.where(mask, rmin, torch.full_like(rmin, -1)).unsqueeze(1)  # [B,1,H_r,W_r]
    proj_mask  = mask.float().unsqueeze(1)                                        # [B,1,H_r,W_r]

    # # （可选）安全检查：只在有效像素上检查 cam_idx 范围
    # # ok = proj_mask.squeeze(1) > 0
    # # assert torch.all((proj_pix[...,0][ok] >= 0) & (proj_pix[...,0][ok] < V)), "cam_idx out of range"
    # import matplotlib.pyplot as plt

    # # 在函数末尾可视化 proj_range（以第一个 batch 为例）
    # range_img = proj_range[0, 0].cpu().numpy()
    # plt.figure(figsize=(8, 6))
    # plt.imshow(range_img, cmap='seismic')
    # plt.title('Projected Range View')
    # plt.colorbar(label='Distance (m)')
    # plt.axis('off')
    # plt.show()
    return proj_pix, proj_mask, proj_range



# @torch.no_grad()
# def build_proj_pix_from_depth_torch(depth, K, T_E_from_C, H_r, W_r, fov_up, fov_down):
#     """
#     向量化 Z-buffer 投影（硬投影，谁离得近谁占坑）
#     depth: [B,V,1,H,W]  绝对深度(米)
#     K:     [B,V,3,3]
#     T_E_from_C:[B,V,4,4]  (Cam->Ego)
#     返回:
#       proj_pix:   [B,H_r,W_r,3]  (cam_idx,u,v)  long
#       proj_mask:  [B,1,H_r,W_r]  {0,1}
#       proj_range: [B,1,H_r,W_r]  float
#     """
#     B,V,_,H,W = depth.shape
#     device = depth.device
#     dtype  = depth.dtype

#     # 1) 回投到相机，再到Ego
#     # 像素网格
#     u = torch.arange(W, device=device).view(1,1,1,1,W).expand(B,V,1,H,W)
#     v = torch.arange(H, device=device).view(1,1,1,H,1).expand(B,V,1,H,W)
#     ones = torch.ones_like(u, dtype=dtype)
#     pix  = torch.cat([u, v, ones], dim=2).to(dtype)               # [B,V,3,H,W]

#     Kinv = torch.inverse(K)                                       # [B,V,3,3]
#     rays = (Kinv @ pix.view(B,V,3,H*W)).view(B,V,3,H,W)           # [B,V,3,H,W]
#     Xc   = rays * depth                                           # [B,V,3,H,W]

#     Xh = torch.cat([Xc, torch.ones((B,V,1,H,W), device=device, dtype=dtype)], 2)  # [B,V,4,H,W]
#     Xe = (T_E_from_C @ Xh.view(B,V,4,H*W)).view(B,V,4,H,W)[:, :, :3]              # [B,V,3,H,W]

#     x, y, z = Xe[:, :, 0], Xe[:, :, 1], Xe[:, :, 2]
#     r = torch.sqrt(x*x + y*y + z*z) + 1e-6                                      # [B,V,H,W]
#     yaw   = -torch.atan2(y, x)
#     pitch = torch.asin((z / r).clamp(-1,1))

#     fov_up   = torch.as_tensor(fov_up,   device=device, dtype=dtype) * torch.pi/180.0
#     fov_down = torch.as_tensor(fov_down, device=device, dtype=dtype) * torch.pi/180.0
#     fov = torch.abs(fov_down) + torch.abs(fov_up)

#     px = (0.5*(yaw/torch.pi + 1.0) * (W_r-1)).round().clamp_(0, W_r-1).long()   # [B,V,H,W]
#     py = ((1.0 - (pitch + torch.abs(fov_down))/fov) * (H_r-1)).round().clamp_(0, H_r-1).long()

#     # 2) 展平到 1D，构造“桶”索引 key = b*HWr + y*Wr + x
#     N  = H*W
#     pxf = px.reshape(B, V, N)
#     pyf = py.reshape(B, V, N)
#     rf  = r.reshape(B, V, N)

#     bins_per_b = H_r * W_r
#     key = (torch.arange(B, device=device).view(B,1,1) * bins_per_b) + (pyf * W_r + pxf)  # [B,V,N]
#     key = key.view(-1)                     # [B*V*N]
#     rf  = rf.view(-1)                      # [B*V*N]

#     # 3) 分桶“最小值” + 返回 argmin，用于回取 cam_idx / 源 uv
#     if not HAS_SCATTER:
#         raise RuntimeError("Please install torch-scatter for fast z-buffer. See https://github.com/rusty1s/pytorch_scatter")

#     rmin, argmin = torch_scatter.scatter_min(rf, key, dim=0, dim_size=B*bins_per_b)   # [B*HWr], [B*HWr]
#     valid = argmin >= 0
#     # 反解 b, y, x
#     byx = torch.arange(B*bins_per_b, device=device)
#     bx = (byx % bins_per_b) % W_r
#     by = (byx % bins_per_b) // W_r
#     bb = byx // bins_per_b

#     # 从扁平索引解出 (b,v,n)
#     # 我们的展平顺序是 b | v | n
#     VN = V*N
#     b_sel = bb[valid]
#     flat_idx = argmin[valid]              # in [0, B*V*N)
#     v_sel = (flat_idx // N) % V
#     n_sel =  flat_idx %  N
#     u_src = (n_sel % W).long()
#     v_src = (n_sel // W).long()

#     # 4) 写出结果
#     proj_pix   = torch.full((B, H_r, W_r, 3), -1, dtype=torch.long, device=device)
#     proj_range = torch.full((B, 1, H_r, W_r), -1, dtype=dtype, device=device)
#     proj_mask  = torch.zeros((B, 1, H_r, W_r), dtype=torch.float32, device=device)

#     yy = by[valid]; xx = bx[valid]
#     proj_pix[b_sel, yy, xx, 0] = v_sel
#     proj_pix[b_sel, yy, xx, 1] = u_src
#     proj_pix[b_sel, yy, xx, 2] = v_src
#     proj_range[b_sel, 0, yy, xx] = rmin[valid]
#     proj_mask[b_sel, 0, yy, xx]  = 1.0
#     return proj_pix, proj_mask, proj_range

# @torch.no_grad()
# def build_proj_pix_from_depth_torch(
#     depth,               # [B,V,1,H,W]  每个相机像素的绝对深度(米)
#     K,                   # [B,V,3,3]    每个相机的内参
#     T_E_from_C,          # [B,V,4,4]    Cam->Ego 的外参（齐次矩阵）
#     H_r: int, W_r: int,  # range-view 尺寸
#     fov_up: float, fov_down: float      # LiDAR 风格的垂直视场角(度)
# ):
#     """
#     返回:
#       proj_pix:   [B,H_r,W_r,3]  long，(cam_idx, u, v)，u∈[0,W-1], v∈[0,H-1]
#       proj_mask:  [B,1,H_r,W_r] float {0,1}
#       proj_range: [B,1,H_r,W_r] float 距离自车原点的半径 r
#     说明:
#       - 使用 Z-buffer 思路：对映射到同一 range 像素的候选点，选择“最近”的那个。
#       - 这里按“近→远(升序)”遍历，第一次落入空像素即写入，从而保留最近点。
#     """
#     assert depth.dim() == 5 and depth.size(2) == 1, f"depth shape must be [B,V,1,H,W], got {depth.shape}"
#     assert K.shape[-2:] == (3,3) and K.dim() == 4, f"K must be [B,V,3,3], got {K.shape}"
#     assert T_E_from_C.shape[-2:] == (4,4) and T_E_from_C.dim() == 4, f"T_E_from_C must be [B,V,4,4], got {T_E_from_C.shape}"

#     B, V, _, H, W = depth.shape
#     device = depth.device
#     dtype  = depth.dtype

#     # ---------- 1) 相机像素反投影到相机系，再变换到 Ego ----------
#     # backproject_depth_to_cam: 期望 [B,V,1,H,W] & [B,V,3,3] -> [B,V,3,H,W]
#     Xc = backproject_depth_to_cam(depth, K)                 # [B,V,3,H,W]
#     ones = torch.ones((B, V, 1, H, W), device=device, dtype=dtype)
#     Xh = torch.cat([Xc, ones], dim=2).view(B, V, 4, -1)     # [B,V,4,N], N=H*W
#     # Cam->Ego
#     Xe = torch.matmul(T_E_from_C, Xh).view(B, V, 4, -1)[:, :, :3, :]   # [B,V,3,N]

#     x = Xe[:, :, 0]   # [B,V,N]
#     y = Xe[:, :, 1]
#     z = Xe[:, :, 2]
#     r = torch.sqrt(x * x + y * y + z * z) + 1e-6            # [B,V,N]

#     # 无效深度屏蔽（深度<=0不参与竞争）
#     valid = (depth.view(B, V, -1) > 1e-6)                    # [B,V,N]
#     inf = torch.full_like(r, float('inf'))
#     r = torch.where(valid, r, inf)                           # 无效点置为 +inf

#     # ---------- 2) Ego -> 球面 -> range 像素 ----------
#     yaw   = -torch.atan2(y, x)                               # [-pi, pi]
#     pitch = torch.asin((z / r).clamp(-1, 1))                 # [-pi/2, pi/2]

#     fov_up_rad   = torch.tensor(fov_up,   device=device, dtype=dtype) * math.pi / 180.0
#     fov_down_rad = torch.tensor(fov_down, device=device, dtype=dtype) * math.pi / 180.0
#     fov = torch.abs(fov_down_rad) + torch.abs(fov_up_rad)

#     # 像素化（注意使用 H_r-1/W_r-1 再 round，避免边界偏置）
#     px = (0.5 * (yaw / math.pi + 1.0) * (W_r - 1)).round().long().clamp(0, W_r - 1)    # [B,V,N]
#     py = ((1.0 - (pitch + torch.abs(fov_down_rad)) / fov) * (H_r - 1)).round().long().clamp(0, H_r - 1)

#     # ---------- 3) 记录源相机像素 u,v ----------
#     # 构造原图栅格（每个相机都有一份）
#     u_grid = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, V, H, W)  # [B,V,H,W]
#     v_grid = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, V, H, W)  # [B,V,H,W]
#     u_flat = u_grid.reshape(B, V, -1)  # [B,V,N]
#     v_flat = v_grid.reshape(B, V, -1)  # [B,V,N]

#     # ---------- 4) Z-buffer：按 r 升序，首次落点即写入 ----------
#     # 升序：近 -> 远
#     order = torch.argsort(r, dim=-1, descending=False)       # [B,V,N]

#     proj_pix   = torch.full((B, H_r, W_r, 3), -1, dtype=torch.long,  device=device)
#     proj_range = torch.full((B, 1,  H_r, W_r), -1, dtype=dtype,      device=device)
#     proj_mask  = torch.zeros((B, 1,  H_r, W_r), dtype=torch.float32, device=device)

#     # 展示型实现：按“半径从近到远”的顺序遍历；遇到空像素即写入（因此保留最近）
#     N = H * W
#     arange_B = torch.arange(B, device=device)

#     for k in range(N):
#         # sel: [B,V]；用于从 [B,V,N] 的最后一维 gather
#         sel = order[:, :, k].unsqueeze(-1)                   # [B,V,1]

#         py_k = torch.gather(py, 2, sel).squeeze(-1)          # [B,V]
#         px_k = torch.gather(px, 2, sel).squeeze(-1)          # [B,V]
#         r_k  = torch.gather(r,  2, sel).squeeze(-1)          # [B,V]
#         uu_k = torch.gather(u_flat, 2, sel).squeeze(-1)      # [B,V]
#         vv_k = torch.gather(v_flat, 2, sel).squeeze(-1)      # [B,V]

#         # 逐相机写入；一次写一个“近点”
#         for v in range(V):
#             yv = py_k[:, v]         # [B]
#             xv = px_k[:, v]         # [B]
#             rv = r_k[:,  v]         # [B]
#             u0 = uu_k[:, v].long()  # [B]
#             v0 = vv_k[:, v].long()  # [B]

#             # 仅在该像素还未被占据时写入
#             empty = (proj_mask[arange_B, 0, yv, xv] == 0)    # [B] bool
#             if empty.any():
#                 bidx = arange_B[empty]
#                 yy = yv[empty]
#                 xx = xv[empty]

#                 proj_mask[bidx, 0, yy, xx]  = 1.0
#                 proj_range[bidx, 0, yy, xx] = rv[empty]

#                 proj_pix[bidx, yy, xx, 0] = v          # 相机索引
#                 proj_pix[bidx, yy, xx, 1] = u0[empty]  # 源像素 u
#                 proj_pix[bidx, yy, xx, 2] = v0[empty]  # 源像素 v

#         # 小优化：若该 batch 全部像素都已填满，可提前结束
#         if (proj_mask.sum(dim=(1,2,3)) == (H_r * W_r)).all():
#             break

#     return proj_pix, proj_mask, proj_range

