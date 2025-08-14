import torch
import torch.nn.functional as F

@torch.no_grad()
def _build_base_grid(H, W, device, dtype):
    """像素坐标网格（以像素计量，(x,y) = (列,行)）。"""
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    return xs, ys  # [H,W], [H,W]

def _gather_view_features_bilinear(features_list, proj_pix, H_img, W_img):
    """
    从多视角特征图中“按像素坐标”取样（对源视角做双线性采样），得到每个球面像素的源特征。
    Args:
      features_list: list(V) of [B,C,Hf,Wf]
      proj_pix: [B,Hr,Wr,3]  (cam_idx,u,v) on original image plane
      H_img, W_img: 原图(或你的特征下采样前对齐的)尺寸，用于计算 u->uf, v->vf 的缩放
    Returns:
      sel_feat: [B,C,Hr,Wr]
    """
    B, Hr, Wr, _ = proj_pix.shape
    V = len(features_list)
    C, Hf, Wf = features_list[0].shape[1:]
    # 堆叠视角 → 逐视角 grid_sample 再按 mask 相加
    feats = torch.stack(features_list, dim=1)            # [B,V,C,Hf,Wf]
    cam_idx = proj_pix[...,0]                            # [B,Hr,Wr]
    u_orig  = proj_pix[...,1].float()                    # [B,Hr,Wr]
    v_orig  = proj_pix[...,2].float()

    # 连续到特征分辨率（float）
    su = Wf / float(W_img)
    sv = Hf / float(H_img)
    uf = u_orig * su
    vf = v_orig * sv

    # 归一化到 [-1,1]
    x_norm = (2.0 * uf / (Wf - 1.0)) - 1.0               # [B,Hr,Wr]
    y_norm = (2.0 * vf / (Hf - 1.0)) - 1.0
    grid   = torch.stack([x_norm, y_norm], dim=-1)       # [B,Hr,Wr,2]

    out = 0.0
    for v in range(V):
        mask = (cam_idx == v).float().unsqueeze(1)       # [B,1,Hr,Wr]
        if mask.sum() == 0:
            continue
        sampled = F.grid_sample(
            feats[:,v], grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )                                                # [B,C,Hr,Wr]
        out = out + sampled * mask
    return out                                           # [B,C,Hr,Wr]

def gaussian_splat_to_sphere(features_list, proj_pix, proj_xy,
                             H_img, W_img, wrap_horizontal=True, eps=1e-6,
                             return_conf=False,           # 新增：是否返回置信度
                             conf_norm='max',             # 'max' | 'mean' | None
                             var_alpha=10.0               # 方差转置信度的温度
                             ):
    """
    对每个球面像素 (Hr×Wr)：
      1) 从指定视角的源特征图上双线性采样得到 Fsrc
      2) 在球面 (x_f,y_f) 位置做四邻域的双线性 splat（高斯≈双线性权近似）
    Args:
      features_list: list(V) of [B,C,Hf,Wf]
      proj_pix: [B,Hr,Wr,3]   (cam_idx, u, v)   —— 源像素索引（相机平面）
      proj_xy : [B,Hr,Wr,2]   连续球面坐标 (x_f,y_f) ，单位 = “球面像素”
      H_img, W_img:           用来把 (u,v) 映射到特征分辨率
    Returns:
      当 return_conf=False:
          range_feat: [B,C,Hr,Wr]
      当 return_conf=True:
          range_feat: [B,C,Hr,Wr]
          conf:       [B,1,Hr,Wr]   —— 组合置信度（密度 × 逆方差）
          aux: dict(
            den=[B,1,Hr,Wr],        —— 纯密度（权重和）
            conf_density=[B,1,Hr,Wr],
            conf_invvar=[B,1,Hr,Wr] —— 逆方差置信度
          )
    """
    B, Hr, Wr, _ = proj_pix.shape
    device = proj_pix.device
    C = features_list[0].shape[1]

    # 1) 源特征取样（在相机平面）
    Fsrc = _gather_view_features_bilinear(features_list, proj_pix, H_img, W_img)  # [B,C,Hr,Wr]

    # 2) 在球面上做双线性 splat：把 Fsrc(i) 从 (x_f,y_f) 分配到四邻域
    x_f = proj_xy[..., 0].clamp(0, Wr - 1)        # [B,Hr,Wr]
    y_f = proj_xy[..., 1].clamp(0, Hr - 1)

    x0 = torch.floor(x_f).long()
    y0 = torch.floor(y_f).long()
    x1 = x0 + 1
    y1 = y0 + 1

    if wrap_horizontal:
        x0 = torch.remainder(x0, Wr)
        x1 = torch.remainder(x1, Wr)
    else:
        x0 = x0.clamp(0, Wr - 1)
        x1 = x1.clamp(0, Wr - 1)
    y0 = y0.clamp(0, Hr - 1)
    y1 = y1.clamp(0, Hr - 1)

    dx = (x_f - x0.float()).clamp(0, 1)
    dy = (y_f - y0.float()).clamp(0, 1)

    w00 = (1 - dx) * (1 - dy)   # [B,Hr,Wr]
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    # ---- scatter_add 到扁平网格（保留“未归一化的加权和”，用于方差）----
    N = Hr * Wr
    Fsrc_flat = Fsrc.view(B, C, N).permute(0, 2, 1).contiguous()   # [B,N,C]

    def lin(y, x):                 # 线性索引
        return (y * Wr + x).view(B, -1)                            # [B,N]

    idx00 = lin(y0, x0); idx10 = lin(y0, x1)
    idx01 = lin(y1, x0); idx11 = lin(y1, x1)

    # 拼 4 份
    idx_all = torch.stack([idx00, idx10, idx01, idx11], dim=-1).view(B, -1)    # [B, 4N]
    w_all   = torch.stack([w00,  w10,  w01,  w11 ], dim=-1).view(B, -1, 1)     # [B, 4N,1]
    F_all   = (Fsrc_flat.unsqueeze(2) * w_all.view(B, N, 4, 1)).view(B, -1, C) # [B, 4N, C]

    # 目标平面（num = ∑ w*F；den = ∑ w）
    num = torch.zeros(B, N, C, device=device, dtype=Fsrc.dtype)
    den = torch.zeros(B, N, 1, device=device, dtype=Fsrc.dtype)

    num.scatter_add_(1, idx_all.unsqueeze(-1).expand_as(F_all), F_all)  # 通道加权累加
    den.scatter_add_(1, idx_all.unsqueeze(-1), w_all)                    # 权重和

    # 归一化得到特征
    feat = num / (den + eps)                                            # [B,N,C]
    range_feat = feat.permute(0, 2, 1).contiguous().view(B, C, Hr, Wr)  # [B,C,Hr,Wr]

    if not return_conf:
        return range_feat

    # ---------- 置信度（密度 × 逆方差） ----------
    # 1) 密度项：den 映射到 [0,1]
    if conf_norm == 'max':
        # 每个样本内按像素最大值归一化
        den_max = den.amax(dim=1, keepdim=True)  # [B,1,1]
        conf_density = den / (den_max + eps)
    elif conf_norm == 'mean':
        den_mean = den.mean(dim=1, keepdim=True) # [B,1,1]
        conf_density = den / (den_mean + eps)
        conf_density = conf_density.clamp_(max=1.0)
    else:
        conf_density = den  # 不做归一化（上层可自行处理）

    # 2) 逆方差项：E[F^2] - (E[F])^2
    F2_all = F_all * F_all
    num2 = torch.zeros(B, N, C, device=device, dtype=Fsrc.dtype)
    num2.scatter_add_(1, idx_all.unsqueeze(-1).expand_as(F2_all), F2_all)  # ∑ w*F^2

    Ex2 = num2 / (den + eps)              # [B,N,C]
    Ex  = feat                            # [B,N,C] = ∑ wF / ∑ w
    var = (Ex2 - Ex * Ex).clamp_min(0.0)  # 数值稳定
    var_mean = var.mean(dim=-1, keepdim=True)     # 聚合到通道外：[B,N,1]
    conf_invvar = torch.exp(-var_alpha * var_mean) # 方差越小置信度越大 ∈ (0,1]
    conf_invvar = conf_invvar.view(B, 1, Hr, Wr)

    conf_density_img = conf_density.view(B, 1, Hr, Wr)
    conf = (conf_density_img * conf_invvar).clamp(0.0, 1.0)

    aux = dict(
        den=den.view(B, 1, Hr, Wr),
        conf_density=conf_density_img,
        conf_invvar=conf_invvar,
    )
    return range_feat, conf, aux

