# utils/loss.py
import math
import torch
import torch.nn.functional as F

# ---------- 基础鲁棒项 ----------
def charbonnier(x, eps=1e-3):
    # 近似 L1：sqrt(x^2 + eps^2)
    return torch.sqrt(x * x + eps * eps)

def huber(x, delta=0.03):
    # 平滑 L1
    absx = x.abs()
    quad = 0.5 * (absx ** 2) / delta
    lin  = absx - 0.5 * delta
    return torch.where(absx <= delta, quad, lin)

# ---------- 球面等面积权（Equirectangular） ----------
@torch.no_grad()
def _equal_area_weights(H, device, fov_up_deg=8.0, fov_down_deg=-15.0):
    """
    每一行按 cos(pitch) 加权。pitch 是仰角（由垂直像素 y 反投影得到）。
    """
    fov_up   = math.radians(fov_up_deg)
    fov_down = math.radians(fov_down_deg)
    fov      = abs(fov_down) + abs(fov_up)
    # 行心位置 y_c = i + 0.5，更稳定
    y = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    pitch = (1.0 - y / H) * fov - abs(fov_down)  # 与投影公式反向一致
    w = torch.cos(pitch).clamp_min(1e-6)         # 防止除零
    # 形状 [1,1,H,1] 便于广播
    return w.view(1, 1, H, 1)

# ---------- 角度 wrap ----------
def _angle_diff(a, b):
    # wrap 到 (-pi, pi]
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))

# ---------- 单尺度 scale 损失（log 域） ----------
def _scale_loss_single(scale_pred, gt_scale_with_mask,
                       w_conf=None, loss_kind='charbonnier',
                       fov_up_deg=8.0, fov_down_deg=-15.0):

    gt_scale = gt_scale_with_mask[:, 0:1]
    mask     = gt_scale_with_mask[:, 1:2].bool()

    if mask.sum() == 0:
        return scale_pred.sum() * 0.0

    B, _, H, W = scale_pred.shape
    eps = 1e-6

    # 只取有效像素；并确保对数输入为正
    pred_v = scale_pred[mask].clamp_min(eps)    # [N]
    gt_v   = gt_scale[mask].clamp_min(eps)      # [N]

    err = torch.log(pred_v) - torch.log(gt_v)   # [N]

    if loss_kind == 'charbonnier':
        ell = charbonnier(err)
    elif loss_kind == 'huber':
        ell = huber(err, delta=0.03)
    elif loss_kind == 'l1':
        ell = err.abs()
    else:
        raise ValueError(f"Unknown loss_kind: {loss_kind}")

    # 等面积权
    w_area = _equal_area_weights(H, device=scale_pred.device,
                                 fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg
                                 ).expand(B, 1, H, W)[mask]                 # [N]

    # 置信度（若无则全 1）
    if w_conf is None:
        w_conf_v = torch.ones_like(pred_v)
    else:
        w_conf_v = w_conf.to(scale_pred.dtype)[mask]

    w = (w_area * w_conf_v).detach()            # [N]
    denom = w.sum().clamp_min(1e-6)

    loss = (ell * w).sum() / denom
    return loss


# ---------- 单尺度 风险角度损失 ----------
def _risk_loss_single(risk_pred, gt_risk_with_mask,
                      w_conf=None,
                      loss_kind='charbonnier',   # 'charbonnier' | 'huber' | 'l2'
                      fov_up_deg=8.0, fov_down_deg=-15.0):
    """
    risk_pred: [B,1,H,W]（角度，弧度）
    gt_risk_with_mask: [B,2,H,W]，[0]=gt_angle, [1]=valid_mask
    """
    gt   = gt_risk_with_mask[:, 0:1, :, :]
    mask = gt_risk_with_mask[:, 1:2, :, :].bool()

    if mask.sum() == 0:
        # 返回可导的 0，避免断梯度
        return risk_pred.sum() * 0.0

    B, _, H, W = risk_pred.shape

    # 只在有效像素上计算 wrap 后的角度残差
    err = _angle_diff(risk_pred, gt)[mask]          # [N]

    # 鲁棒项
    if loss_kind == 'charbonnier':
        ell = charbonnier(err)                      # [N]
    elif loss_kind == 'huber':
        ell = huber(err, delta=0.05)
    elif loss_kind == 'l2':
        ell = err * err
    else:
        raise ValueError(f"Unknown loss_kind: {loss_kind}")

    # 等面积权（按行 cos(pitch)）
    w_area = _equal_area_weights(H, device=risk_pred.device,
                                 fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg
                                 ).expand(B, 1, H, W)[mask].view(-1)  # [N]

    # 置信度（若无则全 1）
    if w_conf is None:
        w_conf_v = torch.ones_like(w_area)          # [N]
    else:
        w_conf_v = w_conf.to(risk_pred.dtype)[mask].view(-1)  # [N]

    # 总权重（不反传）
    w = (w_area * w_conf_v).detach()                # [N]
    denom = w.sum().clamp_min(1e-6)

    loss = (ell.view(-1) * w).sum() / denom
    return loss

# ---------- 多阶段/多尺度深监督：支持 list 或 Tensor ----------
def get_loss_scale_map(scale_preds,
                       gt_scale_with_mask,
                       w_conf=None,            # 可为 None 或 list[Tensor]/Tensor（与 scale_preds 对齐）
                       gamma: float = 0.9,     # 后期阶段权重大
                       loss_kind='charbonnier',
                       fov_up_deg=8.0, fov_down_deg=-15.0):
    """
    scale_preds: Tensor[B,1,H,W] 或 list[Tensor]（每个 [B,1,H,W]）
    w_conf: 与 scale_preds 对齐的 Tensor 或 list（若无传 None）
    """
    # 统一成 list
    if not isinstance(scale_preds, (list, tuple)):
        scale_preds = [scale_preds]
    if w_conf is None:
        w_conf_list = [None] * len(scale_preds)
    elif isinstance(w_conf, (list, tuple)):
        w_conf_list = list(w_conf)
    else:
        w_conf_list = [w_conf] * len(scale_preds)

    L = len(scale_preds)
    # 指数衰减：后面阶段权重大
    ws = [gamma ** (L - 1 - i) for i in range(L)]
    s = sum(ws) + 1e-8
    ws = [w / s for w in ws]

    loss = 0.0
    for w, sp, wc in zip(ws, scale_preds, w_conf_list):
        loss = loss + w * _scale_loss_single(sp, gt_scale_with_mask,
                                             w_conf=wc,
                                             loss_kind=loss_kind,
                                             fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg)
    return loss


def get_loss_risk_score_map(risk_preds,
                            gt_risk_with_mask,
                            w_conf=None,
                            gamma: float = 0.9,
                            loss_kind='charbonnier',
                            fov_up_deg=8.0, fov_down_deg=-15.0):
    """
    同上，支持多阶段 list
    """
    if not isinstance(risk_preds, (list, tuple)):
        risk_preds = [risk_preds]
    if w_conf is None:
        w_conf_list = [None] * len(risk_preds)
    elif isinstance(w_conf, (list, tuple)):
        w_conf_list = list(w_conf)
    else:
        w_conf_list = [w_conf] * len(risk_preds)

    L = len(risk_preds)
    ws = [gamma ** (L - 1 - i) for i in range(L)]
    s = sum(ws) + 1e-8
    ws = [w / s for w in ws]

    loss = 0.0
    for w, rp, wc in zip(ws, risk_preds, w_conf_list):
        loss = loss + w * _risk_loss_single(rp, gt_risk_with_mask,
                                            w_conf=wc,
                                            loss_kind=loss_kind,
                                            fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg)
    return loss
