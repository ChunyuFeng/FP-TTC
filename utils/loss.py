
from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

def get_loss(scale, flow, dc_gt, flow_gt, valid, epoch):

    gt_dchange = dc_gt[:,0:1,:,:]
    valids = dc_gt[:, 1, :, :].unsqueeze(1).bool()

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 400)

    gt_dchange[gt_dchange<=0] = 10
    gt_dchange[gt_dchange>3] = 10

    maskdc = ((gt_dchange < 3) & (gt_dchange > 0.3) & valids & (scale>0))

    d_loss = (scale.log() - gt_dchange.log()).abs()
    f_loss = (flow-flow_gt).abs()
    #print(scale.log(), gt_dchange.log())
    sloss = (maskdc * d_loss).sum()/maskdc.sum()
    floss = (valid[:, None] * f_loss).sum()/valid.sum()

    loss = 0.8*sloss + 0.2*floss

    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    f1 = torch.mean(1-out[val])

    return sloss, loss, gt_dchange, f1


def get_loss_multi(scale_preds, valid, scale_gt, epoch, gamma=0.9):

    n_predictions = len(scale_preds)
    scale_loss = 0.0

    gt_depth = scale_gt[:,0:1,:,:]
    gt_depth[gt_depth<=0] = 1e6
    gt_f3d =  scale_gt[:,1:,...].clone()
    gt_dchange = (1+gt_f3d[:,2:,...]/gt_depth)

    validx = ((gt_dchange < 3) & (gt_dchange > 0.2) & valid.unsqueeze(1).bool())

    for i in range(n_predictions):
        scale = scale_preds[i]
        i_weight = gamma ** (n_predictions - i - 1)
        # if i==0:
        #     first = n_predictions//2
        #     #print(first)
        #     i_weight = gamma ** (n_predictions - first)
        # else:
        #     i_weight = gamma ** (n_predictions - i - 1)

        mask_nan = ~torch.isnan(scale)
        maskdc = validx & mask_nan
        mask_minus0 = (scale<=0)

        if mask_minus0.sum() == 0:    
            loss1 =  ((((scale.abs()).log()-(gt_dchange.abs()).log()).abs())*maskdc).sum() / (maskdc.sum())
            loss = loss1
            if epoch < 5:
                loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-4)
                loss += loss2
        else:
            loss1 = (((scale-gt_dchange).abs())[mask_nan]).mean()
            loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-4)
            loss = loss1 + loss2

        scale_loss += i_weight * loss.mean()
        if i == n_predictions - 1:
            loss_final = loss.mean()

    return scale_loss, loss_final, gt_dchange, maskdc

def ttc_smooth_loss(img, disp, mask):
    """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
    """
    # normalize
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp

    grad_disp_x = torch.abs(disp - torch.roll(disp, 1, dims=3))
    grad_disp_y = torch.abs(disp - torch.roll(disp, 1, dims=2))
    grad_disp_x[:,:,:,0] = 0
    grad_disp_y[:,:,0,:] = 0

    # grad_disp_xx = torch.abs(torch.roll(grad_disp_x, -1, dims=3) - grad_disp_x)
    # grad_disp_yy = torch.abs(torch.roll(grad_disp_y, -1, dims=3) - grad_disp_y)
    # grad_disp_xx[:,:,:,0] = 0
    # grad_disp_yy[:,:,0,:] = 0
    # grad_disp_xx[:,:,:,-1] = 0
    # grad_disp_yy[:,:,-1,:] = 0

    grad_img_x = torch.mean(torch.abs(img - torch.roll(img, 1, dims=3)), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img - torch.roll(img, 1, dims=2)), 1, keepdim=True)
    grad_img_x[:,:,:,0] = 0
    grad_img_y[:,:,0,:] = 0

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return (grad_disp_x*mask).sum()/mask.sum() + (grad_disp_y*mask).sum()/mask.sum()


def get_loss_selfsup(scale, valid, scale_gt):
    return (scale.log()-scale_gt.log()).abs()[valid.bool()].mean()


def self_supervised_gt_affine(flow):

    b,_,lh,lw=flow.shape
    bs, w,h = b, lw, lh
    grid_H = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(bs, 1, h, w).to(device=flow.device, dtype=flow.dtype)
    grid_V = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(bs, 1, h, w).to(device=flow.device, dtype=flow.dtype)
    pref = torch.cat([grid_H, grid_V], dim=1)
    ptar = pref + flow
    pw = 1
    pref = F.unfold(pref, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-pref[:,:,np.newaxis]
    ptar = F.unfold(ptar, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-ptar[:,:,np.newaxis] # b, 2,9,h,w
    pref = pref.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)
    ptar = ptar.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)

    prefprefT = pref.matmul(pref.permute(0,2,1))
    ppdet = prefprefT[:,0,0]*prefprefT[:,1,1]-prefprefT[:,1,0]*prefprefT[:,0,1]
    ppinv = torch.cat((prefprefT[:,1,1:],-prefprefT[:,0,1:], -prefprefT[:,1:,0], prefprefT[:,0:1,0]),1).view(-1,2,2)/ppdet.clamp(1e-10,np.inf)[:,np.newaxis,np.newaxis]

    Affine = ptar.matmul(pref.permute(0,2,1)).matmul(ppinv)
    Error = (Affine.matmul(pref)-ptar).norm(2,1).mean(1).view(b,1,lh,lw)

    Avol = (Affine[:,0,0]*Affine[:,1,1]-Affine[:,1,0]*Affine[:,0,1]).view(b,1,lh,lw).abs().clamp(1e-10,np.inf)
    exp = Avol.sqrt()
    mask = (exp>0.5) & (exp<2)
    mask = mask[:,0]

    exp = exp.clamp(0.5,2)
    # exp[Error>0.1]=1
    return torch.reciprocal(exp)


def self_supervised_gt(flow_f):

    b,_,h,w = flow_f.size()
    grid_H = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(b, 1, h, w).to(device=flow_f.device, dtype=flow_f.dtype)
    grid_V = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(b, 1, h, w).to(device=flow_f.device, dtype=flow_f.dtype)
    grids1_ = torch.cat([grid_H, grid_V], dim=1)

    gw = 2
    pad_dim = (gw,gw,gw,gw)
    grids_pad = F.pad(grids1_, pad_dim, "replicate")
    flow_f_pad = F.pad(flow_f, pad_dim, "replicate")
    grids_w_f = grids_pad + flow_f_pad

    # tm - m
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_y_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # bm - m
    len_ori = torch.abs(grids_pad[...,2*gw:,gw:-gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,2*gw:,gw:-gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_y_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # mr - m
    len_ori = torch.abs(grids_pad[...,gw:-gw,2*gw:] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,2*gw:] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_x_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # ml - m
    len_ori = torch.abs(grids_pad[...,gw:-gw,0:-2*gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,0:-2*gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_x_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # tm - mr
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,2*gw:]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,2*gw:]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_l_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # ml - bm
    len_ori = torch.abs(grids_pad[...,gw:-gw,0:-2*gw] - grids_pad[...,2*gw:,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,0:-2*gw] - grids_w_f[...,2*gw:,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_l_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # tm - ml
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,0:-2*gw]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,0:-2*gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_r_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # mr - bm
    len_ori = torch.abs(grids_pad[...,gw:-gw,2*gw:] - grids_pad[...,2*gw:,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,2*gw:] - grids_w_f[...,2*gw:,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_r_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)


    exp_x_len = torch.min(torch.stack([exp_x_len1,exp_x_len2],dim=1),dim=1)[0]
    exp_y_len = torch.min(torch.stack([exp_y_len1,exp_y_len2],dim=1),dim=1)[0]
    exp_l_len = torch.min(torch.stack([exp_l_len1,exp_l_len2],dim=1),dim=1)[0]
    exp_r_len = torch.min(torch.stack([exp_r_len1,exp_r_len2],dim=1),dim=1)[0]

    exp_corner_len = torch.max(torch.stack([exp_l_len,exp_r_len],dim=1),dim=1)[0]
    exp_f_xy_len = torch.max(exp_x_len,exp_y_len)

    threshold = 0.15
    ttc_range = 0.95
    mask_exp1 = torch.logical_and(exp_corner_len<1. ,(exp_corner_len/exp_f_xy_len)>1)
    mask_exp2 = torch.logical_and((exp_x_len/exp_y_len)>(1-threshold),(exp_x_len/exp_y_len)<(1+threshold))
    mask_exp2 = torch.logical_and(mask_exp2,exp_f_xy_len<ttc_range*torch.mean(exp_f_xy_len))
    mask_exp = torch.logical_and(mask_exp1, mask_exp2)

    scale_gt = mask_exp*exp_corner_len+ ~mask_exp*exp_f_xy_len
    scale_gt[...,0:1,:] = 1
    scale_gt[...,-1:,:] = 1
    scale_gt[...,:,0:1] = 1
    scale_gt[...,:,-1:] = 1

    return scale_gt

def get_loss_nusc(scale, gt_scale, valid):
    # 移除单个维度，使所有张量形状为 (320, 640)
    scale = scale.squeeze(1)  # 将形状从 [1, 1, 320, 640] 压缩为 [320, 640]
    gt_scale = gt_scale.squeeze(0)       # 将形状从 [1, 320, 640] 压缩为 [320, 640]
    valid = valid.squeeze(0)             # 将形状从 [1, 320, 640] 压缩为 [320, 640]

    gt_scale = torch.nan_to_num(gt_scale, nan=10)

    # 处理 gt_scale 的有效范围
    gt_scale[gt_scale <= 0] = 10
    gt_scale[gt_scale > 3] = 10

    # 创建有效的深度变化掩膜
    maskdc = (gt_scale < 3) & (gt_scale > 0.3) & valid & (scale > 0)

    if maskdc.sum() == 0:
        return torch.tensor(0.0, device=scale.device)  # 返回 0 作为损失

    # 确保 scale 和 gt_scale 中的值都大于一个非常小的正数
    epsilon = 1e-6  # 预防性的小值
    scale = torch.clamp(scale, min=epsilon)
    gt_scale = torch.clamp(gt_scale, min=epsilon)

    # 计算 scale loss
    d_loss = (scale.log() - gt_scale.log()).abs()

    sloss = (maskdc * d_loss).sum() / maskdc.sum()

    # 返回 sloss 作为最终的损失
    return sloss

def get_loss_mix(scale, gt_scale_with_mask):
    # 移除单个维度，使所有张量形状为 (320, 640)
    scale = scale.squeeze(1)  # 将形状从 [batch_size, 1, 320, 640] 压缩为 [batch_size, 320, 640]
    gt_scale = gt_scale_with_mask[:,0,:,:]
    valid = gt_scale_with_mask[:,1,:,:].bool()

    gt_scale = torch.nan_to_num(gt_scale, nan=10)

    # 处理 gt_scale 的有效范围
    gt_scale[gt_scale <= 0] = 10
    gt_scale[gt_scale > 3] = 10

    # 创建有效的深度变化掩膜
    maskdc = (gt_scale < 3) & (gt_scale > 0.3) & valid & (scale > 0)

    if maskdc.sum() == 0:
        return torch.tensor(0.0, device=scale.device)  # 返回 0 作为损失

    # 确保 scale 和 gt_scale 中的值都大于一个非常小的正数
    epsilon = 1e-6  # 预防性的小值
    scale = torch.clamp(scale, min=epsilon)
    gt_scale = torch.clamp(gt_scale, min=epsilon)

    # 计算 scale loss
    d_loss = (scale.log() - gt_scale.log()).abs()

    # sloss = (maskdc * d_loss).sum() / maskdc.sum()
    sloss = d_loss[maskdc].mean()
    # 返回 sloss 作为最终的损失
    return sloss, valid

def get_loss_scale_map(scale, gt_scale_with_mask, loss_weight_alpha=0.0):
    # 移除单个维度，使所有张量形状为 (B, H, W)
    scale = scale.squeeze(1)
    gt_scale = gt_scale_with_mask[:, 0, :, :]
    mask = gt_scale_with_mask[:, 1, :, :].bool()

    if mask.sum() == 0:
        print("[WARN]:Scale branch no valid area, return 0 loss")
        return scale.sum() * 0.0

    epsilon = 1e-6
    log_scale_m = torch.log(scale[mask] + epsilon)
    log_gt_scale_m = torch.log(gt_scale[mask].clamp(min=epsilon))

    loss = (log_scale_m - log_gt_scale_m).abs()

    if loss_weight_alpha > 0:
        with torch.no_grad():
            weights = 1.0 + loss_weight_alpha * log_gt_scale_m.abs()
        loss = (loss * weights).sum() / weights.sum()
    else:
        loss = loss.mean()

    return loss

def get_loss_risk_score_map(risk_score, gt_risk_score_with_mask):
    # 移除单个维度，使所有张量形状为 (B, H, W)
    risk_score = risk_score.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    gt_risk_score = gt_risk_score_with_mask[:, 0, :, :]
    mask = gt_risk_score_with_mask[:, 1, :, :].bool()

    # 如果没有有效区域，则返回可导的 0（避免断梯度）
    if mask.sum() == 0:
        return risk_score.sum() * 0.0
        print("[WARN]:Risk score branch no valid area, return 0 loss")

    criterion = torch.nn.MSELoss(reduction='mean')
    
    loss = criterion(risk_score[mask], gt_risk_score[mask])
    return loss

@torch.no_grad()
def compute_errs_ttc_from_scale(
    scale_pred: torch.Tensor,
    gt_scale_tensor: torch.Tensor,
    valid_mask: torch.Tensor = None,      # 可选：True 表示该像素参与评估
    thresholds=(1.0, 2.0, 5.0),           # 单位：秒
    eps: float = 1e-6
):
    """
    返回一个 dict: {'Err-1': float, 'Err-2': float, 'Err-5': float}
    约定：TTC<0 视为远离（非风险），在阈值分类中当作 >max(thresholds) 处理。
    """

    assert scale_pred.shape == gt_scale_tensor.shape, "pred/gt 尺寸需一致"

    # 计算 TTC = 1 / (1 - scale)，并做数值稳定处理
    def scale_to_ttc(s: torch.Tensor) -> torch.Tensor:
        denom = 1.0 - s
        # 避免除零（scale=1）或极小分母
        denom = torch.where(torch.abs(denom) < eps, torch.sign(denom) * eps, denom)
        ttc = 0.1 / denom
        # 负 TTC（远离）按评测习惯视为“很大”（不应落入 <T 的正类）
        ttc = torch.where(ttc < 0, torch.tensor(float('inf'), device=ttc.device), ttc)
        return ttc

    ttc_pred = scale_to_ttc(scale_pred)
    ttc_gt   = scale_to_ttc(gt_scale_tensor)

    # 有效掩码：默认忽略 NaN/Inf
    base_valid = torch.isfinite(ttc_pred) & torch.isfinite(ttc_gt)
    if valid_mask is not None:
        base_valid = base_valid & (valid_mask.bool())

    num_valid = base_valid.sum().item()
    if num_valid == 0:
        return {f"Err-{int(t)}": float('nan') for t in thresholds}

    errs = {}
    for T in thresholds:
        y_gt   = (ttc_gt < T)
        y_pred = (ttc_pred < T)
        wrong  = (y_gt != y_pred) & base_valid
        errs[f"Err-{int(T)}"] = wrong.float().sum().item() / (scale_pred.shape[0]*scale_pred.shape[1])

    return errs

@torch.no_grad()
def eval_orientation_and_highrisk_stats(
    ori_pred: torch.Tensor,          # \hat{\theta}(p) \in [0, \pi]
    ori_gt: torch.Tensor,            # \theta_{gt}(p) \in [0, \pi]
    scale_pred: torch.Tensor,        # \widehat{scale}(p)
    scale_gt: torch.Tensor,          # scale_{gt}(p)
    valid_mask: torch.Tensor = None, # True 表示参与评估
    tau_theta: float = 3.14159265/12.0,  # 方向容差 \tau_\theta（注意此处=pi/6）
    tau_T: float = 2.0,              # TTC 阈值（秒）
    delta_t: float = 0.1,            # 帧间隔（秒）
    eps: float = 1e-12               # 数值稳定
):
    """
    返回可跨帧累加的统计量：
      {
        'mae_num': float,   # sum |Δθ|
        'acc_num': float,   # sum 1[|Δθ|<=τθ]
        'N': float,         # 有效像素数
        'TP': float, 'FP': float, 'FN': float
      }
    """

    assert ori_pred.shape == ori_gt.shape == scale_pred.shape == scale_gt.shape, \
        "pred/gt 张量形状需一致"

    # --- scale -> TTC 稳定转换 ---
    def scale_to_ttc(s: torch.Tensor) -> torch.Tensor:
        denom = 1.0 - s
        tiny = torch.abs(denom) < eps
        denom = torch.where(tiny, torch.sign(denom) * eps, denom)
        ttc = delta_t / denom
        # 负 TTC(远离) 记为 +inf，不进入短TTC正类
        ttc = torch.where(ttc < 0, torch.tensor(float('inf'), device=ttc.device), ttc)
        return ttc

    ttc_pred = scale_to_ttc(scale_pred)
    ttc_gt   = scale_to_ttc(scale_gt)

    # 有效掩码：过滤 NaN/Inf，并与外部 valid_mask 取交
    finite_mask = (
        torch.isfinite(ori_pred) & torch.isfinite(ori_gt) &
        torch.isfinite(ttc_pred) & torch.isfinite(ttc_gt)
    )
    if valid_mask is not None:
        finite_mask = finite_mask & valid_mask.bool()

    N = finite_mask.sum().item()
    if N == 0:
        return {'mae_num': 0.0, 'acc_num': 0.0, 'N': 0.0,
                'TP': 0.0, 'FP': 0.0, 'FN': 0.0}

    # 裁剪到有效像素
    op = ori_pred[finite_mask]
    og = ori_gt[finite_mask]
    tp = ttc_pred[finite_mask]
    tg = ttc_gt[finite_mask]

    # |Δθ| 与 Acc@τθ 的计数
    abs_dtheta = torch.abs(op - og)
    mae_num = torch.sum(abs_dtheta).item()
    acc_num = torch.sum((abs_dtheta <= tau_theta).float()).item()

    # 高风险二分类标签
    y_gt  = (tg < tau_T) & (og <= tau_theta)
    y_hat = (tp < tau_T) & (op <= tau_theta)

    TP = torch.sum(y_hat & y_gt).item()
    FP = torch.sum(y_hat & (~y_gt)).item()
    FN = torch.sum((~y_hat) & y_gt).item()

    return {'mae_num': mae_num, 'acc_num': acc_num, 'N': float(N),
            'TP': float(TP), 'FP': float(FP), 'FN': float(FN)}

def reduce_metrics_across_frames(stats_list, beta=2.0, eps=1e-12):
    # 累加原子统计量
    mae_num = sum(d['mae_num'] for d in stats_list)
    acc_num = sum(d['acc_num'] for d in stats_list)
    N       = sum(d['N']       for d in stats_list)

    TP = sum(d['TP'] for d in stats_list)
    FP = sum(d['FP'] for d in stats_list)
    FN = sum(d['FN'] for d in stats_list)

    # micro-average（先合并，再算比值）
    mae = mae_num / max(1.0, N)
    acc = acc_num / max(1.0, N)

    prec = TP / max(eps, TP + FP)
    rec  = TP / max(eps, TP + FN)

    b2 = beta * beta
    fbeta = (1 + b2) * prec * rec / max(eps, b2 * prec + rec)

    return {
        'MAE_theta': mae,
        'Acc@tau_theta': acc,
        'Precision': prec,
        'Recall': rec,
        'HR-Fbeta': fbeta
    }
