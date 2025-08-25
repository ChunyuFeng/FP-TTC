# ===== utils/optim_finetune.py =====
import torch
from torch import nn

SCALE_PREFIXES = ("cnet.", "featnet.", "corrnet.", "conv_corr.", "scale_net.")

def split_wd_nowd(named_params, base_lr, wd):
    wd_params, nowd_params = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        # bias / norm 不做 weight decay
        if n.endswith(".bias") or "bn" in n.lower() or "norm" in n.lower():
            nowd_params.append(p)
        else:
            wd_params.append(p)
    groups = []
    if wd_params:
        groups.append({"params": wd_params, "lr": base_lr, "weight_decay": wd})
    if nowd_params:
        groups.append({"params": nowd_params, "lr": base_lr, "weight_decay": 0.0})
    return groups

def build_optimizer_finetune(model, args):
    """
    - 冻结 DA 编码器（DINOv2）
    - 训练 DA 的 DPT head + scale 分支（按名字筛）
    - head 与 scale 分别设定 lr
    """
    # 1) 冻结 encoder
    if hasattr(model, "da") and hasattr(model.da, "pretrained"):
        for p in model.da.pretrained.parameters():
            p.requires_grad = False
        model.da.pretrained.eval()

    # 2) 设定 lr
    lr_head  = min(2e-5, args.lr)  # DPT head 小一点
    lr_scale = args.lr             # scale 分支用原 lr

    param_groups = []
    max_lrs = []  # 为 OneCycleLR 准备

    # 2.1 DA head
    head_named = []
    for n, p in model.named_parameters():
        if n.startswith("da.depth_head"):
            head_named.append((n, p))
    param_groups += split_wd_nowd(head_named, lr_head, args.weight_decay)
    max_lrs += [lr_head] * len(split_wd_nowd(head_named, lr_head, args.weight_decay))

    # 2.2 scale 分支（按前缀匹配）
    scale_named = []
    for n, p in model.named_parameters():
        if n.startswith(SCALE_PREFIXES):
            scale_named.append((n, p))
    param_groups += split_wd_nowd(scale_named, lr_scale, args.weight_decay)
    max_lrs += [lr_scale] * len(split_wd_nowd(scale_named, lr_scale, args.weight_decay))

    # 3) 构造优化器
    optimizer = torch.optim.AdamW(param_groups, lr=lr_scale, weight_decay=args.weight_decay)
    return optimizer, max_lrs
