# train.py
from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import datetime
import torch.distributed as dist

import dataloader.dataset as datasets
from fpttc.fp_ttc import FpTTC
from utils.trainer import TTCTrainer
from utils.dist import is_main_process

import neptune

# ============== timestamp / out_dir ==============
time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
out_dir = "./log/%s_surround_ttc" % (time_stamp)

# ================== argparse =====================
parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--checkpoint_dir', default='tmp', type=str)
parser.add_argument('--stage', default='chairs', type=str)
parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+')
parser.add_argument('--max_flow', default=400, type=int)
parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+')
parser.add_argument('--padding_factor', default=16, type=int)

# evaluation
parser.add_argument('--eval', action='store_true')
parser.add_argument('--save_eval_to_file', action='store_true')
parser.add_argument('--evaluate_matched_unmatched', action='store_true')
parser.add_argument('--val_things_clean_only', action='store_true')
parser.add_argument('--with_speed_metric', action='store_true')

# training
parser.add_argument('--epoch', default=25, type=int)
parser.add_argument('--lr', default=4e-4, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_steps', default=100000, type=int)
parser.add_argument('--seed', default=326, type=int)
parser.add_argument('--summary_freq', default=100, type=int)
parser.add_argument('--val_freq', default=10000, type=int)
parser.add_argument('--save_ckpt_freq', default=10000, type=int)
parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

# ======= Resume / Pretrained（两条互斥路径） =======
parser.add_argument('--full_resume', type=str, default=None,
                    help='Load full training state (model+optimizer+scheduler+epoch+global_step) to continue.')
parser.add_argument('--strict_full_resume', action='store_true',
                    help='strict=True for full_resume load_state_dict.')

parser.add_argument('--pretrained', type=str, default=None,
                    help='Load only model weights as initialization (no optimizer/scheduler).')
parser.add_argument('--pretrained_include', type=str, default=None,
                    help='comma-separated prefixes to include, e.g. "cnet.,featnet.,corrnet."')
parser.add_argument('--pretrained_exclude', type=str, default=None,
                    help='comma-separated prefixes to exclude')

# model: learnable parameters
parser.add_argument('--num_scales', default=1, type=int)
parser.add_argument('--num_head', default=1, type=int)
parser.add_argument('--feature_channels', default=128, type=int)
parser.add_argument('--upsample_factor', default=8, type=int)
parser.add_argument('--ffn_dim_expansion', default=4, type=int)
parser.add_argument('--num_transformer_layers', default=6, type=int)
parser.add_argument('--reg_refine', action='store_true',
                    help='optional task-specific local regression refinement')
parser.add_argument('--parallel', action='store_true',
                    help='use distributed data parallel for training')

# model: parameter-free
parser.add_argument('--attn_type', default='swin', type=str,
                    help='attention function')
parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                    help='number of splits in attention')
parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                    help='correlation radius for matching, -1 indicates global matching')
parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                    help='self-attention radius for propagation, -1 indicates global attention')
parser.add_argument('--num_reg_refine', default=1, type=int,
                    help='number of additional local regression refinement')
parser.add_argument('--radial_sampling_num', default=8, type=int,
                    help='number of radial sampling points for spherical coordinates')

# loss
parser.add_argument('--gamma', default=0.9, type=float,
                    help='exponential weighting')

# predict / inference
parser.add_argument('--kittidataset', default='/mnt/pool2/lcl/data/data_scene_flow/training/', type=str)
parser.add_argument('--drivingdataset', default='/mnt/pool2/lcl/data/Driving/', type=str)
parser.add_argument('--submission', action='store_true',
                    help='submission to sintel or kitti test sets')
parser.add_argument('--output_path', default='output', type=str,
                    help='where to save the prediction results')
parser.add_argument('--save_vis_flow', action='store_true',
                    help='visualize flow prediction as .png image')
parser.add_argument('--no_save_flo', action='store_true',
                    help='not save flow as .flo if only visualization is needed')

parser.add_argument('--inference_dir', default=None, type=str)
parser.add_argument('--inference_video', default=None, type=str)
parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                    help='can specify the inference size for the input to the network')
parser.add_argument('--save_flo_flow', action='store_true')
parser.add_argument('--pred_bidir_flow', action='store_true',
                    help='predict bidirectional flow')
parser.add_argument('--pred_bwd_flow', action='store_true',
                    help='predict backward flow only')
parser.add_argument('--fwd_bwd_check', action='store_true',
                    help='forward backward consistency check with bidirection flow')
parser.add_argument('--save_video', action='store_true')
parser.add_argument('--concat_flow_img', action='store_true')

# distributed training
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

# train 2 branch sequentially
parser.add_argument('--train_stage', choices=['scale','risk','both'], default='both',
                    help='which branch(s) to train: scale only, risk only, or both sequentially')
parser.add_argument('--scale_epochs', type=int, default=1001,
                    help='number of epochs to train the scale branch')
parser.add_argument('--risk_epochs',  type=int, default=1001,
                    help='number of epochs to train the risk branch')
parser.add_argument('--scale_batch_size', type=int, default=1,
                    help='batch size for scale-only stage')
parser.add_argument('--risk_batch_size',  type=int, default=1,
                    help='batch size for risk-only stage')

# neptune
parser.add_argument('--neptune', action='store_true')

args = parser.parse_args()

# ------- DDP init -------
if args.parallel:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    parallel = True
else:
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    local_rank = 0
    parallel = False

# ============== utils ==============
def get_base_model(model):
    return model.module if hasattr(model, 'module') else model

def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ('model', 'net', 'state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    else:
        raise ValueError("Checkpoint format not supported")

def _clean_keys(sd):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}

def _parse_prefixes(csv_or_none):
    if not csv_or_none:
        return None
    return tuple([p.strip() for p in csv_or_none.split(',') if p.strip()])

def _filter_by_prefix(sd, include_prefixes=None, exclude_prefixes=None):
    if include_prefixes:
        sd = {k: v for k, v in sd.items() if k.startswith(include_prefixes)}
    if exclude_prefixes:
        sd = {k: v for k, v in sd.items() if not k.startswith(exclude_prefixes)}
    return sd

def build_optimizer(model, args):
    max_lr = args.lr
    ini_lr = max_lr / 25
    min_lr = ini_lr / 1e4
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "max_lr": max_lr, "initial_lr": ini_lr, "min_lr": min_lr}],
        lr=max_lr, weight_decay=args.weight_decay
    )
    return optimizer

def build_loader(dataset, batch_size, num_workers, parallel):
    if not parallel:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True
        )
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True
        )
    return loader, sampler

def load_full_checkpoint(model, ckpt_path, strict=False, rank0=True):
    """无缝续训：返回 (epoch, global_step, opt_state, sch_state) 并打印加载报告"""
    ckpt = torch.load(ckpt_path, map_location=device)
    sd_raw = _extract_state_dict(ckpt)
    sd = _clean_keys(sd_raw)

    base = get_base_model(model)
    load_info = base.load_state_dict(sd, strict=strict)

    if rank0:
        print(f"[FULL-RESUME] load_info = {load_info}")
        print(f"[FULL-RESUME] missing_keys: {len(load_info.missing_keys)}, unexpected_keys: {len(load_info.unexpected_keys)}")
        unexpected = set(load_info.unexpected_keys)
        loaded = sorted([k for k in sd.keys() if k not in unexpected])
        print(f"[FULL-RESUME] Loaded {len(loaded)} params")
        for k in loaded:
            print("   ", k)
        if load_info.missing_keys:
            print("[FULL-RESUME] Missing:")
            for k in load_info.missing_keys:
                print("   ", k)
        if load_info.unexpected_keys:
            print("[FULL-RESUME] Unexpected:")
            for k in load_info.unexpected_keys:
                print("   ", k)

    epoch = ckpt.get('epoch', 0)
    gstep = ckpt.get('global_step', 0)
    opt_state = ckpt.get('optimizer', None)
    sch_state = ckpt.get('scheduler', None)
    return epoch, gstep, opt_state, sch_state

def load_pretrained_weights(model, ckpt_path, include_prefixes=None, exclude_prefixes=None, rank0=True):
    """仅加载模型权重（支持前缀过滤）"""
    ckpt = torch.load(ckpt_path, map_location=device)
    sd_raw = _extract_state_dict(ckpt)
    sd = _clean_keys(sd_raw)
    sd = _filter_by_prefix(sd, include_prefixes, exclude_prefixes)

    base = get_base_model(model)
    load_info = base.load_state_dict(sd, strict=False)

    if rank0:
        print(f"[PRETRAINED] missing_keys: {len(load_info.missing_keys)}, unexpected_keys: {len(load_info.unexpected_keys)}")
        unexpected = set(load_info.unexpected_keys)
        loaded = sorted([k for k in sd.keys() if k not in unexpected])
        print(f"[PRETRAINED] Loaded {len(loaded)} params")
        for k in loaded:
            print("   ", k)
        if load_info.missing_keys:
            print("[PRETRAINED] Missing:")
            for k in load_info.missing_keys:
                print("   ", k)
        if load_info.unexpected_keys:
            print("[PRETRAINED] Unexpected:")
            for k in load_info.unexpected_keys:
                print("   ", k)

# ================== main =======================
def main():

    # Neptune
    if args.neptune and ((not args.parallel) or dist.get_rank() == 0):
        run = neptune.init_run(project="fengchunyu/FPTTC")
    else:
        run = None
    
    # 0) 互斥检查
    if args.full_resume and args.pretrained:
        raise ValueError("Do not specify both --full_resume and --pretrained. Choose one.")

    # 1) 构建模型
    model = FpTTC(
        num_scales             = args.num_scales,
        feature_channels       = args.feature_channels,
        upsample_factor        = args.upsample_factor,
        num_head               = args.num_head,
        ffn_dim_expansion      = args.ffn_dim_expansion,
        num_transformer_layers = args.num_transformer_layers,
        reg_refine             = args.reg_refine,
    ).cuda()

    # 2) 先加载权重（在 DDP 包装前）
    start_epoch_full = 0
    global_step_full = 0
    opt_state_full = None
    sch_state_full = None

    if args.full_resume:
        start_epoch_full, global_step_full, opt_state_full, sch_state_full = load_full_checkpoint(
            model, args.full_resume, strict=args.strict_full_resume, rank0=is_main_process()
        )
    elif args.pretrained:
        inc = _parse_prefixes(args.pretrained_include)
        exc = _parse_prefixes(args.pretrained_exclude)
        load_pretrained_weights(
            model, args.pretrained, include_prefixes=inc, exclude_prefixes=exc, rank0=is_main_process()
        )

    # 3) DDP 包装
    if args.parallel:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            output_device=local_rank, find_unused_parameters=True
        )
        if is_main_process():
            os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)

    if is_main_process():
        print('Start Loading ...')

    # 4) 数据集 & Loader（统一在外面建）
    dataset = datasets.fetch_dataloader(args)

    # ===================== Stage 1: train scale branch =====================
    if args.train_stage in ('scale', 'both'):
        # 冻结风险分支
        freeze_prefixes = ('conv_corr_risk.', 'risk_net.')
        for name, p in model.named_parameters():
            bare = name[7:] if name.startswith('module.') else name
            p.requires_grad = not any(bare.startswith(pref) for pref in freeze_prefixes)

        # 专用 batch_size
        args.batch_size = args.scale_batch_size
        train_loader, train_sampler = build_loader(dataset, args.batch_size, args.num_workers, parallel)

        # 优化器
        optimizer = build_optimizer(model, args)
        # LR 调度器（OneCycleLR 需要 steps_per_epoch）
        steps_per_epoch = len(train_loader)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.scale_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=-1,
        )

        # 若是 full resume 且现在训练的是 scale 阶段 → 恢复 optimizer/scheduler/epoch/gstep
        start_epoch = 0
        init_global_step = 0
        if args.full_resume and args.train_stage in ('scale'):
            if opt_state_full is not None:
                try:
                    optimizer.load_state_dict(opt_state_full)
                    if is_main_process():
                        print("[FULL-RESUME] Optimizer state loaded (scale).")
                except Exception as e:
                    if is_main_process():
                        print(f"[FULL-RESUME] Skip loading optimizer (scale): {e}")
            if sch_state_full is not None:
                try:
                    lr_scheduler.load_state_dict(sch_state_full)
                    if is_main_process():
                        print("[FULL-RESUME] Scheduler state loaded (scale).")
                except Exception as e:
                    if is_main_process():
                        print(f"[FULL-RESUME] Skip loading scheduler (scale): {e}")
            start_epoch = start_epoch_full
            init_global_step = global_step_full

        if is_main_process():
            print("Learning rate (scale): ", optimizer.state_dict()['param_groups'][0]['lr'])
            print(f"Training scale branch for {args.scale_epochs} epochs...")

        trainer = TTCTrainer(
            model         = model,
            train_loader  = train_loader,
            train_sampler = train_sampler,
            optimizer     = optimizer,
            lr_scheduler  = lr_scheduler,
            args          = args,
            start_epoch   = start_epoch,
            init_global_step = init_global_step,
            device        = device,
            parallel      = parallel,
            time_stamp    = time_stamp,
            neptune_run   = run,
            scale_only    = True,
        )
        trainer.train()

    # ===================== Stage 2: train risk branch ======================
    if args.train_stage in ('risk'):
        # 冻结尺度分支
        freeze_prefixes = ('cnet.', 'featnet.', 'corrnet.', 'conv_corr.', 'scale_net.')
        for name, p in model.named_parameters():
            bare = name[7:] if name.startswith('module.') else name
            p.requires_grad = not any(bare.startswith(pref) for pref in freeze_prefixes)

        # 专用 batch_size
        args.batch_size = args.risk_batch_size
        train_loader, train_sampler = build_loader(dataset, args.batch_size, args.num_workers, parallel)

        optimizer = build_optimizer(model, args)
        steps_per_epoch = len(train_loader)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.risk_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=-1,
        )

        # full_resume 且当前只训练 risk 时，才恢复 opt/sch/epoch/gstep
        start_epoch = 0
        init_global_step = 0
        if args.full_resume and args.train_stage in ('risk',):
            if opt_state_full is not None:
                try:
                    optimizer.load_state_dict(opt_state_full)
                    if is_main_process():
                        print("[FULL-RESUME] Optimizer state loaded (risk).")
                except Exception as e:
                    if is_main_process():
                        print(f"[FULL-RESUME] Skip loading optimizer (risk): {e}")
            if sch_state_full is not None:
                try:
                    lr_scheduler.load_state_dict(sch_state_full)
                    if is_main_process():
                        print("[FULL-RESUME] Scheduler state loaded (risk).")
                except Exception as e:
                    if is_main_process():
                        print(f"[FULL-RESUME] Skip loading scheduler (risk): {e}")
            start_epoch = start_epoch_full
            init_global_step = global_step_full

        if is_main_process():
            print("Learning rate (risk): ", optimizer.state_dict()['param_groups'][0]['lr'])
            print(f"Training risk branch for {args.risk_epochs} epochs...")

        trainer = TTCTrainer(
            model         = model,
            train_loader  = train_loader,
            train_sampler = train_sampler,
            optimizer     = optimizer,
            lr_scheduler  = lr_scheduler,
            args          = args,
            start_epoch   = start_epoch,       # 默认 0；如果 full_resume+只训risk，则为 ckpt 的 epoch
            init_global_step = init_global_step,
            device        = device,
            parallel      = parallel,
            time_stamp    = time_stamp,
            neptune_run   = run,
            scale_only    = False,
        )
        trainer.train()

if __name__ == "__main__":
    main()