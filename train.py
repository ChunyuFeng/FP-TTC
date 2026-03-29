from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import datetime
import json
import socket
import subprocess
import torch.distributed as dist

import dataloader.dataset as datasets
from fpttc.fp_ttc import FpTTC
from utils.trainer import TTCTrainer
from utils.dist import is_main_process

import neptune

time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
out_dir = "./log/%s_surround_ttc"%(time_stamp)

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                    help='where to save the training log and models')
parser.add_argument('--stage', default='chairs', type=str,
                    help='training stage on different datasets')
parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                    help='validation datasets')
parser.add_argument('--max_flow', default=400, type=int,
                    help='exclude very large motions during training')
parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                    help='image size for training')
parser.add_argument('--padding_factor', default=16, type=int,
                    help='the input should be divisible by padding_factor, otherwise do padding or resizing')
parser.add_argument('--train_info_path', default='./Datasets/nuscenes/2_trainval_test_infos/train', type=str,
                    help='path to the train pkl directory')
parser.add_argument('--train_info_file', default='nusc_train_infos_key_frames_160_1920_fov_8_15.pkl', type=str,
                    help='train pkl file name')
parser.add_argument('--require_complete_depth', action='store_true',
                    help='only keep samples whose 12 depth maps all exist')
parser.add_argument('--max_train_samples', default=None, type=int,
                    help='limit the number of train samples after filtering')
parser.add_argument('--proj_cache_root',
                    default='./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_v1',
                    type=str,
                    help='root directory of precomputed proj_pix cache')

# evaluation
parser.add_argument('--eval', action='store_true',
                    help='evaluation after training done')
parser.add_argument('--save_eval_to_file', action='store_true')
parser.add_argument('--evaluate_matched_unmatched', action='store_true')
parser.add_argument('--val_things_clean_only', action='store_true')
parser.add_argument('--with_speed_metric', action='store_true',
                    help='with speed methic when evaluation')

# training
parser.add_argument('--load_flow_param', action='store_true')
parser.add_argument('--epoch', default=25, type=int)
parser.add_argument('--lr', default=4e-4, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--grad_clip', default=1.0, type=float)
parser.add_argument('--num_steps', default=100000, type=int)
parser.add_argument('--seed', default=326, type=int)
parser.add_argument('--summary_freq', default=100, type=int)
parser.add_argument('--val_freq', default=10000, type=int)
parser.add_argument('--save_ckpt_freq', default=10000, type=int)
parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

# resume pretrained model or resume training
parser.add_argument('--resume', default=None, type=str,
                    help='resume from pretrained model or resume from unexpectedly terminated training')
parser.add_argument('--strict_resume', action='store_true',
                    help='strict resume while loading pretrained weights')
parser.add_argument('--no_resume_optimizer', action='store_true')

# model: learnable parameters
parser.add_argument('--num_scales', default=1, type=int,
                    help='feature scales: 1/8 or 1/8 + 1/4')
parser.add_argument('--num_head', default=1, type=int)
parser.add_argument('--feature_channels', default=128, type=int)
parser.add_argument('--upsample_factor', default=8, type=int)
parser.add_argument('--ffn_dim_expansion', default=4, type=int)
parser.add_argument('--num_transformer_layers', default=6, type=int)
parser.add_argument('--reg_refine', action='store_true',
                    help='optional task-specific local regression refinement')
parser.add_argument('--parallel', action='store_true',
                    help='use distributed data parallel for training')
parser.add_argument('--load_opt', action='store_true',
                    help='optional task-specific local regression refinement')

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

# predict on sintel and kitti test set for submission
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

# inference on images or videos
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

# depth-free / distillation
parser.add_argument('--no_depth', action='store_true',
                    help='train without depth input (3ch RGB only); required on depthfree-rvt-v1')
parser.add_argument('--use_teacher_distill', action='store_true',
                    help='use teacher (proj_pix hard projection) for distillation')
parser.add_argument('--lambda_feat_distill', type=float, default=1.0,
                    help='weight for feature distillation loss')
parser.add_argument('--lambda_corr_distill', type=float, default=0.5,
                    help='weight for correlation distillation loss')
parser.add_argument('--distill_end_pct', type=float, default=0.7,
                    help='fraction of training at which distillation weight reaches 0')
parser.add_argument('--student_tail_epochs', type=int, default=0,
                    help='extra student-only fine-tuning epochs after distillation')
parser.add_argument('--loss_weight_alpha', type=float, default=0.0,
                    help='distance-based scale loss reweighting alpha; 0=off')
parser.add_argument('--new_module_lr_mult', type=float, default=1.0,
                    help='learning-rate multiplier for non-pretrained modules relative to args.lr')
parser.add_argument('--edge_loss_weight', type=float, default=0.0,
                    help='weight for optional masked scale-gradient loss; 0=off')
parser.add_argument('--use_internal_depth_guidance', action='store_true',
                    help='train with an internal depth-distribution head that guides RVT bin selection')
parser.add_argument('--depth_loss_weight', type=float, default=0.5,
                    help='weight for internal depth-distribution supervision')
parser.add_argument('--depth_selection_mode', choices=['soft', 'hard_topk'], default='hard_topk',
                    help='how depth prior influences RVT candidate selection')
parser.add_argument('--bootstrap_topk', type=int, default=4,
                    help='number of (camera,bin) candidates kept during geom bootstrap in hard_topk mode')
parser.add_argument('--attn_topk', type=int, default=8,
                    help='number of (camera,bin) candidates kept per attention head in hard_topk mode')
parser.add_argument('--bootstrap_prior_scale', type=float, default=2.0,
                    help='scale applied to log depth prior during geom bootstrap selection')
parser.add_argument('--attn_prior_scale', type=float, default=2.0,
                    help='scale applied to log depth prior when fused into attention logits')
parser.add_argument('--depth_prior_eps', type=float, default=1e-6,
                    help='epsilon for stable log(depth_prior)')
parser.add_argument('--activation_checkpointing', action='store_true',
                    help='enable activation checkpointing on the heaviest transformer stacks')

# 加载预训练的单分支模型：
parser.add_argument(
    '--scale_pretrained_ckpt',
    type=str, default=None,
    help='path to pretrained scale‑only model (.pth or .pth.tar)'
)

# neptune
parser.add_argument('--neptune', action='store_true',
                    help='use neptune for logging')

args = parser.parse_args()

if args.parallel:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    parallel = True
else:
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    parallel = False

OLD_MODULE_PREFIXES = ('cnet.', 'featnet.', 'corrnet.')


def strip_module_prefix(name):
    return name[len('module.'):] if name.startswith('module.') else name


def print_optimizer_group_summary(optimizer):
    summaries = getattr(optimizer, '_fp_ttc_lr_group_summary', [])
    if not summaries:
        for i, pg in enumerate(optimizer.param_groups):
            print(f"  param group {i} lr = {pg['lr']:.3e}")
        return

    for idx, summary in enumerate(summaries):
        print(
            f"  [{summary['name']}] group={idx} "
            f"tensors={summary['tensor_count']} "
            f"params={summary['parameter_count']} "
            f"init_lr={summary['initial_lr']:.3e} "
            f"max_lr={summary['max_lr']:.3e}"
        )


def build_optimizer(model, args):
    """
    构造只作用于 requires_grad=True 参数的 AdamW 优化器

    Args:
      model:  FpTTC 模型实例
      args:  包含 lr, weight_decay 等超参数的命令行 args
    """
    old_max_lr = args.lr
    old_ini_lr = old_max_lr / 25
    old_min_lr = old_ini_lr / 1e4

    new_max_lr = args.lr * args.new_module_lr_mult
    new_ini_lr = new_max_lr / 25
    new_min_lr = new_ini_lr / 1e4

    grouped = {
        'old': {
            'params': [],
            'prefixes': list(OLD_MODULE_PREFIXES),
            'max_lr': old_max_lr,
            'initial_lr': old_ini_lr,
            'min_lr': old_min_lr,
        },
        'new': {
            'params': [],
            'prefixes': [],
            'max_lr': new_max_lr,
            'initial_lr': new_ini_lr,
            'min_lr': new_min_lr,
        },
    }
    tensor_counts = {'old': 0, 'new': 0}
    param_counts = {'old': 0, 'new': 0}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        bare_name = strip_module_prefix(name)
        group_name = 'old' if any(bare_name.startswith(prefix) for prefix in OLD_MODULE_PREFIXES) else 'new'
        grouped[group_name]['params'].append(p)
        tensor_counts[group_name] += 1
        param_counts[group_name] += p.numel()

    param_groups = []
    group_summary = []
    for group_name in ('old', 'new'):
        spec = grouped[group_name]
        if not spec['params']:
            continue
        param_groups.append({
            "params": spec['params'],
            "lr": spec['initial_lr'],
            "max_lr": spec['max_lr'],
            "initial_lr": spec['initial_lr'],
            "min_lr": spec['min_lr'],
        })
        group_summary.append({
            'name': group_name,
            'tensor_count': tensor_counts[group_name],
            'parameter_count': int(param_counts[group_name]),
            'max_lr': float(spec['max_lr']),
            'initial_lr': float(spec['initial_lr']),
            'min_lr': float(spec['min_lr']),
            'module_prefixes': list(spec['prefixes']),
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=old_max_lr,
        weight_decay=args.weight_decay,
    )
    optimizer._fp_ttc_lr_group_summary = group_summary
    optimizer._fp_ttc_lr_group_names = [item['name'] for item in group_summary]
    return optimizer


def align_state_dict_module_prefix(sd, model_keys):
    """Align checkpoint keys to the current model keyspace with respect to DDP `module.` prefix."""
    model_has_module = any(k.startswith('module.') for k in model_keys)
    ckpt_has_module = any(k.startswith('module.') for k in sd.keys())

    if model_has_module and not ckpt_has_module:
        return {'module.' + k: v for k, v in sd.items()}
    if not model_has_module and ckpt_has_module:
        return {k.replace('module.', '', 1): v for k, v in sd.items()}
    return sd


def extract_checkpoint_state_dict(checkpoint):
    if 'model' in checkpoint:
        return checkpoint['model']
    if 'net' in checkpoint:
        return checkpoint['net']
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def build_load_report(label, path, sd, load_info, extra=None):
    loaded_keys = sorted(set(sd.keys()) - set(load_info.unexpected_keys))
    report = {
        'label': label,
        'path': path,
        'loaded_key_count': len(loaded_keys),
        'missing_key_count': len(load_info.missing_keys),
        'unexpected_key_count': len(load_info.unexpected_keys),
        'loaded_keys': loaded_keys,
        'missing_keys': list(load_info.missing_keys),
        'unexpected_keys': list(load_info.unexpected_keys),
    }
    if extra:
        report.update(extra)
    return report


def write_json(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def get_git_metadata():
    def _git_cmd(args):
        try:
            return subprocess.check_output(
                ['git'] + args,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    return {
        'commit': _git_cmd(['rev-parse', 'HEAD']),
        'branch': _git_cmd(['rev-parse', '--abbrev-ref', 'HEAD']),
    }

def main():

    if args.neptune and ((not args.parallel) or dist.get_rank() == 0):
        run = neptune.init_run(project="fengchunyu/FPTTC")
    else:
        run = None

    if not args.no_depth:
        raise ValueError(
            "depthfree-rvt-v1 only supports RGB-only training. "
            "Please pass --no_depth; the legacy RGBD / teacher-init forward is not maintained on this branch."
        )
    
    model = FpTTC(
        num_scales             = args.num_scales,
        feature_channels       = args.feature_channels,
        upsample_factor        = args.upsample_factor,
        num_head               = args.num_head,
        ffn_dim_expansion      = args.ffn_dim_expansion,
        num_transformer_layers = args.num_transformer_layers,
        reg_refine             = args.reg_refine,
        no_depth               = args.no_depth,
        activation_checkpointing = args.activation_checkpointing,
    ).cuda()

    start_epoch = 0

    # 从预训练的模型加载参数
    model_init_info = {
        'no_depth': args.no_depth,
        'use_teacher_distill': args.use_teacher_distill,
        'activation_checkpointing': args.activation_checkpointing,
        'resume': None,
        'scale_pretrained_ckpt': None,
        'optimizer_groups': {},
    }

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        if args.load_opt:
            optimizer = build_optimizer(model, args)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint.get('epoch', 0)
                optimizer_loaded = True
            except ValueError as exc:
                optimizer_loaded = False
                if is_main_process():
                    print(f"[WARN] Skipped optimizer state loading due to param-group mismatch: {exc}")
        else:
            optimizer_loaded = False

        sd = extract_checkpoint_state_dict(checkpoint)

        model_keys = model.state_dict().keys()
        sd = align_state_dict_module_prefix(sd, model_keys)

        # 2) 处理 conv1 通道不匹配（4ch→3ch 截取 RGB）
        conv1_key = 'module.cnet.conv1.weight' if any(k.startswith('module.') for k in model_keys) else 'cnet.conv1.weight'
        if args.no_depth and conv1_key in sd:
            ckpt_conv1 = sd[conv1_key]
            if ckpt_conv1.shape[1] == 4:
                sd[conv1_key] = ckpt_conv1[:, :3, :, :]
                if is_main_process():
                    print(f'[INFO] Sliced {conv1_key} from {ckpt_conv1.shape} to {sd[conv1_key].shape}')

        # 3) 载入并接收加载报告
        load_info = model.load_state_dict(sd, strict=False)
        model_init_info['resume'] = build_load_report(
            label='resume',
            path=args.resume,
            sd=sd,
            load_info=load_info,
            extra={
                'optimizer_loaded': bool(optimizer_loaded),
                'resume_epoch': int(start_epoch),
            },
        )

        # 4) 打印一下各类 key
        if is_main_process():
            # 成功匹配到的 keys = 原来 sd 里所有 keys，扣掉 “unexpected_keys”
            loaded_keys = set(sd.keys()) - set(load_info.unexpected_keys)
            print(f"[INFO] Loaded ({len(loaded_keys)}) keys:")
            for k in sorted(loaded_keys):
                print(f"    {k}")

            print(f"[WARN] Missing ({len(load_info.missing_keys)}) keys (not found in checkpoint):")
            for k in load_info.missing_keys:
                print(f"    {k}")

            print(f"[WARN] Unexpected ({len(load_info.unexpected_keys)}) keys (not used by model):")
            for k in load_info.unexpected_keys:
                print(f"    {k}")
    
    if parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], \
                        output_device=local_rank, find_unused_parameters=False)

        if is_main_process():
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)

    if is_main_process():
        git_metadata = get_git_metadata()
        write_json(os.path.join(out_dir, 'run_config.json'), {
            'args': vars(args),
            'git': git_metadata,
            'time_stamp': time_stamp,
            'out_dir': out_dir,
            'hostname': socket.gethostname(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        })
        write_json(os.path.join(out_dir, 'model_init.json'), model_init_info)

    if is_main_process():
        print('Start Loading ...')

    dataset = datasets.fetch_dataloader(args) 
    
    # stage 1: train scale branch
    if args.train_stage in ('scale', 'both'):
        ########################### LOAD PRETRAINED SCALE MODEL ###########################
        if args.scale_pretrained_ckpt is not None:
            ckpt = torch.load(args.scale_pretrained_ckpt, map_location=device)
            # 提取 state_dict
            sd = extract_checkpoint_state_dict(ckpt)

            model_dict = model.state_dict()
            model_keys = model_dict.keys()
            sd = align_state_dict_module_prefix(sd, model_keys)

            # 只挑出 cnet/featnet/corrnet 的参数
            if any(k.startswith('module.') for k in model_keys):
                prefixes = ('module.cnet.', 'module.featnet.', 'module.corrnet.')
            else:
                prefixes = ('cnet.', 'featnet.', 'corrnet.')
            filtered = {}
            for k, v in sd.items():
                if any(k.startswith(pref) for pref in prefixes):
                    filtered[k] = v

            conv1_key = 'module.cnet.conv1.weight' if any(k.startswith('module.') for k in model_keys) else 'cnet.conv1.weight'
            if args.no_depth and conv1_key in filtered:
                ckpt_conv1 = filtered[conv1_key]
                if ckpt_conv1.shape[1] == 4:
                    filtered[conv1_key] = ckpt_conv1[:, :3, :, :]
                    if is_main_process():
                        print(
                            f'[SCALE-INIT] Sliced {conv1_key} from '
                            f'{ckpt_conv1.shape} to {filtered[conv1_key].shape}'
                        )

            # 注入到当前模型里
            model_dict.update(filtered)
            load_info = model.load_state_dict(model_dict, strict=False)
            loaded = set(filtered.keys()) - set(load_info.unexpected_keys)
            model_init_info['scale_pretrained_ckpt'] = build_load_report(
                label='scale_pretrained_ckpt',
                path=args.scale_pretrained_ckpt,
                sd=filtered,
                load_info=load_info,
                extra={
                    'filtered_prefixes': list(prefixes),
                    'filtered_key_count': len(filtered),
                    'loaded_filtered_key_count': len(loaded),
                },
            )
            if is_main_process():
                write_json(os.path.join(out_dir, 'model_init.json'), model_init_info)
            if is_main_process():
                print(f"[SCALE-INIT] Loaded {len(loaded)} keys for scale backbone:")
                for k in sorted(loaded):
                    print("   ", k)
                if load_info.missing_keys:
                    print(f"[SCALE-INIT] Missing keys: {load_info.missing_keys}")
            ##########################################################################
        # 1) freeze risk branch only; shared trunk remains trainable
        freeze_prefixes = (
            'conv_corr_risk.', 'risk_net.',
        )

        for name, p in model.named_parameters():
            # remove "module." prefix if using DDP
            bare_name = name
            if bare_name.startswith('module.'):
                bare_name = bare_name[len('module.'):]
            
            if any(bare_name.startswith(prefix) for prefix in freeze_prefixes):
                p.requires_grad = False
            else:
                p.requires_grad = True

        optimizer = build_optimizer(model, args)
        model_init_info['optimizer_groups']['scale'] = {
            'new_module_lr_mult': float(args.new_module_lr_mult),
            'old_module_prefixes': list(OLD_MODULE_PREFIXES),
            'groups': getattr(optimizer, '_fp_ttc_lr_group_summary', []),
        }
        if is_main_process():
            write_json(os.path.join(out_dir, 'model_init.json'), model_init_info)

        if is_main_process():
            print_optimizer_group_summary(optimizer)

        args.batch_size = args.scale_batch_size  # scale-only stage batch size

        total_scale_epochs = args.scale_epochs + args.student_tail_epochs
        print(
            f"Training scale branch for {total_scale_epochs} epochs "
            f"(main={args.scale_epochs}, tail={args.student_tail_epochs})..."
        )

        # 2) compute scale loss
        trainer = TTCTrainer(model       = model,
                             dataset     = dataset,
                             optimizer   = optimizer,
                             args        = args,
                             start_epoch = start_epoch,
                             device      = device,
                             parallel    = parallel,
                             time_stamp  = time_stamp,
                             neptune_run = run,
                             scale_only  = True,
                             no_depth    = args.no_depth,
                             use_teacher_distill = args.use_teacher_distill,
                             lambda_feat_distill = args.lambda_feat_distill,
                             lambda_corr_distill = args.lambda_corr_distill,
                             distill_end_pct     = args.distill_end_pct,
                             student_tail_epochs = args.student_tail_epochs,
                             loss_weight_alpha   = args.loss_weight_alpha,
                             edge_loss_weight    = args.edge_loss_weight,
                             use_internal_depth_guidance = args.use_internal_depth_guidance,
                             depth_loss_weight   = args.depth_loss_weight,
                             depth_selection_mode = args.depth_selection_mode,
                             bootstrap_topk      = args.bootstrap_topk,
                             attn_topk           = args.attn_topk,
                             bootstrap_prior_scale = args.bootstrap_prior_scale,
                             attn_prior_scale    = args.attn_prior_scale,
                             depth_prior_eps     = args.depth_prior_eps,
                             )
        trainer.train()

    # stage 2: train risk branch
    if args.train_stage in ('risk', 'both'):
        # 1) freeze scale head only; shared trunk remains trainable
        freeze_prefixes = (
            'conv_corr_rvt_out.','scale_net.'
        )

        for name, p in model.named_parameters():
            bare_name = name
            if bare_name.startswith('module.'):
                bare_name = bare_name[len('module.'):]
            
            if any(bare_name.startswith(prefix) for prefix in freeze_prefixes):
                p.requires_grad = False
            else:
                p.requires_grad = True

        optimizer = build_optimizer(model, args)
        model_init_info['optimizer_groups']['risk'] = {
            'new_module_lr_mult': float(args.new_module_lr_mult),
            'old_module_prefixes': list(OLD_MODULE_PREFIXES),
            'groups': getattr(optimizer, '_fp_ttc_lr_group_summary', []),
        }
        if is_main_process():
            write_json(os.path.join(out_dir, 'model_init.json'), model_init_info)

        if is_main_process():
            print_optimizer_group_summary(optimizer)

        args.batch_size = args.risk_batch_size  # risk-only stage batch size

        total_risk_epochs = args.risk_epochs + args.student_tail_epochs
        print(
            f"Training risk branch for {total_risk_epochs} epochs "
            f"(main={args.risk_epochs}, tail={args.student_tail_epochs})..."
        )
        # 2) compute risk loss
        trainer = TTCTrainer(model       = model,
                             dataset     = dataset,
                             optimizer   = optimizer,
                             args        = args,
                             start_epoch = start_epoch,
                             device      = device,
                             parallel    = parallel,
                             time_stamp  = time_stamp,
                             neptune_run = run,
                             scale_only  = False,
                             no_depth    = args.no_depth,
                             use_teacher_distill = args.use_teacher_distill,
                             lambda_feat_distill = args.lambda_feat_distill,
                             lambda_corr_distill = args.lambda_corr_distill,
                             distill_end_pct     = args.distill_end_pct,
                             student_tail_epochs = args.student_tail_epochs,
                             loss_weight_alpha   = args.loss_weight_alpha,
                             edge_loss_weight    = args.edge_loss_weight,
                             use_internal_depth_guidance = args.use_internal_depth_guidance,
                             depth_loss_weight   = args.depth_loss_weight,
                             depth_selection_mode = args.depth_selection_mode,
                             bootstrap_topk      = args.bootstrap_topk,
                             attn_topk           = args.attn_topk,
                             bootstrap_prior_scale = args.bootstrap_prior_scale,
                             attn_prior_scale    = args.attn_prior_scale,
                             depth_prior_eps     = args.depth_prior_eps,
                             )
        trainer.train()


if __name__ == "__main__":
    main()
