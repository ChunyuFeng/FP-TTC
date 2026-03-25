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
parser.add_argument('--val_info_path', default='./Datasets/nuscenes/2_trainval_test_infos/val', type=str,
                    help='path to the val pkl directory')
parser.add_argument('--val_info_file', default='nusc_val_infos_key_frames_160_1920_fov_8_15.pkl', type=str,
                    help='val pkl file name')
parser.add_argument('--require_complete_depth', action='store_true',
                    help='only keep samples whose 12 depth maps all exist')
parser.add_argument('--max_train_samples', default=None, type=int,
                    help='limit the number of train samples after filtering')
parser.add_argument('--proj_cache_root',
                    default='./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_hardproj_v1',
                    type=str,
                    help='root directory of precomputed proj_pix cache')
parser.add_argument('--val_proj_cache_root',
                    default='./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_hardproj_v1',
                    type=str,
                    help='root directory of precomputed val proj_pix cache')

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
parser.add_argument('--val_freq', default=1, type=int)
parser.add_argument('--save_ckpt_freq', default=10000, type=int)
parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
parser.add_argument('--val_batch_size', default=1, type=int)
parser.add_argument('--save_best', default=True, action=argparse.BooleanOptionalAction,
                    help='save best checkpoint based on validation loss')

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

def build_optimizer(model, args):
    """
    构造只作用于 requires_grad=True 参数的 AdamW 优化器

    Args:
      model:  FpTTC 模型实例
      args:  包含 lr, weight_decay 等超参数的命令行 args
    """
    max_lr = args.lr
    ini_lr = max_lr / 25
    min_lr = ini_lr / 1e4
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": trainable_params,
                "max_lr": max_lr,
                "initial_lr": ini_lr,
                "min_lr": min_lr
            }
        ],
        lr=max_lr, weight_decay=args.weight_decay)
    return optimizer

def main():

    if args.neptune and ((not args.parallel) or dist.get_rank() == 0):
        run = neptune.init_run(project="fengchunyu/FPTTC")
    else:
        run = None
    
    model = FpTTC(
        num_scales             = args.num_scales,
        feature_channels       = args.feature_channels,
        upsample_factor        = args.upsample_factor,
        num_head               = args.num_head,
        ffn_dim_expansion      = args.ffn_dim_expansion,
        num_transformer_layers = args.num_transformer_layers,
        reg_refine             = args.reg_refine,
    ).cuda()

    start_epoch = 0

    # 从预训练的模型加载参数
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        if args.load_opt:
            optimizer = build_optimizer(model, args)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0)

        if 'model' in checkpoint:
            sd = checkpoint['model']
        elif 'net' in checkpoint:
            sd = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint

        # 2) 载入并接收加载报告
        load_info = model.load_state_dict(sd, strict=False)

        # 3) 打印一下各类 key
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
                        output_device=local_rank, find_unused_parameters=True)

        if is_main_process():
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)

    if is_main_process():
        print('Start Loading ...')

    train_dataset = datasets.fetch_dataloader(args)
    val_dataset = datasets.fetch_val_dataloader(args)

    if is_main_process():
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples:   {len(val_dataset)}")
    
    # stage 1: train scale branch
    if args.train_stage in ('scale', 'both'):
        ########################### LOAD PRETRAINED SCALE MODEL ###########################
        if args.scale_pretrained_ckpt is not None:
            ckpt = torch.load(args.scale_pretrained_ckpt, map_location=device)
            # 提取 state_dict
            if 'model' in ckpt:
                sd = ckpt['model']
            elif 'net' in ckpt:
                # 去掉 DDP 的 module. 前缀
                sd = {k.replace('module.', ''): v for k, v in ckpt['net'].items()}
            elif 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            else:
                sd = ckpt

            # 只挑出 cnet/featnet/corrnet 的参数
            prefixes = ('cnet.', 'featnet.', 'corrnet.')
            filtered = {}
            for k, v in sd.items():
                if any(k.startswith(pref) for pref in prefixes):
                    filtered[k] = v

            # 注入到当前模型里
            model_dict = model.state_dict()
            model_dict.update(filtered)
            load_info = model.load_state_dict(model_dict, strict=False)
            if is_main_process():
                loaded = set(filtered.keys()) - set(load_info.unexpected_keys)
                print(f"[SCALE-INIT] Loaded {len(loaded)} keys for scale backbone:")
                for k in sorted(loaded):
                    print("   ", k)
                if load_info.missing_keys:
                    print(f"[SCALE-INIT] Missing keys: {load_info.missing_keys}")
            ##########################################################################
        # 1) freeze risk branch
        freeze_prefixes = (
            'conv_corr_risk.','risk_net.'
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

        if is_main_process():
            print("Learning rate: ", optimizer.state_dict()['param_groups'][0]['lr'])      

        args.batch_size = args.scale_batch_size  # scale-only stage batch size

        print(f"Training scale branch for {args.scale_epochs} epochs...")

        # 2) compute scale loss
        trainer = TTCTrainer(model       = model,
                             dataset     = train_dataset,
                             val_dataset = val_dataset,
                             optimizer   = optimizer,
                             args        = args,
                             start_epoch = start_epoch,
                             device      = device,
                             parallel    = parallel,
                             time_stamp  = time_stamp,
                             neptune_run = run,
                             scale_only  = True,
                             )
        trainer.train()

    # stage 2: train risk branch
    if args.train_stage in ('risk', 'both'):
        # 1) freeze scale branch
        freeze_prefixes = (
            'cnet.','featnet.','corrnet.','conv_corr.','scale_net.'
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

        if is_main_process():
            print("Learning rate: ", optimizer.state_dict()['param_groups'][0]['lr']) 

        args.batch_size = args.risk_batch_size  # risk-only stage batch size

        print(f"Training risk branch for {args.risk_epochs} epochs...")
        # 2) compute risk loss
        trainer = TTCTrainer(model       = model,
                             dataset     = train_dataset,
                             val_dataset = val_dataset,
                             optimizer   = optimizer,
                             args        = args,
                             start_epoch = start_epoch,
                             device      = device,
                             parallel    = parallel,
                             time_stamp  = time_stamp,
                             neptune_run = run,
                             scale_only  = False,
                             )
        trainer.train()


if __name__ == "__main__":
    main()
