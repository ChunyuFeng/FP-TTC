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

# misc
parser.add_argument('--count_time', action='store_true',
                    help='measure the inference time')

parser.add_argument('--debug', action='store_true')

# 加载预训练的cnet
parser.add_argument('--load_cnet', action='store_true',
                    help='load pretrained cnet')
parser.add_argument('--load_cnet_path', default='./pretrained/fpttc_mix.pth.tar', type=str,
                    help='load pretrained cnet path')
parser.add_argument('--freeze_cnet', action='store_true',
                    help='freeze cnet')
parser.add_argument('--fine_tune_cnet', action='store_true',
                    help='fine tune cnet')

# neptune
parser.add_argument('--neptune', action='store_true',
                    help='use neptune for logging')

args = parser.parse_args()
if args.freeze_cnet and args.fine_tune_cnet:
    raise ValueError("freeze_cnet and fine_tune_cnet cannot be used simultaneously")

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

def main():

    if args.neptune and ((not args.parallel) or dist.get_rank() == 0):
        run = neptune.init_run(project="fengchunyu/FPTTC")
    else:
        run = None

    model = FpTTC(num_scales             = args.num_scales,
                  feature_channels       = args.feature_channels,
                  upsample_factor        = args.upsample_factor,
                  num_head               = args.num_head,
                  ffn_dim_expansion      = args.ffn_dim_expansion,
                  num_transformer_layers = args.num_transformer_layers,
                  reg_refine             = args.reg_refine,
                  train                  = True).cuda()
    
    max_lr = args.lr
    ini_lr = max_lr / 25
    min_lr = ini_lr / 1e4

    optimizer = torch.optim.AdamW([
        {
            "params":model.parameters(),
            "max_lr":max_lr,
            "initial_lr":ini_lr,
            "min_lr":min_lr
        }
    ],
    lr=max_lr, weight_decay=args.weight_decay)

    epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        if args.load_opt:
            epoch = checkpoint.get('epoch', 0)
            optimizer.load_state_dict(checkpoint['optimizer'])

        if 'model' in checkpoint:
            sd = checkpoint['model']
        elif 'net' in checkpoint:
            sd = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint
        
        new_sd = {}
        for k, v in sd.items():
            new_sd[k] = v
            # 这一层的输入由 RGB 改为 RGBD
            # 如果是 cnet.conv1.weight，就把它的第 4 个通道设置为原来 3 个通道的均值
            if k == 'cnet.conv1.weight':
                out, c, h, w = v.shape
                new_w = torch.zeros(out, 4, h, w)
                new_w[:, :3, :, :] = v                      # 复用原来那 3 个通道
                new_w[:, 3:4, :, :] = v.mean(dim=1, keepdim=True)  # 第 4 通道取均值
                new_sd[k] = new_w
            # 如果是 featnet 的权重，就也复制到 featnet_risk
            if k.startswith("featnet."):
                new_sd["featnet_risk." + k[len("featnet."):]] = v
            # corrnet → corrnet_risk
            if k.startswith("corrnet."):
                new_sd["corrnet_risk." + k[len("corrnet."):]] = v
            # conv_corr → conv_corr_risk
            if k.startswith("conv_corr."):
                new_sd["conv_corr_risk." + k[len("conv_corr."):]] = v


        # 2) 载入并接收加载报告
        load_info = model.load_state_dict(new_sd, strict=False)

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

        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], \
        #                 output_device=local_rank)
        if torch.distributed.get_rank()==0:
                # torch.distributed.barrier()
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)
                    loss_txt = out_dir + '/0.txt'
                    file = open(loss_txt,'w')
                    file.close()

    else:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)  # 创建多层目录
            loss_txt = os.path.join(out_dir, '0.txt')  # 构建文件路径
            with open(loss_txt, 'w') as file:
                file.close()

    
    start = time.time()
    if is_main_process():
        print('Start Loading ...')
    dataset = datasets.fetch_dataloader(args)
    if is_main_process():
        print('Done ', time.time()-start)
        print("Learning rate: ", optimizer.state_dict()['param_groups'][0]['lr'])        

    trainer = TTCTrainer(model       = model,
                         dataset     = dataset,
                         optimizer   = optimizer, 
                         args        = args, 
                         start_epoch = epoch,
                         device      = device, 
                         model_path  = args.resume, 
                         parallel    = parallel,
                         time_stamp  = time_stamp, 
                         max_lr      = max_lr,
                         neptune_run = run)
    
    trainer.train()

if __name__ == "__main__":
    main()