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

import neptune

time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
out_dir = "./log/%s_selfcon_ttc"%(time_stamp)

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
                    help='optional task-specific local regression refinement')
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


parser.add_argument('--finetune', action='store_true',
                    help='run two-stage fine-tuning')
parser.add_argument('--ft_epoch_s1', type=int, default=50,
                    help='stage1 epochs (freeze cnet)')
parser.add_argument('--ft_epoch_s2', type=int, default=150,
                    help='stage2 epochs (unfreeze cnet)')

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

    model = FpTTC(num_scales=args.num_scales,
                  feature_channels=args.feature_channels,
                  upsample_factor=args.upsample_factor,
                  num_head=args.num_head,
                  ffn_dim_expansion=args.ffn_dim_expansion,
                  num_transformer_layers=args.num_transformer_layers,
                  range_image_feat_shape=[(20, 240), (40, 480)],
                  reg_refine=args.reg_refine,
                  load_cnet=args.load_cnet,
                  pretrained_cnet_path=args.load_cnet_path,
                  freeze_cnet=args.freeze_cnet,
                  train=True).cuda()
    
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
    
    if args.load_cnet and args.fine_tune_cnet:
        ########################### 微调 cnet ############################
        net = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        
        # 为 cnet 设置更小的学习率，仅仅微调
        cnet_lr = max_lr * 0.1
        cnet_ini_lr = ini_lr * 0.1
        cnet_min_lr = min_lr * 0.1

        other_lr = max_lr
        other_ini_lr = ini_lr
        other_min_lr = min_lr

        optimizer = torch.optim.AdamW([
            {
                "params": net.cnet.parameters(),
                "max_lr": cnet_lr,
                "initial_lr": cnet_ini_lr,
                "min_lr": cnet_min_lr
            },
            {
                "params": [p for n,p in net.named_parameters() if not n.startswith("cnet.")],
                "max_lr": other_lr,
                "initial_lr": other_ini_lr,
                "min_lr": other_min_lr
            },
        ],
        lr=max_lr, weight_decay=args.weight_decay)
        ########################### 微调 cnet ############################

    epoch = 0

    '''
    # if args.resume is not None:
    #     checkpoint = torch.load(args.resume, map_location=device)
    #     if args.load_opt:
    #         epoch = checkpoint['epoch']
    #         optimizer.load_state_dict(checkpoint['optimizer'])

    #     if 'model' in checkpoint:
    #         model.load_state_dict(checkpoint['model'])
    #     elif 'net' in checkpoint:
    #         # model.load_state_dict(data['net'])
    #         #print(data['net'].items())
    #         model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    #     elif 'state_dict' in checkpoint:
    #         model.load_state_dict(checkpoint['state_dict'])
    #     else:
    #         model.load_state_dict(checkpoint)
    '''

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        if args.load_opt:
            epoch = checkpoint.get('epoch', 0)
            optimizer.load_state_dict(checkpoint['optimizer'])

        # 先挑出 state_dict
        if 'model' in checkpoint:
            sd = checkpoint['model']
        elif 'net' in checkpoint:
            sd = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint

        # 加载时用 strict=False，跳过缺失的 scale_fuser 权重
        model.load_state_dict(sd, strict=False)
        if is_main_process(parallel):
            print("[WARN] Resuming with strict=False, missing keys (scale_fuser) ignored.")
    
    if parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], \
                        output_device=local_rank, find_unused_parameters=True)

        base_model = model.module if hasattr(model, "module") else model

        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], \
        #                 output_device=local_rank)
        if torch.distributed.get_rank()==0:
                # torch.distributed.barrier()
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)
                    loss_txt = out_dir + '/0.txt'
                    file = open(loss_txt,'w')
                    file.close()
        
        # else:
        #     torch.distributed.barrier()
    else:
        if not os.path.isdir(out_dir):
            # os.mkdir(out_dir)
            # loss_txt = out_dir + '/0.txt'
            # file = open(loss_txt,'w')
            # file.close()
            os.makedirs(out_dir)  # 创建多层目录
            loss_txt = os.path.join(out_dir, '0.txt')  # 构建文件路径
            with open(loss_txt, 'w') as file:
                file.close()

    
    start = time.time()
    if is_main_process(args.parallel):
        print('Start Loading ...')
    dataset = datasets.fetch_dataloader(args)
    if is_main_process(args.parallel):
        print('Done ', time.time()-start)
    if is_main_process(args.parallel):
        print("Learning rate: ", optimizer.state_dict()['param_groups'][0]['lr'])

    # trainer = TTCTrainer(model=model, dataset=dataset, optimizer=optimizer, args=args, 
    #                     start_epoch=epoch, device=device, model_path=args.resume, 
    #                     parallel=parallel, time_stamp=time_stamp, max_lr=max_lr,
    #                     neptune_run=run)
    
    # trainer.train()

    trainer = TTCTrainer(model=model, dataset=dataset,
                        optimizer=optimizer, args=args, start_epoch=epoch,
                        device=device, model_path=args.resume, parallel=parallel,
                        time_stamp=time_stamp, max_lr=max_lr,
                        neptune_run=run)
    
    if args.finetune:
        base_lr = args.lr
        steps_per_epoch = len(dataset)

        # 全网可训练（不再 freeze）
        for p in model.parameters():
            p.requires_grad = True

        # 拆分三档参数组
        params_cnet   = list(base_model.cnet.parameters())
        params_fusion = list(base_model.scale_fuser.parameters())
        # 中间层：除 cnet、fusion 以外的所有参数
        fusion_ids = {id(p) for p in params_fusion}
        cnet_ids   = {id(p) for p in params_cnet}
        params_mid = [p for p in base_model.parameters()
                      if id(p) not in fusion_ids and id(p) not in cnet_ids]

        optimizer = torch.optim.AdamW([
            {'params': params_cnet,
             'max_lr': base_lr * 0.1,
             'initial_lr': (base_lr*0.1)/25,
             'min_lr': (base_lr*0.1)/1e4},
            {'params': params_mid,
             'max_lr': base_lr * 0.5,
             'initial_lr': (base_lr*0.5)/25,
             'min_lr': (base_lr*0.5)/1e4},
            {'params': params_fusion,
             'max_lr': base_lr,
             'initial_lr': base_lr/25,
             'min_lr': base_lr/1e4},
        ], lr=base_lr * 0.1, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[base_lr*0.1, base_lr*0.5, base_lr],
            epochs=args.ft_epoch_s2,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=-1,
        )

        trainer.optimizer     = optimizer
        trainer.lr_scheduler3 = scheduler
        for ep in range(args.ft_epoch_s2):
            if is_main_process(parallel):
                print(f"[Fine-tune] Epoch {ep+1}/{args.ft_epoch_s2}")
            trainer.train_epoch(ep)
    else:
        trainer.train()

'''
    # if args.finetune:
    #     base_lr = args.lr  # e.g. 4e-4
    #     steps_per_epoch = len(dataset)  # 或 len(train_loader)

    #     # —— Stage 1：冻结 cnet，仅 train 老模块 + 新模块 —— 
    #     for name, p in model.named_parameters():
    #         if name.startswith("cnet"):
    #             p.requires_grad = False

    #     # 参数分组：老模块 vs 新模块
    #     params_fusion = list(model.scale_fuser.parameters())
    #     params_old = [p for n,p in model.named_parameters()
    #                   if p.requires_grad and all(not n.startswith("cnet")) 
    #                   and p not in params_fusion]

    #     optimizer1 = torch.optim.AdamW([
    #         {'params': params_old,
    #          'max_lr': base_lr * 0.5,
    #          'initial_lr': (base_lr*0.5)/25,
    #          'min_lr': (base_lr*0.5)/1e4},
    #         {'params': params_fusion,
    #          'max_lr': base_lr,
    #          'initial_lr': base_lr/25,
    #          'min_lr': base_lr/1e4}
    #     ], lr=base_lr * 0.5, weight_decay=args.weight_decay)

    #     scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer1,
    #         max_lr=[base_lr*0.5, base_lr],
    #         epochs=args.ft_epoch_s1,
    #         steps_per_epoch=steps_per_epoch,
    #         pct_start=0.05,
    #         cycle_momentum=False,
    #         anneal_strategy='cos',
    #         last_epoch=-1,
    #     )

    #     trainer.optimizer = optimizer1
    #     trainer.lr_scheduler3 = scheduler1
    #     for ep in range(args.ft_epoch_s1):
    #         if is_main_process(parallel):
    #             print(f"[Fine-tune Stage1] Epoch {ep+1}/{args.ft_epoch_s1}")
    #         trainer.train_epoch(ep)

    #     # —— Stage 2：解冻 cnet，三组参数分档 LR —— 
    #     for p in model.cnet.parameters():
    #         p.requires_grad = True

    #     params_cnet   = list(model.cnet.parameters())
    #     params_mid    = params_old
    #     params_fusion = list(model.scale_fuser.parameters())

    #     optimizer2 = torch.optim.AdamW([
    #         {'params': params_cnet,
    #          'max_lr': base_lr * 0.1,
    #          'initial_lr': (base_lr*0.1)/25,
    #          'min_lr': (base_lr*0.1)/1e4},
    #         {'params': params_mid,
    #          'max_lr': base_lr * 0.5,
    #          'initial_lr': (base_lr*0.5)/25,
    #          'min_lr': (base_lr*0.5)/1e4},
    #         {'params': params_fusion,
    #          'max_lr': base_lr,
    #          'initial_lr': base_lr/25,
    #          'min_lr': base_lr/1e4},
    #     ], lr=base_lr * 0.1, weight_decay=args.weight_decay)

    #     scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer2,
    #         max_lr=[base_lr*0.1, base_lr*0.5, base_lr],
    #         epochs=args.ft_epoch_s2,
    #         steps_per_epoch=steps_per_epoch,
    #         pct_start=0.05,
    #         cycle_momentum=False,
    #         anneal_strategy='cos',
    #         last_epoch=-1,
    #     )

    #     trainer.optimizer = optimizer2
    #     trainer.lr_scheduler3 = scheduler2
    #     for ep in range(args.ft_epoch_s2):
    #         idx = args.ft_epoch_s1 + ep
    #         if is_main_process(parallel):
    #             print(f"[Fine-tune Stage2] Epoch {idx+1}/{args.ft_epoch_s1+args.ft_epoch_s2}")
    #         trainer.train_epoch(idx)
    # else:
    #     trainer.train()
'''

def is_main_process(parallel: bool) -> bool:
    """
    返回 True 当且仅当：
      - 没有并行模式，或
      - 并行模式下当前进程 rank == 0
    """
    if not parallel:
        return True
    return dist.get_rank() == 0

if __name__ == "__main__":
    main()