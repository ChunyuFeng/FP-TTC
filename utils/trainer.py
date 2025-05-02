import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import cv2
import sys
import numpy as np
import datetime
import random
import math
import  time
import torch.optim as optim
import torch.nn as nn
from random import sample, shuffle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F 
import torch.distributed as dist

from PIL import Image
from .loss import get_loss, get_loss_multi, get_loss_nusc, get_loss_mix, get_loss_scale_map, get_loss_risk_score_map
from .draw import disp2rgb_normalized, flow_uv_to_colors, flow_to_image, visual_scale_map_range_image, visual_risk_score_map_range_image
from dataloader.load import load_calib_cam_to_cam, readFlowKITTI, disparity_loader, triangulation

def is_main_process(parallel: bool) -> bool:
    if not parallel:
        return True
    return dist.get_rank() == 0

class TTCTrainer(object):
    def __init__(self, model, dataset, optimizer, args, device, model_path = None, 
                start_epoch=0, parallel=False, time_stamp=None, max_lr=1e-4, crop_size=[352,1152],
                neptune_run=None):
        self.model = model
        self.parallel = parallel
        self.batch_size = args.batch_size
        self.parallel = parallel
        self.train_sampler = None
        if not self.parallel:
            self.train_loader = DataLoader(dataset, batch_size= args.batch_size, shuffle=True, 
                            num_workers=args.batch_size, drop_last=True, pin_memory=True)
        else:
            self.train_sampler = DistributedSampler(dataset)
            self.train_loader = DataLoader(dataset,batch_size=args.batch_size, \
                                sampler=self.train_sampler, shuffle=False, pin_memory=True, num_workers=2)

        self.epoch = args.epoch
        # if optimizer is None:
        #     self.optimizer = self.get_optimizer()
        # else :
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        
        steps_per_epoch = int(len(self.train_loader))
        #print(self.epoch, len(self.train_loader), self.batch_size, steps_per_epoch)
        starte = -1
        if self.start_epoch>0:
            starte = self.start_epoch - 1
        
        if args.load_cnet and args.fine_tune_cnet:
            self.lr_scheduler3 = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[max_lr*0.1, max_lr], # 微调 cnet 的学习率，以及其他模型正常学习率
                epochs=self.epoch,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.05,
                cycle_momentum=False,
                anneal_strategy='cos',
                last_epoch=max(steps_per_epoch*starte,-1),
            )
        else:
            self.lr_scheduler3 = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=self.epoch,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.05,
                cycle_momentum=False,
                anneal_strategy='cos',
                last_epoch=max(steps_per_epoch*starte,-1),
            )

        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
        #                 milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110,\
        #                 115, 120, 125, 130, 135, 140, 145, 160, 170, 180, 190, 200, 210], gamma=0.5, last_epoch=max((self.start_epoch-1),-1))
        # self.lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
        #                 milestones=[200], gamma=2, last_epoch=max((self.start_epoch-1),-1))

        self.device = device
        self.train_loss_history = []
        self.plt_train_epoch = []
        if time_stamp is None:
            self.time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
            out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
        else:
            self.time_stamp = time_stamp
        self.model_path = model_path
        
        self.attn_type=args.attn_type
        self.attn_splits_list=args.attn_splits_list
        self.corr_radius_list=args.corr_radius_list
        self.prop_radius_list=args.prop_radius_list
        self.num_reg_refine=args.num_reg_refine

        self.grad_clip = 1.0
        self.checkpoint_interval = 20 # 保存模型的间隔

        self.loss_per_epoch = 0
        self.loss_sum_per_epoch = 0
        self.iters = 0

        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}

        self.neptune_run = neptune_run

    def get_optimizer(self):
        params = list(self.model.named_parameters())
        param_group = [
            {'params':[p for n,p in params if 'featnet' in n],'lr':1e-5},
            {'params':[p for n,p in params if 'flownet' in n],'lr':1e-5},
            {'params':[p for n,p in params if 'scalenet' in n],'lr':1e-4},
        ]
        optimizer = torch.optim.Adam(param_group,lr=1e-5)
        return optimizer

    def train(self):
        out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)

        if is_main_process(self.parallel):
            for i, pg in enumerate(self.optimizer.param_groups):
                print(f"  param group {i} lr = {pg['lr']:.3e}")

        for epoch in range(self.start_epoch, self.epoch):
            if is_main_process(self.parallel):
                print('Epoch:', epoch)
            self.loss_per_epoch = 0
            self.loss_sum_per_epoch = 0
            self.iters = 0
            self.train_epoch(epoch)

            if (self.parallel and torch.distributed.get_rank() == 0) or (not self.parallel):

                if (epoch < self.epoch and epoch % self.checkpoint_interval == 0):
                    checkpoint = {
                        "net": self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        "epoch": epoch + 1
                    }
                    # # 用当前的时间戳为模型保存路径命名
                    temp_pth = os.path.join(out_dir, f'{epoch}y.pth.tar')
                    torch.save(checkpoint, temp_pth)
                if is_main_process(self.parallel):
                    print("Loss in epoch", epoch, ":", self.loss_per_epoch / max(1, self.iters))
                    for i, pg in enumerate(self.optimizer.param_groups):
                        print(f"  param group {i} lr = {pg['lr']:.3e}")

    
    def train_epoch(self, epoch):
        total_samples = len(self.train_loader.dataset)
        if self.parallel:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        
        epoch_loss = 0
        steps = 0

        out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)
        # save_index = 400 if not self.parallel else random.randint(int(4000/self.batch_size),int(8000/self.batch_size))
        # save_index = 400 if not self.parallel else random.randint(int(400/self.batch_size),int(1200/self.batch_size))
        save_index = 1000
        for i, data in enumerate(self.train_loader):

            (prev_surr_view_imgs_tensor,
             curr_surr_view_imgs_tensor,
             gt_scale_map_with_mask,
             gt_risk_score_map_with_mask,
             gt_depth_map_with_mask,
             sensor_meta) = data
            
            prev_surr_view_imgs_tensor = prev_surr_view_imgs_tensor.to(self.device)
            curr_surr_view_imgs_tensor = curr_surr_view_imgs_tensor.to(self.device)
            gt_scale_map_with_mask = gt_scale_map_with_mask.to(self.device)
            gt_risk_score_map_with_mask = gt_risk_score_map_with_mask.to(self.device)


            # with torch.no_grad():
            #     dummy = torch.randn(1,6,3,160,320).to(self.device)
            #     dummy_scale, dummy_risk = self.model(dummy, dummy, sensor_meta,
            #                         attn_type=self.attn_type,
            #                         attn_splits_list=self.attn_splits_list,
            #                         corr_radius_list=self.corr_radius_list,
            #                         prop_radius_list=self.prop_radius_list,
            #                         num_reg_refine=self.num_reg_refine,
            #                         testing=True,
            #                         )
            #     print(">>> dummy test scale:", dummy_scale.min().item(), dummy_scale.max().item())
            #     print(">>> dummy test risk:", dummy_risk.min().item(), dummy_risk.max().item())
            self.optimizer.zero_grad()
            # 在多卡模式下，从 self.model.module 调用 forward_with_loss，否则直接调用
            if hasattr(self.model, "module"):
                scale, risk_score, loss_s, loss_r = self.model.module.forward_with_loss(
                    prev_surr_view_imgs_tensor,
                    curr_surr_view_imgs_tensor,
                    sensor_meta,
                    gt_scale_map_with_mask=gt_scale_map_with_mask,
                    gt_risk_score_map_with_mask=gt_risk_score_map_with_mask,
                    attn_type=self.attn_type,
                    attn_splits_list=self.attn_splits_list,
                    corr_radius_list=self.corr_radius_list,
                    prop_radius_list=self.prop_radius_list,
                    num_reg_refine=self.num_reg_refine
                )
            else:
                scale, risk_score, loss_s, loss_r = self.model.forward_with_loss(
                    prev_surr_view_imgs_tensor,
                    curr_surr_view_imgs_tensor,
                    sensor_meta,
                    gt_scale_map_with_mask=gt_scale_map_with_mask,
                    gt_risk_score_map_with_mask=gt_risk_score_map_with_mask,
                    attn_type=self.attn_type,
                    attn_splits_list=self.attn_splits_list,
                    corr_radius_list=self.corr_radius_list,
                    prop_radius_list=self.prop_radius_list,
                    num_reg_refine=self.num_reg_refine
                )

            loss_last = None

            # —— 1. 定义各自参数关键字 —— 
            scale_keys = [
                "featnet_scale",
                "corrnet_scale",
                "conv_corr_scale",
                "scale_net"
            ]
            risk_keys = [k.replace("scale", "risk") for k in scale_keys]

            # —— 2. 收集参数 —— 
            # 如果用了 DataParallel/DistributedDataParallel 要取 self.model.module
            model_ref = getattr(self.model, "module", self.model)
            scale_params = [
                p for n, p in model_ref.named_parameters()
                if any(k in n for k in scale_keys)
            ]
            risk_params = [
                p for n, p in model_ref.named_parameters()
                if any(k in n for k in risk_keys)
            ]

            # —— 3. 只计算各自分支的梯度（保留图） —— 
            grads_s = torch.autograd.grad(loss_s, scale_params,
                                          retain_graph=True, allow_unused=True)
            grads_r = torch.autograd.grad(loss_r,  risk_params,
                                          retain_graph=True, allow_unused=True)
            
            # for p, g in zip(scale_params, grads_s):
            #     print(p.shape, "→ grad is", None if g is None else g.norm().item())
            
            grads_s = [g if g is not None else torch.zeros_like(p)
                       for g, p in zip(grads_s, scale_params)]
            grads_r = [g if g is not None else torch.zeros_like(p)
                       for g, p in zip(grads_r, risk_params)]

            norm_s = torch.sqrt(sum((g**2).sum() for g in grads_s))
            norm_r = torch.sqrt(sum((g**2).sum() for g in grads_r))

            # —— 4. 反比权重 —— 
            eps = 1e-6
            w_s = norm_r / (norm_s + norm_r + eps)
            w_r = norm_s / (norm_s + norm_r + eps)

            # —— 5. 合并 loss, 反向并更新 —— 
            loss = w_s * loss_s + w_r * loss_r

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.lr_scheduler3.step()
            epoch_loss += loss.item()
            steps += 1

            gt_scale = gt_scale_map_with_mask[:,0,:,:]
            gt_scale_valid_mask = gt_scale_map_with_mask[:,1,:,:]

            if type(scale) == list:
                scale = scale[-1]
            if i%int(save_index)==0 and is_main_process(self.parallel):

                # 可视化 prediction_scale 和 gt_scale
                gt_scale_np = gt_scale[:1].detach().squeeze(0).cpu().numpy()
                gt_scale_valid_mask_np = gt_scale_valid_mask[:1].squeeze(0).cpu().detach().bool()
                normalized_gt = visual_scale_map_range_image(gt_scale_np, gt_scale_valid_mask_np)

                scale_np = scale[0].detach().squeeze(0).cpu().numpy()
                pred_valid_mask = scale_np > 0
                normalized_pred = visual_scale_map_range_image(scale_np, pred_valid_mask)

                # 可视化 risk_score 和 gt_risk_score
                gt_risk_score = gt_risk_score_map_with_mask[:,0,:,:]
                gt_risk_score_np = gt_risk_score[:1].detach().squeeze(0).cpu().numpy()
                normalized_gt_risk_score = visual_risk_score_map_range_image(gt_risk_score_np)

                risk_score_np = risk_score[0].detach().squeeze(0).cpu().detach().numpy()
                normalized_pred_risk_score = visual_risk_score_map_range_image(risk_score_np)

                # 保存可视化结果
                plt.imsave(os.path.join(out_dir, f"{epoch}_{i}_pred.png"), -normalized_pred, cmap='seismic', vmin=-1, vmax=1)
                plt.imsave(os.path.join(out_dir, f"{epoch}_{i}_gt.png"), -normalized_gt, cmap='seismic', vmin=-1, vmax=1)
                plt.imsave(os.path.join(out_dir, f"{epoch}_{i}_pred_risk.png"), normalized_pred_risk_score, cmap='seismic', vmin=-1, vmax=1)
                plt.imsave(os.path.join(out_dir, f"{epoch}_{i}_gt_risk.png"), normalized_gt_risk_score, cmap='seismic', vmin=-1, vmax=1)

            # If an auxiliary loss (loss_last) is available, use it for reporting.
            if loss_last is not None:
                self.loss_per_epoch += loss.item()
                self.loss_sum_per_epoch += loss_last.item()
                if i % 10 == 0 and is_main_process(self.parallel):
                    print(
                        f"[{i * self.train_loader.batch_size:5}/{total_samples:5} "
                        f"({100 * i / len(self.train_loader):3.0f}%)]  "
                        f"Loss_now (aux): {loss_last.item():6.4f}    "
                        f"Loss: {loss.item():6.4f}    "
                    )
            else:
                self.loss_per_epoch += loss.item()
                if i % 10 == 0 and is_main_process(self.parallel):
                    print(
                        f"[{i * self.train_loader.batch_size:5}/{total_samples:5} "
                        f"({100 * i / len(self.train_loader):3.0f}%)]  "
                        f"Loss_now: {loss.item():6.4f}"
                    )

            self.iters += 1

            # 记录每个 batch 的 loss 和全局步数
            global_step = epoch * len(self.train_loader) + i
            if self.neptune_run is not None and is_main_process(self.parallel):
                self.neptune_run["train/batch_loss"].append(loss.item(), step=global_step)
                self.neptune_run["train/batch_loss_scale"].append(loss_s.item(), step=global_step)
                self.neptune_run["train/batch_loss_risk"].append(loss_r.item(), step=global_step)
                self.neptune_run["train/batch_loss_scale_uncertainty"].append(loss_s_term.item(), step=global_step)
                self.neptune_run["train/batch_loss_risk_uncertainty"].append(loss_r_term.item(), step=global_step)
            if self.neptune_run is not None and is_main_process(self.parallel):
                for i, pg in enumerate(self.optimizer.param_groups):
                    tag = f"train/batch_learning_rate_group_{i}"
                    self.neptune_run[tag].append(pg["lr"], step=global_step)
        
        avg_loss = epoch_loss / steps

        if self.neptune_run is not None and is_main_process(self.parallel):
            self.neptune_run["train/epoch_loss"].append(avg_loss, step=epoch)
            for i, pg in enumerate(self.optimizer.param_groups):
                tag = f"train/epoch_learning_rate_group_{i}"
                self.neptune_run[tag].append(pg["lr"], step=epoch)
        
    @torch.no_grad()
    def eval_epoch(self, eval_path='/mnt/pool/lcl/data/kitti/data_scene_flow/training/'):

        img0x, img1x, flow0 = self.eva_dataloader(eval_path)
        disp0 = [i.replace('flow_occ','disp_occ_0') for i in flow0]
        disp1 = [i.replace('flow_occ','disp_occ_1') for i in flow0]
        calib = [i.replace('flow_occ','calib')[:-7]+'.txt' for i in flow0]

        #w0,h0 = 1152,320
        w0,h0 = 960,288

        total_loss=0
        t=0

        self.model.eval()

        for i in range(len(img0x)):
            # print(flow0[i])
            flow, valid = readFlowKITTI(flow0[i])
            ints = load_calib_cam_to_cam(calib[i])
            fl = ints['K_cam2'][0,0]
            cx = ints['K_cam2'][0,2]
            cy = ints['K_cam2'][1,2]
            bl = ints['b20']-ints['b30']
            d1 = disparity_loader(disp0[i])
            d2 = disparity_loader(disp1[i])

            flow = np.ascontiguousarray(flow,dtype=np.float32)
            flow[np.isnan(flow)] = 1e6 # set to max
            valid = np.logical_and(np.logical_and(valid>0.99, d1>0), d2>0)
            d1[d1<=0] = 1e6
            d2[d2<=0] = 1e6

            shape = d1.shape
            mesh = np.meshgrid(range(shape[1]),range(shape[0]))
            xcoord = mesh[0].astype(float)
            ycoord = mesh[1].astype(float)
            
            # triangulation in two frames
            P0 = triangulation(d1, xcoord, ycoord, bl=bl, fl = fl, cx = cx, cy = cy)
            P1 = triangulation(d2, xcoord + flow[:,:,0], ycoord + flow[:,:,1], bl=bl, fl = fl, cx = cx, cy = cy)
            dis0 = P0[2]
            dis1 = P1[2]

            change_size =  dis0.reshape(shape).astype(np.float32)
            valid = np.logical_and(valid, change_size>0).astype(float)
            flow3d = (P1-P0)[:3].reshape((3,)+shape).transpose((1,2,0))
            change_size = np.concatenate((change_size[:,:,np.newaxis],flow3d),2)
            scale_gt = (np.array(change_size).astype(np.float32))

            gt_depth = scale_gt[...,0:1]
            # gt_depth[gt_depth<=0] = 1e6
            gt_f3d =  scale_gt[...,1:]
            gt_dchange = (1+gt_f3d[...,2:]/gt_depth)
            maskdc = (gt_dchange < 3) & (gt_dchange > 0.3) & np.expand_dims(valid.astype(bool),axis=2)


            file_1 = img0x[i]
            file_2 = img1x[i]

            image1 = Image.open(file_1).convert('RGB')
            image2 = Image.open(file_2).convert('RGB')
            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)          
            image1 = cv2.resize(image1, (w0,h0))
            image2 = cv2.resize(image2, (w0,h0))
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)


            padding_factor = 32
            inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                            int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

            ori_size = image1.shape[-2:]
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                        align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                        align_corners=True)
            scale, flow, _ = self.model(image1, image2,
                                    attn_type=self.attn_type,
                                    attn_splits_list=self.attn_splits_list,
                                    corr_radius_list=self.corr_radius_list,
                                    prop_radius_list=self.prop_radius_list,
                                    num_reg_refine=self.num_reg_refine,
                                    testing=True,
                                    )
            if type(scale) == list:
                scale = scale[-1]
            flow_pr = scale
            # resize back
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow_pr = flow_pr * ori_size[-1] / inference_size[-1]
            
            scale = flow_pr[0,0].detach().cpu().numpy()
            h,w,_ = gt_dchange.shape
            if scale.shape[0]!=h and scale.shape[1]!=w:
                scale = cv2.resize(scale, (w,h))
            scale = np.expand_dims(scale, axis=2)
            loss =  (np.abs((np.log(scale)-np.log(gt_dchange))))[maskdc].mean()
            total_loss += loss.mean()
            t += 1

        loss = total_loss / float(t)

        return loss

    def eva_dataloader(self, filepath):

        img_file_path = filepath
        left_fold  = 'image_2/'
        flow_noc   = 'flow_occ/'

        train_img = [img for img in os.listdir(img_file_path+left_fold) if img.find('_10') > -1]
        train_img = [i for i in train_img if int(i.split('_')[0])%5==0]
        train = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
        train = [i for i in train if int(i.split('_')[0])%5==0]

        l0_train  = [img_file_path+left_fold+img for img in train_img]
        l1_train = [img_file_path+left_fold+img.replace('_10','_11') for img in train_img]
        flow_train = [filepath+flow_noc+img for img in train]

        return sorted(l0_train), sorted(l1_train), sorted(flow_train)

    def average_gradients(self):  ##每个gpu上的梯度求平均
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data,op = dist.reduce_op.SUM)
                param.grad.data /= size
