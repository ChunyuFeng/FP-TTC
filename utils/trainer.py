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
from utils.dist import DistributedEvalSampler, is_main_process

from PIL import Image
from .draw import visual_scale_map_range_image, visual_risk_score_map_range_image
from dataloader.load import load_calib_cam_to_cam, readFlowKITTI, disparity_loader, triangulation


def _normalize_to_uint8(array):
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value > min_value:
        array = (array - min_value) / (max_value - min_value)
    else:
        array = np.zeros_like(array, dtype=np.float32)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def _feature_pca_rgb(feature_tensor):
    feature = feature_tensor.detach().float().cpu().numpy()
    c, h, w = feature.shape
    flat = feature.reshape(c, -1).T
    flat = flat - flat.mean(axis=0, keepdims=True)

    if flat.shape[0] == 0 or flat.shape[1] == 0 or np.allclose(flat, 0.0):
        return np.zeros((h, w, 3), dtype=np.uint8)

    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        basis = vt[:3].T
        proj = flat @ basis
    except np.linalg.LinAlgError:
        proj = flat[:, :min(3, flat.shape[1])]

    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])), mode='constant')

    proj = proj.reshape(h, w, 3)
    channels = [_normalize_to_uint8(proj[..., idx]) for idx in range(3)]
    return np.stack(channels, axis=-1)


def _feature_norm_rgb(feature_tensor, cmap_name='viridis'):
    feature = feature_tensor.detach().float().cpu().numpy()
    norm_map = np.linalg.norm(feature, axis=0)
    norm_map = _normalize_to_uint8(norm_map).astype(np.float32) / 255.0
    colored = plt.get_cmap(cmap_name)(norm_map)[..., :3]
    return np.clip(colored * 255.0, 0, 255).astype(np.uint8)


def _rgb_tensor_to_uint8(image_tensor):
    image = image_tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0, 255).astype(np.uint8)


def _make_horizontal_montage(image_list):
    return np.concatenate(image_list, axis=1)


class TTCTrainer(object):
    def __init__(self, model, dataset, optimizer, args, start_epoch, device,
                val_dataset=None,
                parallel=False, time_stamp=None, 
                neptune_run=None, scale_only=False):
        self.model = model
        self.parallel = parallel
        self.batch_size = args.batch_size
        self.train_sampler = None
        self.val_sampler = None
        self.scale_only = scale_only
        if not self.parallel:
            self.train_loader = DataLoader(dataset, 
                                           batch_size  = args.batch_size, 
                                           shuffle     = True, 
                                           num_workers = args.num_workers, 
                                           drop_last   = True, 
                                           pin_memory  = True)
        else:
            self.train_sampler = DistributedSampler(dataset)
            self.train_loader = DataLoader(dataset,
                                           batch_size  = args.batch_size,
                                           sampler     = self.train_sampler,
                                           shuffle     = False, 
                                           pin_memory  = True, 
                                           num_workers = args.num_workers)
        self.val_loader = None
        if val_dataset is not None:
            if not self.parallel:
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    drop_last=False,
                    pin_memory=True,
                )
            else:
                self.val_sampler = DistributedEvalSampler(val_dataset)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    sampler=self.val_sampler,
                    shuffle=False,
                    num_workers=args.num_workers,
                    drop_last=False,
                    pin_memory=True,
                )

        if self.scale_only:
            self.epoch = args.scale_epochs
        else:
            self.epoch = args.risk_epochs
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        
        steps_per_epoch = int(len(self.train_loader))
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        starte = -1
        if self.start_epoch>0:
            starte = self.start_epoch - 1
        
        # 构建每个 param group 的 max_lr 列表（支持分层学习率）
        max_lr_list = [pg.get('max_lr', args.lr) for pg in self.optimizer.param_groups]
        pct_start = getattr(args, 'pct_start', 0.05)

        # 若 grad_accum > 1，每 " grad_accum_steps 次 forward" 才算 1 步 scheduler
        effective_steps = steps_per_epoch // self.grad_accum_steps
        effective_steps = max(effective_steps, 1)

        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr_list,
            epochs=self.epoch,
            steps_per_epoch=effective_steps,
            pct_start=pct_start,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=max(effective_steps*starte,-1),
        )

        self.device = device
        self.train_loss_history = []
        self.plt_train_epoch = []
        if time_stamp is None:
            self.time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
        else:
            self.time_stamp = time_stamp  
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

        self.neptune_run = neptune_run
        self.val_freq = max(1, args.val_freq)
        self.save_best = args.save_best
        self.best_val_loss = float('inf')
        self.out_dir = "./log/%s_surround_ttc"%(self.time_stamp)
        self.best_ckpt_name = 'best_scale.pth.tar' if self.scale_only else 'best_risk.pth.tar'
        self.debug_visualize_projection = bool(getattr(args, 'debug_visualize_projection', False))
        self.debug_visualize_dir = getattr(args, 'debug_visualize_dir', None)
        self._projection_debug_saved = False


    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        if is_main_process():
            for i, pg in enumerate(self.optimizer.param_groups):
                print(f"  param group {i} lr = {pg['lr']:.3e}")

        for epoch in range(self.start_epoch, self.epoch):
            if is_main_process():
                print('Epoch:', epoch)
            self.loss_per_epoch = 0
            self.loss_sum_per_epoch = 0
            self.iters = 0
            train_loss = self.train_epoch(epoch)
            val_loss = None
            if self.val_loader is not None and epoch % self.val_freq == 0:
                val_loss = self.validate_epoch(epoch)

            if is_main_process():
                if (epoch < self.epoch and epoch % self.checkpoint_interval == 0):
                    if self.scale_only:
                        temp_pth = os.path.join(self.out_dir, f'{epoch}_scale.pth.tar')
                    else:
                        temp_pth = os.path.join(self.out_dir, f'{epoch}.pth.tar')
                    self._save_checkpoint(temp_pth, epoch)

                if self.save_best and val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(self.out_dir, self.best_ckpt_name)
                    self._save_checkpoint(best_path, epoch)
                    print(f"Best checkpoint updated: {best_path} (val_loss={val_loss:.6f})")

                print("Train loss in epoch", epoch, ":", train_loss)
                if val_loss is not None:
                    print("Val loss in epoch", epoch, ":", val_loss)
                    print("Best val loss so far:", self.best_val_loss)
                for i, pg in enumerate(self.optimizer.param_groups):
                    print(f"  param group {i} lr = {pg['lr']:.3e}")

    
    def train_epoch(self, epoch):
        total_samples = len(self.train_loader.dataset)
        if self.parallel:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        
        epoch_loss = 0
        steps = 0

        save_index = 1000
        for i, data in enumerate(self.train_loader):
            
            (prev_surr_view_imgs_tensor,
             curr_surr_view_imgs_tensor,
             prev_surr_view_depths_tensor,
             curr_surr_view_depths_tensor,
             proj_pix_prev_tensor,
             proj_pix_curr_tensor,
             gt_scale_map_with_mask,
             gt_risk_score_map_with_mask) = data
            
            prev_surr_view_imgs_tensor   = prev_surr_view_imgs_tensor.to(self.device)
            curr_surr_view_imgs_tensor   = curr_surr_view_imgs_tensor.to(self.device)
            prev_surr_view_depths_tensor = prev_surr_view_depths_tensor.to(self.device)
            curr_surr_view_depths_tensor = curr_surr_view_depths_tensor.to(self.device)
            proj_pix_prev_tensor         = proj_pix_prev_tensor.to(self.device)
            proj_pix_curr_tensor         = proj_pix_curr_tensor.to(self.device)
            gt_scale_map_with_mask       = gt_scale_map_with_mask.to(self.device)
            gt_risk_score_map_with_mask  = gt_risk_score_map_with_mask.to(self.device)
            # affine_matrix                = affine_matrix.to(self.device)

            capture_projection_debug = (
                self.debug_visualize_projection
                and (not self._projection_debug_saved)
                and epoch == 0
                and i == 0
                and is_main_process()
            )

            if i % self.grad_accum_steps == 0:
                self.optimizer.zero_grad()
            # 在多卡模式下，从 self.model.module 调用 forward_with_loss，否则直接调用
            if hasattr(self.model, "module"):
                model_outputs = self.model.module.forward_with_loss(
                    img_prev                      = prev_surr_view_imgs_tensor,
                    img_curr                      = curr_surr_view_imgs_tensor,
                    depth_prev                    = prev_surr_view_depths_tensor,
                    depth_curr                    = curr_surr_view_depths_tensor,
                    proj_pix_prev                 = proj_pix_prev_tensor,
                    proj_pix_curr                 = proj_pix_curr_tensor,
                    gt_scale_map_with_mask        = gt_scale_map_with_mask,
                    gt_risk_score_map_with_mask   = gt_risk_score_map_with_mask,
                    attn_type                     = self.attn_type,
                    attn_splits_list              = self.attn_splits_list,
                    corr_radius_list              = self.corr_radius_list,
                    prop_radius_list              = self.prop_radius_list,
                    num_reg_refine                = self.num_reg_refine,
                    scale_only                    = self.scale_only,
                    return_debug                  = capture_projection_debug,
                )
            else:
                model_outputs = self.model.forward_with_loss(
                    img_prev                      = prev_surr_view_imgs_tensor,
                    img_curr                      = curr_surr_view_imgs_tensor,
                    depth_prev                    = prev_surr_view_depths_tensor,
                    depth_curr                    = curr_surr_view_depths_tensor,
                    proj_pix_prev                 = proj_pix_prev_tensor,
                    proj_pix_curr                 = proj_pix_curr_tensor,
                    gt_scale_map_with_mask        = gt_scale_map_with_mask,
                    gt_risk_score_map_with_mask   = gt_risk_score_map_with_mask,
                    attn_type                     = self.attn_type,
                    attn_splits_list              = self.attn_splits_list,
                    corr_radius_list              = self.corr_radius_list,
                    prop_radius_list              = self.prop_radius_list,
                    num_reg_refine                = self.num_reg_refine,
                    scale_only                    = self.scale_only,
                    return_debug                  = capture_projection_debug,
                )

            debug_dict = None
            if capture_projection_debug:
                scale, risk_score, loss_s, loss_r, debug_dict = model_outputs
            else:
                scale, risk_score, loss_s, loss_r = model_outputs
            
            loss = loss_s if self.scale_only else loss_r

            # 支持 gradient accumulation
            if self.grad_accum_steps > 1:
                loss = loss / self.grad_accum_steps

            loss.backward()

            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.lr_scheduler.step()
            epoch_loss += loss.item()
            steps += 1

            if capture_projection_debug and debug_dict is not None:
                self._save_projection_debug(
                    prev_surr_view_imgs_tensor[:1],
                    curr_surr_view_imgs_tensor[:1],
                    debug_dict,
                    batch_index=i,
                )
                self._projection_debug_saved = True

            ### 可视化结果
            
            if type(scale) == list:
                scale = scale[-1]
            if i%int(save_index)==0 and is_main_process():

                if self.scale_only:
                    gt_scale = gt_scale_map_with_mask[:,0,:,:]
                    gt_scale_valid_mask = gt_scale_map_with_mask[:,1,:,:]

                    # 可视化 prediction_scale 和 gt_scale
                    gt_scale_np = gt_scale[:1].detach().squeeze(0).cpu().numpy()
                    gt_scale_valid_mask_np = gt_scale_valid_mask[:1].squeeze(0).cpu().detach().bool()
                    normalized_gt = visual_scale_map_range_image(gt_scale_np, gt_scale_valid_mask_np)

                    scale_np = scale[0].detach().squeeze(0).cpu().numpy()
                    pred_scale_valid_mask = scale_np > 0
                    normalized_pred = visual_scale_map_range_image(scale_np, pred_scale_valid_mask)

                    plt.imsave(os.path.join(self.out_dir, f"{epoch}_{i}_pred.png"), -normalized_pred, cmap='seismic', vmin=-1, vmax=1)
                    plt.imsave(os.path.join(self.out_dir, f"{epoch}_{i}_gt.png"), -normalized_gt, cmap='seismic', vmin=-1, vmax=1)

                else: 
                    gt_risk_score = gt_risk_score_map_with_mask[:,0,:,:]
                    gt_risk_score_valid_mask = gt_risk_score_map_with_mask[:,1,:,:]
                    # 可视化 risk_score 和 gt_risk_score
                    gt_risk_score_np = gt_risk_score[:1].detach().squeeze(0).cpu().numpy()
                    gt_risk_score_valid_mask_np = gt_risk_score_valid_mask[:1].squeeze(0).cpu().detach().bool()
                    normalized_gt_risk_score = visual_risk_score_map_range_image(gt_risk_score_np, gt_risk_score_valid_mask_np)

                    risk_score_np = risk_score[0].detach().squeeze(0).cpu().detach().numpy()
                    normalized_pred_risk_score = visual_risk_score_map_range_image(risk_score_np, None)

                    # 保存可视化结果
                    plt.imsave(os.path.join(self.out_dir, f"{epoch}_{i}_pred_risk.png"), -normalized_pred_risk_score, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)
                    plt.imsave(os.path.join(self.out_dir, f"{epoch}_{i}_gt_risk.png"), -normalized_gt_risk_score, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)

            loss_last = None
            # If an auxiliary loss (loss_last) is available, use it for reporting.
            if loss_last is not None:
                self.loss_per_epoch += loss.item()
                self.loss_sum_per_epoch += loss_last.item()
                if i % 10 == 0 and is_main_process():
                    print(
                        f"[{i * self.train_loader.batch_size:5}/{total_samples:5} "
                        f"({100 * i / len(self.train_loader):3.0f}%)]  "
                        f"Loss_now (aux): {loss_last.item():6.4f}    "
                        f"Loss: {loss.item():6.4f}    "
                    )
            else:
                self.loss_per_epoch += loss.item()
                if i % 10 == 0 and is_main_process():
                    print(
                        f"[{i * self.train_loader.batch_size:5}/{total_samples:5} "
                        f"({100 * i / len(self.train_loader):3.0f}%)]  "
                        f"Loss_now: {loss.item():6.4f}"
                    )

            self.iters += 1

            # 记录每个 batch 的 loss 和全局步数
            global_step = epoch * len(self.train_loader) + i
            if self.neptune_run is not None and is_main_process():
                if self.scale_only:
                    self.neptune_run["train/batch_loss_scale"].append(loss_s.item(), step=global_step)
                else:
                    self.neptune_run["train/batch_loss_risk"].append(loss_r.item(), step=global_step)
                for group_idx, pg in enumerate(self.optimizer.param_groups):
                    tag = f"train/batch_learning_rate_group_{group_idx}"
                    self.neptune_run[tag].append(pg["lr"], step=global_step)
        
        avg_loss = epoch_loss / steps

        if self.neptune_run is not None and is_main_process():
            self.neptune_run["train/epoch_loss"].append(avg_loss, step=epoch)
            for group_idx, pg in enumerate(self.optimizer.param_groups):
                tag = f"train/epoch_learning_rate_group_{group_idx}"
                self.neptune_run[tag].append(pg["lr"], step=epoch)

        return avg_loss

    def _save_projection_debug(self, prev_imgs, curr_imgs, debug_dict, batch_index):
        output_dir = self.debug_visualize_dir
        if output_dir is None:
            output_dir = os.path.join(self.out_dir, 'debug_projection')
        os.makedirs(output_dir, exist_ok=True)

        prev_rgb = [_rgb_tensor_to_uint8(prev_imgs[0, cam_idx]) for cam_idx in range(prev_imgs.shape[1])]
        curr_rgb = [_rgb_tensor_to_uint8(curr_imgs[0, cam_idx]) for cam_idx in range(curr_imgs.shape[1])]
        Image.fromarray(_make_horizontal_montage(prev_rgb)).save(
            os.path.join(output_dir, f'train_batch{batch_index}_prev_rgb.png')
        )
        Image.fromarray(_make_horizontal_montage(curr_rgb)).save(
            os.path.join(output_dir, f'train_batch{batch_index}_curr_rgb.png')
        )

        for frame_name, feat_key, range_key in [
            ('prev', 'multi_level_feats_prev', 'multi_level_ranges_prev'),
            ('curr', 'multi_level_feats_curr', 'multi_level_ranges_curr'),
        ]:
            for scale_idx, feat_level in enumerate(debug_dict[feat_key]):
                view_pca = []
                view_norm = []
                for cam_idx in range(feat_level.shape[1]):
                    view_feature = feat_level[0, cam_idx]
                    view_pca.append(_feature_pca_rgb(view_feature))
                    view_norm.append(_feature_norm_rgb(view_feature))

                Image.fromarray(_make_horizontal_montage(view_pca)).save(
                    os.path.join(output_dir, f'{frame_name}_scale{scale_idx}_views_pca.png')
                )
                Image.fromarray(_make_horizontal_montage(view_norm)).save(
                    os.path.join(output_dir, f'{frame_name}_scale{scale_idx}_views_norm.png')
                )

                range_feature = debug_dict[range_key][scale_idx][0]
                Image.fromarray(_feature_pca_rgb(range_feature)).save(
                    os.path.join(output_dir, f'{frame_name}_scale{scale_idx}_range_pca.png')
                )
                Image.fromarray(_feature_norm_rgb(range_feature)).save(
                    os.path.join(output_dir, f'{frame_name}_scale{scale_idx}_range_norm.png')
                )

    @torch.no_grad()
    def validate_epoch(self, epoch):
        if self.val_loader is None:
            return None

        if is_main_process():
            print(f"Validation epoch {epoch} ...")

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for data in self.val_loader:
            (prev_surr_view_imgs_tensor,
             curr_surr_view_imgs_tensor,
             prev_surr_view_depths_tensor,
             curr_surr_view_depths_tensor,
             proj_pix_prev_tensor,
             proj_pix_curr_tensor,
             gt_scale_map_with_mask,
             gt_risk_score_map_with_mask) = data

            prev_surr_view_imgs_tensor   = prev_surr_view_imgs_tensor.to(self.device)
            curr_surr_view_imgs_tensor   = curr_surr_view_imgs_tensor.to(self.device)
            prev_surr_view_depths_tensor = prev_surr_view_depths_tensor.to(self.device)
            curr_surr_view_depths_tensor = curr_surr_view_depths_tensor.to(self.device)
            proj_pix_prev_tensor         = proj_pix_prev_tensor.to(self.device)
            proj_pix_curr_tensor         = proj_pix_curr_tensor.to(self.device)
            gt_scale_map_with_mask       = gt_scale_map_with_mask.to(self.device)
            gt_risk_score_map_with_mask  = gt_risk_score_map_with_mask.to(self.device)

            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            _, _, loss_s, loss_r = model_ref.forward_with_loss(
                img_prev                    = prev_surr_view_imgs_tensor,
                img_curr                    = curr_surr_view_imgs_tensor,
                depth_prev                  = prev_surr_view_depths_tensor,
                depth_curr                  = curr_surr_view_depths_tensor,
                proj_pix_prev               = proj_pix_prev_tensor,
                proj_pix_curr               = proj_pix_curr_tensor,
                gt_scale_map_with_mask      = gt_scale_map_with_mask,
                gt_risk_score_map_with_mask = gt_risk_score_map_with_mask,
                attn_type                   = self.attn_type,
                attn_splits_list            = self.attn_splits_list,
                corr_radius_list            = self.corr_radius_list,
                prop_radius_list            = self.prop_radius_list,
                num_reg_refine              = self.num_reg_refine,
                scale_only                  = self.scale_only
            )

            loss = loss_s if self.scale_only else loss_r
            batch_samples = prev_surr_view_imgs_tensor.shape[0]
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

        if self.parallel and dist.is_available() and dist.is_initialized():
            stats = torch.tensor([total_loss, total_samples], dtype=torch.float64, device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss = float(stats[0].item())
            total_samples = int(stats[1].item())

        avg_loss = total_loss / max(1, total_samples)
        self.model.train()

        if self.neptune_run is not None and is_main_process():
            tag = "val/epoch_loss_scale" if self.scale_only else "val/epoch_loss_risk"
            self.neptune_run[tag].append(avg_loss, step=epoch)

        return avg_loss

    def _save_checkpoint(self, path, epoch):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": epoch + 1
        }
        torch.save(checkpoint, path)
    
    def _weighted_loss(self, keys_s, keys_r, loss_s, loss_r, model_ref):
        """
        根据两个分支的梯度范数，动态计算权重 w_s, w_r，
        并返回合并后的 loss = w_s * loss_s + w_r * loss_r。

        Args:
            keys_s (list): scale 分支参数的关键字列表
            keys_r (list): risk 分支参数的关键字列表
            loss_s (Tensor): scale 分支的 loss
            loss_r (Tensor): risk 分支的 loss
            model_ref (nn.Module): 用于提取参数的模型引用
        Returns:
            loss (Tensor): 加权合并后的 loss
            w_s (Tensor), w_r (Tensor): 分别作用于 loss_s, loss_r 的权重
        """

        # 收集参数
        scale_params = []
        risk_params  = []
        for name, param in model_ref.named_parameters():
            # 如果参数名中包含任意一个 scale_keys，就归到 scale_params
            if any(key in name for key in keys_s):
                scale_params.append(param)
            # 否则如果包含任意一个 risk_keys，就归到 risk_params
            elif any(key in name for key in keys_r):
                risk_params.append(param)

        # 只计算各自分支的梯度（保留图）
        raw_grads_s = torch.autograd.grad(loss_s, 
                                          scale_params,
                                          retain_graph=True, 
                                          allow_unused=True)
        raw_grads_r = torch.autograd.grad(loss_r, 
                                          risk_params,
                                          retain_graph=True, 
                                          allow_unused=True)

        # 对于 None 的梯度，补零
        grads_s = []
        for grad, param in zip(raw_grads_s, scale_params):
            if grad is None:
                # 如果 grad=None，说明这个参数对 loss_s 没有影响，直接补 0 Tensor
                grads_s.append(torch.zeros_like(param))
            else:
                grads_s.append(grad)

        grads_r = []
        for grad, param in zip(raw_grads_r, risk_params):
            if grad is None:
                grads_r.append(torch.zeros_like(param))
            else:
                grads_r.append(grad)

        # 计算范数
        norm_s = torch.sqrt(sum((g**2).sum() for g in grads_s))
        norm_r = torch.sqrt(sum((g**2).sum() for g in grads_r))

        # 防止除零
        eps = 1e-6
        w_s = norm_r / (norm_s + norm_r + eps)
        w_r = norm_s / (norm_s + norm_r + eps)

        # 合并 loss
        loss = w_s * loss_s + w_r * loss_r
        return loss, w_s, w_r
    
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
