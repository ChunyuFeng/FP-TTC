# utils/trainer.py
import os
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils.dist import is_main_process
from PIL import Image
from .draw import visual_scale_map_range_image, visual_risk_score_map_range_image

class TTCTrainer(object):
    def __init__(self,
                 model,
                 train_loader,
                 train_sampler,
                 optimizer,
                 lr_scheduler,
                 args,
                 start_epoch,
                 init_global_step,
                 device,
                 parallel=False,
                 time_stamp=None,
                 neptune_run=None,
                 scale_only=False):
        self.model = model
        self.parallel = parallel
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scale_only = scale_only
        self.device = device

        # 训练总 epoch
        self.epoch = args.scale_epochs if self.scale_only else args.risk_epochs

        # 起始 epoch & 步数（由 train.py 决定）
        self.start_epoch = int(start_epoch)
        self.global_step = int(init_global_step)

        self.train_loss_history = []
        self.plt_train_epoch = []
        self.time_stamp = time_stamp or datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")

        self.attn_type        = args.attn_type
        self.attn_splits_list = args.attn_splits_list
        self.corr_radius_list = args.corr_radius_list
        self.prop_radius_list = args.prop_radius_list
        self.num_reg_refine   = args.num_reg_refine

        # 不做梯度裁剪（应你的要求）
        self.grad_clip = None
        self.checkpoint_interval = 20  # 保存模型的间隔（按 epoch）

        self.loss_per_epoch = 0.0
        self.loss_sum_per_epoch = 0.0
        self.iters = 0

        self.neptune_run = neptune_run

        if is_main_process():
            print(f"[LR] steps_per_epoch = {len(self.train_loader)}, "
                  f"start_epoch = {self.start_epoch}, "
                  f"global_step(init) = {self.global_step}")

    def train(self):
        out_dir = f"./log/{self.time_stamp}_surround_ttc"
        os.makedirs(out_dir, exist_ok=True)

        if is_main_process():
            for i, pg in enumerate(self.optimizer.param_groups):
                print(f"  param group {i} lr(init) = {pg['lr']:.3e}")

        for epoch in range(self.start_epoch, self.epoch):
            if is_main_process():
                print('Epoch:', epoch)

            self.loss_per_epoch = 0.0
            self.loss_sum_per_epoch = 0.0
            self.iters = 0
            self.train_epoch(epoch)

            if is_main_process():
                # 保存 ckpt（含 scheduler/global_step，便于 full_resume）
                if (epoch < self.epoch and epoch % self.checkpoint_interval == 0):
                    checkpoint = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.lr_scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "global_step": self.global_step,
                    }
                    temp_pth = os.path.join(out_dir, f'{epoch}_{"scale" if self.scale_only else "risk"}.pth.tar')
                    torch.save(checkpoint, temp_pth)

                print("Loss in epoch", epoch, ":", self.loss_per_epoch / max(1, self.iters))
                for i, pg in enumerate(self.optimizer.param_groups):
                    print(f"  param group {i} lr(end) = {pg['lr']:.3e}")

    def train_epoch(self, epoch):
        total_samples = len(self.train_loader.dataset)
        if self.parallel and isinstance(self.train_sampler, DistributedSampler):
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        epoch_loss = 0.0
        steps = 0

        out_dir = f"./log/{self.time_stamp}_surround_ttc"
        save_index = 1000

        for i, data in enumerate(self.train_loader):
            (prev_surr_view_imgs_tensor,
             curr_surr_view_imgs_tensor,
             prev_surr_view_depths_tensor,
             curr_surr_view_depths_tensor,
             proj_pix_prev_tensor,
             proj_pix_curr_tensor,
             proj_xy_prev_tensor,
             proj_xy_curr_tensor,
             gt_scale_map_with_mask,
             gt_risk_score_map_with_mask) = data
            
            prev_surr_view_imgs_tensor   = prev_surr_view_imgs_tensor.to(self.device, non_blocking=True)
            curr_surr_view_imgs_tensor   = curr_surr_view_imgs_tensor.to(self.device, non_blocking=True)
            prev_surr_view_depths_tensor = prev_surr_view_depths_tensor.to(self.device, non_blocking=True)
            curr_surr_view_depths_tensor = curr_surr_view_depths_tensor.to(self.device, non_blocking=True)
            proj_pix_prev_tensor         = proj_pix_prev_tensor.to(self.device, non_blocking=True)
            proj_pix_curr_tensor         = proj_pix_curr_tensor.to(self.device, non_blocking=True)
            proj_xy_prev_tensor          = proj_xy_prev_tensor.to(self.device, non_blocking=True)
            proj_xy_curr_tensor          = proj_xy_curr_tensor.to(self.device, non_blocking=True)
            gt_scale_map_with_mask       = gt_scale_map_with_mask.to(self.device, non_blocking=True)
            gt_risk_score_map_with_mask  = gt_risk_score_map_with_mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if hasattr(self.model, "module"):
                scale, risk_score, loss_s, loss_r = self.model.module.forward_with_loss(
                    img_prev                      = prev_surr_view_imgs_tensor,
                    img_curr                      = curr_surr_view_imgs_tensor,
                    depth_prev                    = prev_surr_view_depths_tensor,
                    depth_curr                    = curr_surr_view_depths_tensor,
                    proj_pix_prev                 = proj_pix_prev_tensor,
                    proj_pix_curr                 = proj_pix_curr_tensor,
                    proj_xy_prev                  = proj_xy_prev_tensor,
                    proj_xy_curr                  = proj_xy_curr_tensor,  
                    gt_scale_map_with_mask        = gt_scale_map_with_mask,
                    gt_risk_score_map_with_mask   = gt_risk_score_map_with_mask,
                    attn_type                     = self.attn_type,
                    attn_splits_list              = self.attn_splits_list,
                    corr_radius_list              = self.corr_radius_list,
                    prop_radius_list              = self.prop_radius_list,
                    num_reg_refine                = self.num_reg_refine,
                    scale_only                    = self.scale_only
                )
            else:
                scale, risk_score, loss_s, loss_r = self.model.forward_with_loss(
                    img_prev                      = prev_surr_view_imgs_tensor,
                    img_curr                      = curr_surr_view_imgs_tensor,
                    depth_prev                    = prev_surr_view_depths_tensor,
                    depth_curr                    = curr_surr_view_depths_tensor,
                    proj_pix_prev                 = proj_pix_prev_tensor,
                    proj_pix_curr                 = proj_pix_curr_tensor,
                    proj_xy_prev                  = proj_xy_prev_tensor,
                    proj_xy_curr                  = proj_xy_curr_tensor,
                    gt_scale_map_with_mask        = gt_scale_map_with_mask,
                    gt_risk_score_map_with_mask   = gt_risk_score_map_with_mask,
                    attn_type                     = self.attn_type,
                    attn_splits_list              = self.attn_splits_list,
                    corr_radius_list              = self.corr_radius_list,
                    prop_radius_list              = self.prop_radius_list,
                    num_reg_refine                = self.num_reg_refine,
                    scale_only                    = self.scale_only
                )
            
            loss = loss_s if self.scale_only else loss_r
            loss.backward()

            # 不做梯度裁剪，直接 step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.global_step += 1

            epoch_loss += float(loss.item())
            steps += 1

            # === 可视化保存 ===
            if isinstance(scale, list):
                scale = scale[-1]
            if isinstance(risk_score, list):
                risk_score = risk_score[-1]

            if i % int(save_index) == 0 and is_main_process():
                if self.scale_only:
                    gt_scale = gt_scale_map_with_mask[:,0,:,:]
                    gt_scale_valid_mask = gt_scale_map_with_mask[:,1,:,:]

                    gt_scale_np = gt_scale[:1].detach().squeeze(0).cpu().numpy()
                    gt_scale_valid_mask_np = gt_scale_valid_mask[:1].squeeze(0).cpu().detach().bool()
                    normalized_gt = visual_scale_map_range_image(gt_scale_np, gt_scale_valid_mask_np)

                    scale_np = scale[0].detach().squeeze(0).cpu().numpy()
                    pred_scale_valid_mask = scale_np > 0
                    normalized_pred = visual_scale_map_range_image(scale_np, pred_scale_valid_mask)

                    plt.imsave(os.path.join(out_dir, f"{epoch}_{i}_pred.png"), -normalized_pred, cmap='seismic', vmin=-1, vmax=1)
                    plt.imsave(os.path.join(out_dir, f"{epoch}_{i}_gt.png"),   -normalized_gt,   cmap='seismic', vmin=-1, vmax=1)
                else:
                    gt_risk_score = gt_risk_score_map_with_mask[:,0,:,:]
                    gt_risk_score_valid_mask = gt_risk_score_map_with_mask[:,1,:,:]

                    gt_risk_score_np = gt_risk_score[:1].detach().squeeze(0).cpu().numpy()
                    gt_risk_score_valid_mask_np = gt_risk_score_valid_mask[:1].squeeze(0).cpu().detach().bool()
                    normalized_gt_risk_score = visual_risk_score_map_range_image(gt_risk_score_np, gt_risk_score_valid_mask_np)

                    risk_score_np = risk_score[0].detach().squeeze(0).cpu().detach().numpy()
                    normalized_pred_risk_score = visual_risk_score_map_range_image(risk_score_np, None)

                    plt.imsave(os.path.join(out_dir, f"{epoch}_{i}_pred_risk.png"), -normalized_pred_risk_score, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)
                    plt.imsave(os.path.join(out_dir, f"{epoch}_{i}_gt_risk.png"),   -normalized_gt_risk_score,   cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)

            # 打印 / 记录
            self.loss_per_epoch += float(loss.item())
            if (i % 10 == 0) and is_main_process():
                print(
                    f"[{i * self.train_loader.batch_size:5}/{total_samples:5} "
                    f"({100 * i / len(self.train_loader):3.0f}%)]  "
                    f"Loss_now: {loss.item():6.4f}   "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.3e}   "
                    f"gstep: {self.global_step}"
                )

            self.iters += 1

            # Neptune 记录
            if self.neptune_run is not None and is_main_process():
                if self.scale_only:
                    self.neptune_run["train/batch_loss_scale"].append(loss_s.item(),
                                                                      step=self.global_step)
                else:
                    self.neptune_run["train/batch_loss_risk"].append(loss_r.item(),
                                                                     step=self.global_step)
                for group_idx, pg in enumerate(self.optimizer.param_groups):
                    tag = f"train/batch_learning_rate_group_{group_idx}"
                    self.neptune_run[tag].append(pg["lr"], step=self.global_step)
        
        avg_loss = epoch_loss / max(steps, 1)

        if self.neptune_run is not None and is_main_process():
            self.neptune_run["train/epoch_loss"].append(avg_loss, step=epoch)
            for group_idx, pg in enumerate(self.optimizer.param_groups):
                tag = f"train/epoch_learning_rate_group_{group_idx}"
                self.neptune_run[tag].append(pg["lr"], step=epoch)

    def _weighted_loss(self, keys_s, keys_r, loss_s, loss_r, model_ref):
        scale_params = []
        risk_params  = []
        for name, param in model_ref.named_parameters():
            if any(key in name for key in keys_s):
                scale_params.append(param)
            elif any(key in name for key in keys_r):
                risk_params.append(param)

        raw_grads_s = torch.autograd.grad(loss_s, scale_params,
                                          retain_graph=True, allow_unused=True)
        raw_grads_r = torch.autograd.grad(loss_r, risk_params,
                                          retain_graph=True, allow_unused=True)

        grads_s = [torch.zeros_like(p) if g is None else g for g, p in zip(raw_grads_s, scale_params)]
        grads_r = [torch.zeros_like(p) if g is None else g for g, p in zip(raw_grads_r, risk_params)]

        norm_s = torch.sqrt(sum((g**2).sum() for g in grads_s))
        norm_r = torch.sqrt(sum((g**2).sum() for g in grads_r))

        eps = 1e-6
        w_s = norm_r / (norm_s + norm_r + eps)
        w_r = norm_s / (norm_s + norm_r + eps)

        loss = w_s * loss_s + w_r * loss_r
        return loss, w_s, w_r
