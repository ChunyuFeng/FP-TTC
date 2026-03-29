import os
import csv
import json

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
from utils.dist import is_main_process

from PIL import Image
from .draw import visual_scale_map_range_image, visual_risk_score_map_range_image
from dataloader.load import load_calib_cam_to_cam, readFlowKITTI, disparity_loader, triangulation

class TTCTrainer(object):
    def __init__(self, model, dataset, optimizer, args, start_epoch, device,
                parallel=False, time_stamp=None, 
                neptune_run=None, scale_only=False,
                no_depth=False, use_teacher_distill=False,
                lambda_feat_distill=1.0, lambda_corr_distill=0.5,
                distill_end_pct=0.7, student_tail_epochs=0,
                loss_weight_alpha=0.0, edge_loss_weight=0.0,
                use_internal_depth_guidance=False, depth_loss_weight=0.5,
                depth_selection_mode='hard_topk', bootstrap_topk=4, attn_topk=8,
                bootstrap_prior_scale=2.0, attn_prior_scale=2.0, depth_prior_eps=1e-6):
        self.model = model
        self.parallel = parallel
        self.batch_size = args.batch_size
        self.train_sampler = None
        self.scale_only = scale_only
        self.no_depth = no_depth
        self.use_teacher_distill = use_teacher_distill
        self.lambda_feat_distill = lambda_feat_distill
        self.lambda_corr_distill = lambda_corr_distill
        self.distill_end_pct = distill_end_pct
        self.student_tail_epochs = max(0, int(student_tail_epochs))
        self.loss_weight_alpha = loss_weight_alpha
        self.edge_loss_weight = edge_loss_weight
        self.use_internal_depth_guidance = use_internal_depth_guidance
        self.depth_loss_weight = depth_loss_weight
        self.depth_selection_mode = depth_selection_mode
        self.bootstrap_topk = int(bootstrap_topk)
        self.attn_topk = int(attn_topk)
        self.bootstrap_prior_scale = float(bootstrap_prior_scale)
        self.attn_prior_scale = float(attn_prior_scale)
        self.depth_prior_eps = float(depth_prior_eps)
        self.new_module_lr_mult = getattr(args, 'new_module_lr_mult', 1.0)
        self.activation_checkpointing = bool(getattr(args, 'activation_checkpointing', False))
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

        if self.scale_only:
            self.main_epochs = args.scale_epochs
        else:
            self.main_epochs = args.risk_epochs
        self.stage_a_end = max(0, min(self.main_epochs, int(math.ceil(self.distill_end_pct * self.main_epochs))))
        self.total_epochs = self.main_epochs + self.student_tail_epochs
        self.epoch = self.total_epochs
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        
        steps_per_epoch = int(len(self.train_loader))
        #print(self.epoch, len(self.train_loader), self.batch_size, steps_per_epoch)
        starte = -1
        if self.start_epoch>0:
            starte = self.start_epoch - 1
        
        max_lrs = [pg.get('max_lr', pg['lr']) for pg in self.optimizer.param_groups]
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lrs,
            epochs=self.epoch,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=max(steps_per_epoch*starte,-1),
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
        self.dataset = dataset
        self.out_dir = "./log/%s_surround_ttc" % (self.time_stamp)
        self.train_branch = 'scale' if self.scale_only else 'risk'
        self.batch_metrics_interval = 50
        self.num_probe_samples = 4
        self.probe_seed = 326
        self.fixed_probe_samples = []
        self.latest_feature_drift = float('nan')

    def _write_json(self, path, payload):
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _load_json(self, path, default):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return default

    def _append_jsonl(self, path, record):
        with open(path, 'a') as f:
            f.write(json.dumps(record, sort_keys=True) + '\n')

    def _append_json_array(self, path, record):
        records = self._load_json(path, [])
        records.append(record)
        self._write_json(path, records)

    def _append_csv_row(self, path, fieldnames, row):
        file_exists = os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _model_ref(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _current_lrs(self):
        group_names = getattr(self.optimizer, '_fp_ttc_lr_group_names', [])
        lrs = {}
        for idx, pg in enumerate(self.optimizer.param_groups):
            name = group_names[idx] if idx < len(group_names) else f'group_{idx}'
            lrs[name] = float(pg['lr'])

        if self.optimizer.param_groups:
            fallback_lr = float(self.optimizer.param_groups[0]['lr'])
            lrs.setdefault('old', fallback_lr)
            lrs.setdefault('new', fallback_lr)
        else:
            lrs.setdefault('old', 0.0)
            lrs.setdefault('new', 0.0)
        return lrs

    def _reset_cuda_peak_stats(self):
        if torch.cuda.is_available() and 'cuda' in str(self.device):
            torch.cuda.reset_peak_memory_stats(self.device)

    def _cuda_peak_stats_mb(self):
        if torch.cuda.is_available() and 'cuda' in str(self.device):
            return {
                'cuda_max_memory_allocated_mb': float(
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                ),
                'cuda_max_memory_reserved_mb': float(
                    torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
                ),
            }
        return {
            'cuda_max_memory_allocated_mb': 0.0,
            'cuda_max_memory_reserved_mb': 0.0,
        }

    def _log_current_lrs(self):
        group_names = getattr(self.optimizer, '_fp_ttc_lr_group_names', [])
        if group_names:
            for idx, name in enumerate(group_names):
                print(f"  [{name}] lr = {self.optimizer.param_groups[idx]['lr']:.3e}")
        else:
            for i, pg in enumerate(self.optimizer.param_groups):
                print(f"  param group {i} lr = {pg['lr']:.3e}")

    def _sample_meta(self, sample_index):
        meta = {
            'sample_index': int(sample_index),
            'original_index': None,
            'scene_indice': None,
            'gt_map_path': None,
            'prev_sample_token': None,
            'curr_sample_token': None,
        }
        if not hasattr(self.dataset, 'samples'):
            return meta

        original_index, info = self.dataset.samples[sample_index]
        meta['original_index'] = int(original_index)
        meta['scene_indice'] = info.get('scene_indice')
        meta['gt_map_path'] = info.get('gt_map_path')
        prev_front = info.get('prev_camera_data', {}).get('CAM_FRONT', {})
        curr_front = info.get('curr_camera_data', {}).get('CAM_FRONT', {})
        meta['prev_sample_token'] = prev_front.get('sample_token')
        meta['curr_sample_token'] = curr_front.get('sample_token')
        return meta

    def _init_fixed_probe_samples(self):
        if self.fixed_probe_samples:
            return

        probe_path = os.path.join(self.out_dir, 'fixed_probe_samples.json')
        if os.path.exists(probe_path):
            self.fixed_probe_samples = self._load_json(probe_path, [])
            return

        total = len(self.dataset)
        probe_count = min(self.num_probe_samples, total)
        rng = random.Random(self.probe_seed)
        chosen_indices = list(range(total)) if probe_count == total else sorted(rng.sample(range(total), probe_count))
        self.fixed_probe_samples = [self._sample_meta(idx) for idx in chosen_indices]
        self._write_json(probe_path, self.fixed_probe_samples)

    def _update_train_manifest(self):
        manifest_path = os.path.join(self.out_dir, 'train_manifest.json')
        manifest = self._load_json(manifest_path, {
            'train_samples': len(self.dataset),
            'checkpoint_interval': self.checkpoint_interval,
            'batch_metrics_interval': self.batch_metrics_interval,
            'branches': {},
        })
        manifest['train_samples'] = len(self.dataset)
        manifest['checkpoint_interval'] = self.checkpoint_interval
        manifest['batch_metrics_interval'] = self.batch_metrics_interval
        manifest['branches'][self.train_branch] = {
            'main_epochs': self.main_epochs,
            'stage_a_end': self.stage_a_end,
            'total_epochs': self.total_epochs,
            'scale_only': bool(self.scale_only),
            'use_teacher_distill': bool(self.use_teacher_distill),
            'use_internal_depth_guidance': bool(self.use_internal_depth_guidance),
            'student_tail_epochs': self.student_tail_epochs,
            'edge_loss_weight': float(self.edge_loss_weight),
            'depth_loss_weight': float(self.depth_loss_weight),
            'depth_selection_mode': self.depth_selection_mode,
            'bootstrap_topk': int(self.bootstrap_topk),
            'attn_topk': int(self.attn_topk),
            'bootstrap_prior_scale': float(self.bootstrap_prior_scale),
            'attn_prior_scale': float(self.attn_prior_scale),
            'depth_prior_eps': float(self.depth_prior_eps),
            'new_module_lr_mult': float(self.new_module_lr_mult),
            'activation_checkpointing': bool(self.activation_checkpointing),
            'optimizer_groups': getattr(self.optimizer, '_fp_ttc_lr_group_summary', []),
        }
        self._write_json(manifest_path, manifest)

    def _initialize_run_artifacts(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._update_train_manifest()
        self._init_fixed_probe_samples()

        for filename, default in (
            ('checkpoints.json', []),
            ('visualizations.json', []),
            ('probe_metrics.json', []),
        ):
            path = os.path.join(self.out_dir, filename)
            if not os.path.exists(path):
                self._write_json(path, default)

        for filename in ('batch_metrics.jsonl',):
            path = os.path.join(self.out_dir, filename)
            if not os.path.exists(path):
                open(path, 'a').close()

    def _batchify_probe_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.unsqueeze(0)
        if isinstance(value, dict):
            return {k: self._batchify_probe_value(v) for k, v in value.items()}
        return value

    def _load_probe_sample(self, sample_index):
        np_state = np.random.get_state()
        py_state = random.getstate()
        np.random.seed(self.probe_seed + int(sample_index))
        random.seed(self.probe_seed + int(sample_index))
        try:
            sample = self.dataset[sample_index]
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
        return sample

    def _move_sensor_metas_to_device(self, sensor_metas):
        for frame_key in sensor_metas:
            for channel in sensor_metas[frame_key]:
                for key in sensor_metas[frame_key][channel]:
                    sensor_metas[frame_key][channel][key] = sensor_metas[frame_key][channel][key].to(self.device)
        return sensor_metas

    def _log_visualization_record(self, record):
        self._append_json_array(os.path.join(self.out_dir, 'visualizations.json'), record)

    def _save_depth_panorama(self, depth_expected, path):
        if depth_expected is None:
            return None
        depth_map = depth_expected[0].detach().cpu().numpy()  # [V,H,W]
        pano = np.concatenate([depth_map[v] for v in range(depth_map.shape[0])], axis=1)
        log_min = math.log(1.0)
        log_max = math.log(40.0)
        pano_norm = np.clip((np.log(np.clip(pano, 1e-6, None)) - log_min) / (log_max - log_min + 1e-6), 0.0, 1.0)
        plt.imsave(path, pano_norm, cmap='viridis', vmin=0.0, vmax=1.0)
        return path

    def _probe_feature_ref_path(self):
        return os.path.join(self.out_dir, 'probe_feature_refs.pt')

    def _save_prediction_visuals(self, epoch, step, phase, scale, risk_score,
                                 gt_scale_map_with_mask, gt_risk_score_map_with_mask,
                                 record_extra=None, prefix=None,
                                 depth_expected_prev=None, depth_expected_curr=None,
                                 feature_drift=None):
        base_prefix = prefix if prefix is not None else f"{epoch}_{step}"
        extra = record_extra or {}

        if self.scale_only:
            gt_scale = gt_scale_map_with_mask[:, 0, :, :]
            gt_scale_valid_mask = gt_scale_map_with_mask[:, 1, :, :]

            gt_scale_np = gt_scale[:1].detach().squeeze(0).cpu().numpy()
            gt_scale_valid_mask_np = gt_scale_valid_mask[:1].squeeze(0).cpu().detach().bool()
            normalized_gt = visual_scale_map_range_image(gt_scale_np, gt_scale_valid_mask_np)

            scale_np = scale[0].detach().squeeze(0).cpu().numpy()
            pred_scale_valid_mask = scale_np > 0
            normalized_pred = visual_scale_map_range_image(scale_np, pred_scale_valid_mask)

            pred_path = os.path.join(self.out_dir, f"{base_prefix}_pred.png")
            gt_path = os.path.join(self.out_dir, f"{base_prefix}_gt.png")
            plt.imsave(pred_path, -normalized_pred, cmap='seismic', vmin=-1, vmax=1)
            plt.imsave(gt_path, -normalized_gt, cmap='seismic', vmin=-1, vmax=1)
        else:
            gt_risk_score = gt_risk_score_map_with_mask[:, 0, :, :]
            gt_risk_score_valid_mask = gt_risk_score_map_with_mask[:, 1, :, :]

            gt_risk_score_np = gt_risk_score[:1].detach().squeeze(0).cpu().numpy()
            gt_risk_score_valid_mask_np = gt_risk_score_valid_mask[:1].squeeze(0).cpu().detach().bool()
            normalized_gt_risk_score = visual_risk_score_map_range_image(gt_risk_score_np, gt_risk_score_valid_mask_np)

            risk_score_np = risk_score[0].detach().squeeze(0).cpu().detach().numpy()
            normalized_pred_risk_score = visual_risk_score_map_range_image(risk_score_np, None)

            pred_path = os.path.join(self.out_dir, f"{base_prefix}_pred_risk.png")
            gt_path = os.path.join(self.out_dir, f"{base_prefix}_gt_risk.png")
            plt.imsave(pred_path, -normalized_pred_risk_score, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)
            plt.imsave(gt_path, -normalized_gt_risk_score, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)

        depth_prev_path = None
        depth_curr_path = None
        if depth_expected_prev is not None:
            depth_prev_path = os.path.join(self.out_dir, f"{base_prefix}_depth_prev.png")
            self._save_depth_panorama(depth_expected_prev, depth_prev_path)
        if depth_expected_curr is not None:
            depth_curr_path = os.path.join(self.out_dir, f"{base_prefix}_depth_curr.png")
            self._save_depth_panorama(depth_expected_curr, depth_curr_path)

        self._log_visualization_record({
            'kind': extra.get('kind', 'train_step'),
            'train_branch': self.train_branch,
            'epoch': int(epoch),
            'step': int(step),
            'phase': phase,
            'pred_path': os.path.relpath(pred_path, self.out_dir),
            'gt_path': os.path.relpath(gt_path, self.out_dir),
            'depth_prev_path': os.path.relpath(depth_prev_path, self.out_dir) if depth_prev_path else None,
            'depth_curr_path': os.path.relpath(depth_curr_path, self.out_dir) if depth_curr_path else None,
            'feature_drift': None if feature_drift is None or not np.isfinite(feature_drift) else float(feature_drift),
            'sample_index': extra.get('sample_index'),
            'original_index': extra.get('original_index'),
            'scene_indice': extra.get('scene_indice'),
            'prev_sample_token': extra.get('prev_sample_token'),
            'curr_sample_token': extra.get('curr_sample_token'),
        })

    @torch.no_grad()
    def _save_probe_snapshots(self, epoch, phase, cur_use_distill, cur_lambda_feat, cur_lambda_corr):
        if not self.fixed_probe_samples:
            return

        probe_dir = os.path.join(self.out_dir, 'probes', f'epoch_{epoch:03d}')
        os.makedirs(probe_dir, exist_ok=True)
        ref_path = self._probe_feature_ref_path()
        feature_refs = torch.load(ref_path, map_location='cpu') if os.path.exists(ref_path) else {}
        probe_drifts = []

        model_ref = self._model_ref()
        was_training = self.model.training
        self.model.eval()
        try:
            for probe_idx, probe_meta in enumerate(self.fixed_probe_samples):
                sample = self._load_probe_sample(probe_meta['sample_index'])
                (prev_imgs,
                 curr_imgs,
                 prev_depths,
                 curr_depths,
                 proj_prev,
                 proj_curr,
                 gt_scale,
                 gt_risk,
                 sensor_metas) = sample

                prev_imgs = self._batchify_probe_value(prev_imgs).to(self.device)
                curr_imgs = self._batchify_probe_value(curr_imgs).to(self.device)
                prev_depths = self._batchify_probe_value(prev_depths).to(self.device)
                curr_depths = self._batchify_probe_value(curr_depths).to(self.device)
                proj_prev = self._batchify_probe_value(proj_prev).to(self.device)
                proj_curr = self._batchify_probe_value(proj_curr).to(self.device)
                gt_scale = self._batchify_probe_value(gt_scale).to(self.device)
                gt_risk = self._batchify_probe_value(gt_risk).to(self.device)
                sensor_metas = self._move_sensor_metas_to_device(self._batchify_probe_value(sensor_metas))

                scale, risk_score, _, _, loss_dict = model_ref.forward_with_loss(
                    img_prev=prev_imgs,
                    img_curr=curr_imgs,
                    depth_prev=prev_depths,
                    depth_curr=curr_depths,
                    proj_pix_prev=proj_prev,
                    proj_pix_curr=proj_curr,
                    gt_scale_map_with_mask=gt_scale,
                    gt_risk_score_map_with_mask=gt_risk,
                    sensor_metas=sensor_metas,
                    attn_type=self.attn_type,
                    attn_splits_list=self.attn_splits_list,
                    corr_radius_list=self.corr_radius_list,
                    prop_radius_list=self.prop_radius_list,
                    num_reg_refine=self.num_reg_refine,
                    scale_only=self.scale_only,
                    no_depth=self.no_depth,
                    use_teacher_distill=cur_use_distill,
                    lambda_feat_distill=cur_lambda_feat,
                    lambda_corr_distill=cur_lambda_corr,
                    loss_weight_alpha=self.loss_weight_alpha,
                    edge_loss_weight=self.edge_loss_weight,
                    use_internal_depth_guidance=self.use_internal_depth_guidance,
                    depth_loss_weight=self.depth_loss_weight,
                    depth_selection_mode=self.depth_selection_mode,
                    bootstrap_topk=self.bootstrap_topk,
                    attn_topk=self.attn_topk,
                    bootstrap_prior_scale=self.bootstrap_prior_scale,
                    attn_prior_scale=self.attn_prior_scale,
                    depth_prior_eps=self.depth_prior_eps,
                    return_aux=True,
                )

                if isinstance(scale, list):
                    scale = scale[-1]

                aux = loss_dict.get('aux', {})
                feature_monitor_prev = aux.get('feature_monitor_prev')
                feature_monitor_curr = aux.get('feature_monitor_curr')
                feature_drift = float('nan')
                if feature_monitor_prev is not None and feature_monitor_curr is not None:
                    current_feature = torch.stack(
                        [feature_monitor_prev[0].cpu(), feature_monitor_curr[0].cpu()],
                        dim=0,
                    )
                    ref_key = str(probe_idx)
                    if ref_key not in feature_refs:
                        feature_refs[ref_key] = current_feature
                    ref_feature = feature_refs[ref_key]
                    feature_drift = torch.sqrt(torch.mean((current_feature.float() - ref_feature.float()) ** 2)).item()
                    probe_drifts.append(feature_drift)

                self._save_prediction_visuals(
                    epoch=epoch,
                    step=probe_idx,
                    phase=phase,
                    scale=scale,
                    risk_score=risk_score,
                    gt_scale_map_with_mask=gt_scale,
                    gt_risk_score_map_with_mask=gt_risk,
                    record_extra={**probe_meta, 'kind': 'probe'},
                    prefix=os.path.join('probes', f'epoch_{epoch:03d}', f'probe_{probe_idx:02d}'),
                    depth_expected_prev=aux.get('depth_expected_prev'),
                    depth_expected_curr=aux.get('depth_expected_curr'),
                    feature_drift=feature_drift,
                )
        finally:
            if was_training:
                self.model.train()
        torch.save(feature_refs, ref_path)
        mean_drift = float(np.mean(probe_drifts)) if probe_drifts else float('nan')
        self.latest_feature_drift = mean_drift
        self._append_json_array(os.path.join(self.out_dir, 'probe_metrics.json'), {
            'train_branch': self.train_branch,
            'epoch': int(epoch),
            'phase': phase,
            'feature_drift': None if not np.isfinite(mean_drift) else mean_drift,
        })


    def train(self):
        if is_main_process():
            self._initialize_run_artifacts()

        if is_main_process():
            self._log_current_lrs()

        for epoch in range(self.start_epoch, self.epoch):
            stage_name = self._stage_name(epoch)
            if is_main_process():
                print(f'Epoch: {epoch} [{stage_name}]')
            self.loss_per_epoch = 0
            self.loss_sum_per_epoch = 0
            self.iters = 0
            epoch_metrics = self.train_epoch(epoch)

            if is_main_process():
                if (epoch < self.epoch and epoch % self.checkpoint_interval == 0):
                    checkpoint = {
                        "net": self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "metrics": epoch_metrics,
                    }
                    # # 用当前的时间戳为模型保存路径命名
                    if self.scale_only:
                        temp_pth = os.path.join(self.out_dir, f'{epoch}_scale.pth.tar')
                    else:
                        temp_pth = os.path.join(self.out_dir, f'{epoch}.pth.tar')
                    torch.save(checkpoint, temp_pth)
                    self._append_json_array(os.path.join(self.out_dir, 'checkpoints.json'), {
                        'train_branch': self.train_branch,
                        'epoch': int(epoch),
                        'phase': stage_name,
                        'path': os.path.relpath(temp_pth, self.out_dir),
                        'lr': epoch_metrics['lr_old'],
                        'lr_old': epoch_metrics['lr_old'],
                        'lr_new': epoch_metrics['lr_new'],
                        'train_loss_total': epoch_metrics['train_loss_total'],
                        'train_loss_task': epoch_metrics['train_loss_task'],
                    })
                    self._save_probe_snapshots(
                        epoch=epoch,
                        phase=stage_name,
                        cur_use_distill=bool(epoch_metrics['distill_scale'] > 0),
                        cur_lambda_feat=epoch_metrics['lambda_feat_distill'],
                        cur_lambda_corr=epoch_metrics['lambda_corr_distill'],
                    )

                print("Loss in epoch", epoch, ":", epoch_metrics['train_loss_total'])
                print(
                    f"Peak CUDA memory in epoch {epoch}: "
                    f"allocated={epoch_metrics['cuda_max_memory_allocated_mb']:.1f} MB, "
                    f"reserved={epoch_metrics['cuda_max_memory_reserved_mb']:.1f} MB"
                )
                self._log_current_lrs()

    
    def train_epoch(self, epoch):
        total_samples = len(self.train_loader.dataset)
        if self.parallel:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        self._reset_cuda_peak_stats()
        
        if self.use_teacher_distill and self.stage_a_end > 0 and epoch < self.stage_a_end:
            distill_scale = 1.0 - (epoch / self.stage_a_end)
        else:
            distill_scale = 0.0

        cur_lambda_feat = self.lambda_feat_distill * distill_scale
        cur_lambda_corr = self.lambda_corr_distill * distill_scale
        cur_use_distill = self.use_teacher_distill and (distill_scale > 0)

        if is_main_process():
            if cur_use_distill:
                print(f"  [Distill] epoch={epoch}, scale={distill_scale:.3f}, "
                      f"lambda_feat={cur_lambda_feat:.4f}, lambda_corr={cur_lambda_corr:.4f}")
            else:
                print(f"  [Student-Tail] epoch={epoch}, distill disabled")

        phase = self._stage_name(epoch)
        epoch_totals = {
            'total': 0.0,
            'task': 0.0,
            'scale': 0.0,
            'risk': 0.0,
            'feat_distill': 0.0,
            'corr_distill': 0.0,
            'edge': 0.0,
            'depth': 0.0,
            'depth_entropy': 0.0,
            'depth_valid_ratio': 0.0,
            'depth_top1_prob': 0.0,
            'depth_top1_margin': 0.0,
            'bootstrap_active_candidates': 0.0,
            'attn_active_candidates': 0.0,
            'depth_selection_sparsity': 0.0,
        }
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
             gt_risk_score_map_with_mask,
             sensor_metas) = data
            
            prev_surr_view_imgs_tensor   = prev_surr_view_imgs_tensor.to(self.device)
            curr_surr_view_imgs_tensor   = curr_surr_view_imgs_tensor.to(self.device)
            prev_surr_view_depths_tensor = prev_surr_view_depths_tensor.to(self.device)
            curr_surr_view_depths_tensor = curr_surr_view_depths_tensor.to(self.device)
            proj_pix_prev_tensor         = proj_pix_prev_tensor.to(self.device)
            proj_pix_curr_tensor         = proj_pix_curr_tensor.to(self.device)
            gt_scale_map_with_mask       = gt_scale_map_with_mask.to(self.device)
            gt_risk_score_map_with_mask  = gt_risk_score_map_with_mask.to(self.device)
            sensor_metas = self._move_sensor_metas_to_device(sensor_metas)

            self.optimizer.zero_grad(set_to_none=True)
            # 在多卡模式下，从 self.model.module 调用 forward_with_loss，否则直接调用
            if hasattr(self.model, "module"):
                scale, risk_score, loss_s, loss_r, loss_dict = self.model.module.forward_with_loss(
                    img_prev                      = prev_surr_view_imgs_tensor,
                    img_curr                      = curr_surr_view_imgs_tensor,
                    depth_prev                    = prev_surr_view_depths_tensor,
                    depth_curr                    = curr_surr_view_depths_tensor,
                    proj_pix_prev                 = proj_pix_prev_tensor,
                    proj_pix_curr                 = proj_pix_curr_tensor,
                    gt_scale_map_with_mask        = gt_scale_map_with_mask,
                    gt_risk_score_map_with_mask   = gt_risk_score_map_with_mask,
                    sensor_metas                  = sensor_metas,
                    attn_type                     = self.attn_type,
                    attn_splits_list              = self.attn_splits_list,
                    corr_radius_list              = self.corr_radius_list,
                    prop_radius_list              = self.prop_radius_list,
                    num_reg_refine                = self.num_reg_refine,
                    scale_only                    = self.scale_only,
                    no_depth                      = self.no_depth,
                    use_teacher_distill           = cur_use_distill,
                    lambda_feat_distill           = cur_lambda_feat,
                    lambda_corr_distill           = cur_lambda_corr,
                    loss_weight_alpha             = self.loss_weight_alpha,
                    edge_loss_weight              = self.edge_loss_weight,
                    use_internal_depth_guidance   = self.use_internal_depth_guidance,
                    depth_loss_weight             = self.depth_loss_weight,
                    depth_selection_mode          = self.depth_selection_mode,
                    bootstrap_topk                = self.bootstrap_topk,
                    attn_topk                     = self.attn_topk,
                    bootstrap_prior_scale         = self.bootstrap_prior_scale,
                    attn_prior_scale              = self.attn_prior_scale,
                    depth_prior_eps               = self.depth_prior_eps,
                )
            else:
                scale, risk_score, loss_s, loss_r, loss_dict = self.model.forward_with_loss(
                    img_prev                      = prev_surr_view_imgs_tensor,
                    img_curr                      = curr_surr_view_imgs_tensor,
                    depth_prev                    = prev_surr_view_depths_tensor,
                    depth_curr                    = curr_surr_view_depths_tensor,
                    proj_pix_prev                 = proj_pix_prev_tensor,
                    proj_pix_curr                 = proj_pix_curr_tensor,
                    gt_scale_map_with_mask        = gt_scale_map_with_mask,
                    gt_risk_score_map_with_mask   = gt_risk_score_map_with_mask,
                    sensor_metas                  = sensor_metas,
                    attn_type                     = self.attn_type,
                    attn_splits_list              = self.attn_splits_list,
                    corr_radius_list              = self.corr_radius_list,
                    prop_radius_list              = self.prop_radius_list,
                    num_reg_refine                = self.num_reg_refine,
                    scale_only                    = self.scale_only,
                    no_depth                      = self.no_depth,
                    use_teacher_distill           = cur_use_distill,
                    lambda_feat_distill           = cur_lambda_feat,
                    lambda_corr_distill           = cur_lambda_corr,
                    loss_weight_alpha             = self.loss_weight_alpha,
                    edge_loss_weight              = self.edge_loss_weight,
                    use_internal_depth_guidance   = self.use_internal_depth_guidance,
                    depth_loss_weight             = self.depth_loss_weight,
                    depth_selection_mode          = self.depth_selection_mode,
                    bootstrap_topk                = self.bootstrap_topk,
                    attn_topk                     = self.attn_topk,
                    bootstrap_prior_scale         = self.bootstrap_prior_scale,
                    attn_prior_scale              = self.attn_prior_scale,
                    depth_prior_eps               = self.depth_prior_eps,
                )
            
            loss = loss_dict['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.lr_scheduler.step()
            batch_metrics = {
                'total': float(loss_dict['total'].item()),
                'task': float(loss_dict['task'].item()),
                'scale': float(loss_dict['scale'].item()),
                'risk': float(loss_dict['risk'].item()),
                'feat_distill': float(loss_dict['feat_distill'].item()),
                'corr_distill': float(loss_dict['corr_distill'].item()),
                'edge': float(loss_dict['edge'].item()),
                'depth': float(loss_dict['depth'].item()),
                'depth_entropy': float(loss_dict['depth_entropy'].item()),
                'depth_valid_ratio': float(loss_dict['depth_valid_ratio'].item()),
                'depth_top1_prob': float(loss_dict['depth_top1_prob'].item()),
                'depth_top1_margin': float(loss_dict['depth_top1_margin'].item()),
                'bootstrap_active_candidates': float(loss_dict['bootstrap_active_candidates'].item()),
                'attn_active_candidates': float(loss_dict['attn_active_candidates'].item()),
                'depth_selection_sparsity': float(loss_dict['depth_selection_sparsity'].item()),
            }
            for key, value in batch_metrics.items():
                epoch_totals[key] += value
            steps += 1

            ### 可视化结果
            
            if type(scale) == list:
                scale = scale[-1]
            if i%int(save_index)==0 and is_main_process():
                self._save_prediction_visuals(
                    epoch=epoch,
                    step=i,
                    phase=phase,
                    scale=scale,
                    risk_score=risk_score,
                    gt_scale_map_with_mask=gt_scale_map_with_mask,
                    gt_risk_score_map_with_mask=gt_risk_score_map_with_mask,
                )

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
                        f"Loss_now: {batch_metrics['total']:6.4f}  "
                        f"(task={batch_metrics['task']:6.4f}, "
                        f"feat={batch_metrics['feat_distill']:6.4f}, "
                        f"corr={batch_metrics['corr_distill']:6.4f}, "
                        f"edge={batch_metrics['edge']:6.4f}, "
                        f"depth={batch_metrics['depth']:6.4f}, "
                        f"top1={batch_metrics['depth_top1_prob']:6.4f}, "
                        f"margin={batch_metrics['depth_top1_margin']:6.4f})"
                    )

            self.iters += 1

            # 记录每个 batch 的 loss 和全局步数
            global_step = epoch * len(self.train_loader) + i
            if is_main_process() and (i == 0 or global_step % self.batch_metrics_interval == 0):
                current_lrs = self._current_lrs()
                peak_stats = self._cuda_peak_stats_mb()
                self._append_jsonl(
                    os.path.join(self.out_dir, 'batch_metrics.jsonl'),
                    {
                        'train_branch': self.train_branch,
                        'global_step': int(global_step),
                        'epoch': int(epoch),
                        'step_in_epoch': int(i),
                        'phase': phase,
                        'lr': current_lrs['old'],
                        'lr_old': current_lrs['old'],
                        'lr_new': current_lrs['new'],
                        'loss_total': batch_metrics['total'],
                        'loss_task': batch_metrics['task'],
                        'loss_scale': batch_metrics['scale'],
                        'loss_risk': batch_metrics['risk'],
                        'loss_feat_distill': batch_metrics['feat_distill'],
                        'loss_corr_distill': batch_metrics['corr_distill'],
                        'loss_edge': batch_metrics['edge'],
                        'loss_depth': batch_metrics['depth'],
                        'depth_entropy': batch_metrics['depth_entropy'],
                        'depth_valid_ratio': batch_metrics['depth_valid_ratio'],
                        'depth_top1_prob': batch_metrics['depth_top1_prob'],
                        'depth_top1_margin': batch_metrics['depth_top1_margin'],
                        'bootstrap_active_candidates': batch_metrics['bootstrap_active_candidates'],
                        'attn_active_candidates': batch_metrics['attn_active_candidates'],
                        'depth_selection_sparsity': batch_metrics['depth_selection_sparsity'],
                        'distill_scale': float(distill_scale),
                        'lambda_feat_distill': float(cur_lambda_feat),
                        'lambda_corr_distill': float(cur_lambda_corr),
                        'cuda_max_memory_allocated_mb': peak_stats['cuda_max_memory_allocated_mb'],
                        'cuda_max_memory_reserved_mb': peak_stats['cuda_max_memory_reserved_mb'],
                    },
                )
            if self.neptune_run is not None and is_main_process():
                if self.scale_only:
                    self.neptune_run["train/batch_loss_scale"].append(batch_metrics['scale'], step=global_step)
                else:
                    self.neptune_run["train/batch_loss_risk"].append(batch_metrics['risk'], step=global_step)
                for group_idx, pg in enumerate(self.optimizer.param_groups):
                    tag = f"train/batch_learning_rate_group_{group_idx}"
                    self.neptune_run[tag].append(pg["lr"], step=global_step)
        
        avg_metrics = {key: value / max(1, steps) for key, value in epoch_totals.items()}
        current_lrs = self._current_lrs()
        peak_stats = self._cuda_peak_stats_mb()
        epoch_metrics = {
            'train_branch': self.train_branch,
            'epoch': int(epoch),
            'phase': phase,
            'lr': current_lrs['old'],
            'lr_old': current_lrs['old'],
            'lr_new': current_lrs['new'],
            'train_loss_total': avg_metrics['total'],
            'train_loss_task': avg_metrics['task'],
            'train_loss_scale': avg_metrics['scale'],
            'train_loss_risk': avg_metrics['risk'],
            'train_loss_feat_distill': avg_metrics['feat_distill'],
            'train_loss_corr_distill': avg_metrics['corr_distill'],
            'train_loss_edge': avg_metrics['edge'],
            'train_loss_depth': avg_metrics['depth'],
            'depth_entropy': avg_metrics['depth_entropy'],
            'depth_valid_ratio': avg_metrics['depth_valid_ratio'],
            'depth_top1_prob': avg_metrics['depth_top1_prob'],
            'depth_top1_margin': avg_metrics['depth_top1_margin'],
            'bootstrap_active_candidates': avg_metrics['bootstrap_active_candidates'],
            'attn_active_candidates': avg_metrics['attn_active_candidates'],
            'depth_selection_sparsity': avg_metrics['depth_selection_sparsity'],
            'feature_drift': self.latest_feature_drift,
            'distill_scale': float(distill_scale),
            'lambda_feat_distill': float(cur_lambda_feat),
            'lambda_corr_distill': float(cur_lambda_corr),
            'cuda_max_memory_allocated_mb': peak_stats['cuda_max_memory_allocated_mb'],
            'cuda_max_memory_reserved_mb': peak_stats['cuda_max_memory_reserved_mb'],
            'num_steps': int(steps),
        }

        if is_main_process():
            self._append_csv_row(
                os.path.join(self.out_dir, 'epoch_metrics.csv'),
                [
                    'train_branch',
                    'epoch',
                    'phase',
                    'lr',
                    'lr_old',
                    'lr_new',
                    'train_loss_total',
                    'train_loss_task',
                    'train_loss_scale',
                    'train_loss_risk',
                    'train_loss_feat_distill',
                    'train_loss_corr_distill',
                    'train_loss_edge',
                    'train_loss_depth',
                    'depth_entropy',
                    'depth_valid_ratio',
                    'depth_top1_prob',
                    'depth_top1_margin',
                    'bootstrap_active_candidates',
                    'attn_active_candidates',
                    'depth_selection_sparsity',
                    'feature_drift',
                    'distill_scale',
                    'lambda_feat_distill',
                    'lambda_corr_distill',
                    'cuda_max_memory_allocated_mb',
                    'cuda_max_memory_reserved_mb',
                    'num_steps',
                ],
                epoch_metrics,
            )

        if self.neptune_run is not None and is_main_process():
            self.neptune_run["train/epoch_loss"].append(epoch_metrics['train_loss_total'], step=epoch)
            for group_idx, pg in enumerate(self.optimizer.param_groups):
                tag = f"train/epoch_learning_rate_group_{group_idx}"
                self.neptune_run[tag].append(pg["lr"], step=epoch)
        
        return epoch_metrics
    
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

    def _stage_name(self, epoch):
        if self.use_teacher_distill and epoch < self.stage_a_end:
            return 'distill'
        return 'student-tail'
    
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
