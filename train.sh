#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ──────────────────────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────────────────────
export PYTHONPATH=.
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

# ──────────────────────────────────────────────────────────────
#  GPU / Distributed
# ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MASTER_PORT="${MASTER_PORT:-29501}"
nproc_per_node=6

# ──────────────────────────────────────────────────────────────
#  Dataset paths
# ──────────────────────────────────────────────────────────────
train_info_path=./Datasets/nuscenes/2_trainval_test_infos/train
train_info_file=nusc_train_infos_key_frames_160_1920_fov_8_15.pkl
val_info_path=./Datasets/nuscenes/2_trainval_test_infos/val
val_info_file=nusc_val_infos_key_frames_160_1920_fov_8_15.pkl

proj_cache_root=./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_hardproj_v1
val_proj_cache_root=./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_hardproj_v1

# ──────────────────────────────────────────────────────────────
#  Checkpoint
# ──────────────────────────────────────────────────────────────
resume_ckpt=./log/26_03_25-21_54_22_surround_ttc/best_scale.pth.tar

# ──────────────────────────────────────────────────────────────
#  Training hyper-parameters
# ──────────────────────────────────────────────────────────────
lr=8e-5
pct_start=0.08
new_module_lr_mult=5.0
grad_accum_steps=1

risk_epochs=300
risk_batch_size=1
val_freq=5
num_workers=2

# ──────────────────────────────────────────────────────────────
#  Launch training
# ──────────────────────────────────────────────────────────────
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${nproc_per_node}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --parallel \
  --stage nuscenes_range_image \
  \
  --train_info_path "${train_info_path}" \
  --train_info_file "${train_info_file}" \
  --val_info_path "${val_info_path}" \
  --val_info_file "${val_info_file}" \
  --require_complete_depth \
  --proj_cache_root "${proj_cache_root}" \
  --val_proj_cache_root "${val_proj_cache_root}" \
  \
  --image_size 160 320 \
  --padding_factor 32 \
  --upsample_factor 4 \
  --num_scales 2 \
  --attn_splits_list 2 8 \
  --corr_radius_list -1 4 \
  --prop_radius_list -1 1 \
  \
  --lr "${lr}" \
  --pct_start "${pct_start}" \
  --new_module_lr_mult "${new_module_lr_mult}" \
  --grad_accum_steps "${grad_accum_steps}" \
  \
  --train_stage risk \
  --risk_epochs "${risk_epochs}" \
  --risk_batch_size "${risk_batch_size}" \
  --val_batch_size 1 \
  --val_freq "${val_freq}" \
  --num_workers "${num_workers}" \
  --save_best \
  --resume "${resume_ckpt}"
