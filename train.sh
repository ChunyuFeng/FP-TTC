#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH=.
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MASTER_PORT="${MASTER_PORT:-29501}"

TRAIN_INFO_PATH="${TRAIN_INFO_PATH:-./Datasets/nuscenes/2_trainval_test_infos/train}"
TRAIN_INFO_FILE="${TRAIN_INFO_FILE:-nusc_train_infos_key_frames_160_1920_fov_8_15.pkl}"
VAL_INFO_PATH="${VAL_INFO_PATH:-./Datasets/nuscenes/2_trainval_test_infos/val}"
VAL_INFO_FILE="${VAL_INFO_FILE:-nusc_val_infos_key_frames_160_1920_fov_8_15.pkl}"
PROJ_CACHE_ROOT="${PROJ_CACHE_ROOT:-./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_hardproj_v1}"
VAL_PROJ_CACHE_ROOT="${VAL_PROJ_CACHE_ROOT:-./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_hardproj_v1}"
SCALE_CKPT="${SCALE_CKPT:-pretrained/fpttc_mix.pth.tar}"

SCALE_EPOCHS="${SCALE_EPOCHS:-500}"
VAL_FREQ="${VAL_FREQ:-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --parallel \
  --stage nuscenes_range_image \
  --train_info_path "${TRAIN_INFO_PATH}" \
  --train_info_file "${TRAIN_INFO_FILE}" \
  --val_info_path "${VAL_INFO_PATH}" \
  --val_info_file "${VAL_INFO_FILE}" \
  --require_complete_depth \
  --proj_cache_root "${PROJ_CACHE_ROOT}" \
  --val_proj_cache_root "${VAL_PROJ_CACHE_ROOT}" \
  --padding_factor 32 \
  --upsample_factor 4 \
  --num_scales 2 \
  --attn_splits_list 2 8 \
  --corr_radius_list -1 4 \
  --prop_radius_list -1 1 \
  --lr 4e-5 \
  --batch_size 1 \
  --num_workers "${NUM_WORKERS}" \
  --image_size 160 320 \
  --train_stage scale \
  --scale_epochs "${SCALE_EPOCHS}" \
  --risk_epochs 1 \
  --scale_batch_size 1 \
  --risk_batch_size 1 \
  --val_batch_size 1 \
  --val_freq "${VAL_FREQ}" \
  --save_best \
  --scale_pretrained_ckpt "${SCALE_CKPT}"
