#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH=.
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MASTER_PORT="${MASTER_PORT:-29501}"

TRAIN_INFO_PATH="${TRAIN_INFO_PATH:-./Datasets/nuscenes/2_trainval_test_infos/train}"
TRAIN_INFO_FILE="${TRAIN_INFO_FILE:-nusc_train_infos_key_frames_160_1920_fov_8_15.pkl}"
PROJ_CACHE_ROOT="${PROJ_CACHE_ROOT:-./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_v1}"
SCALE_CKPT="${SCALE_CKPT:-pretrained/1120_scale_kbins_8.pth.tar}"

SCALE_EPOCHS="${SCALE_EPOCHS:-160}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
LR="${LR:-3e-5}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"

cmd=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  train.py
  --parallel
  --stage nuscenes_range_image
  --train_info_path "${TRAIN_INFO_PATH}"
  --train_info_file "${TRAIN_INFO_FILE}"
  --require_complete_depth
  --proj_cache_root "${PROJ_CACHE_ROOT}"
  --padding_factor 32
  --upsample_factor 4
  --num_scales 2
  --attn_splits_list 2 8
  --corr_radius_list -1 4
  --prop_radius_list -1 1
  --lr "${LR}"
  --batch_size 4
  --num_workers "${NUM_WORKERS}"
  --image_size 160 320
  --train_stage scale
  --scale_epochs "${SCALE_EPOCHS}"
  --risk_epochs 1
  --scale_batch_size 4
  --risk_batch_size 1
  --scale_pretrained_ckpt "${SCALE_CKPT}"
)

if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  cmd+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi

"${cmd[@]}"
