#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH=.
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

TRAIN_INFO_PATH="${TRAIN_INFO_PATH:-./Datasets/nuscenes/2_trainval_test_infos/train}"
TRAIN_INFO_FILE="${TRAIN_INFO_FILE:-nusc_train_infos_key_frames_160_1920_fov_8_15.pkl}"
PROJ_CACHE_ROOT="${PROJ_CACHE_ROOT:-./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_hardproj_v1}"
SCALE_CKPT="${SCALE_CKPT:-pretrained/1120_scale_kbins_8.pth.tar}"
DA_CKPT="${DA_CKPT:-pretrained/depth_anything_v2_metric_vkitti_vits.pth}"

SCALE_EPOCHS="${SCALE_EPOCHS:-240}"
LOSS_WEIGHT_ALPHA="${LOSS_WEIGHT_ALPHA:-3.0}"
NEW_MODULE_LR_MULT="${NEW_MODULE_LR_MULT:-1.0}"
EDGE_LOSS_WEIGHT="${EDGE_LOSS_WEIGHT:-0.0}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"

NUM_WORKERS="${NUM_WORKERS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
LR="${LR:-4e-5}"
SCALE_BATCH_SIZE="${SCALE_BATCH_SIZE:-4}"

EXTRA_ARGS=()
if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
  EXTRA_ARGS+=(--activation_checkpointing)
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --parallel \
  --no_depth \
  --backbone_type dual_backbone \
  --da_branch_mode off \
  --da_pretrained_ckpt "${DA_CKPT}" \
  --aggregation_mode hardproj_cache \
  --stage nuscenes_range_image \
  --train_info_path "${TRAIN_INFO_PATH}" \
  --train_info_file "${TRAIN_INFO_FILE}" \
  --proj_cache_root "${PROJ_CACHE_ROOT}" \
  --padding_factor 32 \
  --upsample_factor 4 \
  --num_scales 2 \
  --attn_splits_list 2 8 \
  --corr_radius_list -1 4 \
  --prop_radius_list -1 1 \
  --lr "${LR}" \
  --new_module_lr_mult "${NEW_MODULE_LR_MULT}" \
  --loss_weight_alpha "${LOSS_WEIGHT_ALPHA}" \
  --edge_loss_weight "${EDGE_LOSS_WEIGHT}" \
  "${EXTRA_ARGS[@]}" \
  --batch_size 1 \
  --num_workers "${NUM_WORKERS}" \
  --image_size 160 320 \
  --train_stage scale \
  --scale_epochs "${SCALE_EPOCHS}" \
  --scale_batch_size "${SCALE_BATCH_SIZE}" \
  --risk_batch_size 1 \
  --scale_pretrained_ckpt "${SCALE_CKPT}"