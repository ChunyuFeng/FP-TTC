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
PROJ_CACHE_ROOT="${PROJ_CACHE_ROOT:-./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_v1}"
SCALE_CKPT="${SCALE_CKPT:-pretrained/rvt/1120_scale_kbins_8.pth.tar}"

SCALE_EPOCHS="${SCALE_EPOCHS:-240}"
STUDENT_TAIL_EPOCHS="${STUDENT_TAIL_EPOCHS:-0}"
DISTILL_END_PCT="${DISTILL_END_PCT:-0.7}"
LAMBDA_FEAT_DISTILL="${LAMBDA_FEAT_DISTILL:-1.0}"
LAMBDA_CORR_DISTILL="${LAMBDA_CORR_DISTILL:-0.5}"
LOSS_WEIGHT_ALPHA="${LOSS_WEIGHT_ALPHA:-0.0}"
NEW_MODULE_LR_MULT="${NEW_MODULE_LR_MULT:-5.0}"

NUM_WORKERS="${NUM_WORKERS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
LR="${LR:-4e-5}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --parallel \
  --no_depth \
  --use_teacher_distill \
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
  --batch_size 1 \
  --num_workers "${NUM_WORKERS}" \
  --image_size 160 320 \
  --train_stage scale \
  --scale_epochs "${SCALE_EPOCHS}" \
  --student_tail_epochs "${STUDENT_TAIL_EPOCHS}" \
  --distill_end_pct "${DISTILL_END_PCT}" \
  --lambda_feat_distill "${LAMBDA_FEAT_DISTILL}" \
  --lambda_corr_distill "${LAMBDA_CORR_DISTILL}" \
  --risk_epochs 1 \
  --scale_batch_size 1 \
  --risk_batch_size 1 \
  --scale_pretrained_ckpt "${SCALE_CKPT}"
