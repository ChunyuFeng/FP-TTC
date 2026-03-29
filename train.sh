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
PROJ_CACHE_ROOT="${PROJ_CACHE_ROOT:-./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_hardproj_v1}"
SCALE_CKPT="${SCALE_CKPT:-pretrained/1120_scale_kbins_8.pth.tar}"

SCALE_EPOCHS="${SCALE_EPOCHS:-240}"
STUDENT_TAIL_EPOCHS="${STUDENT_TAIL_EPOCHS:-0}"
DISTILL_END_PCT="${DISTILL_END_PCT:-0.5}"
LAMBDA_FEAT_DISTILL="${LAMBDA_FEAT_DISTILL:-0.3}"
LAMBDA_CORR_DISTILL="${LAMBDA_CORR_DISTILL:-0.5}"
LOSS_WEIGHT_ALPHA="${LOSS_WEIGHT_ALPHA:-3.0}"
NEW_MODULE_LR_MULT="${NEW_MODULE_LR_MULT:-5.0}"
EDGE_LOSS_WEIGHT="${EDGE_LOSS_WEIGHT:-0.1}"
DEPTH_LOSS_WEIGHT="${DEPTH_LOSS_WEIGHT:-0.5}"
DEPTH_SELECTION_MODE="${DEPTH_SELECTION_MODE:-hard_topk}"
BOOTSTRAP_TOPK="${BOOTSTRAP_TOPK:-4}"
ATTN_TOPK="${ATTN_TOPK:-8}"
BOOTSTRAP_PRIOR_SCALE="${BOOTSTRAP_PRIOR_SCALE:-2.0}"
ATTN_PRIOR_SCALE="${ATTN_PRIOR_SCALE:-2.0}"
DEPTH_PRIOR_EPS="${DEPTH_PRIOR_EPS:-1e-6}"

NUM_WORKERS="${NUM_WORKERS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
LR="${LR:-4e-5}"
SCALE_BATCH_SIZE="${SCALE_BATCH_SIZE:-2}"
RISK_BATCH_SIZE="${RISK_BATCH_SIZE:-1}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --parallel \
  --no_depth \
  --use_internal_depth_guidance \
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
  --edge_loss_weight "${EDGE_LOSS_WEIGHT}" \
  --depth_loss_weight "${DEPTH_LOSS_WEIGHT}" \
  --depth_selection_mode "${DEPTH_SELECTION_MODE}" \
  --bootstrap_topk "${BOOTSTRAP_TOPK}" \
  --attn_topk "${ATTN_TOPK}" \
  --bootstrap_prior_scale "${BOOTSTRAP_PRIOR_SCALE}" \
  --attn_prior_scale "${ATTN_PRIOR_SCALE}" \
  --depth_prior_eps "${DEPTH_PRIOR_EPS}" \
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
  --scale_batch_size "${SCALE_BATCH_SIZE}" \
  --risk_batch_size "${RISK_BATCH_SIZE}" \
  --scale_pretrained_ckpt "${SCALE_CKPT}"
