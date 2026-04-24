#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH=.
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MASTER_PORT="${MASTER_PORT:-29501}"

TRAIN_INFO_PATH="${TRAIN_INFO_PATH:-./Datasets/nuscenes/2_trainval_test_infos/train}"
TRAIN_INFO_FILE="${TRAIN_INFO_FILE:-nusc_train_infos_key_frames_160_1920_fov_8_15.pkl}"
VAL_INFO_PATH="${VAL_INFO_PATH:-./Datasets/nuscenes/2_trainval_test_infos/val}"
VAL_INFO_FILE="${VAL_INFO_FILE:-nusc_val_infos_key_frames_160_1920_fov_8_15.pkl}"
PROJ_CACHE_ROOT="${PROJ_CACHE_ROOT:-./Datasets/nuscenes/5_proj_cache/nusc_150_keyframes_160x320_fov8_15_hardproj_v1}"
SCALE_CKPT="${SCALE_CKPT-pretrained/1120_scale_kbins_8.pth.tar}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./hyy-results/query-init-ablation-2adb}"
RVT_QUERY_INIT="${RVT_QUERY_INIT:-zero}"

SCALE_EPOCHS="${SCALE_EPOCHS:-480}"
STUDENT_TAIL_EPOCHS="${STUDENT_TAIL_EPOCHS:-0}"
DISTILL_END_PCT="${DISTILL_END_PCT:-0.5}"
DISTILL_END_EPOCH="${DISTILL_END_EPOCH:-120}"
LAMBDA_FEAT_DISTILL="${LAMBDA_FEAT_DISTILL:-0.3}"
LAMBDA_CORR_DISTILL="${LAMBDA_CORR_DISTILL:-0.5}"
LOSS_WEIGHT_ALPHA="${LOSS_WEIGHT_ALPHA:-3.0}"
NEW_MODULE_LR_MULT="${NEW_MODULE_LR_MULT:-5.0}"
EDGE_LOSS_WEIGHT="${EDGE_LOSS_WEIGHT:-0.0}"
RUN_NAME="${RUN_NAME:-queryinit-${RVT_QUERY_INIT}-diag-e${SCALE_EPOCHS}-distill${DISTILL_END_EPOCH}}"

NUM_WORKERS="${NUM_WORKERS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
LR="${LR:-4e-5}"
SEED="${SEED:-326}"
VAL_FREQ="${VAL_FREQ:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
METRIC_DELTA_T="${METRIC_DELTA_T:-0.5}"
SCALE_BATCH_SIZE="${SCALE_BATCH_SIZE:-1}"
RISK_BATCH_SIZE="${RISK_BATCH_SIZE:-1}"

EXTRA_ARGS=()
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max_val_samples "${MAX_VAL_SAMPLES}")
fi
if [[ -n "${SCALE_CKPT}" ]]; then
  EXTRA_ARGS+=(--scale_pretrained_ckpt "${SCALE_CKPT}")
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --parallel \
  --no_depth \
  --use_teacher_distill \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --run_name "${RUN_NAME}" \
  --stage nuscenes_range_image \
  --train_info_path "${TRAIN_INFO_PATH}" \
  --train_info_file "${TRAIN_INFO_FILE}" \
  --val_info_path "${VAL_INFO_PATH}" \
  --val_info_file "${VAL_INFO_FILE}" \
  --proj_cache_root "${PROJ_CACHE_ROOT}" \
  --padding_factor 32 \
  --upsample_factor 4 \
  --num_scales 2 \
  --rvt_query_init "${RVT_QUERY_INIT}" \
  --attn_splits_list 2 8 \
  --corr_radius_list -1 4 \
  --prop_radius_list -1 1 \
  --lr "${LR}" \
  --seed "${SEED}" \
  --val_freq "${VAL_FREQ}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --metric_delta_t "${METRIC_DELTA_T}" \
  --new_module_lr_mult "${NEW_MODULE_LR_MULT}" \
  --loss_weight_alpha "${LOSS_WEIGHT_ALPHA}" \
  --edge_loss_weight "${EDGE_LOSS_WEIGHT}" \
  --batch_size 1 \
  --num_workers "${NUM_WORKERS}" \
  --image_size 160 320 \
  --train_stage scale \
  --scale_epochs "${SCALE_EPOCHS}" \
  --student_tail_epochs "${STUDENT_TAIL_EPOCHS}" \
  --distill_end_pct "${DISTILL_END_PCT}" \
  --distill_end_epoch "${DISTILL_END_EPOCH}" \
  --lambda_feat_distill "${LAMBDA_FEAT_DISTILL}" \
  --lambda_corr_distill "${LAMBDA_CORR_DISTILL}" \
  --risk_epochs 1 \
  --scale_batch_size "${SCALE_BATCH_SIZE}" \
  --risk_batch_size "${RISK_BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"
