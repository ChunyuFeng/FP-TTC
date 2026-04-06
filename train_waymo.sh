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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MASTER_PORT="${MASTER_PORT:-29501}"
nproc_per_node=8

# ──────────────────────────────────────────────────────────────
#  Dataset paths
# ──────────────────────────────────────────────────────────────
waymo_dataset_root="${WAYMO_DATASET_ROOT:-./Datasets/waymo}"
train_info_path="${waymo_dataset_root}/2_trainval_test_infos/train"
train_info_file=waymo_train_infos_scene_flow_key_frames_top_lidar_160_1920_fov_2_17_nusc_layout_vehrot_occsup_fgfg_sector.pkl
val_info_path="${waymo_dataset_root}/2_trainval_test_infos/val"
val_info_file=waymo_val_infos_scene_flow_key_frames_top_lidar_160_1920_fov_2_17_nusc_layout_vehrot_occsup_fgfg_sector.pkl

proj_cache_root="${waymo_dataset_root}/5_proj_cache/waymo_scene150_keyframes_160x320_proj40x400_fov2_17_toplidar_nusc_vehrot_v1"
val_proj_cache_root="${waymo_dataset_root}/5_proj_cache/waymo_scene150_keyframes_160x320_proj40x400_fov2_17_toplidar_nusc_vehrot_v1"

# ──────────────────────────────────────────────────────────────
#  Checkpoint
# ──────────────────────────────────────────────────────────────
resume_ckpt="${RESUME_CKPT:-./pretrained/26_04_02-02_48_59_surround_ttc/best_risk.pth.tar}"

# ──────────────────────────────────────────────────────────────
#  Training hyper-parameters
# ──────────────────────────────────────────────────────────────
lr="${LR:-8e-5}"
pct_start="${PCT_START:-0.08}"             # OneCycleLR warm-up fraction
new_module_lr_mult="${NEW_MODULE_LR_MULT:-5.0}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-1}"

train_stage="${TRAIN_STAGE:-scale}"
scale_epochs="${SCALE_EPOCHS:-300}"
risk_epochs="${RISK_EPOCHS:-300}"
scale_batch_size="${SCALE_BATCH_SIZE:-1}"
risk_batch_size="${RISK_BATCH_SIZE:-1}"
val_freq="${VAL_FREQ:-5}"
num_workers="${NUM_WORKERS:-2}"

cmd=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${nproc_per_node}"
  --master_port="${MASTER_PORT}"
  train.py
  --parallel
  --stage waymo_range_image
  --waymo_dataset_root "${waymo_dataset_root}"
  --train_info_path "${train_info_path}"
  --train_info_file "${train_info_file}"
  --val_info_path "${val_info_path}"
  --val_info_file "${val_info_file}"
  --require_complete_depth
  --proj_cache_root "${proj_cache_root}"
  --val_proj_cache_root "${val_proj_cache_root}"
  --image_size 160 320
  --padding_factor 32
  --upsample_factor 4
  --num_scales 2
  --attn_splits_list 2 8
  --corr_radius_list -1 4
  --prop_radius_list -1 1
  --lr "${lr}"
  --pct_start "${pct_start}"
  --new_module_lr_mult "${new_module_lr_mult}"
  --grad_accum_steps "${grad_accum_steps}"
  --train_stage "${train_stage}"
  --scale_epochs "${scale_epochs}"
  --risk_epochs "${risk_epochs}"
  --scale_batch_size "${scale_batch_size}"
  --risk_batch_size "${risk_batch_size}"
  --val_batch_size 1
  --val_freq "${val_freq}"
  --num_workers "${num_workers}"
  --save_best
)

if [[ -n "${resume_ckpt}" ]]; then
  cmd+=(--resume "${resume_ckpt}")
fi

"${cmd[@]}"
