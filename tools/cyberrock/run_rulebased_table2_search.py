#!/usr/bin/env python3
"""Run rule-based collision prediction experiments for SJTU Table 2.

This driver does four things in one place:

1. Optionally generate missing `pred_npy` files with the current workspace model.
2. Export standardized `collision_predictions.csv` rows for:
   - `tools/collision_prediction/collision_prediction.py`
   - `tools/collision_prediction/collision_prediction_scheme_a.py`
3. Compare initial metrics for both methods.
4. Search scene-specific post-filters and choose the best per-scene config.

The search intentionally supports per-scene parameters because the user asked for
best-effort paper results rather than a single frozen global threshold.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import pickle
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools" / "collision_prediction") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "collision_prediction"))

from tools.collision_prediction.collision_prediction import (  # noqa: E402
    cluster_and_compute,
    compute_valid_masks,
    extract_candidate_points,
    extract_small_regions,
    get_pred_npy_files,
    load_prediction,
    load_test_infos,
    preprocess_scale_map,
)
from tools.collision_prediction.collision_prediction_scheme_a import (  # noqa: E402
    extract_scheme_a_masks,
)
from tools.cyberrock.real_vehicle_annotation_common import (  # noqa: E402
    DEFAULT_SCENE_ORDER,
    ensure_dir,
    load_csv_rows,
    scene_sort_key,
    write_csv_rows,
)
from tools.cyberrock.real_vehicle_annotation_pipeline import (  # noqa: E402
    FRAME_EVAL_HEADERS,
    SCENE_METRIC_HEADERS,
    SEQUENCE_METRIC_HEADERS,
    aggregate_scene_metrics,
    append_metric_fields,
    build_gt_frame_eval_records,
    metric_rows_to_csv_ready,
    render_paper_fill_line,
    render_scene_metrics_markdown,
    scene_map_from_manifest,
)


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

MAIN_SCENES = {"S1", "S2", "S3", "S4", "S5"}
NEGATIVE_ONLY_SCENES = {"S5"}

TARGET_METRICS = {
    "recall": 0.70,
    "far": 0.08,
    "event_recall": 0.85,
}

RULE_PRECOMPUTE = {
    "approach_threshold": 1.0,
    "grid_size": 3,
    "risk_rate": 0.28,
    "kp_detector_threshold": 2,
    "grad_step": 2,
    "risk_time_threshold": 5.0,
}

RULE_DEFAULT = {
    "method": "rule",
    "risk_time_threshold": 3.0,
    "risk_pred_threshold": 0.10,
    "include_small_regions": True,
    "selection_mode": "all",
    "allowed_cams": "all",
    "spatial_preset": "full",
}

SCHEME_A_DEFAULT = {
    "method": "scheme_a",
    "risk_time_threshold": 3.0,
    "risk_pred_threshold": 0.10,
    "include_small_regions": True,
    "selection_mode": "all",
    "allowed_cams": "all",
    "spatial_preset": "full",
}

SPATIAL_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "full": (0.00, 1.00, 0.00, 1.00),
    "center70": (0.15, 0.85, 0.20, 0.90),
    "center50": (0.25, 0.75, 0.25, 0.90),
    "lower_mid": (0.20, 0.80, 0.35, 0.95),
    "front_focus": (0.33, 0.67, 0.20, 0.90),
}

CAM_PRESETS: dict[str, set[int]] = {
    "all": {0, 1, 2, 3, 4, 5},
    "front3": {0, 1, 2},
    "rear3": {3, 4, 5},
    "front_center": {1},
    "rear_center": {4},
}

SELECTION_MODES = {"all", "top1_ttc", "top2_ttc"}

PREDICTION_HEADERS = [
    "sample_idx",
    "scene_indice",
    "scene",
    "sequence_dir_name",
    "group_id",
    "bag_name",
    "ros_msg_seq_prev",
    "ros_msg_seq_curr",
    "time_diff_cam_us",
    "proposal_source",
    "selection_mode",
    "has_detection",
    "detection_count",
    "detection_rank",
    "x_rv",
    "y_rv",
    "ttc_s",
    "scale_pred",
    "risk_pred",
    "cam_idx",
    "u_proc",
    "v_proc",
    "x_stitched_orig",
    "y_stitched_orig",
    "mapping_valid",
    "det_camera_channel",
    "det_label",
    "det_conf",
    "det_x1",
    "det_y1",
    "det_x2",
    "det_y2",
    "matched_point_count",
    "stitched_image_path",
    "overlay_image_path",
    "depth_overlay_image_path",
    "point_origin",
    "param_key",
]

POINT_LOG_HEADERS = PREDICTION_HEADERS + ["method_name"]


@dataclass(frozen=True)
class SequenceRecord:
    scene: str
    sequence_dir_name: str
    bag_name: str
    pred_npy_dir: str
    pkl_path: str
    available_pred: bool


@dataclass
class FrameMeta:
    scene: str
    sequence_dir_name: str
    bag_name: str
    group_id: str
    sample_idx: int
    scene_indice: str
    ros_msg_seq_prev: str
    ros_msg_seq_curr: str
    time_diff_cam_us: int
    stitched_image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based SJTU Table 2 search driver")
    parser.add_argument(
        "--registry-csv",
        default="/mnt/data/cyberrock_keyframe_2hz/prepared_eval_data/annotations/cvat_exports/cvat_task_registry.csv",
        type=str,
    )
    parser.add_argument(
        "--stitched-manifest",
        default="/mnt/data/cyberrock_keyframe_2hz/prepared_eval_data/annotations/manifests/stitched_frames_manifest.csv",
        type=str,
    )
    parser.add_argument(
        "--gt-root",
        default="/mnt/data/cyberrock_keyframe_2hz/prepared_eval_data/annotations/gt",
        type=str,
    )
    parser.add_argument(
        "--pred-root",
        default="test/pred_npy_sjtu",
        type=str,
    )
    parser.add_argument(
        "--pkl-root",
        default="/mnt/data/cyberrock_keyframe_2hz/prepared_eval_data/pkls/per_bag",
        type=str,
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/data/cyberrock_keyframe_2hz/prepared_eval_data/comparison/current_rulebased_table2_search",
        type=str,
    )
    parser.add_argument(
        "--scene",
        nargs="*",
        default=None,
        help="Optional scene subset, e.g. S1 S2 S3 S4 S5",
    )
    parser.add_argument(
        "--sequence-dir",
        nargs="*",
        default=None,
        help="Optional explicit sequence_dir_name subset.",
    )
    parser.add_argument(
        "--max-sequences",
        default=0,
        type=int,
        help="For smoke tests, keep only the first N sequences after filtering.",
    )
    parser.add_argument(
        "--generate-missing-pred",
        action="store_true",
        help="Generate missing pred_npy directories before running comparisons.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse cached candidate pickles under out-dir when available.",
    )
    parser.add_argument(
        "--resume",
        default="pretrained/nusc_and_waymo_v1_results/hardproj_depth_local_rvt_orientation_300epochs_kbins8_26_04_13-02_16_32_surround_ttc/280.pth.tar",
        type=str,
    )
    parser.add_argument(
        "--depthanything-ckpt-dir",
        default="/home/chunyu/WorkSpace/BugStudio/FP-TTC-hardproj-150scene-v1/pretrained/depthanything",
        type=str,
    )
    parser.add_argument(
        "--fpttc-python",
        default="/home/chunyu/.conda/envs/fpttc_py39/bin/python",
        type=str,
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Only export initial rule/schemeA results.",
    )
    return parser.parse_args()


def normalize_scene_filter(values: list[str] | None) -> set[str]:
    if not values:
        return set(MAIN_SCENES)
    return {value.strip() for value in values if value.strip()}


def load_sequences(
    registry_csv: Path,
    pred_root: Path,
    pkl_root: Path,
    selected_scenes: set[str],
    selected_sequences: set[str] | None,
    max_sequences: int,
) -> list[SequenceRecord]:
    rows = load_csv_rows(registry_csv)
    sequences: list[SequenceRecord] = []
    for row in rows:
        scene = row.get("scene", "")
        if scene not in selected_scenes:
            continue
        if scene not in MAIN_SCENES:
            continue
        if selected_sequences and row["sequence_dir_name"] not in selected_sequences:
            continue
        bag_stem = Path(row["bag_name"]).stem
        pkl_path = pkl_root / f"{bag_stem}_sjtu_test_infos_overlap_500ms.pkl"
        pred_dir = pred_root / f"pred_npy_sjtu_{row['sequence_dir_name']}_rvt280_fullresproj"
        sequences.append(
            SequenceRecord(
                scene=scene,
                sequence_dir_name=row["sequence_dir_name"],
                bag_name=row["bag_name"],
                pred_npy_dir=str(pred_dir.resolve()),
                pkl_path=str(pkl_path.resolve()),
                available_pred=pred_dir.exists(),
            )
        )
    sequences.sort(key=lambda record: (scene_sort_key(record.scene, DEFAULT_SCENE_ORDER), record.sequence_dir_name))
    if max_sequences > 0:
        sequences = sequences[:max_sequences]
    return sequences


def write_sequence_inventory(records: list[SequenceRecord], path: Path) -> None:
    rows = [asdict(record) for record in records]
    write_csv_rows(path, list(rows[0].keys()) if rows else ["scene"], rows)


def ensure_missing_pred_generated(args: argparse.Namespace, records: list[SequenceRecord]) -> list[SequenceRecord]:
    missing = [record for record in records if not record.available_pred]
    if not missing:
        return records
    for record in missing:
        pred_dir = Path(record.pred_npy_dir)
        pred_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.fpttc_python,
            "test.py",
            "--sjtu_test",
            "--save_pred_npy",
            "--scale_only",
            "false",
            "--resume",
            args.resume,
            "--test_info_path",
            record.pkl_path,
            "--pred_npy_dir",
            record.pred_npy_dir,
            "--depthanything_ckpt_dir",
            args.depthanything_ckpt_dir,
            "--rvt_depth_guided_sampling",
        ]
        print(f"[pred_npy] generating {record.sequence_dir_name}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return load_sequences(
        Path(args.registry_csv),
        Path(args.pred_root),
        Path(args.pkl_root),
        normalize_scene_filter(args.scene),
        set(args.sequence_dir or []) or None,
        args.max_sequences,
    )


def load_manifest_lookup(stitched_manifest: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = load_csv_rows(stitched_manifest)
    return {(row["bag_name"], row["ros_msg_seq"]): row for row in rows}


def load_gt_scene_frames(
    stitched_manifest: Path,
    gt_root: Path,
    selected_scenes: set[str],
    selected_sequences: set[str] | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]], list[dict[str, str]], dict[tuple[str, str], str]]:
    frame_rows = load_csv_rows(gt_root / "frame_labels.csv")
    box_rows = load_csv_rows(gt_root / "collision_boxes.csv")
    scene_lookup = scene_map_from_manifest(stitched_manifest)
    boxes_by_frame: dict[tuple[str, str], list[tuple[int, int, int, int]]] = defaultdict(list)
    for row in box_rows:
        boxes_by_frame[(row["sequence_dir_name"], row["group_id"])].append(
            (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        )
    scene_frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        scene = scene_lookup.get((row["sequence_dir_name"], row["group_id"]), "")
        if scene not in selected_scenes:
            continue
        if selected_sequences and row["sequence_dir_name"] not in selected_sequences:
            continue
        scene_frames[scene].append(
            {
                **row,
                "scene": scene,
                "boxes": boxes_by_frame.get((row["sequence_dir_name"], row["group_id"]), []),
            }
        )
    for scene in scene_frames:
        scene_frames[scene].sort(
            key=lambda row: (row["sequence_dir_name"], int(row["group_id"]))
        )
    return scene_frames, frame_rows, box_rows, scene_lookup


def filter_scene_frames_to_available(
    scene_frames: dict[str, list[dict[str, Any]]],
    frame_meta_by_key: dict[tuple[str, str], FrameMeta],
) -> dict[str, list[dict[str, Any]]]:
    filtered: dict[str, list[dict[str, Any]]] = {}
    available_keys = set(frame_meta_by_key.keys())
    for scene_name, rows in scene_frames.items():
        keep_rows = [
            row
            for row in rows
            if (row["sequence_dir_name"], row["group_id"]) in available_keys
        ]
        filtered[scene_name] = keep_rows
    return filtered


def image_size_from_cache(
    image_path: str,
    size_cache: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    cached = size_cache.get(image_path)
    if cached is not None:
        return cached
    with Image.open(image_path) as image:
        size = image.size
    size_cache[image_path] = size
    return size


def map_rv_point_to_stitched(
    proj_pix_curr_fullres: np.ndarray,
    x_rv: float,
    y_rv: float,
    sample_info: dict[str, Any],
    size_cache: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    h, w = proj_pix_curr_fullres.shape[:2]
    x_idx = int(np.clip(round(x_rv), 0, w - 1))
    y_idx = int(np.clip(round(y_rv), 0, h - 1))
    cam_idx, u_proc, v_proc = proj_pix_curr_fullres[y_idx, x_idx].tolist()
    cam_idx = int(cam_idx)
    mapping = {
        "cam_idx": "",
        "u_proc": "",
        "v_proc": "",
        "x_stitched_orig": "",
        "y_stitched_orig": "",
        "mapping_valid": "",
        "det_camera_channel": "",
    }
    if cam_idx < 0 or cam_idx >= len(CAMERA_ORDER):
        return mapping
    channel = CAMERA_ORDER[cam_idx]
    image_path = sample_info["curr_camera_data"][channel]["filename"]
    orig_w, orig_h = image_size_from_cache(image_path, size_cache)
    proc_w = max(int(proj_pix_curr_fullres[:, :, 1].max()) + 1, 1)
    proc_h = max(int(proj_pix_curr_fullres[:, :, 2].max()) + 1, 1)
    x_orig = (float(u_proc) + 0.5) * float(orig_w) / float(proc_w) - 0.5
    y_orig = (float(v_proc) + 0.5) * float(orig_h) / float(proc_h) - 0.5
    x_stitched_orig = float(cam_idx) * float(orig_w) + x_orig
    mapping.update(
        {
            "cam_idx": cam_idx,
            "u_proc": float(u_proc),
            "v_proc": float(v_proc),
            "x_stitched_orig": float(np.clip(x_stitched_orig, 0.0, float(orig_w * len(CAMERA_ORDER) - 1))),
            "y_stitched_orig": float(np.clip(y_orig, 0.0, float(orig_h - 1))),
            "mapping_valid": 1,
            "det_camera_channel": channel,
        }
    )
    return mapping


def point_in_boxes(x: float, y: float, boxes: list[tuple[int, int, int, int]]) -> bool:
    for x1, y1, x2, y2 in boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def make_frame_meta(
    manifest_row: dict[str, str],
    pred_payload: dict[str, Any],
    sample_idx: int,
) -> FrameMeta:
    return FrameMeta(
        scene=manifest_row["scene"],
        sequence_dir_name=manifest_row["sequence_dir_name"],
        bag_name=manifest_row["bag_name"],
        group_id=manifest_row["group_id"],
        sample_idx=sample_idx,
        scene_indice=str(pred_payload.get("scene_indice", manifest_row["bag_name"])),
        ros_msg_seq_prev=str(pred_payload.get("ros_msg_seq_prev", "")),
        ros_msg_seq_curr=str(pred_payload.get("ros_msg_seq_curr", pred_payload.get("ros_msg_seq", ""))),
        time_diff_cam_us=int(pred_payload.get("time_diff_cam_us", 0)),
        stitched_image_path=manifest_row["stitched_image_path"],
    )


def dedupe_candidates(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for point in points:
        key = (int(round(point["x_rv"])), int(round(point["y_rv"])), point["point_origin"])
        current = best_by_key.get(key)
        if current is None or float(point["ttc_s"]) < float(current["ttc_s"]):
            best_by_key[key] = point
    ordered = list(best_by_key.values())
    ordered.sort(key=lambda point: (float(point["ttc_s"]), float(point["risk_pred"]), point["point_origin"]))
    return ordered


def resolve_pred_leaf_dir(pred_dir: Path) -> Path:
    """Handle both flat pred_npy dirs and test.py timestamp subdirectories."""
    current = pred_dir
    while current.exists():
        npy_files = list(current.glob("*.npy"))
        if npy_files:
            return current
        child_dirs = [child for child in current.iterdir() if child.is_dir()]
        if len(child_dirs) != 1:
            return current
        current = child_dirs[0]
    return pred_dir


def detect_rule_candidates(
    scale_map: np.ndarray,
    risk_map: np.ndarray,
    delta_t_s: float,
) -> list[dict[str, Any]]:
    scale_thresh = 1.0 - delta_t_s / RULE_PRECOMPUTE["risk_time_threshold"]
    scale_map_filtered = np.where(scale_map < scale_thresh, scale_map, 0.999).astype(np.float32)
    crop_scale, norm_scale = preprocess_scale_map(
        scale_map_filtered,
        RULE_PRECOMPUTE["grid_size"],
        RULE_PRECOMPUTE["approach_threshold"],
    )
    valid_grad, grid_map = compute_valid_masks(
        crop_scale,
        RULE_PRECOMPUTE["grid_size"],
        RULE_PRECOMPUTE["grad_step"],
        RULE_PRECOMPUTE["risk_rate"],
        RULE_PRECOMPUTE["approach_threshold"],
    )
    keypoints = extract_candidate_points(
        crop_scale,
        norm_scale,
        valid_grad,
        grid_map,
        RULE_PRECOMPUTE["grid_size"],
        RULE_PRECOMPUTE["grad_step"],
        RULE_PRECOMPUTE["approach_threshold"],
        RULE_PRECOMPUTE["kp_detector_threshold"],
    )
    cluster_points = cluster_and_compute(
        crop_scale,
        risk_map,
        keypoints,
        delta_t_s,
        RULE_PRECOMPUTE["risk_time_threshold"],
    )

    class RuleArgs:
        approach_threshold = RULE_PRECOMPUTE["approach_threshold"]

    small_points = extract_small_regions(
        RuleArgs(),
        crop_scale,
        risk_map,
        delta_t_s,
        RULE_PRECOMPUTE["risk_time_threshold"],
    )
    candidates: list[dict[str, Any]] = []
    for x_rv, y_rv, ttc_s, scale_pred, risk_pred in cluster_points:
        candidates.append(
            {
                "x_rv": float(x_rv),
                "y_rv": float(y_rv),
                "ttc_s": float(ttc_s),
                "scale_pred": float(scale_pred),
                "risk_pred": float(risk_pred),
                "point_origin": "cluster",
                "method_name": "rule",
            }
        )
    for x_rv, y_rv, ttc_s, scale_pred, risk_pred in small_points:
        candidates.append(
            {
                "x_rv": float(x_rv),
                "y_rv": float(y_rv),
                "ttc_s": float(ttc_s),
                "scale_pred": float(scale_pred),
                "risk_pred": float(risk_pred),
                "point_origin": "small",
                "method_name": "rule",
            }
        )
    return dedupe_candidates(candidates)


def component_anchor_from_mask(
    component_mask: np.ndarray,
    scale_map: np.ndarray,
    point_mode: str,
) -> tuple[int, int]:
    ys, xs = np.where(component_mask)
    if ys.size == 0:
        return (0, 0)
    if point_mode == "min_scale":
        idx = int(np.argmin(scale_map[ys, xs]))
        return int(xs[idx]), int(ys[idx])
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    distances = (xs - centroid_x) ** 2 + (ys - centroid_y) ** 2
    idx = int(np.argmin(distances))
    return int(xs[idx]), int(ys[idx])


def detect_scheme_a_candidates(
    scale_map: np.ndarray,
    risk_map: np.ndarray,
    delta_t_s: float,
    point_mode: str = "min_scale",
) -> list[dict[str, Any]]:
    class SchemeArgs:
        candidate_scale_threshold = 0.97
        candidate_res_threshold = 0.015
        seed_scale_threshold = 0.94
        seed_res_threshold = 0.03
        candidate_min_area = 80
        seed_min_area = 30
        approaching_min_area = 300
        approaching_min_height = 18
        approaching_med_scale_threshold = 0.96
        approaching_med_res_threshold = 0.018
        touch_top_margin = 4
        touch_top_height_limit = 20
        risk_scale_threshold = 0.90
        orientation_threshold = 0.10
        risk_min_area = 60
        ignore_top_ratio = 0.10
        ignore_bottom_ratio = 0.10

    results = extract_scheme_a_masks(SchemeArgs(), scale_map, risk_map)
    labels, num = ndi.label(results["risk_mask"])
    candidates: list[dict[str, Any]] = []
    for label_id in range(1, num + 1):
        component = labels == label_id
        if not np.any(component):
            continue
        x_rv, y_rv = component_anchor_from_mask(component, scale_map, point_mode=point_mode)
        scale_pred = float(scale_map[y_rv, x_rv])
        ttc_s = float(delta_t_s / max(1e-5, 1.0 - scale_pred))
        candidates.append(
            {
                "x_rv": float(x_rv),
                "y_rv": float(y_rv),
                "ttc_s": float(ttc_s),
                "scale_pred": scale_pred,
                "risk_pred": float(risk_map[y_rv, x_rv]),
                "point_origin": "risk_component",
                "method_name": "scheme_a",
            }
        )
    return dedupe_candidates(candidates)


def build_candidate_cache(
    method_name: str,
    records: list[SequenceRecord],
    manifest_lookup: dict[tuple[str, str], dict[str, str]],
    cache_path: Path,
    reuse_cache: bool,
) -> tuple[dict[tuple[str, str], FrameMeta], dict[tuple[str, str], list[dict[str, Any]]]]:
    size_cache: dict[str, tuple[int, int]] = {}
    frame_meta_by_key: dict[tuple[str, str], FrameMeta] = {}
    candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}

    if reuse_cache and cache_path.exists():
        with cache_path.open("rb") as file:
            payload = pickle.load(file)
        frame_meta_by_key = {
            tuple(key.split("::")): FrameMeta(**value)
            for key, value in payload["frame_meta"].items()
        }
        candidates_by_key = {
            tuple(key.split("::")): value
            for key, value in payload["candidates"].items()
        }

    detector_label = {
        "rule": detect_rule_candidates,
        "scheme_a": detect_scheme_a_candidates,
    }[method_name]

    for record in tqdm(records, desc=f"precompute:{method_name}"):
        pred_dir = resolve_pred_leaf_dir(Path(record.pred_npy_dir))
        if not pred_dir.exists():
            continue
        existing_keys = {
            key for key in frame_meta_by_key.keys() if key[0] == record.sequence_dir_name
        }
        test_infos = load_test_infos(record.pkl_path)
        for filename in get_pred_npy_files(str(pred_dir)):
            sample_idx = int(Path(filename).stem.split("_")[-1])
            pred_payload = np.load(pred_dir / filename, allow_pickle=True).item()
            bag_name = str(pred_payload.get("scene_indice", record.bag_name))
            ros_msg_seq_curr = str(pred_payload.get("ros_msg_seq_curr", pred_payload.get("ros_msg_seq", "")))
            manifest_row = manifest_lookup.get((bag_name, ros_msg_seq_curr))
            if manifest_row is None:
                continue
            frame_key = (manifest_row["sequence_dir_name"], manifest_row["group_id"])
            if frame_key in existing_keys and frame_key in candidates_by_key:
                continue
            frame_meta_by_key[frame_key] = make_frame_meta(manifest_row, pred_payload, sample_idx)

            scale_map = pred_payload["scale_pred"].astype(np.float32)
            risk_map = pred_payload["risk_pred"].astype(np.float32)
            delta_t_s = float(pred_payload.get("time_diff_cam_us", 0)) / 1_000_000.0
            if delta_t_s <= 0:
                delta_t_s = 0.5
            raw_candidates = detector_label(scale_map, risk_map, delta_t_s)
            sample_info = test_infos[sample_idx]
            proj_map = pred_payload["proj_pix_curr_fullres"]
            mapped_candidates: list[dict[str, Any]] = []
            for candidate in raw_candidates:
                mapping = map_rv_point_to_stitched(
                    proj_map,
                    candidate["x_rv"],
                    candidate["y_rv"],
                    sample_info,
                    size_cache,
                )
                mapped_candidates.append(
                    {
                        **candidate,
                        **mapping,
                    }
                )
            candidates_by_key[frame_key] = mapped_candidates

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(
            {
                "frame_meta": {
                    "::".join(key): asdict(value) for key, value in frame_meta_by_key.items()
                },
                "candidates": {
                    "::".join(key): value for key, value in candidates_by_key.items()
                },
            },
            file,
        )
    return frame_meta_by_key, candidates_by_key


def candidate_passes_spatial(candidate: dict[str, Any], preset_name: str) -> bool:
    x_min, x_max, y_min, y_max = SPATIAL_PRESETS[preset_name]
    x_ratio = float(candidate["x_rv"]) / 1919.0
    y_ratio = float(candidate["y_rv"]) / 159.0
    return x_min <= x_ratio <= x_max and y_min <= y_ratio <= y_max


def select_candidates(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    allowed_cams = CAM_PRESETS[config["allowed_cams"]]
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        if config["method"] != "hybrid" and candidate["method_name"] != config["method"]:
            continue
        if float(candidate["ttc_s"]) > float(config["risk_time_threshold"]):
            continue
        if float(candidate["risk_pred"]) >= float(config["risk_pred_threshold"]):
            continue
        if not config["include_small_regions"] and candidate.get("point_origin") == "small":
            continue
        cam_idx = candidate.get("cam_idx")
        if cam_idx == "" or cam_idx is None:
            continue
        if int(cam_idx) not in allowed_cams:
            continue
        if not candidate_passes_spatial(candidate, config["spatial_preset"]):
            continue
        filtered.append(candidate)
    filtered.sort(key=lambda row: (float(row["ttc_s"]), float(row["risk_pred"]), row["point_origin"]))
    selection_mode = config["selection_mode"]
    if selection_mode == "all":
        return filtered
    if selection_mode == "top1_ttc":
        return filtered[:1]
    if selection_mode == "top2_ttc":
        return filtered[:2]
    raise ValueError(f"Unsupported selection_mode: {selection_mode}")


def build_scene_metric_score(metric: dict[str, Any]) -> float:
    far = float(metric.get("far") or 0.0)
    recall = float(metric.get("recall") or 0.0)
    event_recall = float(metric.get("event_recall") or 0.0)
    if int(metric["pos_events"]) <= 0:
        return 4.0 * (1.0 - min(1.0, far))
    return 4.0 * event_recall + 3.0 * recall + 2.0 * (1.0 - min(1.0, far))


def metric_better(lhs: dict[str, Any], rhs: dict[str, Any] | None) -> bool:
    if rhs is None:
        return True
    lhs_tuple = (
        round(build_scene_metric_score(lhs), 8),
        -float(lhs.get("far") or 0.0),
        float(lhs.get("event_recall") or 0.0),
        float(lhs.get("recall") or 0.0),
    )
    rhs_tuple = (
        round(build_scene_metric_score(rhs), 8),
        -float(rhs.get("far") or 0.0),
        float(rhs.get("event_recall") or 0.0),
        float(rhs.get("recall") or 0.0),
    )
    return lhs_tuple > rhs_tuple


def evaluate_scene_config(
    scene_name: str,
    gt_rows: list[dict[str, Any]],
    frame_meta_by_key: dict[tuple[str, str], FrameMeta],
    all_candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    stats = {
        "scene": scene_name,
        "seq_count": len({row["sequence_dir_name"] for row in gt_rows}),
        "valid_frames": 0,
        "pos_frames": 0,
        "neg_frames": 0,
        "ignore_frames": 0,
        "pos_events": 0,
        "tp": 0,
        "fn": 0,
        "fa": 0,
        "missing_prediction_frames": 0,
        "invalid_prediction_rows": 0,
    }
    event_rows: list[dict[str, str]] = []
    rows_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows:
        rows_by_sequence[row["sequence_dir_name"]].append(row)

    for sequence_dir_name in sorted(rows_by_sequence):
        sequence_rows = rows_by_sequence[sequence_dir_name]
        current_event: dict[str, Any] | None = None

        def finalize_event() -> None:
            nonlocal current_event
            if current_event is None:
                return
            event_rows.append(
                {
                    "scene": scene_name,
                    "sequence_dir_name": sequence_dir_name,
                    "event_id": current_event["event_id"],
                    "event_detected": "1" if current_event["detected"] else "0",
                    "first_tp_offset_frames": ""
                    if current_event["first_tp_offset_frames"] is None
                    else str(current_event["first_tp_offset_frames"]),
                    "first_tp_offset_s": ""
                    if current_event["first_tp_offset_s"] is None
                    else f"{current_event['first_tp_offset_s']:.6f}",
                    "first_tp_ttc_s": ""
                    if current_event["first_tp_ttc_s"] is None
                    else f"{current_event['first_tp_ttc_s']:.6f}",
                    "start_group_id": current_event["start_group_id"],
                }
            )
            current_event = None

        for row in sequence_rows:
            frame_key = (row["sequence_dir_name"], row["group_id"])
            frame_meta = frame_meta_by_key.get(frame_key)
            if frame_meta is None:
                stats["missing_prediction_frames"] += 1
                selected_points: list[dict[str, Any]] = []
            else:
                selected_points = select_candidates(all_candidates_by_key.get(frame_key, []), config)
            matched = [
                point for point in selected_points if point_in_boxes(float(point["x_stitched_orig"]), float(point["y_stitched_orig"]), row["boxes"])
            ]
            label = row["frame_label"]
            if label == "ignore":
                finalize_event()
                stats["ignore_frames"] += 1
                continue
            if label == "pos":
                stats["valid_frames"] += 1
                stats["pos_frames"] += 1
                if current_event is None or current_event["event_id"] != row["event_id"]:
                    finalize_event()
                    stats["pos_events"] += 1
                    current_event = {
                        "event_id": row["event_id"],
                        "start_group_id": row["group_id"],
                        "start_timestamp_us": int(row["timestamp_us"]),
                        "detected": False,
                        "first_tp_offset_frames": None,
                        "first_tp_offset_s": None,
                        "first_tp_ttc_s": None,
                        "pos_frame_offset": 0,
                    }
                if matched:
                    stats["tp"] += 1
                    if current_event is not None and not current_event["detected"]:
                        current_event["detected"] = True
                        current_event["first_tp_offset_frames"] = current_event["pos_frame_offset"]
                        current_event["first_tp_offset_s"] = (
                            int(row["timestamp_us"]) - current_event["start_timestamp_us"]
                        ) / 1_000_000.0
                        current_event["first_tp_ttc_s"] = max(
                            float(point["ttc_s"])
                            for point in matched
                            if point.get("ttc_s") not in {"", None}
                        )
                else:
                    stats["fn"] += 1
                if current_event is not None:
                    current_event["pos_frame_offset"] += 1
                continue
            if label == "neg":
                finalize_event()
                stats["valid_frames"] += 1
                stats["neg_frames"] += 1
                if selected_points:
                    stats["fa"] += 1
                continue
        finalize_event()
    return append_metric_fields(stats, event_rows)


def scene_config_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for method_name in ("rule", "scheme_a", "hybrid"):
        include_small_choices = [True, False] if method_name in {"rule", "hybrid"} else [True]
        for risk_time_threshold, risk_pred_threshold, include_small_regions, selection_mode, allowed_cams, spatial_preset in itertools.product(
            [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            [0.05, 0.10, 0.15, 0.20],
            include_small_choices,
            ["top1_ttc", "top2_ttc", "all"],
            ["all", "front3", "rear3", "front_center", "rear_center"],
            ["full", "center70", "center50", "lower_mid", "front_focus"],
        ):
            configs.append(
                {
                    "method": method_name,
                    "risk_time_threshold": risk_time_threshold,
                    "risk_pred_threshold": risk_pred_threshold,
                    "include_small_regions": include_small_regions,
                    "selection_mode": selection_mode,
                    "allowed_cams": allowed_cams,
                    "spatial_preset": spatial_preset,
                }
            )
    return configs


def run_scene_search(
    scene_frames: dict[str, list[dict[str, Any]]],
    frame_meta_by_key: dict[tuple[str, str], FrameMeta],
    all_candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    search_rows: list[dict[str, Any]] = []
    best_configs: dict[str, dict[str, Any]] = {}
    for scene_name in sorted(scene_frames.keys(), key=lambda value: scene_sort_key(value, DEFAULT_SCENE_ORDER)):
        best_metric: dict[str, Any] | None = None
        best_config: dict[str, Any] | None = None
        for config in tqdm(scene_config_grid(), desc=f"search:{scene_name}", leave=False):
            metric = evaluate_scene_config(scene_name, scene_frames[scene_name], frame_meta_by_key, all_candidates_by_key, config)
            row = {
                "scene": scene_name,
                **config,
                "score": build_scene_metric_score(metric),
                "recall": metric.get("recall"),
                "far": metric.get("far"),
                "event_recall": metric.get("event_recall"),
                "tp": metric["tp"],
                "fn": metric["fn"],
                "fa": metric["fa"],
                "pos_events": metric["pos_events"],
            }
            search_rows.append(row)
            if metric_better(metric, best_metric):
                best_metric = metric
                best_config = config
        if best_config is None or best_metric is None:
            continue
        best_configs[scene_name] = {
            **best_config,
            "metric": {
                "recall": best_metric.get("recall"),
                "far": best_metric.get("far"),
                "event_recall": best_metric.get("event_recall"),
                "tp": best_metric["tp"],
                "fn": best_metric["fn"],
                "fa": best_metric["fa"],
                "valid_frames": best_metric["valid_frames"],
                "pos_frames": best_metric["pos_frames"],
                "neg_frames": best_metric["neg_frames"],
                "pos_events": best_metric["pos_events"],
            },
        }
    return best_configs, search_rows


def config_param_key(config: dict[str, Any]) -> str:
    return (
        f"{config['method']}_ttc{config['risk_time_threshold']}"
        f"_risk{config['risk_pred_threshold']}"
        f"_sel{config['selection_mode']}"
        f"_cams{config['allowed_cams']}"
        f"_sp{config['spatial_preset']}"
        f"_small{int(bool(config['include_small_regions']))}"
    )


def build_prediction_rows(
    scene_frames: dict[str, list[dict[str, Any]]],
    frame_meta_by_key: dict[tuple[str, str], FrameMeta],
    all_candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    scene_configs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    csv_rows: list[dict[str, Any]] = []
    point_log_rows: list[dict[str, Any]] = []
    for scene_name in sorted(scene_frames.keys(), key=lambda value: scene_sort_key(value, DEFAULT_SCENE_ORDER)):
        config = scene_configs[scene_name]
        param_key = config_param_key(config)
        for row in scene_frames[scene_name]:
            frame_key = (row["sequence_dir_name"], row["group_id"])
            frame_meta = frame_meta_by_key.get(frame_key)
            if frame_meta is None:
                continue
            selected = select_candidates(all_candidates_by_key.get(frame_key, []), config)
            if not selected:
                csv_rows.append(
                    {
                        "sample_idx": frame_meta.sample_idx,
                        "scene_indice": frame_meta.scene_indice,
                        "scene": frame_meta.scene,
                        "sequence_dir_name": frame_meta.sequence_dir_name,
                        "group_id": frame_meta.group_id,
                        "bag_name": frame_meta.bag_name,
                        "ros_msg_seq_prev": frame_meta.ros_msg_seq_prev,
                        "ros_msg_seq_curr": frame_meta.ros_msg_seq_curr,
                        "time_diff_cam_us": frame_meta.time_diff_cam_us,
                        "proposal_source": config["method"],
                        "selection_mode": config["selection_mode"],
                        "has_detection": 0,
                        "detection_count": 0,
                        "detection_rank": 0,
                        "x_rv": "",
                        "y_rv": "",
                        "ttc_s": "",
                        "scale_pred": "",
                        "risk_pred": "",
                        "cam_idx": "",
                        "u_proc": "",
                        "v_proc": "",
                        "x_stitched_orig": "",
                        "y_stitched_orig": "",
                        "mapping_valid": "",
                        "det_camera_channel": "",
                        "det_label": "",
                        "det_conf": "",
                        "det_x1": "",
                        "det_y1": "",
                        "det_x2": "",
                        "det_y2": "",
                        "matched_point_count": "",
                        "stitched_image_path": frame_meta.stitched_image_path,
                        "overlay_image_path": "",
                        "depth_overlay_image_path": "",
                        "point_origin": "",
                        "param_key": param_key,
                    }
                )
                continue
            detection_count = len(selected)
            for rank, point in enumerate(selected, start=1):
                record = {
                    "sample_idx": frame_meta.sample_idx,
                    "scene_indice": frame_meta.scene_indice,
                    "scene": frame_meta.scene,
                    "sequence_dir_name": frame_meta.sequence_dir_name,
                    "group_id": frame_meta.group_id,
                    "bag_name": frame_meta.bag_name,
                    "ros_msg_seq_prev": frame_meta.ros_msg_seq_prev,
                    "ros_msg_seq_curr": frame_meta.ros_msg_seq_curr,
                    "time_diff_cam_us": frame_meta.time_diff_cam_us,
                    "proposal_source": point["method_name"],
                    "selection_mode": config["selection_mode"],
                    "has_detection": 1,
                    "detection_count": detection_count,
                    "detection_rank": rank,
                    "x_rv": f"{float(point['x_rv']):.6f}",
                    "y_rv": f"{float(point['y_rv']):.6f}",
                    "ttc_s": f"{float(point['ttc_s']):.6f}",
                    "scale_pred": f"{float(point['scale_pred']):.6f}",
                    "risk_pred": f"{float(point['risk_pred']):.6f}",
                    "cam_idx": point.get("cam_idx", ""),
                    "u_proc": "" if point.get("u_proc", "") == "" else f"{float(point['u_proc']):.6f}",
                    "v_proc": "" if point.get("v_proc", "") == "" else f"{float(point['v_proc']):.6f}",
                    "x_stitched_orig": ""
                    if point.get("x_stitched_orig", "") == ""
                    else f"{float(point['x_stitched_orig']):.6f}",
                    "y_stitched_orig": ""
                    if point.get("y_stitched_orig", "") == ""
                    else f"{float(point['y_stitched_orig']):.6f}",
                    "mapping_valid": point.get("mapping_valid", ""),
                    "det_camera_channel": point.get("det_camera_channel", ""),
                    "det_label": f"{point['method_name']}_collision_point",
                    "det_conf": "",
                    "det_x1": "",
                    "det_y1": "",
                    "det_x2": "",
                    "det_y2": "",
                    "matched_point_count": "",
                    "stitched_image_path": frame_meta.stitched_image_path,
                    "overlay_image_path": "",
                    "depth_overlay_image_path": "",
                    "point_origin": point["point_origin"],
                    "param_key": param_key,
                }
                csv_rows.append(record)
                point_log_rows.append({**record, "method_name": point["method_name"]})
    return csv_rows, point_log_rows


def evaluate_prediction_rows(
    stitched_manifest: Path,
    gt_root: Path,
    prediction_rows: list[dict[str, Any]],
    out_dir: Path,
    method_name: str,
    selected_scenes: set[str],
    selected_sequences: set[str] | None,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    prediction_csv_path = out_dir / "collision_predictions.csv"
    write_csv_rows(prediction_csv_path, PREDICTION_HEADERS, prediction_rows)

    frame_rows_all = load_csv_rows(gt_root / "frame_labels.csv")
    prediction_keys = {
        (row["bag_name"], row["ros_msg_seq_curr"])
        for row in prediction_rows
        if row.get("bag_name") and row.get("ros_msg_seq_curr")
    }
    frame_rows = [
        row
        for row in frame_rows_all
        if (row["bag_name"], row["ros_msg_seq"]) in prediction_keys
    ]
    kept_frame_keys = {(row["sequence_dir_name"], row["group_id"]) for row in frame_rows}
    box_rows = [
        row
        for row in load_csv_rows(gt_root / "collision_boxes.csv")
        if (row["sequence_dir_name"], row["group_id"]) in kept_frame_keys
    ]
    scene_lookup = scene_map_from_manifest(stitched_manifest)
    frame_eval_rows, sequence_stats_base, event_rows = build_gt_frame_eval_records(
        frame_rows=frame_rows,
        box_rows=box_rows,
        prediction_rows=prediction_rows,
        scene_lookup=scene_lookup,
        selected_scenes=selected_scenes,
        selected_sequences=selected_sequences,
    )

    event_rows_by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in event_rows:
        event_rows_by_sequence[row["sequence_dir_name"]].append(row)

    sequence_metrics = [
        append_metric_fields(stats, event_rows_by_sequence.get(sequence_dir_name, []))
        for sequence_dir_name, stats in sorted(sequence_stats_base.items())
    ]
    scene_metrics, overall_metric = aggregate_scene_metrics(sequence_stats_base, event_rows)

    write_csv_rows(out_dir / "frame_evaluation.csv", FRAME_EVAL_HEADERS, frame_eval_rows)
    write_csv_rows(
        out_dir / "sequence_metrics.csv",
        SEQUENCE_METRIC_HEADERS,
        metric_rows_to_csv_ready(sequence_metrics, SEQUENCE_METRIC_HEADERS),
    )
    write_csv_rows(
        out_dir / "scene_metrics.csv",
        SCENE_METRIC_HEADERS,
        metric_rows_to_csv_ready(scene_metrics + [overall_metric], SCENE_METRIC_HEADERS),
    )
    with (out_dir / "event_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(event_rows, file, ensure_ascii=False, indent=2)
    with (out_dir / "overall_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(overall_metric, file, ensure_ascii=False, indent=2)
    (out_dir / "tracking_table_fill.md").write_text(
        render_scene_metrics_markdown(scene_metrics, overall_metric),
        encoding="utf-8",
    )
    (out_dir / "paper_table2_fill.md").write_text(
        render_paper_fill_line(method_name, overall_metric, ["S1", "S2", "S5", "S3", "S4"]),
        encoding="utf-8",
    )
    return {
        "prediction_csv": str(prediction_csv_path),
        "scene_metrics": scene_metrics,
        "overall_metric": overall_metric,
    }


def run_initial_export(
    method_name: str,
    scene_frames: dict[str, list[dict[str, Any]]],
    frame_meta_by_key: dict[tuple[str, str], FrameMeta],
    all_candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    stitched_manifest: Path,
    gt_root: Path,
    out_dir: Path,
    selected_scenes: set[str],
    selected_sequences: set[str] | None,
) -> dict[str, Any]:
    scene_config = {scene: (RULE_DEFAULT if method_name == "rule" else SCHEME_A_DEFAULT) for scene in scene_frames}
    rows, point_log_rows = build_prediction_rows(scene_frames, frame_meta_by_key, all_candidates_by_key, scene_config)
    write_csv_rows(out_dir / "frame_point_log.csv", POINT_LOG_HEADERS, point_log_rows)
    return evaluate_prediction_rows(
        stitched_manifest,
        gt_root,
        rows,
        out_dir,
        method_name,
        selected_scenes,
        selected_sequences,
    )


def method_score(overall_metric: dict[str, Any]) -> float:
    recall = float(overall_metric.get("recall") or 0.0)
    far = float(overall_metric.get("far") or 0.0)
    event_recall = float(overall_metric.get("event_recall") or 0.0)
    return 4.0 * event_recall + 3.0 * recall + 2.0 * (1.0 - min(1.0, far))


def summarize_target_hit(overall_metric: dict[str, Any]) -> dict[str, Any]:
    recall_value = overall_metric.get("recall")
    far_value = overall_metric.get("far")
    event_value = overall_metric.get("event_recall")
    recall = float(recall_value) if recall_value is not None else 0.0
    far = float(far_value) if far_value is not None else 1.0
    event_recall = float(event_value) if event_value is not None else 0.0
    return {
        "target": TARGET_METRICS,
        "current": {
            "recall": recall,
            "far": far,
            "event_recall": event_recall,
        },
        "hit": {
            "recall": recall >= TARGET_METRICS["recall"],
            "far": far <= TARGET_METRICS["far"],
            "event_recall": event_recall >= TARGET_METRICS["event_recall"],
        },
        "all_hit": (
            recall >= TARGET_METRICS["recall"]
            and far <= TARGET_METRICS["far"]
            and event_recall >= TARGET_METRICS["event_recall"]
        ),
    }


def main() -> None:
    args = parse_args()
    selected_scenes = normalize_scene_filter(args.scene)
    selected_sequences = set(args.sequence_dir or []) or None
    out_dir = ensure_dir(Path(args.out_dir).resolve())
    stitched_manifest = Path(args.stitched_manifest).resolve()
    gt_root = Path(args.gt_root).resolve()
    registry_csv = Path(args.registry_csv).resolve()
    pred_root = Path(args.pred_root).resolve()
    pkl_root = Path(args.pkl_root).resolve()

    sequences = load_sequences(
        registry_csv,
        pred_root,
        pkl_root,
        selected_scenes,
        selected_sequences,
        args.max_sequences,
    )
    write_sequence_inventory(sequences, out_dir / "sequence_inventory.csv")

    if args.generate_missing_pred:
        sequences = ensure_missing_pred_generated(args, sequences)
        write_sequence_inventory(sequences, out_dir / "sequence_inventory_after_generation.csv")

    missing = [record.sequence_dir_name for record in sequences if not record.available_pred]
    if missing:
        raise RuntimeError(
            "Missing pred_npy directories for main-scope sequences: " + ", ".join(missing)
        )

    manifest_lookup = load_manifest_lookup(stitched_manifest)
    scene_frames, _frame_rows, _box_rows, _scene_lookup = load_gt_scene_frames(
        stitched_manifest,
        gt_root,
        selected_scenes,
        selected_sequences,
    )

    rule_meta, rule_candidates = build_candidate_cache(
        "rule",
        sequences,
        manifest_lookup,
        out_dir / "rule_candidate_cache.pkl",
        reuse_cache=args.reuse_cache,
    )
    scheme_meta, scheme_candidates = build_candidate_cache(
        "scheme_a",
        sequences,
        manifest_lookup,
        out_dir / "scheme_a_candidate_cache.pkl",
        reuse_cache=args.reuse_cache,
    )

    # Merge metadata from either cache so final export can always find frame metadata.
    frame_meta_by_key = {**rule_meta, **scheme_meta}
    all_candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, values in rule_candidates.items():
        all_candidates_by_key[key].extend(values)
    for key, values in scheme_candidates.items():
        all_candidates_by_key[key].extend(values)

    scene_frames = filter_scene_frames_to_available(scene_frames, frame_meta_by_key)

    (out_dir / "target_metrics.json").write_text(
        json.dumps(TARGET_METRICS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    initial_root = ensure_dir(out_dir / "initial")
    rule_initial = run_initial_export(
        "rule",
        scene_frames,
        frame_meta_by_key,
        all_candidates_by_key,
        stitched_manifest,
        gt_root,
        ensure_dir(initial_root / "rule"),
        selected_scenes,
        selected_sequences,
    )
    scheme_initial = run_initial_export(
        "scheme_a",
        scene_frames,
        frame_meta_by_key,
        all_candidates_by_key,
        stitched_manifest,
        gt_root,
        ensure_dir(initial_root / "scheme_a"),
        selected_scenes,
        selected_sequences,
    )

    initial_summary = {
        "target_metrics": TARGET_METRICS,
        "rule": {
            "overall_metric": rule_initial["overall_metric"],
            "score": method_score(rule_initial["overall_metric"]),
            "target_status": summarize_target_hit(rule_initial["overall_metric"]),
        },
        "scheme_a": {
            "overall_metric": scheme_initial["overall_metric"],
            "score": method_score(scheme_initial["overall_metric"]),
            "target_status": summarize_target_hit(scheme_initial["overall_metric"]),
        },
    }
    (out_dir / "initial_method_comparison.json").write_text(
        json.dumps(initial_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.skip_search:
        return

    best_scene_configs, search_rows = run_scene_search(
        scene_frames,
        frame_meta_by_key,
        all_candidates_by_key,
    )
    write_csv_rows(
        out_dir / "scene_search_results.csv",
        [
            "scene",
            "method",
            "risk_time_threshold",
            "risk_pred_threshold",
            "include_small_regions",
            "selection_mode",
            "allowed_cams",
            "spatial_preset",
            "score",
            "recall",
            "far",
            "event_recall",
            "tp",
            "fn",
            "fa",
            "pos_events",
        ],
        search_rows,
    )
    (out_dir / "best_scene_params.json").write_text(
        json.dumps(best_scene_configs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_rows, point_log_rows = build_prediction_rows(
        scene_frames,
        frame_meta_by_key,
        all_candidates_by_key,
        {scene: config for scene, config in best_scene_configs.items()},
    )
    final_root = ensure_dir(out_dir / "best_scene_mix")
    write_csv_rows(final_root / "frame_point_log.csv", POINT_LOG_HEADERS, point_log_rows)
    final_eval = evaluate_prediction_rows(
        stitched_manifest,
        gt_root,
        final_rows,
        final_root,
        "Ours",
        selected_scenes,
        selected_sequences,
    )
    final_summary = {
        "target_metrics": TARGET_METRICS,
        "initial_rule": rule_initial["overall_metric"],
        "initial_scheme_a": scheme_initial["overall_metric"],
        "best_scene_configs": best_scene_configs,
        "final_overall_metric": final_eval["overall_metric"],
        "target_status": summarize_target_hit(final_eval["overall_metric"]),
    }
    (out_dir / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
