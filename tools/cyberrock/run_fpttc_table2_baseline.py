#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


REPO_ROOT = Path(__file__).resolve().parents[2]
ANN_PIPELINE = REPO_ROOT / "tools" / "cyberrock" / "real_vehicle_annotation_pipeline.py"

CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

PREDICTION_CSV_HEADERS = [
    "sample_idx",
    "scene_indice",
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run monocular FP-TTC on the S1-S7 real-vehicle Table-2 subset.",
    )
    parser.add_argument(
        "--comparison-repo",
        default="/home/chunyu/WorkSpace/BugStudio/comparison_exp/FP-TTC",
        help="Path to the comparison FP-TTC repo.",
    )
    parser.add_argument(
        "--prepared-eval-root",
        default=str(REPO_ROOT / "Datasets" / "cyberrock_keyframe_2hz" / "prepared_eval_data"),
        help="prepared_eval_data root containing annotations/, manifests/, pkls/.",
    )
    parser.add_argument(
        "--ckpt",
        default="/home/chunyu/WorkSpace/BugStudio/comparison_exp/FP-TTC/pretrained/fpttc_mix.pth.tar",
        help="Checkpoint used for FP-TTC inference.",
    )
    parser.add_argument(
        "--conda-env",
        default="fpttc_py39",
        help="Conda env used to run the FP-TTC inference script.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[320, 640],
        metavar=("H", "W"),
        help="Inference size passed to test_sjtu.py.",
    )
    parser.add_argument(
        "--padding-factor",
        type=int,
        default=32,
        help="Padding factor passed to test_sjtu.py.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Fallback FPS used when time_diff_cam_us is missing.",
    )
    parser.add_argument(
        "--orig-camera-size",
        type=int,
        nargs=2,
        default=[1080, 1920],
        metavar=("H", "W"),
        help="Original per-camera image size used to map detections back to stitched coordinates.",
    )
    parser.add_argument(
        "--export-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel workers used for collision point export.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output dir. Defaults to prepared_eval_data/comparison/fpttc_table2_s1s7_320x640.",
    )
    parser.add_argument(
        "--include-nr",
        action="store_true",
        help="Include the NR night/rain sequence. Default keeps the main-table S1-S7 scope.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Reuse an existing pred_npy directory under out-dir instead of running test_sjtu.py.",
    )
    parser.add_argument(
        "--skip-export-csv",
        action="store_true",
        help="Reuse an existing collision_predictions.csv under out-dir.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Stop after creating collision_predictions.csv.",
    )
    parser.add_argument(
        "--method-name",
        default="FP-TTC",
        help="Method name passed to evaluate-predictions.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def truthy_scene(scene_name: str, include_nr: bool) -> bool:
    if include_nr:
        return True
    return not scene_name.startswith("S6/S7/S8")


def collect_main_scope(registry_path: Path, include_nr: bool) -> tuple[set[str], set[str]]:
    rows = load_csv_rows(registry_path)
    selected_bags = set()
    selected_sequences = set()
    for row in rows:
        if not truthy_scene(row.get("scene", ""), include_nr):
            continue
        selected_bags.add(row["bag_name"])
        selected_sequences.add(row["sequence_dir_name"])
    return selected_bags, selected_sequences


def build_filtered_pair_subset(
    pkl_path: Path,
    manifest_path: Path,
    frame_labels_path: Path,
    collision_boxes_path: Path,
    selected_bags: set[str],
    out_dir: Path,
) -> dict[str, Any]:
    pkl_rows = load_pickle(pkl_path)
    filtered_pkl_rows = [row for row in pkl_rows if row.get("scene_indice", "") in selected_bags]

    selected_frame_keys = {
        (row["scene_indice"], str(row.get("ros_msg_seq_curr", row.get("ros_msg_seq", ""))))
        for row in filtered_pkl_rows
    }

    manifest_rows = load_csv_rows(manifest_path)
    filtered_manifest_rows = [
        row
        for row in manifest_rows
        if (row["bag_name"], row["ros_msg_seq"]) in selected_frame_keys
    ]

    frame_rows = load_csv_rows(frame_labels_path)
    filtered_frame_rows = [
        row
        for row in frame_rows
        if (row["bag_name"], row["ros_msg_seq"]) in selected_frame_keys
    ]
    kept_frame_ids = {(row["sequence_dir_name"], row["group_id"]) for row in filtered_frame_rows}

    box_rows = load_csv_rows(collision_boxes_path)
    filtered_box_rows = [
        row
        for row in box_rows
        if (row["sequence_dir_name"], row["group_id"]) in kept_frame_ids
    ]

    subset_root = out_dir / "subset_pair_500ms"
    filtered_pkl_path = subset_root / "sjtu_test_infos_s1_s7_pair_500ms.pkl"
    filtered_manifest_path = subset_root / "stitched_frames_manifest_s1_s7_pair_500ms.csv"
    filtered_gt_root = subset_root / "gt"
    filtered_frame_labels_path = filtered_gt_root / "frame_labels.csv"
    filtered_collision_boxes_path = filtered_gt_root / "collision_boxes.csv"

    dump_pickle(filtered_pkl_path, filtered_pkl_rows)
    write_csv_rows(filtered_manifest_path, filtered_manifest_rows, list(manifest_rows[0].keys()))
    write_csv_rows(filtered_frame_labels_path, filtered_frame_rows, list(frame_rows[0].keys()))
    write_csv_rows(filtered_collision_boxes_path, filtered_box_rows, list(box_rows[0].keys()))

    summary = {
        "pair_pkl_path": str(filtered_pkl_path),
        "stitched_manifest_path": str(filtered_manifest_path),
        "gt_root": str(filtered_gt_root),
        "pair_frame_count": len(filtered_pkl_rows),
        "manifest_frame_count": len(filtered_manifest_rows),
        "gt_frame_count": len(filtered_frame_rows),
        "gt_box_count": len(filtered_box_rows),
    }
    return summary


def run_subprocess(cmd: list[str], workdir: Path) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"[Run] {printable}")
    subprocess.run(cmd, cwd=str(workdir), check=True)


def run_fpttc_inference(
    comparison_repo: Path,
    conda_env: str,
    ckpt_path: Path,
    filtered_pkl_path: Path,
    pred_root: Path,
    image_h: int,
    image_w: int,
    padding_factor: int,
) -> Path:
    pred_root.mkdir(parents=True, exist_ok=True)
    before = {p for p in pred_root.glob("pred_npy_*") if p.is_dir()}
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "test_sjtu.py",
        "--resume",
        str(ckpt_path),
        "--test_info_path",
        str(filtered_pkl_path),
        "--save_pred_npy",
        "--pred_npy_dir",
        str(pred_root),
        "--inference_size",
        str(image_h),
        str(image_w),
        "--padding_factor",
        str(padding_factor),
        "--upsample_factor",
        "4",
        "--num_scales",
        "2",
        "--num_head",
        "1",
        "--attn_splits_list",
        "2",
        "8",
        "--corr_radius_list",
        "-1",
        "4",
        "--prop_radius_list",
        "-1",
        "1",
    ]
    run_subprocess(cmd, comparison_repo)
    after = {p for p in pred_root.glob("pred_npy_*") if p.is_dir()}
    new_dirs = sorted(after - before)
    if new_dirs:
        return new_dirs[-1]
    candidates = sorted(after, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No pred_npy_* output found in {pred_root}")
    return candidates[-1]


def find_existing_pred_dir(pred_root: Path) -> Path:
    candidates = sorted(pred_root.glob("pred_npy_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No pred_npy_* output found in {pred_root}")
    return candidates[-1]


def get_pred_npy_files(pred_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, _dirs, filenames in os.walk(pred_dir):
        root_path = Path(root)
        for filename in filenames:
            if filename.endswith(".npy"):
                files.append(root_path / filename)
    return sorted(files)


def load_prediction(pred_path: Path) -> tuple[dict[str, np.ndarray] | None, str, int]:
    try:
        data = np.load(pred_path, allow_pickle=True).item()
        scene = str(data.get("scene", "unknown"))
        idx = int(data.get("index", -1))
        predictions_dict = data.get("predictions")
        if predictions_dict is None and "scale_pred" in data:
            predictions_dict = {"CAM_FRONT": data["scale_pred"]}
        if predictions_dict is None:
            return None, scene, -1
        return predictions_dict, scene, idx
    except Exception:
        return None, "unknown", -1


def get_second_grad(data: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gy = data[stride:, :] - data[:-stride, :]
    gx = data[:, stride:] - data[:, :-stride]
    gyy = gy[stride:, :] - gy[:-stride, :]
    _grad_y, gxx = np.gradient(gx)
    return gy, gx, gyy, gxx


def get_ttc_var(scale: np.ndarray, grid_len: int = 5) -> np.ndarray:
    h, w = scale.shape
    h_var = math.ceil(h / grid_len) - 1
    w_var = math.ceil(w / grid_len) - 1
    var = np.zeros((h_var, w_var), dtype=np.float32)
    for i in range(h_var):
        for j in range(w_var):
            var[i, j] = np.var(scale[i * grid_len : (i + 1) * grid_len, j * grid_len : (j + 1) * grid_len])
    return var


def get_grid_ttc(scale: np.ndarray, scale_var: np.ndarray, grid_len: int = 5) -> tuple[np.ndarray, np.ndarray]:
    var_mean = 0.3 * np.mean(scale_var)
    valid_grid = np.zeros((scale.shape[0], scale.shape[1]), dtype=np.float32)
    valid_map = np.zeros((scale_var.shape[0], scale_var.shape[1]), dtype=np.float32)
    h, w = scale_var.shape
    for i in range(h):
        for j in range(w):
            if scale_var[i, j] < var_mean:
                valid_grid[i * grid_len : (i + 1) * grid_len, j * grid_len : (j + 1) * grid_len] = 1
                valid_map[i, j] = 1
    return valid_map, valid_grid


def preprocess_scale_map(scale_map: np.ndarray, grid_size: int, approach_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = scale_map.shape
    proc_h = (math.ceil(h / grid_size) - 1) * grid_size
    proc_w = (math.ceil(w / grid_size) - 1) * grid_size
    cropped = scale_map[:proc_h, :proc_w]
    min_s = np.min(cropped)
    norm_scale = (cropped - min_s) / (approach_threshold - min_s + 1e-6) * 255.0
    return cropped, norm_scale.astype(np.uint8)


def compute_valid_masks(
    scale_map: np.ndarray,
    grid_size: int,
    grad_step: int,
    risk_rate: float,
    approach_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    variance = get_ttc_var(scale_map, grid_len=grid_size)
    gy, _gx, _gy2, _gx2 = get_second_grad(scale_map, stride=grad_step)
    mean_grad = np.mean(np.abs(gy)) if gy.size > 0 else 1.0
    mask_grad = np.abs(gy) < (risk_rate * mean_grad)
    mask_thresh1 = scale_map[grad_step:, :] < approach_threshold
    mask_thresh2 = scale_map[1:-1, :] < approach_threshold
    valid_grad = np.logical_and(mask_grad, mask_thresh1)
    valid_grad = np.logical_and(valid_grad, mask_thresh2)
    grid_map, _ = get_grid_ttc(scale_map, variance, grid_len=grid_size)
    return valid_grad, grid_map.astype(bool)


def extract_candidate_points(
    scale_map: np.ndarray,
    norm_scale: np.ndarray,
    valid_grad_mask: np.ndarray,
    grid_map: np.ndarray,
    grid_size: int,
    grad_step: int,
    approach_threshold: float,
    kp_threshold: int,
) -> np.ndarray:
    fast = cv2.FastFeatureDetector_create(6400)
    fast.setThreshold(kp_threshold)
    keypoints = fast.detect(norm_scale, None)
    height, width = scale_map.shape
    candidates: list[list[float]] = []

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if not (0.2 * height < y < 0.8 * height and 0.05 * width < x < 0.95 * width):
            continue
        area = 9
        y0, y1 = max(y - area // 2, 0), min(y + area // 2 + 1, height)
        x0, x1 = max(x - area // 2, 0), min(x + area // 2 + 1, width)
        window = valid_grad_mask[y0:y1, x0:x1]
        if window.sum() <= 3:
            continue
        orig = scale_map[y0:y1, x0:x1]
        if orig.min() < approach_threshold and orig.max() < 1.04:
            candidates.append([float(x), float(y), float(norm_scale[y0:y1, x0:x1].min())])

    rows, cols = grid_map.shape
    for i in range(rows):
        for j in range(cols):
            if not grid_map[i, j]:
                continue
            cy = int((2 * i + 1) * grid_size / 2)
            cx = int((2 * j + 1) * grid_size / 2)
            if not valid_grad_mask[cy, cx] or cy < 0.4 * height:
                continue
            if scale_map[cy, cx] < approach_threshold:
                local_norm = np.mean(
                    norm_scale[i * grid_size : (i + 1) * grid_size, j * grid_size : (j + 1) * grid_size]
                )
                candidates.append([float(cx), float(cy), float(local_norm)])

    if not candidates:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(candidates, dtype=np.float32)


def cluster_and_compute(
    scale_map: np.ndarray,
    candidates: np.ndarray,
    delta_t: float,
    risk_time_threshold: float,
) -> list[list[float]]:
    collisions: list[list[float]] = []
    if candidates.size == 0:
        return collisions

    cluster_1 = DBSCAN(eps=50, min_samples=5).fit(candidates)
    clusters_1: dict[int, list[np.ndarray]] = {}
    for idx, label in enumerate(cluster_1.labels_):
        if label < 0:
            continue
        clusters_1.setdefault(int(label), []).append(candidates[idx])

    secondary_centers: list[list[float]] = []
    secondary_regions: list[np.ndarray] = []
    for pts in clusters_1.values():
        arr = np.asarray(pts, dtype=np.float32)[:, :2].astype(np.int32)
        (cx, cy), radius = cv2.minEnclosingCircle(arr)
        ttc_s = delta_t / (1.0 - float(scale_map[int(cy), int(cx)]) + 1e-5)
        if radius < 6 and arr.shape[0] < 4:
            continue
        secondary_centers.append([float(cx), float(cy), float(ttc_s * 200.0)])
        secondary_regions.append(arr)

    if not secondary_centers:
        return collisions

    cluster_2 = DBSCAN(eps=100, min_samples=3).fit(np.asarray(secondary_centers, dtype=np.float32))
    regions: dict[int, list[np.ndarray]] = {}
    for idx, label in enumerate(cluster_2.labels_):
        regions.setdefault(int(label), []).append(secondary_regions[idx])

    for label, regs in regions.items():
        if label < 0:
            for reg in regs:
                (cx, cy), _radius = cv2.minEnclosingCircle(reg)
                ttc_s = delta_t / (1.0 - float(scale_map[int(cy), int(cx)]) + 1e-5)
                if 0 < ttc_s < risk_time_threshold:
                    collisions.append([int(cx), int(cy), float(ttc_s), float(scale_map[int(cy), int(cx)])])
            continue

        merged = np.vstack(regs)
        ttc_values: list[float] = []
        for p in merged:
            y, x = int(p[1]), int(p[0])
            ttc_values.append(delta_t / (1.0 - float(scale_map[y, x]) + 1e-5))
        ttc_min = min(ttc_values)
        (cx, cy), _radius = cv2.minEnclosingCircle(merged)
        if 0 < ttc_min < risk_time_threshold:
            collisions.append([int(cx), int(cy), float(ttc_min), float(scale_map[int(cy), int(cx)])])

    return collisions


def extract_small_regions(
    scale_map: np.ndarray,
    approach_threshold: float,
    delta_t: float,
    risk_time_threshold: float,
) -> list[list[float]]:
    thresh_mask = (scale_map < approach_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_closed = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, _labels, _stats, centroids = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    out: list[list[float]] = []
    for idx in range(1, num_labels):
        cx, cy = centroids[idx]
        cx_i = int(cx)
        cy_i = int(cy)
        ttc_s = delta_t / (1.0 - float(scale_map[cy_i, cx_i]) + 1e-5)
        if 0 < ttc_s < risk_time_threshold:
            out.append([cx_i, cy_i, float(ttc_s), float(scale_map[cy_i, cx_i])])
    return out


def detect_collisions_on_scale_map(
    scale_map: np.ndarray,
    approach_threshold: float,
    risk_time_threshold: float,
    grid_size: int,
    risk_rate: float,
    kp_detector_threshold: int,
    grad_step: int,
    delta_t: float,
) -> list[list[float]]:
    scale_thresh = 1.0 - delta_t / risk_time_threshold
    scale_map_filtered = np.where(scale_map < scale_thresh, scale_map, 0.999)
    crop_scale, norm_scale = preprocess_scale_map(scale_map_filtered, grid_size, approach_threshold)
    valid_grad, grid_map = compute_valid_masks(crop_scale, grid_size, grad_step, risk_rate, approach_threshold)
    candidates = extract_candidate_points(
        crop_scale,
        norm_scale,
        valid_grad,
        grid_map,
        grid_size,
        grad_step,
        approach_threshold,
        kp_detector_threshold,
    )
    collisions = cluster_and_compute(crop_scale, candidates, delta_t, risk_time_threshold)
    collisions.extend(extract_small_regions(crop_scale, approach_threshold, delta_t, risk_time_threshold))

    height, width = crop_scale.shape
    y_min, y_max = 0.2 * height, 0.8 * height
    x_min, x_max = 0.05 * width, 0.95 * width
    return [p for p in collisions if x_min < p[0] < x_max and y_min < p[1] < y_max]


def safe_ros_msg_seq_prev(entry: dict[str, Any]) -> str:
    value = entry.get("ros_msg_seq_prev")
    if value is None:
        return ""
    return str(value)


def safe_ros_msg_seq_curr(entry: dict[str, Any]) -> str:
    value = entry.get("ros_msg_seq_curr", entry.get("ros_msg_seq", ""))
    return str(value)


def build_stitched_lookup(manifest_path: Path) -> dict[tuple[str, str], str]:
    rows = load_csv_rows(manifest_path)
    return {(row["bag_name"], row["ros_msg_seq"]): row["stitched_image_path"] for row in rows}


def build_prediction_rows_for_file(
    pred_path: str,
    entry: dict[str, Any],
    stitched_image_path: str,
    image_h: int,
    image_w: int,
    orig_h: int,
    orig_w: int,
    fps: float,
) -> tuple[list[dict[str, Any]], bool, int]:
    predictions_dict, _scene_id, global_index = load_prediction(Path(pred_path))
    if predictions_dict is None or global_index < 0:
        return [], False, 0

    bag_name = entry["scene_indice"]
    ros_msg_seq_curr = safe_ros_msg_seq_curr(entry)
    ros_msg_seq_prev = safe_ros_msg_seq_prev(entry)
    delta_t_us = entry.get("time_diff_cam_us")
    delta_t_s = float(delta_t_us) / 1_000_000.0 if delta_t_us is not None else 1.0 / fps
    frame_detections: list[dict[str, Any]] = []

    for cam_idx, cam_name in enumerate(CAMERA_ORDER):
        scale_map = predictions_dict.get(cam_name)
        if scale_map is None:
            continue
        if scale_map.shape[0] != image_h or scale_map.shape[1] != image_w:
            scale_map = cv2.resize(
                scale_map,
                (image_w, image_h),
                interpolation=cv2.INTER_NEAREST,
            )
        collisions = detect_collisions_on_scale_map(
            scale_map=scale_map,
            approach_threshold=1.0,
            risk_time_threshold=2.0,
            grid_size=2,
            risk_rate=0.15,
            kp_detector_threshold=2,
            grad_step=2,
            delta_t=delta_t_s,
        )
        if not collisions:
            continue

        for collision in collisions:
            x_proc = float(collision[0])
            y_proc = float(collision[1])
            ttc_s = float(collision[2])
            scale_pred = float(collision[3])
            x_orig = x_proc * float(orig_w) / float(image_w)
            y_orig = y_proc * float(orig_h) / float(image_h)
            x_stitched_orig = float(cam_idx) * float(orig_w) + x_orig
            frame_detections.append(
                {
                    "sample_idx": global_index,
                    "scene_indice": bag_name,
                    "bag_name": bag_name,
                    "ros_msg_seq_prev": ros_msg_seq_prev,
                    "ros_msg_seq_curr": ros_msg_seq_curr,
                    "time_diff_cam_us": delta_t_us if delta_t_us is not None else int(round(delta_t_s * 1_000_000.0)),
                    "proposal_source": "fpttc_sjtu",
                    "selection_mode": "collision_detect_sjtu",
                    "has_detection": 1,
                    "detection_count": 0,
                    "detection_rank": 0,
                    "x_rv": "",
                    "y_rv": "",
                    "ttc_s": f"{ttc_s:.6f}",
                    "scale_pred": f"{scale_pred:.6f}",
                    "risk_pred": f"{1.0 - scale_pred:.6f}",
                    "cam_idx": cam_idx,
                    "u_proc": f"{x_proc:.6f}",
                    "v_proc": f"{y_proc:.6f}",
                    "x_stitched_orig": f"{x_stitched_orig:.6f}",
                    "y_stitched_orig": f"{y_orig:.6f}",
                    "mapping_valid": 1,
                    "det_camera_channel": cam_name,
                    "det_label": "fpttc_collision_point",
                    "det_conf": "",
                    "det_x1": "",
                    "det_y1": "",
                    "det_x2": "",
                    "det_y2": "",
                    "matched_point_count": "",
                    "stitched_image_path": stitched_image_path,
                    "overlay_image_path": "",
                    "depth_overlay_image_path": "",
                }
            )

    if not frame_detections:
        return (
            [
                {
                    "sample_idx": global_index,
                    "scene_indice": bag_name,
                    "bag_name": bag_name,
                    "ros_msg_seq_prev": ros_msg_seq_prev,
                    "ros_msg_seq_curr": ros_msg_seq_curr,
                    "time_diff_cam_us": delta_t_us if delta_t_us is not None else int(round(delta_t_s * 1_000_000.0)),
                    "proposal_source": "fpttc_sjtu",
                    "selection_mode": "collision_detect_sjtu",
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
                    "stitched_image_path": stitched_image_path,
                    "overlay_image_path": "",
                    "depth_overlay_image_path": "",
                }
            ],
            False,
            0,
        )

    frame_detections.sort(key=lambda row: float(row["ttc_s"]))
    detection_count = len(frame_detections)
    for rank, row in enumerate(frame_detections, start=1):
        row["detection_count"] = detection_count
        row["detection_rank"] = rank
    return frame_detections, True, detection_count


def export_collision_predictions_csv(
    filtered_pkl_path: Path,
    pred_npy_dir: Path,
    filtered_manifest_path: Path,
    csv_path: Path,
    image_h: int,
    image_w: int,
    orig_h: int,
    orig_w: int,
    fps: float,
    export_workers: int,
) -> dict[str, Any]:
    stitched_lookup = build_stitched_lookup(filtered_manifest_path)
    test_entries = load_pickle(filtered_pkl_path)
    npy_files = get_pred_npy_files(pred_npy_dir)
    rows: list[dict[str, Any]] = []
    frame_detection_count = 0
    det_row_count = 0
    tasks: list[tuple[str, dict[str, Any], str, int, int, int, int, float]] = []
    for pred_path in npy_files:
        predictions_dict, _scene_id, global_index = load_prediction(pred_path)
        if predictions_dict is None or global_index < 0 or global_index >= len(test_entries):
            continue
        entry = test_entries[global_index]
        bag_name = entry["scene_indice"]
        ros_msg_seq_curr = safe_ros_msg_seq_curr(entry)
        stitched_image_path = stitched_lookup.get((bag_name, ros_msg_seq_curr), "")
        tasks.append(
            (
                str(pred_path),
                entry,
                stitched_image_path,
                image_h,
                image_w,
                orig_h,
                orig_w,
                fps,
            )
        )

    if export_workers <= 1:
        iterator = map(lambda task: build_prediction_rows_for_file(*task), tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=export_workers)
        iterator = executor.map(build_prediction_rows_for_file, *zip(*tasks), chunksize=8)

    try:
        for frame_rows, had_detection, detection_count in iterator:
            rows.extend(frame_rows)
            if had_detection:
                frame_detection_count += 1
                det_row_count += detection_count
    finally:
        if export_workers > 1:
            executor.shutdown(wait=True)

    write_csv_rows(csv_path, rows, PREDICTION_CSV_HEADERS)
    return {
        "prediction_csv_path": str(csv_path),
        "prediction_frame_rows": len(rows),
        "frames_with_detection": frame_detection_count,
        "detection_rows": det_row_count,
        "npy_file_count": len(npy_files),
    }


def run_evaluation(
    filtered_manifest_path: Path,
    filtered_gt_root: Path,
    prediction_csv_path: Path,
    out_dir: Path,
    method_name: str,
) -> None:
    cmd = [
        sys.executable,
        str(ANN_PIPELINE),
        "evaluate-predictions",
        "--stitched-manifest",
        str(filtered_manifest_path),
        "--gt-root",
        str(filtered_gt_root),
        "--prediction-csv",
        str(prediction_csv_path),
        "--out-dir",
        str(out_dir),
        "--method-name",
        method_name,
    ]
    run_subprocess(cmd, REPO_ROOT)


def main() -> None:
    args = parse_args()
    comparison_repo = Path(args.comparison_repo).resolve()
    prepared_eval_root = Path(args.prepared_eval_root).resolve()
    ckpt_path = Path(args.ckpt).resolve()

    if not comparison_repo.exists():
        raise FileNotFoundError(comparison_repo)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    image_h, image_w = args.image_size
    orig_h, orig_w = args.orig_camera_size
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (
            prepared_eval_root
            / "comparison"
            / f"fpttc_table2_s1s7_{image_h}x{image_w}"
        ).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations_root = prepared_eval_root / "annotations"
    registry_path = annotations_root / "cvat_exports" / "cvat_task_registry.csv"
    manifest_path = annotations_root / "manifests" / "stitched_frames_manifest.csv"
    frame_labels_path = annotations_root / "gt" / "frame_labels.csv"
    collision_boxes_path = annotations_root / "gt" / "collision_boxes.csv"
    pair_pkl_path = prepared_eval_root / "pkls" / "sjtu_test_infos_keep_t2_overlap_500ms.pkl"

    selected_bags, _selected_sequences = collect_main_scope(registry_path, args.include_nr)
    subset_summary = build_filtered_pair_subset(
        pair_pkl_path,
        manifest_path,
        frame_labels_path,
        collision_boxes_path,
        selected_bags,
        out_dir,
    )

    pred_root = out_dir / "pred_npy"
    if args.skip_inference:
        pred_npy_dir = find_existing_pred_dir(pred_root)
    else:
        pred_npy_dir = run_fpttc_inference(
            comparison_repo=comparison_repo,
            conda_env=args.conda_env,
            ckpt_path=ckpt_path,
            filtered_pkl_path=Path(subset_summary["pair_pkl_path"]),
            pred_root=pred_root,
            image_h=image_h,
            image_w=image_w,
            padding_factor=args.padding_factor,
        )

    prediction_csv_path = out_dir / "collision_predictions.csv"
    if not args.skip_export_csv:
        export_summary = export_collision_predictions_csv(
            filtered_pkl_path=Path(subset_summary["pair_pkl_path"]),
            pred_npy_dir=pred_npy_dir,
            filtered_manifest_path=Path(subset_summary["stitched_manifest_path"]),
            csv_path=prediction_csv_path,
            image_h=image_h,
            image_w=image_w,
            orig_h=orig_h,
            orig_w=orig_w,
            fps=args.fps,
            export_workers=args.export_workers,
        )
    else:
        if not prediction_csv_path.exists():
            raise FileNotFoundError(prediction_csv_path)
        export_summary = {
            "prediction_csv_path": str(prediction_csv_path),
        }

    eval_out_dir = out_dir / "evaluation"
    if not args.skip_eval:
        run_evaluation(
            filtered_manifest_path=Path(subset_summary["stitched_manifest_path"]),
            filtered_gt_root=Path(subset_summary["gt_root"]),
            prediction_csv_path=prediction_csv_path,
            out_dir=eval_out_dir,
            method_name=args.method_name,
        )

    summary = {
        "comparison_repo": str(comparison_repo),
        "prepared_eval_root": str(prepared_eval_root),
        "checkpoint_path": str(ckpt_path),
        "image_size": [image_h, image_w],
        "include_nr": args.include_nr,
        "pred_npy_dir": str(pred_npy_dir),
        "prediction_csv_path": str(prediction_csv_path),
        "evaluation_dir": str(eval_out_dir),
    }
    summary.update(subset_summary)
    summary.update(export_summary)
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
