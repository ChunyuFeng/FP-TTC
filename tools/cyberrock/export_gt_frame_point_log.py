#!/usr/bin/env python3
"""Export per-frame GT collision points for a Table-2 prediction run.

The output mirrors the prediction-side point log workflow:
- one row per evaluated frame when the frame has no GT collision box
- one row per GT box when the frame is positive

GT points are chosen to lie inside the annotation box. When possible, the script
also finds the corresponding RV pixel by inverting the per-frame projection map.
If no RV pixel maps into the GT box, the script still keeps the frame row and
stores the box-center point on the stitched image with empty RV coordinates.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cyberrock.real_vehicle_annotation_common import ensure_dir, load_csv_rows, write_csv_rows


GT_POINT_HEADERS = [
    "sample_idx",
    "scene_indice",
    "scene",
    "sequence_dir_name",
    "group_id",
    "bag_name",
    "ros_msg_seq_prev",
    "ros_msg_seq_curr",
    "time_diff_cam_us",
    "frame_label",
    "event_id",
    "has_gt_point",
    "gt_point_count",
    "gt_point_rank",
    "x_rv",
    "y_rv",
    "cam_idx",
    "u_proc",
    "v_proc",
    "x_stitched_orig",
    "y_stitched_orig",
    "mapping_valid",
    "det_camera_channel",
    "gt_target_id",
    "gt_box_id",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "stitched_image_path",
    "pred_npy_path",
    "proposal_source",
    "point_source",
]

CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GT frame-point CSV for evaluated frames.")
    parser.add_argument("--prediction-csv", required=True, type=str)
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
    parser.add_argument("--out-csv", default="", type=str)
    return parser.parse_args()


def resolve_pred_leaf_dir(pred_dir: Path) -> Path:
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


def load_test_infos_cached(cache: dict[str, list[dict[str, Any]]], pkl_path: Path) -> list[dict[str, Any]]:
    key = str(pkl_path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    with pkl_path.open("rb") as f:
        value = pickle.load(f)
    cache[key] = value
    return value


def image_size_cached(cache: dict[str, tuple[int, int]], image_path: str) -> tuple[int, int]:
    cached = cache.get(image_path)
    if cached is not None:
        return cached
    with Image.open(image_path) as image:
        size = image.size
    cache[image_path] = size
    return size


def build_stitched_lookup(
    proj_pix_curr_fullres: np.ndarray,
    sample_info: dict[str, Any],
    size_cache: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    proc_h, proc_w = proj_pix_curr_fullres.shape[:2]
    cam_idx = proj_pix_curr_fullres[:, :, 0].astype(np.int32)
    u_proc = proj_pix_curr_fullres[:, :, 1].astype(np.float32)
    v_proc = proj_pix_curr_fullres[:, :, 2].astype(np.float32)
    valid = (cam_idx >= 0) & (cam_idx < len(CAMERA_ORDER))

    # SJTU samples use the same camera resolution for all six stitched panels.
    first_channel = CAMERA_ORDER[0]
    first_image = sample_info["curr_camera_data"][first_channel]["filename"]
    orig_w, orig_h = image_size_cached(size_cache, first_image)

    x_orig = (u_proc + 0.5) * float(orig_w) / float(proc_w) - 0.5
    y_orig = (v_proc + 0.5) * float(orig_h) / float(proc_h) - 0.5
    x_stitched = cam_idx.astype(np.float32) * float(orig_w) + x_orig
    y_stitched = y_orig
    return {
        "valid": valid,
        "cam_idx": cam_idx,
        "u_proc": u_proc,
        "v_proc": v_proc,
        "x_stitched": x_stitched,
        "y_stitched": y_stitched,
    }


def make_placeholder_row(frame: dict[str, str], frame_label: str, event_id: str, pred_npy_path: str) -> dict[str, object]:
    return {
        **frame,
        "frame_label": frame_label,
        "event_id": event_id,
        "has_gt_point": 0,
        "gt_point_count": 0,
        "gt_point_rank": 0,
        "x_rv": "",
        "y_rv": "",
        "cam_idx": "",
        "u_proc": "",
        "v_proc": "",
        "x_stitched_orig": "",
        "y_stitched_orig": "",
        "mapping_valid": 0,
        "det_camera_channel": "",
        "gt_target_id": "",
        "gt_box_id": "",
        "gt_x1": "",
        "gt_y1": "",
        "gt_x2": "",
        "gt_y2": "",
        "pred_npy_path": pred_npy_path,
        "proposal_source": "ground_truth",
        "point_source": "no_gt_box",
    }


def main() -> None:
    args = parse_args()
    prediction_csv = Path(args.prediction_csv).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else prediction_csv.with_name("gt_frame_point_log.csv")
    gt_root = Path(args.gt_root).resolve()
    pred_root = Path(args.pred_root).resolve()
    pkl_root = Path(args.pkl_root).resolve()

    prediction_rows = load_csv_rows(prediction_csv)
    frame_first_row: dict[tuple[str, str], dict[str, str]] = {}
    for row in prediction_rows:
        key = (row["sequence_dir_name"], row["group_id"])
        frame_first_row.setdefault(key, row)

    frame_labels = {
        (row["sequence_dir_name"], row["group_id"]): row
        for row in load_csv_rows(gt_root / "frame_labels.csv")
        if (row["sequence_dir_name"], row["group_id"]) in frame_first_row
    }
    boxes_by_frame: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in load_csv_rows(gt_root / "collision_boxes.csv"):
        key = (row["sequence_dir_name"], row["group_id"])
        if key in frame_first_row:
            boxes_by_frame[key].append(row)
    for key in boxes_by_frame:
        boxes_by_frame[key].sort(key=lambda row: (int(row["target_id"]), int(row["box_id"])))

    test_info_cache: dict[str, list[dict[str, Any]]] = {}
    image_size_cache: dict[str, tuple[int, int]] = {}

    output_rows: list[dict[str, object]] = []
    frame_keys = sorted(
        frame_first_row.keys(),
        key=lambda key: (
            frame_first_row[key]["scene"],
            frame_first_row[key]["sequence_dir_name"],
            int(frame_first_row[key]["group_id"]),
        ),
    )
    for frame_key in frame_keys:
        pred_row = frame_first_row[frame_key]
        frame_label_row = frame_labels.get(frame_key)
        frame_label = frame_label_row["frame_label"] if frame_label_row else ""
        event_id = frame_label_row["event_id"] if frame_label_row else ""

        sequence_dir_name = pred_row["sequence_dir_name"]
        bag_stem = Path(pred_row["bag_name"]).stem
        sample_idx = int(pred_row["sample_idx"])

        pred_dir = resolve_pred_leaf_dir(pred_root / f"pred_npy_sjtu_{sequence_dir_name}_rvt280_fullresproj")
        pred_npy_path = pred_dir / f"pred_{sample_idx}.npy"
        if not pred_npy_path.exists():
            output_rows.append(make_placeholder_row(pred_row, frame_label, event_id, str(pred_npy_path)))
            continue

        boxes = boxes_by_frame.get(frame_key, [])
        if not boxes:
            output_rows.append(make_placeholder_row(pred_row, frame_label, event_id, str(pred_npy_path)))
            continue

        test_infos = load_test_infos_cached(test_info_cache, pkl_root / f"{bag_stem}_sjtu_test_infos_overlap_500ms.pkl")
        sample_info = test_infos[sample_idx]
        pred_payload = np.load(pred_npy_path, allow_pickle=True).item()
        lookup = build_stitched_lookup(pred_payload["proj_pix_curr_fullres"], sample_info, image_size_cache)

        for rank, box in enumerate(boxes, start=1):
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
            box_center_x = 0.5 * (x1 + x2)
            box_center_y = 0.5 * (y1 + y2)
            inside_mask = (
                lookup["valid"]
                & (lookup["x_stitched"] >= x1)
                & (lookup["x_stitched"] <= x2)
                & (lookup["y_stitched"] >= y1)
                & (lookup["y_stitched"] <= y2)
            )

            if np.any(inside_mask):
                ys, xs = np.where(inside_mask)
                d2 = (lookup["x_stitched"][ys, xs] - box_center_x) ** 2 + (
                    lookup["y_stitched"][ys, xs] - box_center_y
                ) ** 2
                best_idx = int(np.argmin(d2))
                y_rv = int(ys[best_idx])
                x_rv = int(xs[best_idx])
                cam_idx = int(lookup["cam_idx"][y_rv, x_rv])
                row = {
                    **pred_row,
                    "frame_label": frame_label,
                    "event_id": event_id,
                    "has_gt_point": 1,
                    "gt_point_count": len(boxes),
                    "gt_point_rank": rank,
                    "x_rv": f"{float(x_rv):.6f}",
                    "y_rv": f"{float(y_rv):.6f}",
                    "cam_idx": cam_idx,
                    "u_proc": f"{float(lookup['u_proc'][y_rv, x_rv]):.6f}",
                    "v_proc": f"{float(lookup['v_proc'][y_rv, x_rv]):.6f}",
                    "x_stitched_orig": f"{float(lookup['x_stitched'][y_rv, x_rv]):.6f}",
                    "y_stitched_orig": f"{float(lookup['y_stitched'][y_rv, x_rv]):.6f}",
                    "mapping_valid": 1,
                    "det_camera_channel": CAMERA_ORDER[cam_idx],
                    "gt_target_id": box["target_id"],
                    "gt_box_id": box["box_id"],
                    "gt_x1": box["x1"],
                    "gt_y1": box["y1"],
                    "gt_x2": box["x2"],
                    "gt_y2": box["y2"],
                    "pred_npy_path": str(pred_npy_path),
                    "proposal_source": "ground_truth",
                    "point_source": "rv_inside_gt_box",
                }
            else:
                row = {
                    **pred_row,
                    "frame_label": frame_label,
                    "event_id": event_id,
                    "has_gt_point": 1,
                    "gt_point_count": len(boxes),
                    "gt_point_rank": rank,
                    "x_rv": "",
                    "y_rv": "",
                    "cam_idx": "",
                    "u_proc": "",
                    "v_proc": "",
                    "x_stitched_orig": f"{box_center_x:.6f}",
                    "y_stitched_orig": f"{box_center_y:.6f}",
                    "mapping_valid": 0,
                    "det_camera_channel": "",
                    "gt_target_id": box["target_id"],
                    "gt_box_id": box["box_id"],
                    "gt_x1": box["x1"],
                    "gt_y1": box["y1"],
                    "gt_x2": box["x2"],
                    "gt_y2": box["y2"],
                    "pred_npy_path": str(pred_npy_path),
                    "proposal_source": "ground_truth",
                    "point_source": "stitched_box_center_fallback",
                }
            output_rows.append(row)

    write_csv_rows(out_csv, GT_POINT_HEADERS, output_rows)
    print(f"[ok] wrote {len(output_rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
