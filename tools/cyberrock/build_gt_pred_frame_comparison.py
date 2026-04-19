#!/usr/bin/env python3
"""Build a one-row-per-frame GT/Pred comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cyberrock.real_vehicle_annotation_common import load_csv_rows, write_csv_rows


COMPARISON_HEADERS = [
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
    "frame_case",
    "stitched_image_path",
    "pred_npy_path",
    "pred_has_point",
    "pred_point_count",
    "pred_mapping_valid_count",
    "pred_sources",
    "pred_param_keys",
    "pred_points_json",
    "gt_has_point",
    "gt_point_count",
    "gt_mapping_valid_count",
    "gt_points_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GT/Pred frame comparison CSV.")
    parser.add_argument("--prediction-csv", required=True, type=str)
    parser.add_argument("--pred-point-log", default="", type=str)
    parser.add_argument("--gt-point-log", default="", type=str)
    parser.add_argument("--out-csv", default="", type=str)
    return parser.parse_args()


def frame_case(frame_label: str, pred_has_point: bool) -> str:
    if frame_label == "pos":
        return "pos_tp" if pred_has_point else "pos_fn"
    if frame_label == "neg":
        return "neg_fa" if pred_has_point else "neg_tn"
    if frame_label == "ignore":
        return "ignore"
    return "unknown"


def compact_pred_point(row: dict[str, str]) -> dict[str, object]:
    return {
        "rank": int(row["detection_rank"]) if row.get("detection_rank") else 0,
        "proposal_source": row.get("proposal_source", ""),
        "point_origin": row.get("point_origin", ""),
        "param_key": row.get("param_key", ""),
        "x_rv": float(row["x_rv"]) if row.get("x_rv") else None,
        "y_rv": float(row["y_rv"]) if row.get("y_rv") else None,
        "x_stitched_orig": float(row["x_stitched_orig"]) if row.get("x_stitched_orig") else None,
        "y_stitched_orig": float(row["y_stitched_orig"]) if row.get("y_stitched_orig") else None,
        "mapping_valid": int(row["mapping_valid"]) if row.get("mapping_valid") else 0,
        "cam_idx": int(row["cam_idx"]) if row.get("cam_idx") else None,
        "det_camera_channel": row.get("det_camera_channel", ""),
    }


def compact_gt_point(row: dict[str, str]) -> dict[str, object]:
    return {
        "rank": int(row["gt_point_rank"]) if row.get("gt_point_rank") else 0,
        "point_source": row.get("point_source", ""),
        "x_rv": float(row["x_rv"]) if row.get("x_rv") else None,
        "y_rv": float(row["y_rv"]) if row.get("y_rv") else None,
        "x_stitched_orig": float(row["x_stitched_orig"]) if row.get("x_stitched_orig") else None,
        "y_stitched_orig": float(row["y_stitched_orig"]) if row.get("y_stitched_orig") else None,
        "mapping_valid": int(row["mapping_valid"]) if row.get("mapping_valid") else 0,
        "cam_idx": int(row["cam_idx"]) if row.get("cam_idx") else None,
        "det_camera_channel": row.get("det_camera_channel", ""),
        "gt_target_id": row.get("gt_target_id", ""),
        "gt_box_id": row.get("gt_box_id", ""),
        "gt_x1": int(row["gt_x1"]) if row.get("gt_x1") else None,
        "gt_y1": int(row["gt_y1"]) if row.get("gt_y1") else None,
        "gt_x2": int(row["gt_x2"]) if row.get("gt_x2") else None,
        "gt_y2": int(row["gt_y2"]) if row.get("gt_y2") else None,
    }


def main() -> None:
    args = parse_args()
    prediction_csv = Path(args.prediction_csv).resolve()
    pred_point_log = Path(args.pred_point_log).resolve() if args.pred_point_log else prediction_csv.with_name("frame_point_log.csv")
    gt_point_log = Path(args.gt_point_log).resolve() if args.gt_point_log else prediction_csv.with_name("gt_frame_point_log.csv")
    out_csv = Path(args.out_csv).resolve() if args.out_csv else prediction_csv.with_name("gt_pred_frame_comparison.csv")

    prediction_rows = load_csv_rows(prediction_csv)
    pred_point_rows = load_csv_rows(pred_point_log)
    gt_point_rows = load_csv_rows(gt_point_log)

    frame_base: dict[tuple[str, str], dict[str, str]] = {}
    for row in prediction_rows:
        key = (row["sequence_dir_name"], row["group_id"])
        frame_base.setdefault(key, row)

    pred_points_by_frame: dict[tuple[str, str], list[dict[str, object]]] = {}
    pred_sources_by_frame: dict[tuple[str, str], set[str]] = {}
    pred_params_by_frame: dict[tuple[str, str], set[str]] = {}
    pred_valid_by_frame: dict[tuple[str, str], int] = {}
    for row in pred_point_rows:
        key = (row["sequence_dir_name"], row["group_id"])
        pred_points_by_frame.setdefault(key, []).append(compact_pred_point(row))
        pred_sources_by_frame.setdefault(key, set()).add(row.get("proposal_source", ""))
        pred_params_by_frame.setdefault(key, set()).add(row.get("param_key", ""))
        pred_valid_by_frame[key] = pred_valid_by_frame.get(key, 0) + (1 if row.get("mapping_valid") == "1" else 0)

    gt_points_by_frame: dict[tuple[str, str], list[dict[str, object]]] = {}
    gt_valid_by_frame: dict[tuple[str, str], int] = {}
    gt_base: dict[tuple[str, str], dict[str, str]] = {}
    for row in gt_point_rows:
        key = (row["sequence_dir_name"], row["group_id"])
        gt_base.setdefault(key, row)
        if row.get("has_gt_point") == "1":
            gt_points_by_frame.setdefault(key, []).append(compact_gt_point(row))
            gt_valid_by_frame[key] = gt_valid_by_frame.get(key, 0) + (1 if row.get("mapping_valid") == "1" else 0)

    out_rows: list[dict[str, object]] = []
    for key in sorted(frame_base.keys(), key=lambda item: (frame_base[item]["scene"], frame_base[item]["sequence_dir_name"], int(frame_base[item]["group_id"]))):
        base = frame_base[key]
        gt_row = gt_base.get(key, {})
        pred_points = pred_points_by_frame.get(key, [])
        gt_points = gt_points_by_frame.get(key, [])
        pred_has_point = len(pred_points) > 0
        gt_has_point = len(gt_points) > 0
        frame_label = gt_row.get("frame_label", "")
        row = {
            "sample_idx": base.get("sample_idx", ""),
            "scene_indice": base.get("scene_indice", ""),
            "scene": base.get("scene", ""),
            "sequence_dir_name": base.get("sequence_dir_name", ""),
            "group_id": base.get("group_id", ""),
            "bag_name": base.get("bag_name", ""),
            "ros_msg_seq_prev": base.get("ros_msg_seq_prev", ""),
            "ros_msg_seq_curr": base.get("ros_msg_seq_curr", ""),
            "time_diff_cam_us": base.get("time_diff_cam_us", ""),
            "frame_label": frame_label,
            "event_id": gt_row.get("event_id", ""),
            "frame_case": frame_case(frame_label, pred_has_point),
            "stitched_image_path": base.get("stitched_image_path", ""),
            "pred_npy_path": gt_row.get("pred_npy_path", ""),
            "pred_has_point": int(pred_has_point),
            "pred_point_count": len(pred_points),
            "pred_mapping_valid_count": pred_valid_by_frame.get(key, 0),
            "pred_sources": json.dumps(sorted(s for s in pred_sources_by_frame.get(key, set()) if s), ensure_ascii=False),
            "pred_param_keys": json.dumps(sorted(s for s in pred_params_by_frame.get(key, set()) if s), ensure_ascii=False),
            "pred_points_json": json.dumps(pred_points, ensure_ascii=False),
            "gt_has_point": int(gt_has_point),
            "gt_point_count": len(gt_points),
            "gt_mapping_valid_count": gt_valid_by_frame.get(key, 0),
            "gt_points_json": json.dumps(gt_points, ensure_ascii=False),
        }
        out_rows.append(row)

    write_csv_rows(out_csv, COMPARISON_HEADERS, out_rows)
    print(f"[ok] wrote {len(out_rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
