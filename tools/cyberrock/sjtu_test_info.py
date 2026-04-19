#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cyberrock.sjtu_pipeline_utils import (
    CAMERA_CHANNELS,
    REF_CHANNEL,
    build_sequence_dir_name,
    compute_rectify_map_and_kud,
    ensure_dir,
    load_camera_projections,
    strip_markdown_code,
)


def undistort_image(image_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray):
    h, w = image_bgr.shape[:2]
    K = np.asarray(K, dtype=np.float32)
    dist = np.asarray(dist, dtype=np.float32).reshape(-1)
    if dist.size == 0:
        return image_bgr.copy(), K.copy(), (0, 0, w, h)

    K_undist, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    image_undist = cv2.undistort(image_bgr, K, dist, None, K_undist)
    return image_undist, K_undist.astype(np.float32), roi


def intersect_rois(roi_a, roi_b):
    ax, ay, aw, ah = roi_a
    bx, by, bw, bh = roi_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0, 0, 0, 0
    return left, top, right - left, bottom - top


def parse_args():
    parser = argparse.ArgumentParser(description="Create SJTU test PKLs from aligned manifests.")
    parser.add_argument("--yaml_path", type=str, required=True, help="Camera calibration YAML path.")
    parser.add_argument(
        "--aligned_manifest",
        type=str,
        default=None,
        help="Single aligned manifest CSV path for per-bag PKL generation.",
    )
    parser.add_argument(
        "--rectified_root",
        type=str,
        default=None,
        help="Optional rectified root used to derive paths when aligned manifest omits them.",
    )
    parser.add_argument(
        "--pkl_save_path",
        type=str,
        default=None,
        help="Single-mode output PKL path or directory.",
    )
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=None,
        help="Structured scene manifest CSV used for batch PKL generation.",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default=None,
        help="Root created by rosbag_extract_images.py, containing sequences/<sequence_dir_name>/manifests/*.aligned.csv.",
    )
    parser.add_argument(
        "--pkl_save_root",
        type=str,
        default=None,
        help="Batch-mode PKL output root. Will contain per_bag/ and merged PKLs.",
    )
    parser.add_argument(
        "--scene_idx",
        type=int,
        default=None,
        help="Optional legacy scene index used only in single-mode naming.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional legacy naming hint; no resize is applied.",
    )
    return parser.parse_args()


def load_aligned_manifest(aligned_manifest: Path) -> List[dict]:
    with open(aligned_manifest, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.sort(key=lambda row: int(row["group_id"]))
    return rows


def resolve_rectified_path(row: Dict[str, str], channel: str, rectified_root: Optional[Path]) -> str:
    direct = row.get(f"{channel}_rectified_path", "")
    if direct:
        return str(Path(direct).resolve())
    if rectified_root is None:
        raise ValueError(f"Missing rectified path for {channel} and no --rectified_root provided.")
    raw_path = Path(row[f"{channel}_raw_path"])
    return str((rectified_root / channel / raw_path.name).resolve())


def build_channel_meta(calib: Dict[str, dict], sample_row: Dict[str, str], rectified_root: Optional[Path]):
    per_channel_meta = {}
    for channel in CAMERA_CHANNELS:
        rectified_path = Path(resolve_rectified_path(sample_row, channel, rectified_root))
        image = cv2.imread(str(rectified_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read rectified image: {rectified_path}")
        height, width = image.shape[:2]
        _, _, K_ud = compute_rectify_map_and_kud(
            calib[channel]["K_src"],
            calib[channel]["dist_src"],
            width,
            height,
        )
        R = calib[channel]["R"]
        t = calib[channel]["t"]
        R_l2c = R.T
        t_l2c = -R_l2c @ t
        per_channel_meta[channel] = {
            "K_src": calib[channel]["K_src"],
            "dist_src": calib[channel]["dist_src"],
            "K": K_ud,
            "dist": np.array([], dtype=np.float32),
            "is_rectified": True,
            "image_size_rectified": (height, width),
            "R_l2c": R_l2c,
            "t_l2c": t_l2c,
            "R": R,
            "t": t,
        }
    return per_channel_meta


def build_raw_entries_from_rows(
    rows: List[Dict[str, str]],
    per_channel_meta: Dict[str, dict],
    rectified_root: Optional[Path],
) -> List[dict]:
    if len(rows) < 2:
        return []

    entries = []
    bag_name = rows[0].get("bag_name", "")
    scene_indice = bag_name or rows[0].get("scene", "")
    for index in range(1, len(rows)):
        prev_row = rows[index - 1]
        curr_row = rows[index]
        prev_data = {}
        curr_data = {}
        for channel in CAMERA_CHANNELS:
            prev_data[channel] = {
                "filename": resolve_rectified_path(prev_row, channel, rectified_root),
                "timestamp": int(prev_row[f"{channel}_header_time_us"]),
            }
            curr_data[channel] = {
                "filename": resolve_rectified_path(curr_row, channel, rectified_root),
                "timestamp": int(curr_row[f"{channel}_header_time_us"]),
            }

        ros_msg_seq = int(curr_row.get(f"{REF_CHANNEL}_seq") or curr_row["group_id"])
        time_diff_cam_us = (
            int(curr_row["CAM_FRONT_header_time_us"]) - int(prev_row["CAM_FRONT_header_time_us"])
        )
        entries.append(
            {
                "sjtu_pair_mode": "raw_100ms",
                "prev_camera_data": prev_data,
                "curr_camera_data": curr_data,
                "prev_lidar_data": None,
                "curr_lidar_data": None,
                "sensor_metas_prev": deepcopy(per_channel_meta),
                "sensor_metas_curr": deepcopy(per_channel_meta),
                "gt_map_path": None,
                "scene_flow_path": None,
                "scene_indice": scene_indice,
                "ros_msg_seq": ros_msg_seq,
                "time_diff_cam_us": time_diff_cam_us,
            }
        )
    return entries


def build_overlap_500ms_entries(raw_entries: List[dict]) -> List[dict]:
    if len(raw_entries) < 5:
        return []

    overlap_entries = []
    segment_start = 0
    while segment_start < len(raw_entries):
        scene_indice = raw_entries[segment_start].get("scene_indice")
        segment_end = segment_start + 1
        while (
            segment_end < len(raw_entries)
            and raw_entries[segment_end].get("scene_indice") == scene_indice
        ):
            segment_end += 1

        for start in range(segment_start, max(segment_start, segment_end - 4)):
            end = start + 4
            if end >= segment_end:
                break
            prev_entry = raw_entries[start]
            curr_entry = raw_entries[end]
            front_channel = "CAM_FRONT"
            prev_ts = int(prev_entry["prev_camera_data"][front_channel]["timestamp"])
            curr_ts = int(curr_entry["curr_camera_data"][front_channel]["timestamp"])
            overlap_entries.append(
                {
                    "sjtu_pair_mode": "overlap_500ms",
                    "prev_camera_data": deepcopy(prev_entry["prev_camera_data"]),
                    "curr_camera_data": deepcopy(curr_entry["curr_camera_data"]),
                    "prev_lidar_data": prev_entry.get("prev_lidar_data"),
                    "curr_lidar_data": curr_entry.get("curr_lidar_data"),
                    "sensor_metas_prev": deepcopy(prev_entry["sensor_metas_prev"]),
                    "sensor_metas_curr": deepcopy(curr_entry["sensor_metas_curr"]),
                    "gt_map_path": None,
                    "scene_flow_path": None,
                    "scene_indice": prev_entry.get("scene_indice"),
                    "ros_msg_seq_prev": prev_entry.get("ros_msg_seq"),
                    "ros_msg_seq_curr": curr_entry.get("ros_msg_seq"),
                    "ros_msg_seq": curr_entry.get("ros_msg_seq"),
                    "time_diff_cam_us": curr_ts - prev_ts,
                }
            )
        segment_start = segment_end

    return overlap_entries


def save_pickle(entries: List[dict], pkl_path: Path):
    ensure_dir(pkl_path.parent)
    with open(pkl_path, "wb") as f:
        pickle.dump(entries, f)


def derive_single_output_path(args, aligned_manifest: Path) -> Path:
    if args.pkl_save_path:
        output = Path(args.pkl_save_path)
        if output.suffix == ".pkl":
            return output.resolve()
        bag_stem = aligned_manifest.stem.replace(".aligned", "")
        return (output / f"{bag_stem}_sjtu_test_infos_rectified.pkl").resolve()

    bag_stem = aligned_manifest.stem.replace(".aligned", "")
    if args.scene_idx is not None and args.image_size:
        height, width = args.image_size
        return (aligned_manifest.parent / f"scene_{args.scene_idx}_sjtu_test_infos_{height}_{width * len(CAMERA_CHANNELS)}_rectified.pkl").resolve()
    return (aligned_manifest.parent / f"{bag_stem}_sjtu_test_infos_rectified.pkl").resolve()


def build_per_bag_pkl(
    aligned_manifest: Path,
    calib: Dict[str, dict],
    rectified_root: Optional[Path],
    raw_output_pkl: Path,
    overlap_output_pkl: Path,
) -> Dict[str, List[dict]]:
    rows = load_aligned_manifest(aligned_manifest)
    if not rows:
        print(f"[warn] empty aligned manifest: {aligned_manifest}", file=sys.stderr)
        raw_entries = []
        overlap_entries = []
    else:
        per_channel_meta = build_channel_meta(calib, rows[0], rectified_root)
        raw_entries = build_raw_entries_from_rows(rows, per_channel_meta, rectified_root)
        overlap_entries = build_overlap_500ms_entries(raw_entries)
    save_pickle(raw_entries, raw_output_pkl)
    save_pickle(overlap_entries, overlap_output_pkl)
    print(f"[pkl] saved {len(raw_entries)} raw_100ms entries to {raw_output_pkl}")
    print(f"[pkl] saved {len(overlap_entries)} overlap_500ms entries to {overlap_output_pkl}")
    return {
        "raw_100ms": raw_entries,
        "overlap_500ms": overlap_entries,
    }


def load_scene_manifest(manifest_csv: Path) -> List[Dict[str, str]]:
    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_batch_report(pkl_save_root: Path, batch_summary: dict):
    report_path = pkl_save_root / "pkl_generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)


def candidate_sequence_dirs(processed_root: Path, row: Dict[str, str]) -> List[Path]:
    bag_name = strip_markdown_code(row.get("bag_name", ""))
    bag_stem = Path(bag_name).stem
    scene = strip_markdown_code(row.get("scene", ""))
    tag = strip_markdown_code(row.get("tag", ""))
    manifest_sequence_dir = strip_markdown_code(row.get("sequence_dir_name", ""))
    derived_sequence_dir = build_sequence_dir_name(bag_name, scene, tag)

    candidate_names = []
    for name in [manifest_sequence_dir, derived_sequence_dir, bag_stem]:
        if name and name not in candidate_names:
            candidate_names.append(name)

    candidates = []
    for name in candidate_names:
        candidates.append(processed_root / "sequences" / name)
    for name in [bag_stem]:
        candidates.append(processed_root / "bags" / name)
        candidates.append(processed_root / name)
    return candidates


def resolve_aligned_manifest_path(processed_root: Path, row: Dict[str, str]) -> Path:
    bag_name = strip_markdown_code(row.get("bag_name", ""))
    bag_stem = Path(bag_name).stem
    for sequence_dir in candidate_sequence_dirs(processed_root, row):
        aligned_manifest = sequence_dir / "manifests" / f"{bag_stem}.aligned.csv"
        if aligned_manifest.exists():
            return aligned_manifest
    manifest_sequence_dir = strip_markdown_code(row.get("sequence_dir_name", ""))
    if manifest_sequence_dir:
        return processed_root / "sequences" / manifest_sequence_dir / "manifests" / f"{bag_stem}.aligned.csv"
    return processed_root / "sequences" / bag_stem / "manifests" / f"{bag_stem}.aligned.csv"


def batch_mode(args, calib: Dict[str, dict]):
    if args.manifest_csv is None or args.processed_root is None or args.pkl_save_root is None:
        raise ValueError("Batch mode requires --manifest_csv, --processed_root, and --pkl_save_root.")

    manifest_rows = load_scene_manifest(Path(args.manifest_csv).resolve())
    processed_root = Path(args.processed_root).resolve()
    pkl_save_root = Path(args.pkl_save_root).resolve()
    per_bag_root = ensure_dir(pkl_save_root / "per_bag")

    all_raw_entries = []
    keep_raw_entries = []
    all_overlap_entries = []
    keep_overlap_entries = []
    per_bag_summary = []

    for row in manifest_rows:
        bag_name = row["bag_name"]
        bag_stem = Path(bag_name).stem
        aligned_manifest = resolve_aligned_manifest_path(processed_root, row)
        if not aligned_manifest.exists():
            print(f"[warn] missing aligned manifest, skipped: {aligned_manifest}", file=sys.stderr)
            continue
        keep_flag = row.get("keep_t2", "").strip().upper()
        effective_keep_t2 = "Y" if keep_flag in {"", "?", "Y"} else keep_flag
        raw_output_pkl = per_bag_root / f"{bag_stem}_sjtu_test_infos_raw_100ms.pkl"
        overlap_output_pkl = per_bag_root / f"{bag_stem}_sjtu_test_infos_overlap_500ms.pkl"
        entry_payload = build_per_bag_pkl(aligned_manifest, calib, None, raw_output_pkl, overlap_output_pkl)
        raw_entries = entry_payload["raw_100ms"]
        overlap_entries = entry_payload["overlap_500ms"]
        all_raw_entries.extend(raw_entries)
        all_overlap_entries.extend(overlap_entries)
        if effective_keep_t2 == "Y":
            keep_raw_entries.extend(raw_entries)
            keep_overlap_entries.extend(overlap_entries)
        per_bag_summary.append(
            {
                "bag_name": bag_name,
                "aligned_manifest": str(aligned_manifest),
                "per_bag_raw_100ms_pkl": str(raw_output_pkl),
                "per_bag_overlap_500ms_pkl": str(overlap_output_pkl),
                "raw_entry_count": len(raw_entries),
                "overlap_500ms_entry_count": len(overlap_entries),
                "keep_t2": effective_keep_t2,
            }
        )

    all_raw_pkl = pkl_save_root / "sjtu_test_infos_all_processed_raw_100ms.pkl"
    keep_raw_pkl = pkl_save_root / "sjtu_test_infos_keep_t2_raw_100ms.pkl"
    all_overlap_pkl = pkl_save_root / "sjtu_test_infos_all_processed_overlap_500ms.pkl"
    keep_overlap_pkl = pkl_save_root / "sjtu_test_infos_keep_t2_overlap_500ms.pkl"
    save_pickle(all_raw_entries, all_raw_pkl)
    save_pickle(keep_raw_entries, keep_raw_pkl)
    save_pickle(all_overlap_entries, all_overlap_pkl)
    save_pickle(keep_overlap_entries, keep_overlap_pkl)
    batch_summary = {
        "per_bag": per_bag_summary,
        "all_processed_raw_100ms_entries": len(all_raw_entries),
        "keep_t2_raw_100ms_entries": len(keep_raw_entries),
        "all_processed_overlap_500ms_entries": len(all_overlap_entries),
        "keep_t2_overlap_500ms_entries": len(keep_overlap_entries),
        "all_processed_raw_100ms_pkl": str(all_raw_pkl),
        "keep_t2_raw_100ms_pkl": str(keep_raw_pkl),
        "all_processed_overlap_500ms_pkl": str(all_overlap_pkl),
        "keep_t2_overlap_500ms_pkl": str(keep_overlap_pkl),
    }
    write_batch_report(pkl_save_root, batch_summary)
    print(f"[batch] saved {len(all_raw_entries)} raw_100ms entries to {all_raw_pkl}")
    print(f"[batch] saved {len(keep_raw_entries)} keep_t2 raw_100ms entries to {keep_raw_pkl}")
    print(f"[batch] saved {len(all_overlap_entries)} overlap_500ms entries to {all_overlap_pkl}")
    print(f"[batch] saved {len(keep_overlap_entries)} keep_t2 overlap_500ms entries to {keep_overlap_pkl}")


def main():
    args = parse_args()
    calib = load_camera_projections(args.yaml_path)

    if args.aligned_manifest:
        aligned_manifest = Path(args.aligned_manifest).resolve()
        raw_output_pkl = derive_single_output_path(args, aligned_manifest)
        overlap_output_pkl = raw_output_pkl.with_name(
            raw_output_pkl.name.replace("_rectified.pkl", "_overlap_500ms.pkl")
            if raw_output_pkl.name.endswith("_rectified.pkl")
            else raw_output_pkl.stem + "_overlap_500ms.pkl"
        )
        if raw_output_pkl.name.endswith("_overlap_500ms.pkl"):
            raw_output_pkl = raw_output_pkl.with_name(raw_output_pkl.name.replace("_overlap_500ms.pkl", "_raw_100ms.pkl"))
        rectified_root = Path(args.rectified_root).resolve() if args.rectified_root else None
        build_per_bag_pkl(aligned_manifest, calib, rectified_root, raw_output_pkl, overlap_output_pkl)
        return

    if args.manifest_csv:
        batch_mode(args, calib)
        return

    raise ValueError("Provide either --aligned_manifest for single mode or --manifest_csv for batch mode.")


if __name__ == "__main__":
    main()
