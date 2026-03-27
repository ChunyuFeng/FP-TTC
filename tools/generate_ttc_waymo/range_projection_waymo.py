"""
Generate range-view scale & risk (orientation) ground truth for Waymo.

Mirrors ``tools/generate_ttc_nuscenes/range_projection.py`` but adapted for
the Waymo Open Dataset:
  * 5 cameras (no rear) → W = 1600 (5 × 320) by default
  * Vehicle frame: x-forward, y-left, z-up
  * Camera-LiDAR timestamp matching is trivial (same Frame proto)
  * Calibrations come from the extracted per-segment PKLs

Input:  scene flow pairs (pc_prev.npy, pc_curr.npy) from generate_sf_waymo.py
Output: range_image_prev.npy, range_image_curr.npy + metadata PKL
"""
from __future__ import annotations

import os
import sys
import re
import glob
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.waymo_paths import (
    WAYMO_CAMERAS,
    WAYMO_CAMERA_ID_TO_NAME,
    make_waymo_relative_path,
)


# =====================================================================
# Range projection (spherical) — same formula as nuScenes version
# =====================================================================

def range_projection(
    points: np.ndarray,
    scales: np.ndarray,
    risk_score: np.ndarray,
    H: int = 160,
    W: int = 1600,
    fov_up: float = 5.0,
    fov_down: float = -25.0,
):
    """
    Project 3D points to a 2D spherical range image.

    In Waymo vehicle frame (x-forward, y-left, z-up):
      yaw = -atan2(y, x)  →  forward (x+) maps to yaw=0 → center of image
      pitch = arcsin(z / r)
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("Points should be a numpy array.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points should have shape [m, 3].")

    # Init output arrays
    proj_range = np.full((H, W), -1, dtype=np.float32)
    proj_scale = np.full((H, W), -1, dtype=np.float32)
    proj_risk_score = np.full((H, W), 0.0, dtype=np.float32)
    proj_xyz = np.full((H, W, 3), -1, dtype=np.float32)
    proj_idx = np.full((H, W), -1, dtype=np.int32)

    fov_up_rad = fov_up / 180.0 * np.pi
    fov_down_rad = fov_down / 180.0 * np.pi
    fov = abs(fov_down_rad) + abs(fov_up_rad)

    depth = np.linalg.norm(points, axis=1)

    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(np.clip(scan_z / np.maximum(depth, 1e-8), -1.0, 1.0))

    # Normalize to [0, 1]
    proj_x = 0.5 * (yaw / np.pi + 1.0)
    proj_y = 1.0 - (pitch + abs(fov_down_rad)) / fov

    # Scale to image size
    proj_x = np.floor(proj_x * W).astype(np.int32)
    proj_x = np.clip(proj_x, 0, W - 1)
    proj_y = np.floor(proj_y * H).astype(np.int32)
    proj_y = np.clip(proj_y, 0, H - 1)

    # Sort far-to-near (closer overwrites)
    order = np.argsort(depth)[::-1]
    depth_sorted = depth[order]
    scale_sorted = scales[order]
    risk_sorted = risk_score[order]
    points_sorted = points[order]
    px_sorted = proj_x[order]
    py_sorted = proj_y[order]

    proj_range[py_sorted, px_sorted] = depth_sorted
    proj_scale[py_sorted, px_sorted] = scale_sorted
    proj_risk_score[py_sorted, px_sorted] = risk_sorted
    proj_xyz[py_sorted, px_sorted] = points_sorted
    proj_idx[py_sorted, px_sorted] = order

    proj_mask = (proj_idx >= 0).astype(np.int32)

    return proj_range, proj_scale, proj_risk_score, proj_xyz, proj_idx, proj_mask


# =====================================================================
# Scale & risk computation — identical to nuScenes version
# =====================================================================

def compute_scale(
    points_prev: np.ndarray,
    points_curr: np.ndarray,
    clip_min: float = 0.01,
    clip_max: float = 100.0,
) -> np.ndarray:
    """
    scale = ||p_curr||_xy / ||p_prev||_xy
    """
    eps = 1e-6
    depth_xy_prev = np.linalg.norm(points_prev[:, :2], axis=1)
    depth_xy_curr = np.linalg.norm(points_curr[:, :2], axis=1)
    scale = depth_xy_curr / np.maximum(depth_xy_prev, eps)
    if clip_min is not None and clip_max is not None:
        scale = np.clip(scale, clip_min, clip_max)
    return scale.astype(np.float32)


def preprocess_points(
    points_prev: np.ndarray,
    points_curr: np.ndarray,
    max_points: int = 500_000,
    min_dist: float = 1.0,
    max_dist: float = 80.0,
) -> tuple:
    """
    Filter and downsample point clouds for efficient range projection.
    - Removes points closer than min_dist or farther than max_dist (XY plane)
    - Randomly subsamples to max_points if still too large
    Returns filtered (points_prev, points_curr).
    """
    assert points_prev.shape[0] == points_curr.shape[0]
    n_orig = points_prev.shape[0]

    # Distance filter using prev points (reference frame)
    dist_xy = np.linalg.norm(points_prev[:, :2], axis=1)
    valid = (dist_xy >= min_dist) & (dist_xy <= max_dist)
    points_prev = points_prev[valid]
    points_curr = points_curr[valid]

    # Random subsampling if still too large
    n = points_prev.shape[0]
    if max_points > 0 and n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        idx.sort()  # preserve spatial ordering
        points_prev = points_prev[idx]
        points_curr = points_curr[idx]

    return points_prev, points_curr


def compute_radical_angle(
    points_prev: np.ndarray,
    points_curr: np.ndarray,
    motion_thresh: float = 1e-3,
) -> np.ndarray:
    """
    Angle between the XY motion vector and the radial direction toward origin.
    0 = approaching, π = receding, π/2 = tangential/static.
    """
    p_prev_xy = points_prev[:, :2]
    p_curr_xy = points_curr[:, :2]

    motion_xy = p_curr_xy - p_prev_xy
    motion_mag = np.linalg.norm(motion_xy, axis=1)

    valid = motion_mag >= motion_thresh
    n = points_prev.shape[0]
    angles = np.full(n, np.pi / 2, dtype=np.float32)

    if not np.any(valid):
        return angles

    to_origin = -p_prev_xy[valid]
    dot = np.einsum("ij,ij->i", motion_xy[valid], to_origin)
    norm_motion = motion_mag[valid]
    norm_to_origin = np.linalg.norm(to_origin, axis=1)

    eps = 1e-6
    norm_to_origin = np.maximum(norm_to_origin, eps)
    cos_theta = dot / (norm_motion * norm_to_origin)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles[valid] = np.arccos(cos_theta).astype(np.float32)

    return angles


# =====================================================================
# Folder name parser
# =====================================================================

_SF_FOLDER_PATTERN = re.compile(
    r"^segment_(.+?)_frame_(\d{4})$"
)


def parse_sf_folder(folder_name: str):
    """
    Parse a scene flow folder name.
    Returns (segment_id, frame_idx) or None.
    """
    m = _SF_FOLDER_PATTERN.match(folder_name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def load_pair_meta(sf_path: str):
    """Load explicit prev/curr frame metadata if present."""
    meta_path = os.path.join(sf_path, "pair_meta.pkl")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)


# =====================================================================
# Build frame records from scene flow folders + calibration PKLs
# =====================================================================

def build_frame_records(
    scene_flow_base: str,
    calib_dir: str,
    scene_flow_subdir: str = "scene_flow_key_frames",
):
    """
    Scan scene flow output folders and match them with calibration data.

    Returns a list of frame record dicts, sorted by (segment_id, curr_frame_idx).
    """
    sf_root = os.path.join(scene_flow_base, scene_flow_subdir)
    if not os.path.isdir(sf_root):
        print(f"Scene flow directory not found: {sf_root}")
        return []

    # Index calibration PKLs by segment_id
    calib_by_segment = {}
    if os.path.isdir(calib_dir):
        for pkl_file in glob.glob(os.path.join(calib_dir, "segment_*.pkl")):
            with open(pkl_file, "rb") as f:
                records = pickle.load(f)
            if records:
                sid = records[0]["segment_id"]
                calib_by_frame = {r["frame_idx"]: r for r in records}
                calib_by_segment[sid] = calib_by_frame

    # Scan scene flow folders
    frame_records = []
    for folder in sorted(os.listdir(sf_root)):
        parsed = parse_sf_folder(folder)
        if parsed is None:
            continue
        segment_id, frame_idx_from_folder = parsed

        sf_path = os.path.join(sf_root, folder)
        pc_prev = os.path.join(sf_path, "pc_prev.npy")
        pc_curr = os.path.join(sf_path, "pc_curr.npy")
        if not (os.path.isfile(pc_prev) and os.path.isfile(pc_curr)):
            continue

        pair_meta = load_pair_meta(sf_path)
        if pair_meta is not None:
            prev_frame_idx = int(pair_meta["prev_frame_idx"])
            curr_frame_idx = int(pair_meta["curr_frame_idx"])
            prev_timestamp_us = int(pair_meta.get("prev_timestamp_us", 0))
            curr_timestamp_us = int(pair_meta.get("curr_timestamp_us", 0))
        else:
            curr_frame_idx = frame_idx_from_folder
            prev_frame_idx = max(0, curr_frame_idx - 1)
            prev_timestamp_us = None
            curr_timestamp_us = None

        # Match calibration for both curr and prev frames
        calib_curr = None
        calib_prev = None
        if segment_id in calib_by_segment:
            calib_curr = calib_by_segment[segment_id].get(curr_frame_idx)
            calib_prev = calib_by_segment[segment_id].get(prev_frame_idx)
            if curr_timestamp_us is None and calib_curr is not None:
                curr_timestamp_us = int(calib_curr.get("timestamp_us", 0))
            if prev_timestamp_us is None and calib_prev is not None:
                prev_timestamp_us = int(calib_prev.get("timestamp_us", 0))

        frame_records.append({
            "segment_id": segment_id,
            "frame_idx": curr_frame_idx,
            "prev_frame_idx": prev_frame_idx,
            "curr_timestamp_us": curr_timestamp_us,
            "prev_timestamp_us": prev_timestamp_us,
            "sf_folder": folder,
            "sf_path": sf_path,
            "pc_prev_path": pc_prev,
            "pc_curr_path": pc_curr,
            "calibration": calib_curr,
            "calibration_prev": calib_prev,
        })

    frame_records.sort(key=lambda r: (r["segment_id"], r["frame_idx"]))
    return frame_records


# =====================================================================
# Main
# =====================================================================

def main(args):
    H = args.range_H
    W = args.range_W
    fov_up = args.fov_up
    fov_down = args.fov_down

    scene_flow_base = args.scene_flow_path
    calib_dir = os.path.join(args.dataset_root, "camera_calibrations")
    scene_flow_subdir = "scene_flow_key_frames" if args.keyframe_only else "scene_flow_all_frames"

    # Build frame records
    frame_records = build_frame_records(
        scene_flow_base, calib_dir, scene_flow_subdir
    )
    if not frame_records:
        print("No frame records found!")
        return

    print(f"Found {len(frame_records)} scene flow frames")

    # Output directory naming (mirrors nuScenes convention)
    range_image_dir_name = (
        f"range_image_{scene_flow_subdir}_{H}_{W}_fov_{abs(int(fov_up))}_{abs(int(fov_down))}"
    )
    gt_map_save_path = os.path.join(args.gt_map_path, args.split, range_image_dir_name)
    os.makedirs(gt_map_save_path, exist_ok=True)

    infos_list = []
    pair_count = 0

    # Each scene flow folder already contains a complete (prev, curr) pair.
    # Process each record individually, not by pairing across folders.
    for rec in tqdm(frame_records, desc="Processing frame pairs"):
        # Check if range images already exist (skip if so)
        out_folder = os.path.join(gt_map_save_path, rec["sf_folder"])
        ri_prev_path = os.path.join(out_folder, "range_image_prev.npy")
        ri_curr_path = os.path.join(out_folder, "range_image_curr.npy")
        range_images_exist = os.path.isfile(ri_prev_path) and os.path.isfile(ri_curr_path)

        if not range_images_exist:
            # Load scene flow — both from the SAME folder
            points_prev = np.load(rec["pc_prev_path"])  # (N, 3)
            points_curr = np.load(rec["pc_curr_path"])  # (N, 3)

            assert points_prev.shape == points_curr.shape, (
                f"Shape mismatch in {rec['sf_folder']}: "
                f"prev={points_prev.shape}, curr={points_curr.shape}"
            )

            # Preprocess: distance filter + downsample for performance
            points_prev, points_curr = preprocess_points(
                points_prev, points_curr,
                max_points=args.max_points,
                min_dist=args.min_dist,
                max_dist=args.max_dist,
            )

        if rec["prev_timestamp_us"] is not None and rec["curr_timestamp_us"] is not None:
            time_diff_sf_us = max(1, int(rec["curr_timestamp_us"] - rec["prev_timestamp_us"]))
        else:
            time_diff_sf_us = 100000  # default 0.1s at 10Hz

        # Waymo cameras are synchronized with LiDAR in the same frame.
        time_diff_cam_us = time_diff_sf_us
        time_diff_ratio = float(time_diff_cam_us) / float(time_diff_sf_us)

        if not range_images_exist:
            # Temporal alignment (identity for ratio=1.0)
            points_prev_aligned = points_curr - time_diff_ratio * (points_curr - points_prev)

            # Compute scale and risk
            scale = compute_scale(points_prev_aligned, points_curr)
            risk_score = compute_radical_angle(points_prev_aligned, points_curr)

            # Project prev (aligned) to range image
            (prev_depth, prev_scale, prev_risk, prev_xyz,
             prev_idx, prev_mask) = range_projection(
                points_prev_aligned, scale, risk_score, H=H, W=W,
                fov_up=fov_up, fov_down=fov_down,
            )

            # Project curr to range image
            (curr_depth, curr_scale, curr_risk, curr_xyz,
             curr_idx, curr_mask) = range_projection(
                points_curr, scale, risk_score, H=H, W=W,
                fov_up=fov_up, fov_down=fov_down,
            )

            # Save range images
            os.makedirs(out_folder, exist_ok=True)

            range_image_prev = {
                "depth": prev_depth, "scale": prev_scale,
                "risk_score": prev_risk, "xyz": prev_xyz,
                "idx": prev_idx, "mask": prev_mask,
            }
            range_image_curr = {
                "depth": curr_depth, "scale": curr_scale,
                "risk_score": curr_risk, "xyz": curr_xyz,
                "idx": curr_idx, "mask": curr_mask,
            }

            np.save(ri_prev_path, range_image_prev)
            np.save(ri_curr_path, range_image_curr)

        # Determine prev frame index
        prev_frame_idx = rec["prev_frame_idx"]

        # Build info record
        info = {
            "segment_id": rec["segment_id"],
            "prev_frame_idx": prev_frame_idx,
            "curr_frame_idx": rec["frame_idx"],
            "prev_timestamp_us": rec["prev_timestamp_us"],
            "curr_timestamp_us": rec["curr_timestamp_us"],
            "sf_folder": rec["sf_folder"],
            "gt_map_path": make_waymo_relative_path(out_folder),
            "scene_flow_path": make_waymo_relative_path(rec["sf_path"]),
            "time_diff_cam_us": float(time_diff_cam_us),
            "time_diff_sf_us": float(time_diff_sf_us),
            "time_diff_ratio": float(time_diff_ratio),
        }

        # Add curr frame camera info
        if rec["calibration"]:
            calib = rec["calibration"]
            info["curr_ego_pose"] = calib["ego_pose"]
            info["curr_cameras"] = calib.get("cameras", {})

        # Add prev frame camera info
        if rec["calibration_prev"]:
            calib_prev = rec["calibration_prev"]
            info["prev_ego_pose"] = calib_prev["ego_pose"]
            info["prev_cameras"] = calib_prev.get("cameras", {})

        # Build sensor metas for both frames (for dataloader compatibility)
        def _build_sensor_metas(ego_pose, cameras_dict):
            sm = {"lidar": {"ego_pose": ego_pose}}
            if cameras_dict:
                cam_calibs = {}
                cam_ego_poses = {}
                for cam_name, cam_info in cameras_dict.items():
                    cam_calibs[cam_name] = {
                        "extrinsic": cam_info["extrinsic"],
                        "intrinsic": cam_info["intrinsic"],
                        "width": cam_info["width"],
                        "height": cam_info["height"],
                    }
                    cam_ego_poses[cam_name] = ego_pose
                sm["camera"] = {
                    "calibrated_sensor": cam_calibs,
                    "ego_pose": cam_ego_poses,
                }
            return sm

        info["sensor_metas_curr"] = _build_sensor_metas(
            info.get("curr_ego_pose"), info.get("curr_cameras"))
        info["sensor_metas_prev"] = _build_sensor_metas(
            info.get("prev_ego_pose"), info.get("prev_cameras"))

        infos_list.append(info)
        pair_count += 1

    # Save PKL
    pkl_dir = os.path.join(args.infos_path, args.split)
    os.makedirs(pkl_dir, exist_ok=True)
    pkl_name = (
        f"waymo_{args.split}_infos_{scene_flow_subdir}_{H}_{W}"
        f"_fov_{abs(int(fov_up))}_{abs(int(fov_down))}.pkl"
    )
    pkl_path = os.path.join(pkl_dir, pkl_name)
    with open(pkl_path, "wb") as f:
        pickle.dump(infos_list, f)

    print(f"\nSaved {pair_count} frame pairs")
    print(f"Range images: {gt_map_save_path}")
    print(f"Info PKL: {pkl_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate range-view GT for Waymo")

    parser.add_argument("--dataset_root", type=str, default="./Datasets/waymo",
                        help="Waymo processed dataset root")
    parser.add_argument("--scene_flow_path", type=str,
                        default="./Datasets/waymo/0_scene_flow",
                        help="Scene flow output directory")
    parser.add_argument("--gt_map_path", type=str,
                        default="./Datasets/waymo/1_gt_map",
                        help="Output directory for GT range images")
    parser.add_argument("--infos_path", type=str,
                        default="./Datasets/waymo/2_trainval_test_infos",
                        help="Output directory for info PKLs")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"])

    # Range image params
    parser.add_argument("--range_H", type=int, default=160)
    parser.add_argument("--range_W", type=int, default=1600,
                        help="1600 = 5 cameras × 320 pixels")
    parser.add_argument("--fov_up", type=float, default=5.0)
    parser.add_argument("--fov_down", type=float, default=-25.0)

    parser.add_argument("--keyframe_only", action="store_true", default=True)

    # Point cloud preprocessing
    parser.add_argument("--max_points", type=int, default=500000,
                        help="Max points per cloud after filtering (0=no limit)")
    parser.add_argument("--min_dist", type=float, default=1.0,
                        help="Min XY distance to keep (meters)")
    parser.add_argument("--max_dist", type=float, default=80.0,
                        help="Max XY distance to keep (meters)")

    args = parser.parse_args()
    main(args)
