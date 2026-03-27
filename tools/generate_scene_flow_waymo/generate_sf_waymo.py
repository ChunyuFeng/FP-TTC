"""
Generate scene flow ground truth for the Waymo Open Dataset.

This script mirrors ``tools/generate_scene_flow_nuscenes/generate_sf_nuscenes.py``
but replaces all nuScenes-specific APIs with the official Waymo Open Dataset SDK.

Key differences vs the nuScenes version:
  * Vehicle frame (x-forward, y-left, z-up) instead of LIDAR_TOP frame.
  * 5 LiDARs merged per frame (TOP + 4 SIDE) via the official SDK.
  * Waymo heading = rotation from x-axis (forward), counter-clockwise.
    No π/2 offset needed for mmcv ``points_in_boxes_cpu``.
  * Ego pose is a 4×4 matrix (vehicle → world), not quaternion + translation.
  * Tracking ID is a string (``label.id``), used as the instance token.

Output structure (under ``<save_path>/scene_flow_key_frames/``):
    segment_<id>_frame_<idx>/
        pc_prev.npy   # (N, 3) float32 — aligned point cloud in prev vehicle frame
        pc_curr.npy   # (N, 3) float32 — aligned point cloud in curr vehicle frame
"""
from __future__ import annotations

import os
import sys
import glob
import time
import pickle
import numpy as np
import torch
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation

# mmcv for point-in-box test (same as nuScenes version)
from mmcv.ops.points_in_boxes import points_in_boxes_cpu

# ---- project imports ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tools.generate_scene_flow_waymo.waymo_reader import (
    iter_frames,
    extract_merged_point_cloud,
    extract_boxes,
    extract_ego_pose,
    extract_timestamp_us,
    extract_segment_id,
    extract_camera_images,
    extract_camera_calibrations,
    waymo_boxes_to_mmcv_format,
    vehicle_frame_to_vehicle_frame,
)

# =====================================================================
# Helper: Waymo category → bbox family (for class-aware dilation)
# =====================================================================

WAYMO_TYPE_VEHICLE = 1
WAYMO_TYPE_PEDESTRIAN = 2
WAYMO_TYPE_SIGN = 3
WAYMO_TYPE_CYCLIST = 4


def get_waymo_bbox_family(category_name: str) -> str:
    """Map Waymo category name to a bbox-dilation family."""
    mapping = {
        "vehicle": "vehicle",
        "pedestrian": "pedestrian",
        "cyclist": "two_wheeler",
        "sign": "small_object",
    }
    return mapping.get(category_name, "fallback")


# =====================================================================
# Separation box builder (reused from nuScenes version)
# =====================================================================

def build_separation_boxes(
    base_gt_bbox_3d: np.ndarray,
    object_categories: list,
    bbox_adjustment_mode: str,
    global_bbox_dilation: np.ndarray,
    global_bbox_shift: np.ndarray,
) -> np.ndarray:
    """
    Build enlarged boxes used to separate dynamic vs. static points.
    Identical logic to the nuScenes version.
    """
    separation_gt_bbox_3d = base_gt_bbox_3d.copy()

    if bbox_adjustment_mode == "global":
        separation_gt_bbox_3d[:, :3] += global_bbox_shift
        separation_gt_bbox_3d[:, 3:6] += global_bbox_dilation
        return separation_gt_bbox_3d

    if bbox_adjustment_mode != "class_aware":
        raise ValueError(f"Unsupported bbox_adjustment_mode={bbox_adjustment_mode}")

    family_ratio = {
        "vehicle":      np.asarray([0.25, 0.15, 0.20], dtype=np.float32),
        "two_wheeler":  np.asarray([0.15, 0.12, 0.15], dtype=np.float32),
        "pedestrian":   np.asarray([0.08, 0.08, 0.10], dtype=np.float32),
        "small_object": np.asarray([0.10, 0.10, 0.10], dtype=np.float32),
        "fallback":     np.asarray([0.0,  0.0,  0.0],  dtype=np.float32),
    }
    family_cap = {
        "vehicle":      np.asarray([0.50, 0.50, 0.30], dtype=np.float32),
        "two_wheeler":  np.asarray([0.12, 0.12, 0.12], dtype=np.float32),
        "pedestrian":   np.asarray([0.05, 0.05, 0.10], dtype=np.float32),
        "small_object": np.asarray([0.05, 0.05, 0.08], dtype=np.float32),
        "fallback":     np.asarray([0.0,  0.0,  0.0],  dtype=np.float32),
    }

    dims = separation_gt_bbox_3d[:, 3:6]
    effective_dilation = np.zeros_like(dims, dtype=np.float32)
    effective_shift = np.zeros((separation_gt_bbox_3d.shape[0], 3), dtype=np.float32)

    for i, category_name in enumerate(object_categories):
        family = get_waymo_bbox_family(category_name)
        effective_dilation[i] = np.minimum(family_cap[family], family_ratio[family] * dims[i])

        height = float(dims[i, 2])
        if family == "vehicle":
            effective_shift[i, 2] = min(0.15, 0.10 * height)
        elif family == "two_wheeler":
            effective_shift[i, 2] = min(0.05, 0.05 * height)

    separation_gt_bbox_3d[:, :3] += effective_shift
    separation_gt_bbox_3d[:, 3:6] += effective_dilation
    return separation_gt_bbox_3d


# =====================================================================
# Ground dedup (copied from nuScenes version — algorithm is generic)
# =====================================================================

def dedup_static_ground_layers(
    points_xyz: np.ndarray,
    grid_size: float,
    ground_z_max: float,
    min_points_per_cell: int,
    cell_z_span_max: float,
):
    """BEV grid-based de-layering for near-ground static background points.
    Identical logic to the nuScenes version."""
    keep_mask = np.ones(points_xyz.shape[0], dtype=bool)

    if points_xyz.shape[0] == 0:
        return keep_mask, {
            "ground_candidate_count": 0,
            "collapsed_cell_count": 0,
            "removed_point_count": 0,
        }

    ground_candidate_indices = np.flatnonzero(points_xyz[:, 2] < ground_z_max)
    if ground_candidate_indices.size == 0:
        return keep_mask, {
            "ground_candidate_count": 0,
            "collapsed_cell_count": 0,
            "removed_point_count": 0,
        }

    ground_points = points_xyz[ground_candidate_indices]
    grid_x = np.floor(ground_points[:, 0] / grid_size).astype(np.int32)
    grid_y = np.floor(ground_points[:, 1] / grid_size).astype(np.int32)
    grid_coords = np.stack([grid_x, grid_y], axis=1)

    _, inverse = np.unique(grid_coords, axis=0, return_inverse=True)
    sort_order = np.argsort(inverse, kind="mergesort")
    sorted_group_ids = inverse[sort_order]
    sorted_ground_indices = ground_candidate_indices[sort_order]
    sorted_ground_z = ground_points[sort_order, 2]

    group_starts = np.flatnonzero(np.r_[True, sorted_group_ids[1:] != sorted_group_ids[:-1]])
    group_counts = np.diff(np.r_[group_starts, sorted_group_ids.size])
    group_mins = np.minimum.reduceat(sorted_ground_z, group_starts)
    group_maxs = np.maximum.reduceat(sorted_ground_z, group_starts)
    group_spans = group_maxs - group_mins

    collapsed_cell_count = 0
    removed_point_count = 0

    for start, count, span in zip(group_starts, group_counts, group_spans):
        if count < min_points_per_cell or span > cell_z_span_max:
            continue
        group_slice = slice(start, start + count)
        group_indices = sorted_ground_indices[group_slice]
        group_z = sorted_ground_z[group_slice]
        median_z = np.median(group_z)
        representative_offset = int(np.argmin(np.abs(group_z - median_z)))
        representative_index = group_indices[representative_offset]
        keep_mask[group_indices] = False
        keep_mask[representative_index] = True
        collapsed_cell_count += 1
        removed_point_count += count - 1

    return keep_mask, {
        "ground_candidate_count": int(ground_candidate_indices.size),
        "collapsed_cell_count": int(collapsed_cell_count),
        "removed_point_count": int(removed_point_count),
    }


# =====================================================================
# Main per-segment processing
# =====================================================================

def process_segment(tfrecord_path: str, segment_idx: int, args):
    """
    Process a single Waymo segment (TFRecord file).
    Mirrors ``main()`` in the nuScenes version.
    """

    save_path = args.save_path
    bbox_dilation = np.asarray(args.bbox_dilation, dtype=np.float32)
    bbox_shift = np.asarray(args.bbox_shift, dtype=np.float32)

    segment_id = os.path.basename(tfrecord_path).replace("segment-", "").replace(
        "_with_camera_labels.tfrecord", ""
    )
    print(f"\n{'='*60}")
    print(f"Processing segment {segment_idx}: {segment_id}")
    print(f"TFRecord: {tfrecord_path}")
    print(f"{'='*60}")

    t0 = time.time()

    # ==================================================================
    # Phase 1: Collect all frames in the segment
    # ==================================================================
    dict_list = []
    pose_ref = None  # first frame's ego pose (reference frame)

    for frame_idx, frame in enumerate(iter_frames(tfrecord_path)):
        # ---- Ego pose ----
        pose = extract_ego_pose(frame)
        if pose_ref is None:
            pose_ref = pose.copy()

        # ---- Point cloud (all 5 LiDARs merged, vehicle frame) ----
        pc = extract_merged_point_cloud(frame, return_intensity=False)  # (N, 3)

        # ---- 3D boxes ----
        raw_boxes, tracking_ids, box_types, category_names = extract_boxes(frame)
        # Convert to mmcv format: [cx, cy, z_bottom, l, w, h, heading]
        mmcv_boxes = waymo_boxes_to_mmcv_format(raw_boxes)

        # Build separation boxes (enlarged for static/dynamic separation)
        if mmcv_boxes.shape[0] > 0:
            separation_boxes = build_separation_boxes(
                mmcv_boxes,
                category_names,
                args.bbox_adjustment_mode,
                bbox_dilation,
                bbox_shift,
            )
        else:
            separation_boxes = mmcv_boxes.copy()

        # ---- Separate dynamic / static points ----
        object_points_list = []
        if separation_boxes.shape[0] > 0:
            points_in_boxes = points_in_boxes_cpu(
                torch.from_numpy(pc[:, :3][np.newaxis, :, :]),
                torch.from_numpy(separation_boxes[np.newaxis, :]),
            )
            for j in range(points_in_boxes.shape[-1]):
                obj_mask = points_in_boxes[0][:, j].bool()
                object_points_list.append(pc[obj_mask])

            # Points NOT in any box
            any_box_mask = torch.sum(points_in_boxes, dim=-1).bool()
            points_mask = ~any_box_mask[0]
        else:
            points_mask = torch.ones(pc.shape[0], dtype=torch.bool)

        # ---- Remove ego-vehicle vicinity ----
        self_range = [3.0, 3.0, 3.0]
        ego_mask = torch.from_numpy(
            (np.abs(pc[:, 0]) > self_range[0])
            | (np.abs(pc[:, 1]) > self_range[1])
            | (np.abs(pc[:, 2]) > self_range[2])
        )
        points_mask = points_mask & ego_mask
        static_pc = pc[points_mask.numpy()]

        # ---- Transform static points to the reference frame ----
        static_pc_ref = vehicle_frame_to_vehicle_frame(
            static_pc, pose_src=pose, pose_dst=pose_ref
        )

        # ---- Store frame info ----
        # Store raw_boxes (original Waymo format) for later box placement
        # heading is the last column
        frame_info = {
            "tracking_ids": tracking_ids,
            "object_points_list": object_points_list,
            "static_pc_ref": static_pc_ref,  # (M, 3) in reference frame
            "pose": pose,                     # (4, 4) vehicle → world
            "mmcv_boxes": mmcv_boxes,         # (K, 7) [cx, cy, z_bottom, l, w, h, heading]
            "raw_boxes": raw_boxes,           # (K, 7) [cx, cy, cz, l, w, h, heading]
            "category_names": category_names,
            "frame_idx": frame_idx,
            "timestamp_us": extract_timestamp_us(frame),
            "segment_id": segment_id,
        }
        dict_list.append(frame_info)

    if len(dict_list) < 2:
        print(f"  Segment has < 2 frames, skipping.")
        return

    print(f"  Collected {len(dict_list)} frames in {time.time() - t0:.1f}s")

    # ==================================================================
    # Phase 2: Aggregate object point clouds across all frames
    # ==================================================================
    t1 = time.time()

    # Gather all unique tracking IDs
    object_token_zoo = []
    token_set = set()
    for fi in dict_list:
        for i, tid in enumerate(fi["tracking_ids"]):
            if tid not in token_set and fi["object_points_list"][i].shape[0] > 0:
                token_set.add(tid)
                object_token_zoo.append(tid)

    # Build canonical object point clouds
    # For each object: collect points from all frames, transform to canonical
    # coordinates (subtract box center, un-rotate by heading)
    object_points_dict = {}
    for query_tid in object_token_zoo:
        parts = []
        for fi in dict_list:
            for i, tid in enumerate(fi["tracking_ids"]):
                if tid != query_tid:
                    continue
                obj_pts = fi["object_points_list"][i]
                if obj_pts.shape[0] == 0:
                    continue
                # Transform to canonical: subtract center, un-rotate
                box = fi["raw_boxes"][i]  # [cx, cy, cz, l, w, h, heading]
                center = box[:3]
                heading = box[6]
                local_pts = obj_pts[:, :3] - center
                rot = Rotation.from_euler("z", -heading, degrees=False)
                canonical_pts = rot.apply(local_pts)
                parts.append(canonical_pts)
        if parts:
            object_points_dict[query_tid] = np.concatenate(parts, axis=0).astype(np.float32)

    print(f"  Aggregated {len(object_points_dict)} unique objects in {time.time() - t1:.1f}s")

    # ==================================================================
    # Phase 3: Generate frame pairs
    # ==================================================================
    t2 = time.time()

    if args.keyframe_only:
        # Waymo: all frames have annotations, so "key frames" = all frames
        # But we can subsample: e.g. every N-th frame
        step = max(1, args.keyframe_step)
        keyframe_indices = list(range(0, len(dict_list), step))
        frame_pairs = [
            (keyframe_indices[i - 1], keyframe_indices[i])
            for i in range(1, len(keyframe_indices))
        ]
        progress_desc = f"Processing key frames (step={step})"
        scene_flow_subdir = "scene_flow_key_frames"
    else:
        frame_pairs = [(i - 1, i) for i in range(1, len(dict_list))]
        progress_desc = "Processing frames"
        scene_flow_subdir = "scene_flow_all_frames"

    pairs_saved = 0
    for prev_idx, curr_idx in tqdm(frame_pairs, desc=progress_desc):
        prev_dict = dict_list[prev_idx]
        curr_dict = dict_list[curr_idx]

        # ---- Aggregate static points ----
        if args.static_context_mode == "scene":
            static_dict_list = dict_list
        elif args.static_context_mode == "local":
            wstart = max(0, min(prev_idx, curr_idx) - args.static_context_radius)
            wend = min(len(dict_list), max(prev_idx, curr_idx) + args.static_context_radius + 1)
            static_dict_list = dict_list[wstart:wend]
        else:
            raise ValueError(f"Unsupported static_context_mode={args.static_context_mode}")

        # Concatenate static points (all in reference frame)
        static_pc_ref = np.concatenate(
            [fi["static_pc_ref"] for fi in static_dict_list], axis=0
        )

        # ---- Optional ground dedup ----
        if args.static_ground_dedup == "bev":
            keep_mask, dedup_stats = dedup_static_ground_layers(
                static_pc_ref[:, :3],
                grid_size=args.static_ground_grid_size,
                ground_z_max=args.static_ground_z_max,
                min_points_per_cell=args.static_ground_min_points_per_cell,
                cell_z_span_max=args.static_ground_cell_z_span_max,
            )
            static_pc_ref = static_pc_ref[keep_mask]
        elif args.static_ground_dedup != "off":
            raise ValueError(f"Unsupported static_ground_dedup={args.static_ground_dedup}")

        # ---- Transform static points to prev and curr vehicle frames ----
        prev_point_cloud = vehicle_frame_to_vehicle_frame(
            static_pc_ref, pose_src=pose_ref, pose_dst=prev_dict["pose"]
        )
        curr_point_cloud = vehicle_frame_to_vehicle_frame(
            static_pc_ref, pose_src=pose_ref, pose_dst=curr_dict["pose"]
        )

        # ---- Find common objects in both frames ----
        prev_box_idx_by_tid = {}
        for j, tid in enumerate(prev_dict["tracking_ids"]):
            if tid not in prev_box_idx_by_tid:
                prev_box_idx_by_tid[tid] = j

        curr_box_idx_by_tid = {}
        for j, tid in enumerate(curr_dict["tracking_ids"]):
            if tid not in curr_box_idx_by_tid:
                curr_box_idx_by_tid[tid] = j

        common_tids = []
        seen = set()
        for tid in prev_dict["tracking_ids"]:
            if tid in seen:
                continue
            if tid not in curr_box_idx_by_tid:
                continue
            if tid not in object_points_dict:
                continue
            common_tids.append(tid)
            seen.add(tid)

        # ---- Place canonical object points into prev & curr boxes ----
        prev_raw = prev_dict["raw_boxes"]   # [cx, cy, cz, l, w, h, heading]
        curr_raw = curr_dict["raw_boxes"]

        filtered_prev_obj_pts = []
        filtered_curr_obj_pts = []

        for tid in common_tids:
            pidx = prev_box_idx_by_tid[tid]
            cidx = curr_box_idx_by_tid[tid]
            canonical = object_points_dict[tid]

            # Place into prev frame
            prev_heading = prev_raw[pidx, 6]
            prev_center = prev_raw[pidx, :3]
            prev_rot = Rotation.from_euler("z", prev_heading, degrees=False)
            prev_obj = prev_rot.apply(canonical) + prev_center
            filtered_prev_obj_pts.append(prev_obj)

            # Place into curr frame
            curr_heading = curr_raw[cidx, 6]
            curr_center = curr_raw[cidx, :3]
            curr_rot = Rotation.from_euler("z", curr_heading, degrees=False)
            curr_obj = curr_rot.apply(canonical) + curr_center
            filtered_curr_obj_pts.append(curr_obj)

        # ---- Concatenate static + object points ----
        if filtered_prev_obj_pts:
            prev_scene_points = np.concatenate(
                [prev_point_cloud] + filtered_prev_obj_pts, axis=0
            )
            curr_scene_points = np.concatenate(
                [curr_point_cloud] + filtered_curr_obj_pts, axis=0
            )
        else:
            prev_scene_points = prev_point_cloud
            curr_scene_points = curr_point_cloud

        if prev_scene_points.shape[0] != curr_scene_points.shape[0]:
            raise RuntimeError(
                f"Point count mismatch: segment={segment_id}, prev_idx={prev_idx}, "
                f"curr_idx={curr_idx}, prev={prev_scene_points.shape[0]}, "
                f"curr={curr_scene_points.shape[0]}"
            )

        # ---- Spatial cropping ----
        prev_mask = (
            (np.abs(prev_scene_points[:, 0]) < 50)
            & (np.abs(prev_scene_points[:, 1]) < 50)
            & (prev_scene_points[:, 2] > -5.0)
            & (prev_scene_points[:, 2] < 3.0)
        )
        curr_mask = (
            (np.abs(curr_scene_points[:, 0]) < 50)
            & (np.abs(curr_scene_points[:, 1]) < 50)
            & (curr_scene_points[:, 2] > -5.0)
            & (curr_scene_points[:, 2] < 3.0)
        )
        keep = prev_mask & curr_mask

        prev_scene_points = prev_scene_points[keep].astype(np.float32, copy=False)
        curr_scene_points = curr_scene_points[keep].astype(np.float32, copy=False)

        # ---- Save ----
        folder_name = f"segment_{segment_id}_frame_{curr_dict['frame_idx']:04d}"
        out_dir = os.path.join(save_path, scene_flow_subdir, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "pc_prev.npy"), prev_scene_points)
        np.save(os.path.join(out_dir, "pc_curr.npy"), curr_scene_points)
        pair_meta = {
            "segment_id": segment_id,
            "prev_frame_idx": int(prev_dict["frame_idx"]),
            "curr_frame_idx": int(curr_dict["frame_idx"]),
            "prev_timestamp_us": int(prev_dict["timestamp_us"]),
            "curr_timestamp_us": int(curr_dict["timestamp_us"]),
        }
        with open(os.path.join(out_dir, "pair_meta.pkl"), "wb") as f:
            pickle.dump(pair_meta, f)
        pairs_saved += 1

    print(f"  Saved {pairs_saved} frame pairs in {time.time() - t2:.1f}s")
    print(f"  Total segment time: {time.time() - t0:.1f}s")


# =====================================================================
# Main entry
# =====================================================================

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate scene flow for Waymo Open Dataset")

    # --- paths ---
    parser.add_argument("--dataroot", type=str, default="/mnt/data2/waymo",
                        help="Root directory of raw Waymo data")
    parser.add_argument("--split", type=str, default="training",
                        choices=["training", "validation"],
                        help="Dataset split")
    parser.add_argument("--save_path", type=str,
                        default="./Datasets/waymo/0_scene_flow",
                        help="Output directory for scene flow")
    parser.add_argument("--tfrecord_list", type=str, default=None,
                        help="Optional text file listing TFRecord paths (one per line)")

    # --- segment selection ---
    parser.add_argument("--start", type=int, default=0,
                        help="Start index in the sorted TFRecord list")
    parser.add_argument("--end", type=int, default=5,
                        help="End index (exclusive) in the sorted TFRecord list")

    # --- frame pairing ---
    parser.add_argument("--keyframe_only", action="store_true",
                        help="Only export scene flow between key frames")
    parser.add_argument("--keyframe_step", type=int, default=1,
                        help="Step between key frames (1 = every frame, 2 = every other, etc.)")

    # --- bbox separation ---
    parser.add_argument("--bbox_adjustment_mode", type=str, default="class_aware",
                        choices=["global", "class_aware"],
                        help="How to enlarge boxes for static/dynamic separation")
    parser.add_argument("--bbox_dilation", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],
                        help="Global bbox dilation [dx, dy, dz]")
    parser.add_argument("--bbox_shift", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],
                        help="Global bbox shift [x, y, z]")

    # --- static background ---
    parser.add_argument("--static_context_mode", type=str, default="scene",
                        choices=["scene", "local"],
                        help="Static background aggregation mode")
    parser.add_argument("--static_context_radius", type=int, default=10,
                        help="Number of frames on each side for local mode")
    parser.add_argument("--static_ground_dedup", type=str, default="off",
                        choices=["off", "bev"],
                        help="Ground de-layering mode")
    parser.add_argument("--static_ground_grid_size", type=float, default=0.15)
    parser.add_argument("--static_ground_z_max", type=float, default=-1.0)
    parser.add_argument("--static_ground_min_points_per_cell", type=int, default=3)
    parser.add_argument("--static_ground_cell_z_span_max", type=float, default=0.4)

    args = parser.parse_args()

    # ---- Discover TFRecord files ----
    if args.tfrecord_list:
        with open(args.tfrecord_list, "r") as f:
            tfrecord_files = [line.strip() for line in f if line.strip()]
    else:
        pattern = os.path.join(args.dataroot, args.split, "*.tfrecord")
        tfrecord_files = sorted(glob.glob(pattern))

    if not tfrecord_files:
        print(f"No TFRecord files found at {args.dataroot}/{args.split}/")
        sys.exit(1)

    print(f"Found {len(tfrecord_files)} TFRecord files")

    # ---- Slice ----
    start = max(0, args.start)
    end = min(len(tfrecord_files), args.end)
    selected = tfrecord_files[start:end]
    print(f"Processing segments [{start}, {end}): {len(selected)} segments")

    # ---- Process ----
    for seg_idx, tfrecord_path in enumerate(selected):
        process_segment(tfrecord_path, segment_idx=start + seg_idx, args=args)

    print("\nDone!")
