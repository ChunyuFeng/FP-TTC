import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv.ops.points_in_boxes import points_in_boxes_cpu
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.signal import convolve2d
from scipy.spatial.transform import Rotation

from tools.generate_scene_flow_nuscenes.generate_sf_nuscenes import lidar_to_world_to_lidar
from tools.generate_ttc_nuscenes.range_projection import range_projection
from utils.nusc_paths import resolve_nusc_path


DEFAULT_SCENE24_TAIL = (
    "scene_24_n015-2018-07-18-11-50-34+0800__LIDAR_TOP__1531886088898203"
)
DEFAULT_SCENE24_CONTROL = (
    "scene_24_n015-2018-07-18-11-50-34+0800__LIDAR_TOP__1531886077448002"
)


@dataclass
class SceneContext:
    scene_index: int
    scene_name: str
    records: list
    object_points_xyz_by_token: dict
    first_lidar_calibrated_sensor: dict
    first_lidar_ego_pose: dict
    full_static_exact_first: np.ndarray
    full_static_dilated_first: np.ndarray
    full_static_shell_first: np.ndarray


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_box_membership(points_xyz: np.ndarray, boxes_3d: np.ndarray) -> np.ndarray:
    if boxes_3d.shape[0] == 0:
        return np.zeros((points_xyz.shape[0], 0), dtype=bool)

    points_tensor = torch.from_numpy(points_xyz[np.newaxis, :, :]).float()
    boxes_tensor = torch.from_numpy(boxes_3d[np.newaxis, :, :]).float()
    points_in_boxes = points_in_boxes_cpu(points_tensor, boxes_tensor)
    return points_in_boxes[0].cpu().numpy().astype(bool)


def build_gt_bbox_3d(boxes) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    locs = np.array([b.center for b in boxes], dtype=np.float32).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes], dtype=np.float32).reshape(-1, 3)
    rots = np.array(
        [b.orientation.yaw_pitch_roll[0] for b in boxes],
        dtype=np.float32,
    ).reshape(-1, 1)
    gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
    gt_bbox_3d[:, 6] += np.pi / 2.0
    gt_bbox_3d[:, 2] -= dims[:, 2] / 2.0
    return gt_bbox_3d


def dilate_boxes_3d(boxes_3d: np.ndarray, dilation_xyz: tuple[float, float, float]) -> np.ndarray:
    if boxes_3d.shape[0] == 0:
        return boxes_3d.copy()

    dx, dy, dz = dilation_xyz
    dilated = boxes_3d.copy()
    dilated[:, 3] += dx
    dilated[:, 4] += dy
    dilated[:, 5] += dz
    dilated[:, 2] -= dz / 2.0
    return dilated


def self_mask(points_xyz: np.ndarray) -> np.ndarray:
    self_range = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    return (
        (np.abs(points_xyz[:, 0]) > self_range[0]) |
        (np.abs(points_xyz[:, 1]) > self_range[1]) |
        (np.abs(points_xyz[:, 2]) > self_range[2])
    )


def to_first_lidar(points: np.ndarray, lidar_calibrated_sensor: dict, lidar_ego_pose: dict,
                   first_lidar_calibrated_sensor: dict, first_lidar_ego_pose: dict) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if points.shape[1] == 3:
        intensity = np.ones((points.shape[0], 1), dtype=points.dtype)
        points_for_transform = np.concatenate([points, intensity], axis=1)
    else:
        points_for_transform = points

    lidar_pc = lidar_to_world_to_lidar(
        points_for_transform.copy(),
        lidar_calibrated_sensor.copy(),
        lidar_ego_pose.copy(),
        first_lidar_calibrated_sensor,
        first_lidar_ego_pose,
    )
    return lidar_pc.points.T[:, :3].astype(np.float32, copy=False)


def transform_from_first_to_target(
    points_first: np.ndarray,
    first_lidar_calibrated_sensor: dict,
    first_lidar_ego_pose: dict,
    target_lidar_calibrated_sensor: dict,
    target_lidar_ego_pose: dict,
) -> np.ndarray:
    if points_first.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if points_first.shape[1] == 3:
        intensity = np.ones((points_first.shape[0], 1), dtype=points_first.dtype)
        points_for_transform = np.concatenate([points_first, intensity], axis=1)
    else:
        points_for_transform = points_first

    lidar_pc = lidar_to_world_to_lidar(
        points_for_transform.copy(),
        first_lidar_calibrated_sensor.copy(),
        first_lidar_ego_pose.copy(),
        target_lidar_calibrated_sensor,
        target_lidar_ego_pose,
    )
    return lidar_pc.points.T[:, :3].astype(np.float32, copy=False)


def collect_scene_records(
    nusc: NuScenes,
    scene_index: int,
    dilation_xyz: tuple[float, float, float],
) -> SceneContext:
    scene = nusc.scene[scene_index]
    first_sample = nusc.get("sample", scene["first_sample_token"])
    lidar_data = nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])
    first_lidar_ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    first_lidar_calibrated_sensor = nusc.get(
        "calibrated_sensor", lidar_data["calibrated_sensor_token"]
    )

    records = []
    while True:
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_data["token"])
        lidar_pc = LidarPointCloud.from_file(lidar_path)
        raw_points = lidar_pc.points.T
        points_xyz = raw_points[:, :3]

        gt_bbox_3d = build_gt_bbox_3d(boxes)
        dilated_bbox_3d = dilate_boxes_3d(gt_bbox_3d, dilation_xyz)

        exact_membership = compute_box_membership(points_xyz, gt_bbox_3d)
        dilated_membership = compute_box_membership(points_xyz, dilated_bbox_3d)

        exact_object_points_list = [raw_points[exact_membership[:, j]] for j in range(exact_membership.shape[1])]
        exact_any = exact_membership.any(axis=1) if exact_membership.size > 0 else np.zeros(points_xyz.shape[0], dtype=bool)
        dilated_any = dilated_membership.any(axis=1) if dilated_membership.size > 0 else np.zeros(points_xyz.shape[0], dtype=bool)
        keep_self = self_mask(points_xyz)

        static_exact = raw_points[(~exact_any) & keep_self]
        static_dilated = raw_points[(~dilated_any) & keep_self]
        static_shell = raw_points[(dilated_any & (~exact_any)) & keep_self]

        lidar_ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
        lidar_calibrated_sensor = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        sample_record = nusc.get("sample", lidar_data["sample_token"])
        ann_infos = [nusc.get("sample_annotation", ann_token) for ann_token in sample_record["anns"]]
        object_tokens = [ann["instance_token"] for ann in ann_infos]

        records.append({
            "scene_index": scene_index,
            "scene_name": scene["name"],
            "sample_token": lidar_data["sample_token"],
            "lidar_token": lidar_data["token"],
            "timestamp": lidar_data["timestamp"],
            "is_key_frame": lidar_data["is_key_frame"],
            "pc_file_name": Path(lidar_path).name,
            "folder_name": f"scene_{scene_index}_{Path(lidar_path).name.replace('.pcd.bin', '')}",
            "lidar_ego_pose": lidar_ego_pose,
            "lidar_calibrated_sensor": lidar_calibrated_sensor,
            "gt_bbox_3d": gt_bbox_3d,
            "object_tokens": object_tokens,
            "object_points_list": exact_object_points_list,
            "static_exact_first": to_first_lidar(
                static_exact,
                lidar_calibrated_sensor,
                lidar_ego_pose,
                first_lidar_calibrated_sensor,
                first_lidar_ego_pose,
            ),
            "static_dilated_first": to_first_lidar(
                static_dilated,
                lidar_calibrated_sensor,
                lidar_ego_pose,
                first_lidar_calibrated_sensor,
                first_lidar_ego_pose,
            ),
            "static_shell_first": to_first_lidar(
                static_shell,
                lidar_calibrated_sensor,
                lidar_ego_pose,
                first_lidar_calibrated_sensor,
                first_lidar_ego_pose,
            ),
        })

        next_token = lidar_data["next"]
        if next_token == "":
            break
        lidar_data = nusc.get("sample_data", next_token)

    object_token_zoo = []
    for frame in records:
        for idx, object_token in enumerate(frame["object_tokens"]):
            if object_token in object_token_zoo:
                continue
            if frame["object_points_list"][idx].shape[0] > 0:
                object_token_zoo.append(object_token)

    object_points_dict = {}
    for query_object_token in object_token_zoo:
        canonical_points = []
        for frame in records:
            for idx, object_token in enumerate(frame["object_tokens"]):
                if object_token != query_object_token:
                    continue
                object_points = frame["object_points_list"][idx]
                if object_points.shape[0] == 0:
                    continue
                object_points = object_points[:, :3] - frame["gt_bbox_3d"][idx][:3]
                rot = Rotation.from_euler("z", -frame["gt_bbox_3d"][idx][6], degrees=False)
                canonical_points.append(rot.apply(object_points))
        if canonical_points:
            object_points_dict[query_object_token] = np.concatenate(canonical_points, axis=0).astype(np.float32)

    object_points_xyz_by_token = {
        token: points[:, :3].astype(np.float32, copy=False)
        for token, points in object_points_dict.items()
    }

    return SceneContext(
        scene_index=scene_index,
        scene_name=scene["name"],
        records=records,
        object_points_xyz_by_token=object_points_xyz_by_token,
        first_lidar_calibrated_sensor=first_lidar_calibrated_sensor,
        first_lidar_ego_pose=first_lidar_ego_pose,
        full_static_exact_first=np.concatenate(
            [record["static_exact_first"] for record in records if record["static_exact_first"].size > 0],
            axis=0,
        ).astype(np.float32, copy=False),
        full_static_dilated_first=np.concatenate(
            [record["static_dilated_first"] for record in records if record["static_dilated_first"].size > 0],
            axis=0,
        ).astype(np.float32, copy=False),
        full_static_shell_first=np.concatenate(
            [record["static_shell_first"] for record in records if record["static_shell_first"].size > 0],
            axis=0,
        ).astype(np.float32, copy=False),
    )


def load_info_map(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as file:
        infos = pickle.load(file)

    info_by_folder = {}
    for info in infos:
        folder_name = Path(info["scene_flow_path"]).name
        info_by_folder[folder_name] = info
    return info_by_folder


def get_keyframe_pairs(context: SceneContext, info_by_folder: dict) -> list:
    keyframe_indices = [i for i, record in enumerate(context.records) if record["is_key_frame"]]
    pairs = []
    for pair_idx in range(1, len(keyframe_indices)):
        prev_idx = keyframe_indices[pair_idx - 1]
        curr_idx = keyframe_indices[pair_idx]
        folder_name = context.records[curr_idx]["folder_name"]
        if folder_name not in info_by_folder:
            continue
        pairs.append({
            "pair_idx": pair_idx - 1,
            "prev_idx": prev_idx,
            "curr_idx": curr_idx,
            "folder_name": folder_name,
            "info": info_by_folder[folder_name],
        })
    return pairs


def aggregate_static_first(context: SceneContext, prev_idx: int, curr_idx: int, static_mode: str,
                           radius: int, static_source: str) -> np.ndarray:
    if static_mode == "full":
        return {
            "exact": context.full_static_exact_first,
            "dilated": context.full_static_dilated_first,
            "shell": context.full_static_shell_first,
        }[static_source]

    if static_mode == "single":
        selected_records = [context.records[prev_idx], context.records[curr_idx]]
    elif static_mode == "local":
        start = max(0, min(prev_idx, curr_idx) - radius)
        end = min(len(context.records), max(prev_idx, curr_idx) + radius + 1)
        selected_records = context.records[start:end]
    else:
        raise ValueError(f"Unknown static_mode={static_mode}")

    key = {
        "exact": "static_exact_first",
        "dilated": "static_dilated_first",
        "shell": "static_shell_first",
    }[static_source]
    arrays = [record[key] for record in selected_records if record[key].size > 0]
    if not arrays:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def place_aligned_objects(context: SceneContext, prev_record: dict, curr_record: dict) -> tuple[np.ndarray, np.ndarray]:
    prev_box_idx_by_token = {}
    for idx, object_token in enumerate(prev_record["object_tokens"]):
        if object_token not in prev_box_idx_by_token:
            prev_box_idx_by_token[object_token] = idx

    curr_box_idx_by_token = {}
    for idx, object_token in enumerate(curr_record["object_tokens"]):
        if object_token not in curr_box_idx_by_token:
            curr_box_idx_by_token[object_token] = idx

    common_object_tokens = []
    seen_tokens = set()
    for object_token in prev_record["object_tokens"]:
        if object_token in seen_tokens:
            continue
        if object_token not in curr_box_idx_by_token:
            continue
        if object_token not in context.object_points_xyz_by_token:
            continue
        common_object_tokens.append(object_token)
        seen_tokens.add(object_token)

    prev_objects = []
    curr_objects = []
    for object_token in common_object_tokens:
        prev_box_idx = prev_box_idx_by_token[object_token]
        curr_box_idx = curr_box_idx_by_token[object_token]
        canonical_points = context.object_points_xyz_by_token[object_token]

        prev_rot = Rotation.from_euler(
            "z",
            prev_record["gt_bbox_3d"][prev_box_idx][6],
            degrees=False,
        )
        prev_points = prev_rot.apply(canonical_points) + prev_record["gt_bbox_3d"][prev_box_idx][:3]
        prev_objects.append(prev_points.astype(np.float32, copy=False))

        curr_rot = Rotation.from_euler(
            "z",
            curr_record["gt_bbox_3d"][curr_box_idx][6],
            degrees=False,
        )
        curr_points = curr_rot.apply(canonical_points) + curr_record["gt_bbox_3d"][curr_box_idx][:3]
        curr_objects.append(curr_points.astype(np.float32, copy=False))

    prev_concat = np.concatenate(prev_objects, axis=0) if prev_objects else np.zeros((0, 3), dtype=np.float32)
    curr_concat = np.concatenate(curr_objects, axis=0) if curr_objects else np.zeros((0, 3), dtype=np.float32)
    return prev_concat, curr_concat


def build_pair_scene_points(
    context: SceneContext,
    prev_idx: int,
    curr_idx: int,
    static_mode: str,
    radius: int,
    static_source: str,
    include_objects: bool,
) -> tuple[np.ndarray, np.ndarray]:
    static_first = aggregate_static_first(context, prev_idx, curr_idx, static_mode, radius, static_source)
    prev_record = context.records[prev_idx]
    curr_record = context.records[curr_idx]

    if static_first.size == 0:
        prev_static = np.zeros((0, 3), dtype=np.float32)
        curr_static = np.zeros((0, 3), dtype=np.float32)
    else:
        prev_static = transform_from_first_to_target(
            static_first,
            context.first_lidar_calibrated_sensor,
            context.first_lidar_ego_pose,
            prev_record["lidar_calibrated_sensor"],
            prev_record["lidar_ego_pose"],
        )

        curr_static = transform_from_first_to_target(
            static_first,
            context.first_lidar_calibrated_sensor,
            context.first_lidar_ego_pose,
            curr_record["lidar_calibrated_sensor"],
            curr_record["lidar_ego_pose"],
        )

    if include_objects:
        prev_objects, curr_objects = place_aligned_objects(context, prev_record, curr_record)
    else:
        prev_objects = np.zeros((0, 3), dtype=np.float32)
        curr_objects = np.zeros((0, 3), dtype=np.float32)

    prev_scene_points = np.concatenate([prev_static, prev_objects], axis=0)
    curr_scene_points = np.concatenate([curr_static, curr_objects], axis=0)

    prev_range_mask = (
        (np.abs(prev_scene_points[:, 0]) < 50.0) &
        (np.abs(prev_scene_points[:, 1]) < 50.0) &
        (prev_scene_points[:, 2] > -5.0) &
        (prev_scene_points[:, 2] < 3.0)
    )
    curr_range_mask = (
        (np.abs(curr_scene_points[:, 0]) < 50.0) &
        (np.abs(curr_scene_points[:, 1]) < 50.0) &
        (curr_scene_points[:, 2] > -5.0) &
        (curr_scene_points[:, 2] < 3.0)
    )
    intersection = prev_range_mask & curr_range_mask

    return (
        prev_scene_points[intersection].astype(np.float32, copy=False),
        curr_scene_points[intersection].astype(np.float32, copy=False),
    )


def project_scale_maps(
    points_prev: np.ndarray,
    points_curr: np.ndarray,
    time_diff_ratio: float,
    image_height: int,
    image_width: int,
    fov_up: float,
    fov_down: float,
) -> dict:
    if points_prev.shape[0] == 0 or points_curr.shape[0] == 0:
        raise ValueError("Cannot project empty point clouds.")

    points_prev_aligned = points_curr - time_diff_ratio * (points_curr - points_prev)
    depth_prev = np.linalg.norm(points_prev_aligned[:, :2], axis=1)
    depth_curr = np.linalg.norm(points_curr[:, :2], axis=1)
    depth_prev[depth_prev == 0] = 1e-6
    scales = (depth_curr / depth_prev).astype(np.float32, copy=False)
    risk = np.zeros_like(scales, dtype=np.float32)

    proj_prev = range_projection(
        points_prev_aligned,
        scales,
        risk,
        H=image_height,
        W=image_width,
        fov_up=fov_up,
        fov_down=-fov_down,
    )
    proj_curr = range_projection(
        points_curr,
        scales,
        risk,
        H=image_height,
        W=image_width,
        fov_up=fov_up,
        fov_down=-fov_down,
    )
    return {
        "prev": {
            "depth": proj_prev[0],
            "scale": proj_prev[1],
            "risk_score": proj_prev[2],
            "xyz": proj_prev[3],
            "idx": proj_prev[4],
            "mask": proj_prev[5],
        },
        "curr": {
            "depth": proj_curr[0],
            "scale": proj_curr[1],
            "risk_score": proj_curr[2],
            "xyz": proj_curr[3],
            "idx": proj_curr[4],
            "mask": proj_curr[5],
        },
    }


def compute_ghost_stats(scale_map: np.ndarray, valid_mask: np.ndarray, sparse_neighbors: int,
                        scale_low: float, scale_high: float) -> dict:
    valid = valid_mask.astype(bool)
    extreme = valid & ((scale_map < scale_low) | (scale_map > scale_high))
    neighbor_count = convolve2d(
        valid.astype(np.int32),
        np.ones((3, 3), dtype=np.int32),
        mode="same",
        boundary="fill",
        fillvalue=0,
    )
    sparse = neighbor_count <= sparse_neighbors
    ghost = extreme & sparse

    scale_valid = scale_map[valid]
    scale_extreme = scale_map[extreme]
    return {
        "valid_mask": valid,
        "extreme_mask": extreme,
        "ghost_mask": ghost,
        "neighbor_count": neighbor_count,
        "num_valid_pixels": int(valid.sum()),
        "num_extreme_pixels": int(extreme.sum()),
        "num_ghost_pixels": int(ghost.sum()),
        "extreme_ratio_in_valid": float(extreme.sum() / max(valid.sum(), 1)),
        "ghost_ratio_in_valid": float(ghost.sum() / max(valid.sum(), 1)),
        "scale_min": float(scale_valid.min()) if scale_valid.size else math.nan,
        "scale_max": float(scale_valid.max()) if scale_valid.size else math.nan,
        "scale_mean": float(scale_valid.mean()) if scale_valid.size else math.nan,
        "scale_std": float(scale_valid.std()) if scale_valid.size else math.nan,
        "scale_extreme_q10": float(np.quantile(scale_extreme, 0.1)) if scale_extreme.size else math.nan,
        "scale_extreme_q50": float(np.quantile(scale_extreme, 0.5)) if scale_extreme.size else math.nan,
        "scale_extreme_q90": float(np.quantile(scale_extreme, 0.9)) if scale_extreme.size else math.nan,
    }


def normalize_scale_for_display(scale_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    scale_display = np.copy(scale_map).astype(np.float32)
    valid = valid_mask.astype(bool)
    scale_display[valid] = np.clip(scale_display[valid], 0.5, 1.5)
    deviations = np.zeros_like(scale_display, dtype=np.float32)
    deviations[valid] = scale_display[valid] - 1.0

    normalized = np.zeros_like(deviations, dtype=np.float32)
    pos_mask = deviations > 0
    neg_mask = deviations < 0
    if np.any(pos_mask):
        pos_vals = deviations[pos_mask]
        if pos_vals.max() > pos_vals.min() >= 0:
            normalized[pos_mask] = (pos_vals - pos_vals.min()) / (pos_vals.max() - pos_vals.min())
    if np.any(neg_mask):
        neg_vals = deviations[neg_mask]
        if neg_vals.min() < neg_vals.max() <= 0:
            normalized[neg_mask] = (neg_vals - neg_vals.min()) / (neg_vals.max() - neg_vals.min()) - 1.0
    normalized[~valid] = 0.0
    return normalized


def render_scale_image(scale_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    normalized = normalize_scale_for_display(scale_map, valid_mask)
    rgba = plt.get_cmap("seismic")((-normalized + 1.0) / 2.0)
    rgba[~valid_mask.astype(bool)] = [1.0, 1.0, 1.0, 1.0]
    return rgba[:, :, :3]


def render_binary_mask(mask: np.ndarray, color=(1.0, 0.0, 0.0)) -> np.ndarray:
    rgb = np.ones(mask.shape + (3,), dtype=np.float32)
    rgb[mask.astype(bool)] = color
    return rgb


def render_overlay(base_rgb: np.ndarray, overlay_mask: np.ndarray, color=(1.0, 1.0, 0.0), alpha=0.7) -> np.ndarray:
    out = base_rgb.copy()
    mask = overlay_mask.astype(bool)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return out


def save_scene24_contact_sheet(output_path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(rows):
        axes[row_idx, 0].imshow(row["scale_rgb"])
        axes[row_idx, 0].set_title(f"{row['title']} | scale")
        axes[row_idx, 1].imshow(row["ghost_overlay"])
        axes[row_idx, 1].set_title(f"{row['title']} | ghost overlay")
        axes[row_idx, 2].imshow(row["ghost_rgb"])
        axes[row_idx, 2].set_title(
            f"{row['title']} | ghost={row['ghost_count']} ({row['ghost_ratio']:.3%})"
        )
        for col in range(3):
            axes[row_idx, col].axis("off")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_scene65_examples(output_path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(rows), 5, figsize=(22, 4 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(rows):
        images = [
            row["baseline_scale_rgb"],
            row["baseline_ghost_overlay"],
            row["shell_overlay"],
            row["dilated_scale_rgb"],
            row["dilated_ghost_overlay"],
        ]
        titles = [
            f"{row['title']} | exact scale",
            f"{row['title']} | exact ghost",
            f"{row['title']} | shell overlap",
            f"{row['title']} | dilated scale",
            (
                f"{row['title']} | dilated ghost\n"
                f"drop={row['ghost_drop_ratio']:.1%}"
            ),
        ]
        for col_idx, (image, title) in enumerate(zip(images, titles)):
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].set_title(title)
            axes[row_idx, col_idx].axis("off")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def resolve_gt_map_path(info: dict, gt_map_root: Path) -> Path:
    gt_map_path = resolve_nusc_path(info["gt_map_path"], gt_map_root.parent)
    if gt_map_path.exists():
        return gt_map_path

    folder_name = Path(info["scene_flow_path"]).name
    matches = list(gt_map_root.rglob(f"{folder_name}/range_image_prev.npy"))
    if not matches:
        raise FileNotFoundError(f"Could not resolve gt_map_path for {folder_name} under {gt_map_root}")
    return matches[0].parent


def load_range_image_prev(info: dict, gt_map_root: Path) -> dict:
    range_image_path = resolve_gt_map_path(info, gt_map_root) / "range_image_prev.npy"
    return np.load(range_image_path, allow_pickle=True).item()


def to_serializable(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    return value


def analyze_scene24(
    context: SceneContext,
    pair_by_folder: dict,
    tail_folder: str,
    control_folder: str,
    output_dir: Path,
    gt_map_root: Path,
    local_radii: tuple[int, int],
    sparse_neighbors: int,
    scale_low: float,
    scale_high: float,
    image_height: int,
    image_width: int,
    fov_up: float,
    fov_down: float,
) -> dict:
    ensure_dir(output_dir)
    print(f"[scene24] Analyzing tail sample {tail_folder}")
    tail_pair = pair_by_folder[tail_folder]
    control_pair = pair_by_folder[control_folder]

    tail_existing = load_range_image_prev(tail_pair["info"], gt_map_root)
    control_existing = load_range_image_prev(control_pair["info"], gt_map_root)

    tail_existing_stats = compute_ghost_stats(
        tail_existing["scale"],
        tail_existing["mask"],
        sparse_neighbors,
        scale_low,
        scale_high,
    )
    control_existing_stats = compute_ghost_stats(
        control_existing["scale"],
        control_existing["mask"],
        sparse_neighbors,
        scale_low,
        scale_high,
    )

    small_radius, large_radius = sorted(local_radii)
    ablation_defs = [
        ("full_scene_static", "full", 0),
        (f"local_static_{large_radius}", "local", large_radius),
        (f"local_static_{small_radius}", "local", small_radius),
        ("single_frame_static", "single", 0),
    ]

    ablation_stats = []
    contact_rows = [
        {
            "title": "scene24 control existing",
            "scale_rgb": render_scale_image(control_existing["scale"], control_existing["mask"]),
            "ghost_overlay": render_overlay(
                render_scale_image(control_existing["scale"], control_existing["mask"]),
                control_existing_stats["ghost_mask"],
            ),
            "ghost_rgb": render_binary_mask(control_existing_stats["ghost_mask"]),
            "ghost_count": control_existing_stats["num_ghost_pixels"],
            "ghost_ratio": control_existing_stats["ghost_ratio_in_valid"],
        },
        {
            "title": "scene24 tail existing",
            "scale_rgb": render_scale_image(tail_existing["scale"], tail_existing["mask"]),
            "ghost_overlay": render_overlay(
                render_scale_image(tail_existing["scale"], tail_existing["mask"]),
                tail_existing_stats["ghost_mask"],
            ),
            "ghost_rgb": render_binary_mask(tail_existing_stats["ghost_mask"]),
            "ghost_count": tail_existing_stats["num_ghost_pixels"],
            "ghost_ratio": tail_existing_stats["ghost_ratio_in_valid"],
        },
    ]

    baseline_ghost = max(tail_existing_stats["num_ghost_pixels"], 1)
    baseline_valid = max(tail_existing_stats["num_valid_pixels"], 1)
    scene24_primary_cause = "inconclusive"
    selected_fix = None

    for label, static_mode, radius in ablation_defs:
        print(f"[scene24] Running ablation: {label}")
        prev_points, curr_points = build_pair_scene_points(
            context=context,
            prev_idx=tail_pair["prev_idx"],
            curr_idx=tail_pair["curr_idx"],
            static_mode=static_mode,
            radius=radius,
            static_source="exact",
            include_objects=True,
        )
        projected = project_scale_maps(
            prev_points,
            curr_points,
            tail_pair["info"]["time_diff_ratio"],
            image_height,
            image_width,
            fov_up,
            fov_down,
        )
        stats = compute_ghost_stats(
            projected["prev"]["scale"],
            projected["prev"]["mask"],
            sparse_neighbors,
            scale_low,
            scale_high,
        )
        ghost_drop_ratio = float((baseline_ghost - stats["num_ghost_pixels"]) / baseline_ghost)
        valid_drop_ratio = float((baseline_valid - stats["num_valid_pixels"]) / baseline_valid)
        result = {
            "label": label,
            "static_mode": static_mode,
            "radius": radius,
            "num_valid_pixels": stats["num_valid_pixels"],
            "num_extreme_pixels": stats["num_extreme_pixels"],
            "num_ghost_pixels": stats["num_ghost_pixels"],
            "ghost_ratio_in_valid": stats["ghost_ratio_in_valid"],
            "ghost_drop_ratio_vs_existing": ghost_drop_ratio,
            "valid_drop_ratio_vs_existing": valid_drop_ratio,
        }
        ablation_stats.append(result)
        contact_rows.append({
            "title": label,
            "scale_rgb": render_scale_image(projected["prev"]["scale"], projected["prev"]["mask"]),
            "ghost_overlay": render_overlay(
                render_scale_image(projected["prev"]["scale"], projected["prev"]["mask"]),
                stats["ghost_mask"],
            ),
            "ghost_rgb": render_binary_mask(stats["ghost_mask"]),
            "ghost_count": stats["num_ghost_pixels"],
            "ghost_ratio": stats["ghost_ratio_in_valid"],
        })

        if selected_fix is None and ghost_drop_ratio >= 0.70 and valid_drop_ratio <= 0.05:
            scene24_primary_cause = "static_accumulation"
            selected_fix = label

    save_scene24_contact_sheet(output_dir / "scene24_tail_ablation_contact_sheet.png", contact_rows)

    result = {
        "scene24_primary_cause": scene24_primary_cause,
        "selected_fix_variant": selected_fix,
        "control_existing": {
            "folder_name": control_folder,
            "num_valid_pixels": control_existing_stats["num_valid_pixels"],
            "num_extreme_pixels": control_existing_stats["num_extreme_pixels"],
            "num_ghost_pixels": control_existing_stats["num_ghost_pixels"],
            "ghost_ratio_in_valid": control_existing_stats["ghost_ratio_in_valid"],
        },
        "tail_existing": {
            "folder_name": tail_folder,
            "num_valid_pixels": tail_existing_stats["num_valid_pixels"],
            "num_extreme_pixels": tail_existing_stats["num_extreme_pixels"],
            "num_ghost_pixels": tail_existing_stats["num_ghost_pixels"],
            "ghost_ratio_in_valid": tail_existing_stats["ghost_ratio_in_valid"],
            "time_diff_ratio": float(tail_pair["info"]["time_diff_ratio"]),
        },
        "ablations": ablation_stats,
    }
    with open(output_dir / "scene24_tail_stats.json", "w") as file:
        json.dump(to_serializable(result), file, indent=2)
    return result


def project_shell_mask(
    prev_shell: np.ndarray,
    curr_shell: np.ndarray,
    time_diff_ratio: float,
    image_height: int,
    image_width: int,
    fov_up: float,
    fov_down: float,
) -> np.ndarray:
    if prev_shell.shape[0] == 0 or curr_shell.shape[0] == 0:
        return np.zeros((image_height, image_width), dtype=bool)
    points_prev_aligned = curr_shell - time_diff_ratio * (curr_shell - prev_shell)
    dummy_scale = np.ones(points_prev_aligned.shape[0], dtype=np.float32)
    dummy_risk = np.zeros(points_prev_aligned.shape[0], dtype=np.float32)
    _, _, _, _, _, proj_mask = range_projection(
        points_prev_aligned,
        dummy_scale,
        dummy_risk,
        H=image_height,
        W=image_width,
        fov_up=fov_up,
        fov_down=-fov_down,
    )
    return proj_mask.astype(bool)


def analyze_scene65(
    context: SceneContext,
    pairs: list,
    output_dir: Path,
    gt_map_root: Path,
    sparse_neighbors: int,
    scale_low: float,
    scale_high: float,
    image_height: int,
    image_width: int,
    fov_up: float,
    fov_down: float,
) -> dict:
    ensure_dir(output_dir)
    print(f"[scene65] Scanning {len(pairs)} pairs for bbox leakage")

    ranking = []
    for pair in pairs:
        existing = load_range_image_prev(pair["info"], gt_map_root)
        existing_stats = compute_ghost_stats(
            existing["scale"],
            existing["mask"],
            sparse_neighbors,
            scale_low,
            scale_high,
        )

        prev_shell, curr_shell = build_pair_scene_points(
            context=context,
            prev_idx=pair["prev_idx"],
            curr_idx=pair["curr_idx"],
            static_mode="full",
            radius=0,
            static_source="shell",
            include_objects=False,
        )
        shell_mask = project_shell_mask(
            prev_shell,
            curr_shell,
            pair["info"]["time_diff_ratio"],
            image_height,
            image_width,
            fov_up,
            fov_down,
        )
        overlap_mask = shell_mask & existing_stats["ghost_mask"]
        shell_pixels = int(shell_mask.sum())
        overlap_pixels = int(overlap_mask.sum())
        ghost_pixels = max(existing_stats["num_ghost_pixels"], 1)
        ranking.append({
            "folder_name": pair["folder_name"],
            "pair_idx": pair["pair_idx"],
            "num_ghost_pixels": existing_stats["num_ghost_pixels"],
            "shell_pixels": shell_pixels,
            "shell_overlap_pixels": overlap_pixels,
            "shell_overlap_ratio": float(overlap_pixels / max(shell_pixels, 1)),
            "ghost_explained_ratio": float(overlap_pixels / ghost_pixels),
        })

    ranking.sort(
        key=lambda item: (
            item["shell_overlap_pixels"],
            item["ghost_explained_ratio"],
            item["shell_overlap_ratio"],
        ),
        reverse=True,
    )

    top_examples = []
    scene65_primary_cause = "inconclusive"
    top_pair_ghost_drop_ratio = 0.0

    for entry in ranking[:3]:
        print(f"[scene65] Building top example for {entry['folder_name']}")
        pair = next(pair for pair in pairs if pair["folder_name"] == entry["folder_name"])
        existing = load_range_image_prev(pair["info"], gt_map_root)
        existing_stats = compute_ghost_stats(
            existing["scale"],
            existing["mask"],
            sparse_neighbors,
            scale_low,
            scale_high,
        )

        prev_shell, curr_shell = build_pair_scene_points(
            context=context,
            prev_idx=pair["prev_idx"],
            curr_idx=pair["curr_idx"],
            static_mode="full",
            radius=0,
            static_source="shell",
            include_objects=False,
        )
        shell_mask = project_shell_mask(
            prev_shell,
            curr_shell,
            pair["info"]["time_diff_ratio"],
            image_height,
            image_width,
            fov_up,
            fov_down,
        )

        dilated_prev, dilated_curr = build_pair_scene_points(
            context=context,
            prev_idx=pair["prev_idx"],
            curr_idx=pair["curr_idx"],
            static_mode="full",
            radius=0,
            static_source="dilated",
            include_objects=True,
        )
        dilated_projected = project_scale_maps(
            dilated_prev,
            dilated_curr,
            pair["info"]["time_diff_ratio"],
            image_height,
            image_width,
            fov_up,
            fov_down,
        )
        dilated_stats = compute_ghost_stats(
            dilated_projected["prev"]["scale"],
            dilated_projected["prev"]["mask"],
            sparse_neighbors,
            scale_low,
            scale_high,
        )
        ghost_drop_ratio = float(
            (existing_stats["num_ghost_pixels"] - dilated_stats["num_ghost_pixels"]) /
            max(existing_stats["num_ghost_pixels"], 1)
        )

        top_examples.append({
            "folder_name": pair["folder_name"],
            "pair_idx": pair["pair_idx"],
            "baseline_scale_rgb": render_scale_image(existing["scale"], existing["mask"]),
            "baseline_ghost_overlay": render_overlay(
                render_scale_image(existing["scale"], existing["mask"]),
                existing_stats["ghost_mask"],
            ),
            "shell_overlay": render_overlay(
                render_binary_mask(np.zeros_like(shell_mask, dtype=bool), color=(1.0, 1.0, 1.0)),
                shell_mask | existing_stats["ghost_mask"],
                color=(1.0, 0.8, 0.0),
                alpha=1.0,
            ),
            "dilated_scale_rgb": render_scale_image(
                dilated_projected["prev"]["scale"], dilated_projected["prev"]["mask"]
            ),
            "dilated_ghost_overlay": render_overlay(
                render_scale_image(dilated_projected["prev"]["scale"], dilated_projected["prev"]["mask"]),
                dilated_stats["ghost_mask"],
            ),
            "ghost_drop_ratio": ghost_drop_ratio,
            "title": pair["folder_name"].split("__")[-1],
        })

        entry["dilated_num_ghost_pixels"] = dilated_stats["num_ghost_pixels"]
        entry["dilated_ghost_drop_ratio"] = ghost_drop_ratio
        if entry is ranking[0]:
            top_pair_ghost_drop_ratio = ghost_drop_ratio

    if ranking and top_pair_ghost_drop_ratio >= 0.30:
        scene65_primary_cause = "bbox_leakage"

    with open(output_dir / "scene65_shell_residual_ranking.json", "w") as file:
        json.dump(to_serializable(ranking), file, indent=2)

    save_scene65_examples(output_dir / "scene65_top_bbox_leakage_examples.png", top_examples)

    result = {
        "scene65_primary_cause": scene65_primary_cause,
        "top_pair_ghost_drop_ratio": top_pair_ghost_drop_ratio,
        "ranking_top3": ranking[:3],
    }
    return result


def write_summary(output_dir: Path, scene24_result: dict, scene65_result: dict) -> None:
    lines = [
        "# Ghost Cause Diagnosis",
        "",
        f"- scene24_primary_cause = {scene24_result['scene24_primary_cause']}",
        f"- scene65_primary_cause = {scene65_result['scene65_primary_cause']}",
        (
            f"- scene24 tail ghost pixels: {scene24_result['tail_existing']['num_ghost_pixels']} "
            f"(control: {scene24_result['control_existing']['num_ghost_pixels']})"
        ),
        (
            f"- scene65 top bbox leakage candidate drop ratio after dilated bbox removal: "
            f"{scene65_result['top_pair_ghost_drop_ratio']:.1%}"
        ),
        "",
    ]

    if scene24_result["scene24_primary_cause"] == "static_accumulation" and scene65_result["scene65_primary_cause"] == "bbox_leakage":
        lines.append("- overall = mixed")
        lines.append("- priority = static_accumulation first, bbox_leakage second")
    elif scene24_result["scene24_primary_cause"] == "static_accumulation":
        lines.append("- overall = static_accumulation")
        lines.append("- priority = shorten static accumulation window")
    elif scene65_result["scene65_primary_cause"] == "bbox_leakage":
        lines.append("- overall = bbox_leakage")
        lines.append("- priority = improve dynamic point removal around box boundaries")
    else:
        lines.append("- overall = inconclusive")
        lines.append("- priority = inspect additional samples manually")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_flow_root",
        type=Path,
        default=Path("/mnt/data/fpttc_ground_truth/0_scene_flow_150_keyframes_float32/scene_flow_key_frames"),
    )
    parser.add_argument(
        "--gt_map_root",
        type=Path,
        default=Path("/mnt/data/fpttc_ground_truth/1_gt_map_150_keyframes"),
    )
    parser.add_argument(
        "--test_pkl_path",
        type=Path,
        default=Path("/mnt/data/fpttc_ground_truth/2_trainval_test_infos_150_keyframes/test/nusc_test_infos_key_frames_160_1920_fov_8_15.pkl"),
    )
    parser.add_argument("--dataroot", type=Path, default=Path("./Datasets/nuscenes/"))
    parser.add_argument("--output_dir", type=Path, default=Path("./Datasets/nuscenes_debug/ghost_diagnosis"))
    parser.add_argument("--scene24_tail_folder", type=str, default=DEFAULT_SCENE24_TAIL)
    parser.add_argument("--scene24_control_folder", type=str, default=DEFAULT_SCENE24_CONTROL)
    parser.add_argument("--bbox_dilation", type=float, nargs=3, default=[0.5, 0.5, 0.3])
    parser.add_argument("--local_radii", type=int, nargs=2, default=[5, 10])
    parser.add_argument("--scale_low", type=float, default=0.95)
    parser.add_argument("--scale_high", type=float, default=1.05)
    parser.add_argument("--sparse_neighbors", type=int, default=2)
    parser.add_argument("--image_size", type=int, nargs=2, default=[160, 320])
    parser.add_argument("--fov", type=float, nargs=2, default=[8.0, 15.0])
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    info_by_folder = load_info_map(args.test_pkl_path)
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(args.dataroot), verbose=False)

    print("[scene24] Collecting scene records")
    scene24_context = collect_scene_records(
        nusc=nusc,
        scene_index=24,
        dilation_xyz=tuple(args.bbox_dilation),
    )
    scene24_pairs = get_keyframe_pairs(scene24_context, info_by_folder)
    scene24_pair_by_folder = {pair["folder_name"]: pair for pair in scene24_pairs}

    scene24_result = analyze_scene24(
        context=scene24_context,
        pair_by_folder=scene24_pair_by_folder,
        tail_folder=args.scene24_tail_folder,
        control_folder=args.scene24_control_folder,
        output_dir=args.output_dir,
        gt_map_root=args.gt_map_root,
        local_radii=tuple(args.local_radii),
        sparse_neighbors=args.sparse_neighbors,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        image_height=args.image_size[0],
        image_width=args.image_size[1] * 6,
        fov_up=args.fov[0],
        fov_down=args.fov[1],
    )

    print("[scene65] Collecting scene records")
    scene65_context = collect_scene_records(
        nusc=nusc,
        scene_index=65,
        dilation_xyz=tuple(args.bbox_dilation),
    )
    scene65_pairs = [pair for pair in get_keyframe_pairs(scene65_context, info_by_folder) if pair["info"]["scene_indice"] == "65"]
    scene65_result = analyze_scene65(
        context=scene65_context,
        pairs=scene65_pairs,
        output_dir=args.output_dir,
        gt_map_root=args.gt_map_root,
        sparse_neighbors=args.sparse_neighbors,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        image_height=args.image_size[0],
        image_width=args.image_size[1] * 6,
        fov_up=args.fov[0],
        fov_down=args.fov[1],
    )

    write_summary(args.output_dir, scene24_result, scene65_result)
    print(f"Saved diagnosis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
