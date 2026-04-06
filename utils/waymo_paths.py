from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


DEFAULT_WAYMO_DATASET_ROOT = Path("./Datasets/waymo")
DEFAULT_WAYMO_PROJ_CACHE_NAME = (
    "waymo_scene150_keyframes_160x320_proj40x400_"
    "fov2_17_toplidar_nusc_vehrot_v1"
)

WAYMO_PATH_ANCHORS = (
    "0_scene_flow",
    "1_gt_map",
    "2_trainval_test_infos",
    "3_visualization",
    "4_depth_map",
    "5_proj_cache",
    "camera_images",
    "raw",
)

WAYMO_CAMERAS = [
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
]


PathLike = Union[str, Path]


def _normalize_segment_name(segment_id: str) -> str:
    return segment_id if segment_id.startswith("segment_") else f"segment_{segment_id}"


def resolve_waymo_path(
    path_value: Optional[PathLike],
    dataset_root: PathLike = DEFAULT_WAYMO_DATASET_ROOT,
) -> Optional[Path]:
    if path_value in (None, ""):
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    return Path(dataset_root) / path


def resolve_waymo_depth_pred_path(
    camera_name: str,
    segment_id: str,
    frame_idx: int,
    dataset_root: PathLike = DEFAULT_WAYMO_DATASET_ROOT,
) -> Path:
    segment_name = _normalize_segment_name(segment_id)
    depth_rel = Path("4_depth_map") / camera_name / f"{segment_name}_frame_{int(frame_idx):04d}.npy"
    return Path(dataset_root) / depth_rel


def resolve_waymo_camera_image_path(
    camera_name: str,
    segment_id: str,
    frame_idx: int,
    dataset_root: PathLike = DEFAULT_WAYMO_DATASET_ROOT,
) -> Path:
    segment_name = _normalize_segment_name(segment_id)
    img_rel = Path("camera_images") / camera_name / f"{segment_name}_frame_{int(frame_idx):04d}.jpg"
    return Path(dataset_root) / img_rel


def default_waymo_proj_cache_root(
    dataset_root: PathLike = DEFAULT_WAYMO_DATASET_ROOT,
    cache_name: str = DEFAULT_WAYMO_PROJ_CACHE_NAME,
) -> Path:
    return Path(dataset_root) / "5_proj_cache" / cache_name


def make_waymo_relative_path(path_value: Optional[PathLike]) -> Optional[str]:
    if path_value in (None, ""):
        return path_value

    path = Path(path_value)
    parts = path.parts
    for index, part in enumerate(parts):
        if part in WAYMO_PATH_ANCHORS:
            return Path(*parts[index:]).as_posix()

    return path.as_posix()


def infer_waymo_dataset_root(path_value: PathLike) -> Path:
    path = Path(path_value)
    search_chain = [path] + list(path.parents)
    for candidate in search_chain:
        if candidate.name == "2_trainval_test_infos":
            return candidate.parent
    return path.parent
