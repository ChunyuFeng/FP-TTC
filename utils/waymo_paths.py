"""Path utilities for Waymo dataset, mirroring utils/nusc_paths.py."""
from __future__ import annotations

from pathlib import Path


DEFAULT_WAYMO_DATASET_ROOT = Path("./Datasets/waymo")

WAYMO_PATH_ANCHORS = (
    "0_scene_flow",
    "1_gt_map",
    "2_trainval_test_infos",
    "3_visualization",
    "4_depth_map",
    "camera_images",
    "raw",
)

# Waymo camera names (5 cameras, no rear camera)
WAYMO_CAMERAS = [
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
]

# Camera name enum mapping (Waymo proto CameraName enum values)
WAYMO_CAMERA_NAME_TO_ID = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
}

WAYMO_CAMERA_ID_TO_NAME = {v: k for k, v in WAYMO_CAMERA_NAME_TO_ID.items()}

# LiDAR name enum mapping (Waymo proto LaserName enum values)
WAYMO_LIDAR_NAME_TO_ID = {
    "TOP": 1,
    "FRONT": 2,
    "SIDE_LEFT": 3,
    "SIDE_RIGHT": 4,
    "REAR": 5,
}

WAYMO_LIDAR_ID_TO_NAME = {v: k for k, v in WAYMO_LIDAR_NAME_TO_ID.items()}


def resolve_waymo_path(
    path_value: str | Path | None,
    dataset_root: str | Path = DEFAULT_WAYMO_DATASET_ROOT,
) -> Path | None:
    """Resolve a relative path under the Waymo dataset root."""
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
    dataset_root: str | Path = DEFAULT_WAYMO_DATASET_ROOT,
) -> Path:
    """Construct depth prediction path for a Waymo camera image."""
    depth_rel = Path("4_depth_map") / camera_name / f"{segment_id}_frame_{frame_idx:04d}.npy"
    return Path(dataset_root) / depth_rel


def resolve_waymo_camera_image_path(
    camera_name: str,
    segment_id: str,
    frame_idx: int,
    dataset_root: str | Path = DEFAULT_WAYMO_DATASET_ROOT,
) -> Path:
    """Construct camera image path for a Waymo frame."""
    img_rel = Path("camera_images") / camera_name / f"{segment_id}_frame_{frame_idx:04d}.jpg"
    return Path(dataset_root) / img_rel


def make_waymo_relative_path(path_value: str | Path | None) -> str | None:
    """Strip absolute prefix, keeping only the part from an anchor directory."""
    if path_value in (None, ""):
        return path_value
    path = Path(path_value)
    parts = path.parts
    for index, part in enumerate(parts):
        if part in WAYMO_PATH_ANCHORS:
            return Path(*parts[index:]).as_posix()
    return path.as_posix()


def infer_waymo_dataset_root(path_value: str | Path) -> Path:
    """Walk up from *path_value* until we find 2_trainval_test_infos."""
    path = Path(path_value)
    search_chain = [path] + list(path.parents)
    for candidate in search_chain:
        if candidate.name == "2_trainval_test_infos":
            return candidate.parent
    return path.parent


def get_segment_id_from_tfrecord(tfrecord_path: str | Path) -> str:
    """
    Extract the segment ID from a Waymo TFRecord filename.
    e.g. 'segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'
         -> '10017090168044687777_6380_000_6400_000'
    """
    name = Path(tfrecord_path).stem  # remove .tfrecord
    # Remove 'segment-' prefix and '_with_camera_labels' suffix
    name = name.replace("segment-", "").replace("_with_camera_labels", "")
    return name
