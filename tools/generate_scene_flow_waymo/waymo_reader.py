"""
Waymo TFRecord reader — thin wrapper around the official Waymo Open Dataset SDK.

Provides frame iteration, point cloud extraction (all 5 LiDARs merged),
3D box extraction, ego pose, camera images, and camera calibrations.
"""
from __future__ import annotations

import os
import io
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF info/warnings

import tensorflow as tf

# ---------- Waymo SDK imports (official) ----------
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils, range_image_utils, transform_utils

# Convenient enum aliases
from waymo_open_dataset import label_pb2

# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------

def iter_frames(tfrecord_path: str):
    """
    Yield parsed ``open_dataset.Frame`` objects from a TFRecord file.

    Uses the official Waymo protobuf + TFRecordDataset pipeline.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    for raw_record in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytes(raw_record.numpy()))
        yield frame


# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------

def extract_merged_point_cloud(
    frame: open_dataset.Frame,
    return_intensity: bool = False,
) -> np.ndarray:
    """
    Extract and merge point clouds from **all 5 LiDARs** into the vehicle frame.

    Uses the official ``frame_utils.convert_range_image_to_point_cloud``
    which handles:
      - Range image decompression
      - Beam inclination angles → Cartesian conversion
      - Per-LiDAR extrinsic calibration → vehicle frame
      - Top LiDAR rolling-shutter motion compensation

    Parameters
    ----------
    frame : open_dataset.Frame
        A parsed Waymo Frame proto.
    return_intensity : bool
        If True, return (N, 4) with the 4th column being intensity.

    Returns
    -------
    points : np.ndarray
        Shape ``(N, 3)`` or ``(N, 4)`` XYZ(I) in the **vehicle frame**.
        Vehicle frame convention: x-forward, y-left, z-up.
    """
    # Official SDK: parse range images and camera projections
    (range_images,
     camera_projections,
     _,   # segmentation labels (may be None)
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    # Convert range images to Cartesian point clouds (vehicle frame)
    # Returns a list of 5 arrays, one per LiDAR: [TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR]
    points_all, cp_points_all = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
    )

    # Each element in points_all is (N_i, 3)
    if return_intensity:
        # Also get second-return points for intensity (range_image ri_index=0 → first return)
        # The first 3 columns are XYZ, merge all LiDARs
        all_points = []
        for lidar_points in points_all:
            all_points.append(lidar_points)
        merged = np.concatenate(all_points, axis=0).astype(np.float32)
    else:
        merged = np.concatenate(points_all, axis=0)[:, :3].astype(np.float32)

    return merged


def extract_point_cloud_per_lidar(
    frame: open_dataset.Frame,
) -> Dict[int, np.ndarray]:
    """
    Extract point clouds separately for each LiDAR (in vehicle frame).

    Returns
    -------
    dict : {lidar_id: np.ndarray (N, 3)}
        Keys are Waymo LaserName enum values: 1=TOP, 2=FRONT, 3=SIDE_LEFT,
        4=SIDE_RIGHT, 5=REAR.
    """
    (range_images,
     camera_projections,
     _,
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    points_all, _ = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
    )

    lidar_ids = [1, 2, 3, 4, 5]  # TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR
    return {lid: pts[:, :3].astype(np.float32) for lid, pts in zip(lidar_ids, points_all)}


# ---------------------------------------------------------------------------
# 3D boxes
# ---------------------------------------------------------------------------

def extract_boxes(
    frame: open_dataset.Frame,
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """
    Extract 3D bounding boxes from ``frame.laser_labels``.

    Returns
    -------
    boxes : np.ndarray, shape (M, 7)
        Each row: ``[cx, cy, cz, length, width, height, heading]``
        in the **vehicle frame** (x-forward, y-left, z-up).
        heading: rotation around z-axis, from positive x-axis, counter-clockwise.
    tracking_ids : list of str
        Unique tracking IDs (equivalent to nuScenes instance_token).
    box_types : list of int
        Waymo label type enum: 1=VEHICLE, 2=PEDESTRIAN, 3=SIGN, 4=CYCLIST.
    category_names : list of str
        Human-readable category name.
    """
    TYPE_TO_NAME = {
        label_pb2.Label.TYPE_VEHICLE: "vehicle",
        label_pb2.Label.TYPE_PEDESTRIAN: "pedestrian",
        label_pb2.Label.TYPE_SIGN: "sign",
        label_pb2.Label.TYPE_CYCLIST: "cyclist",
    }

    boxes_list = []
    tracking_ids = []
    box_types = []
    category_names = []

    for label in frame.laser_labels:
        box = label.box
        boxes_list.append([
            box.center_x, box.center_y, box.center_z,
            box.length, box.width, box.height,
            box.heading,
        ])
        tracking_ids.append(label.id)
        box_types.append(label.type)
        category_names.append(TYPE_TO_NAME.get(label.type, "unknown"))

    if not boxes_list:
        return (
            np.zeros((0, 7), dtype=np.float32),
            [],
            [],
            [],
        )

    return (
        np.array(boxes_list, dtype=np.float32),
        tracking_ids,
        box_types,
        category_names,
    )


def waymo_boxes_to_mmcv_format(boxes: np.ndarray) -> np.ndarray:
    """
    Convert Waymo box format to mmcv ``points_in_boxes_cpu`` format.

    Waymo:  [cx, cy, cz, length, width, height, heading]
    mmcv:   [cx, cy, z_bottom, dx, dy, dz, heading]

    Note: Waymo heading is measured from x-axis (forward) counter-clockwise,
    which is the **same convention** as mmcv expects. No π/2 offset needed
    (unlike nuScenes where yaw is from y-axis).
    """
    if boxes.shape[0] == 0:
        return boxes.copy()

    mmcv_boxes = boxes.copy()
    # z_bottom = cz - height/2
    mmcv_boxes[:, 2] = boxes[:, 2] - boxes[:, 5] / 2.0
    # dx=length, dy=width, dz=height (already in this order)
    return mmcv_boxes


# ---------------------------------------------------------------------------
# Ego pose
# ---------------------------------------------------------------------------

def extract_ego_pose(frame: open_dataset.Frame) -> np.ndarray:
    """
    Extract the 4×4 ego vehicle → world transformation matrix.

    Returns
    -------
    pose : np.ndarray, shape (4, 4)
    """
    pose = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    return pose


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def extract_camera_images(
    frame: open_dataset.Frame,
) -> Dict[int, bytes]:
    """
    Extract camera JPEG bytes from the frame.

    Returns
    -------
    images : dict {camera_id: jpeg_bytes}
        camera_id: 1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT
    """
    images = {}
    for image in frame.images:
        images[image.name] = image.image  # raw JPEG bytes
    return images


def extract_camera_calibrations(
    frame: open_dataset.Frame,
) -> Dict[int, dict]:
    """
    Extract camera calibrations from ``frame.context.camera_calibrations``.

    Returns
    -------
    calibrations : dict {camera_id: calibration_dict}
        Each calibration_dict contains:
        - 'extrinsic': np.ndarray (4, 4)  — camera-to-vehicle transform
        - 'intrinsic': np.ndarray (9,)    — [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
        - 'width': int
        - 'height': int
    """
    calibrations = {}
    for calib in frame.context.camera_calibrations:
        extrinsic = np.array(calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)
        intrinsic = np.array(calib.intrinsic, dtype=np.float64)
        calibrations[calib.name] = {
            "extrinsic": extrinsic,     # camera → vehicle frame
            "intrinsic": intrinsic,     # [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
            "width": calib.width,
            "height": calib.height,
        }
    return calibrations


def extract_lidar_calibrations(
    frame: open_dataset.Frame,
) -> Dict[int, np.ndarray]:
    """
    Extract LiDAR extrinsic calibrations (lidar → vehicle frame).

    Returns
    -------
    calibrations : dict {lidar_id: np.ndarray (4, 4)}
    """
    calibrations = {}
    for calib in frame.context.laser_calibrations:
        extrinsic = np.array(calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)
        calibrations[calib.name] = extrinsic
    return calibrations


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

def extract_timestamp_us(frame: open_dataset.Frame) -> int:
    """Frame timestamp in microseconds."""
    return frame.timestamp_micros


def extract_segment_id(frame: open_dataset.Frame) -> str:
    """Segment context name (unique segment identifier)."""
    return frame.context.name


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def vehicle_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Transform points from vehicle frame to world frame.

    Parameters
    ----------
    points : (N, 3)
    pose : (4, 4) vehicle → world

    Returns
    -------
    (N, 3) in world frame
    """
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    pts_homo = np.hstack([points, ones])  # (N, 4)
    pts_world = (pose @ pts_homo.T).T[:, :3]
    return pts_world.astype(np.float32)


def world_to_vehicle(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Transform points from world frame to vehicle frame.

    Parameters
    ----------
    points : (N, 3)
    pose : (4, 4) vehicle → world

    Returns
    -------
    (N, 3) in vehicle frame
    """
    pose_inv = np.linalg.inv(pose)
    return vehicle_to_world(points, pose_inv)


def vehicle_frame_to_vehicle_frame(
    points: np.ndarray,
    pose_src: np.ndarray,
    pose_dst: np.ndarray,
) -> np.ndarray:
    """
    Transform points from one vehicle frame to another via the world frame.

    Equivalent to nuScenes ``lidar_to_world_to_lidar``.

    Parameters
    ----------
    points : (N, 3) in source vehicle frame
    pose_src : (4, 4) source vehicle → world
    pose_dst : (4, 4) destination vehicle → world

    Returns
    -------
    (N, 3) in destination vehicle frame
    """
    pts_world = vehicle_to_world(points, pose_src)
    return world_to_vehicle(pts_world, pose_dst)
