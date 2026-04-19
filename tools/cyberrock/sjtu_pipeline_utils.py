#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml


TOPIC_TO_CHANNEL = OrderedDict(
    [
        ("/multi/camera/wide_len1/compressed", "CAM_FRONT"),
        ("/multi/camera/wide_len2/compressed", "CAM_FRONT_RIGHT"),
        ("/multi/camera/wide_len3/compressed", "CAM_BACK_RIGHT"),
        ("/multi/camera/wide_len4/compressed", "CAM_BACK"),
        ("/multi/camera/wide_len5/compressed", "CAM_BACK_LEFT"),
        ("/multi/camera/wide_len6/compressed", "CAM_FRONT_LEFT"),
    ]
)

CAMERA_CHANNELS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

REF_CHANNEL = "CAM_BACK_LEFT"

SPECIAL_TAG_SEQUENCE_SLUGS = {
    "OE": "s1_occl_emerge",
    "NR": "nr_night_rain",
}

SCENE_SEQUENCE_SLUGS = {
    "S1": "s1_front_static",
    "S2": "s2_rear_static",
    "S3": "s3_side_cutin",
    "S4": "s4_crossing",
    "S5": "s5_passing",
    "S6": "s6_roadside",
    "S7": "s7_clearpath",
    "S8": "s8_lighting_var",
}


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def strip_markdown_code(text: str) -> str:
    return text.strip().strip("`").strip()


def resolve_sequence_slug(scene: str, tag: str) -> str:
    tag_norm = strip_markdown_code(tag).upper()
    if tag_norm in SPECIAL_TAG_SEQUENCE_SLUGS:
        return SPECIAL_TAG_SEQUENCE_SLUGS[tag_norm]

    scene_norm = strip_markdown_code(scene).upper()
    if scene_norm in SCENE_SEQUENCE_SLUGS:
        return SCENE_SEQUENCE_SLUGS[scene_norm]
    if "/" in scene_norm:
        return "multi_scene"
    if not scene_norm or scene_norm == "待填".upper():
        return "unknown_scene"
    return "unknown_scene"


def build_sequence_dir_name(bag_name: str, scene: str, tag: str) -> str:
    bag_stem = Path(strip_markdown_code(bag_name)).stem
    slug = resolve_sequence_slug(scene, tag)
    return f"{slug}__{bag_stem}"


def quat_xyzw_to_rotmat(quat_xyzw) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Quaternion norm is zero.")
    x, y, z, w = quat / norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_camera_projections(yaml_path: str | Path) -> Dict[str, dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    calib = {}
    cameras = config.get("cameras", {})
    for cam_name, cam_cfg in cameras.items():
        intr = cam_cfg["intrinsics"]
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        distortion = intr.get("distortion")
        if distortion:
            dist = np.asarray(distortion, dtype=np.float64)
        else:
            dist = np.array([], dtype=np.float64)

        ext = cam_cfg["extrinsics"]
        t = np.asarray(ext["translation"], dtype=np.float64).reshape(3)
        R = quat_xyzw_to_rotmat(ext["rotation"])

        calib[cam_name] = {
            "K_src": K,
            "dist_src": dist,
            "R": R,
            "t": t,
            "width": intr.get("width"),
            "height": intr.get("height"),
        }

    missing = [ch for ch in CAMERA_CHANNELS if ch not in calib]
    if missing:
        raise KeyError(f"Missing camera calibration for channels: {missing}")
    return calib


def compute_rectify_map_and_kud(
    K_src: np.ndarray,
    dist_src: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    if dist_src is None or np.asarray(dist_src).size == 0:
        return None, None, np.asarray(K_src, dtype=np.float64)

    K_ud, _ = cv2.getOptimalNewCameraMatrix(K_src, dist_src, (width, height), 0, (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(
        K_src,
        dist_src,
        None,
        K_ud,
        (width, height),
        cv2.CV_32FC1,
    )
    return map1, map2, K_ud
