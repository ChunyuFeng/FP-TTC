from __future__ import annotations

import json
import os.path as osp
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .utils.augmentor import NuscRangeImageAugmentor
from utils.waymo_paths import (
    DEFAULT_WAYMO_PROJ_CACHE_NAME,
    WAYMO_CAMERAS,
    WAYMO_RANGE_FOV_DOWN,
    WAYMO_RANGE_FOV_UP,
    default_waymo_proj_cache_root,
    infer_waymo_dataset_root,
    resolve_waymo_camera_image_path,
    resolve_waymo_depth_pred_path,
    resolve_waymo_path,
)


WAYMO_BASE_IMAGE_SIZE = (1920, 1280)  # (W, H)
WAYMO_COORDINATE_FRAME_TOP_LIDAR = "top_lidar"
WAYMO_RANGE_HORIZONTAL_LAYOUT_NUSC_SEMANTIC = "nuscenes_semantic"
WAYMO_RANGE_HORIZONTAL_ALIGNMENT_FRAME_VEHICLE = "vehicle"
WAYMO_RANGE_HORIZONTAL_ALIGNMENT_MODE_ROTATION_ONLY = "rotation_only"
WAYMO_PROJ_CACHE_HW = (40, 400)

WAYMO_CAM_TO_CV = np.array(
    [
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ],
    dtype=np.float64,
)


def _infer_waymo_split_name(train_info_path: str | Path, train_info_file: str) -> str:
    path_name = Path(train_info_path).name.lower()
    if path_name in {"train", "val", "test"}:
        return path_name

    file_name = train_info_file.lower()
    if "waymo_train_" in file_name:
        return "train"
    if "waymo_val_" in file_name:
        return "val"
    if "waymo_test_" in file_name:
        return "test"

    raise ValueError(
        f"Unable to infer split from train_info_path={train_info_path} "
        f"and train_info_file={train_info_file}"
    )


def _load_proj_cache_manifest(cache_root: Path, split_name: str):
    manifest_pkl = cache_root / split_name / "manifest.pkl"
    manifest_json = cache_root / split_name / "manifest.json"

    if manifest_pkl.exists():
        with open(manifest_pkl, "rb") as f:
            manifest = pickle.load(f)
    elif manifest_json.exists():
        manifest = json.loads(manifest_json.read_text())
    else:
        return None

    entries = manifest["entries"] if isinstance(manifest, dict) and "entries" in manifest else manifest
    if not isinstance(entries, list):
        raise ValueError(
            "Unsupported manifest format: "
            f"{manifest_pkl if manifest_pkl.exists() else manifest_json}"
        )

    return {int(entry["original_index"]): entry for entry in entries}


def _float_matches(left, right, tol: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tol


def _resolve_waymo_coordinate_frame(info: dict, frame_key: Optional[str] = None) -> str:
    if "coordinate_frame" in info:
        return info["coordinate_frame"]

    if frame_key is not None:
        sensor_metas = info.get(f"sensor_metas_{frame_key}", {})
        lidar_metas = sensor_metas.get("lidar", {})
        if "coordinate_frame" in lidar_metas:
            return lidar_metas["coordinate_frame"]

    return WAYMO_COORDINATE_FRAME_TOP_LIDAR


def _resolve_waymo_range_fov(info: dict) -> tuple[float, float]:
    return (
        float(info.get("range_fov_up", WAYMO_RANGE_FOV_UP)),
        float(info.get("range_fov_down", WAYMO_RANGE_FOV_DOWN)),
    )


def _resolve_waymo_range_horizontal_layout(info: dict) -> str:
    return info.get("range_horizontal_layout", WAYMO_RANGE_HORIZONTAL_LAYOUT_NUSC_SEMANTIC)


def _resolve_waymo_lidar_to_vehicle_transform(info: dict, frame_key: str) -> np.ndarray:
    key = f"{frame_key}_top_lidar_to_vehicle"
    if key in info:
        return np.asarray(info[key], dtype=np.float64)

    sensor_metas = info.get(f"sensor_metas_{frame_key}", {})
    lidar_metas = sensor_metas.get("lidar", {})
    if "top_lidar_to_vehicle" in lidar_metas:
        return np.asarray(lidar_metas["top_lidar_to_vehicle"], dtype=np.float64)

    raise KeyError(f"Missing {key} in Waymo info.")


def _resolve_waymo_range_horizontal_alignment(
    info: dict,
    frame_key: str,
) -> np.ndarray | None:
    if _resolve_waymo_range_horizontal_layout(info) != WAYMO_RANGE_HORIZONTAL_LAYOUT_NUSC_SEMANTIC:
        return None
    if info.get("range_horizontal_alignment_frame") != WAYMO_RANGE_HORIZONTAL_ALIGNMENT_FRAME_VEHICLE:
        return None
    if info.get("range_horizontal_alignment_mode") != WAYMO_RANGE_HORIZONTAL_ALIGNMENT_MODE_ROTATION_ONLY:
        return None

    if _resolve_waymo_coordinate_frame(info, frame_key) != WAYMO_COORDINATE_FRAME_TOP_LIDAR:
        return np.eye(3, dtype=np.float32)

    transform = _resolve_waymo_lidar_to_vehicle_transform(info, frame_key)
    return np.asarray(transform[:3, :3], dtype=np.float32)


def _validate_waymo_proj_cache_meta(
    cache_root: Path,
    crop_size,
    reference_info: Optional[dict] = None,
):
    meta_path = cache_root / "cache_meta.json"
    if not meta_path.exists():
        return

    meta = json.loads(meta_path.read_text())

    if crop_size is not None:
        expected_crop = [int(crop_size[0]), int(crop_size[1])]
        if meta.get("image_size") != expected_crop:
            raise ValueError(
                "Waymo projection cache image_size mismatch: "
                f"cache={meta.get('image_size')} expected={expected_crop}"
            )

    if meta.get("camera_order") != WAYMO_CAMERAS:
        raise ValueError("Waymo projection cache camera_order mismatch.")

    if [meta.get("H_r"), meta.get("W_r")] != [WAYMO_PROJ_CACHE_HW[0], WAYMO_PROJ_CACHE_HW[1]]:
        raise ValueError("Waymo projection cache range-view resolution mismatch.")

    augmentor_meta = meta.get("augmentor", {})
    if augmentor_meta.get("do_flip") is not False or augmentor_meta.get("rotate") is not False:
        raise ValueError("Waymo projection cache was generated with incompatible augmentation settings.")

    if reference_info is None:
        return

    expected_fov_up, expected_fov_down = _resolve_waymo_range_fov(reference_info)
    if meta.get("coordinate_frame") != _resolve_waymo_coordinate_frame(reference_info):
        raise ValueError("Waymo projection cache coordinate_frame mismatch.")
    if not _float_matches(meta.get("fov_up"), expected_fov_up):
        raise ValueError("Waymo projection cache fov_up mismatch.")
    if not _float_matches(meta.get("fov_down"), expected_fov_down):
        raise ValueError("Waymo projection cache fov_down mismatch.")
    if meta.get("range_horizontal_layout") != _resolve_waymo_range_horizontal_layout(reference_info):
        raise ValueError("Waymo projection cache range_horizontal_layout mismatch.")
    if meta.get("range_horizontal_alignment_frame") != reference_info.get(
        "range_horizontal_alignment_frame",
        WAYMO_RANGE_HORIZONTAL_ALIGNMENT_FRAME_VEHICLE,
    ):
        raise ValueError("Waymo projection cache range_horizontal_alignment_frame mismatch.")
    if meta.get("range_horizontal_alignment_mode") != reference_info.get(
        "range_horizontal_alignment_mode",
        WAYMO_RANGE_HORIZONTAL_ALIGNMENT_MODE_ROTATION_ONLY,
    ):
        raise ValueError("Waymo projection cache range_horizontal_alignment_mode mismatch.")


def _waymo_intrinsic_to_k_and_dist(intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    fx, fy, cx, cy = intrinsic[:4]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = intrinsic[4:9].astype(np.float64)
    return k, dist


def _scale_waymo_k(
    k: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    k_scaled = np.asarray(k, dtype=np.float64).copy()
    k_scaled[0, 0] *= sx
    k_scaled[0, 2] *= sx
    k_scaled[1, 1] *= sy
    k_scaled[1, 2] *= sy
    return k_scaled


def _build_waymo_lidar_to_camera_projection(
    cam_calib: dict,
    lidar_to_vehicle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_cam_to_vehicle = np.asarray(cam_calib["extrinsic"], dtype=np.float64)
    t_vehicle_to_cam_waymo = np.linalg.inv(t_cam_to_vehicle)
    t_lidar_to_cam_waymo = t_vehicle_to_cam_waymo @ np.asarray(lidar_to_vehicle, dtype=np.float64)

    r_l2c = WAYMO_CAM_TO_CV @ t_lidar_to_cam_waymo[:3, :3]
    t_l2c = WAYMO_CAM_TO_CV @ t_lidar_to_cam_waymo[:3, 3]
    k, dist = _waymo_intrinsic_to_k_and_dist(cam_calib["intrinsic"])
    return k, dist, r_l2c.astype(np.float32), t_l2c.astype(np.float32)


def _compute_waymo_range_yaw(
    points_xyz: np.ndarray,
    horizontal_layout: str,
    horizontal_alignment_rotation: np.ndarray | None = None,
) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    aligned_points = points_xyz

    if (
        horizontal_layout == WAYMO_RANGE_HORIZONTAL_LAYOUT_NUSC_SEMANTIC
        and horizontal_alignment_rotation is not None
    ):
        aligned_points = points_xyz @ np.asarray(horizontal_alignment_rotation, dtype=np.float32).T

    x = aligned_points[:, 0]
    y = aligned_points[:, 1]
    if horizontal_layout == WAYMO_RANGE_HORIZONTAL_LAYOUT_NUSC_SEMANTIC:
        return -np.arctan2(x, -y)

    return -np.arctan2(y, x)


def _precompute_pix_orig(h_img: int, w_img: int, affine_matrix: np.ndarray) -> np.ndarray:
    us = np.arange(w_img)
    vs = np.arange(h_img)
    u_grid, v_grid = np.meshgrid(us, vs)
    ones = np.ones_like(u_grid, dtype=np.float32)
    pix_proc = np.stack([u_grid, v_grid, ones], axis=-1).reshape(-1, 3).T
    inv_a = np.linalg.inv(affine_matrix).astype(np.float32)
    pix_orig = inv_a @ pix_proc
    pix_orig /= np.maximum(pix_orig[2:3, :], 1e-8)
    return pix_orig.astype(np.float32)


def _geometry_cam2lidar_from_depth_with_distortion(
    k: np.ndarray,
    dist: np.ndarray,
    r_c2l: np.ndarray,
    t_c2l: np.ndarray,
    pix_orig: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    if depths.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    pts_2d = pix_orig[:2, :].T.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts_2d, k, dist, P=None).reshape(-1, 2).T.astype(np.float32)

    x_c = np.vstack(
        [
            undist[0] * depths,
            undist[1] * depths,
            depths,
        ]
    ).astype(np.float32)
    x_l = (r_c2l @ x_c) + t_c2l.reshape(3, 1).astype(np.float32)
    return x_l.T


def _range_projection_with_mapping_fast(
    points: np.ndarray,
    pix_coords: np.ndarray,
    h: int,
    w: int,
    fov_up: float,
    fov_down: float,
    horizontal_layout: str,
    horizontal_alignment_rotation: np.ndarray | None,
):
    depth = np.linalg.norm(points, axis=1).astype(np.float32)
    safe = np.maximum(depth, 1e-9)
    z = points[:, 2]
    yaw = _compute_waymo_range_yaw(
        points,
        horizontal_layout=horizontal_layout,
        horizontal_alignment_rotation=horizontal_alignment_rotation,
    )
    pitch = np.arcsin(np.clip(z / safe, -1.0, 1.0))

    fov_up_rad = np.deg2rad(fov_up)
    fov_down_rad = np.deg2rad(fov_down)
    fov = abs(fov_down_rad) + abs(fov_up_rad)

    in_fov = (pitch >= fov_down_rad) & (pitch <= fov_up_rad)
    if not np.any(in_fov):
        proj_range = np.full((h, w), -1, np.float32)
        proj_pix = np.full((h, w, 3), -1, np.int32)
        proj_mask = np.zeros((h, w), np.int32)
        return proj_range, proj_pix, proj_mask

    depth = depth[in_fov]
    yaw = yaw[in_fov]
    pitch = pitch[in_fov]
    pix_coords = pix_coords[in_fov]

    proj_x = np.floor(0.5 * (yaw / np.pi + 1.0) * w).astype(np.int32)
    proj_y = np.floor((1.0 - (pitch + abs(fov_down_rad)) / fov) * h).astype(np.int32)
    np.clip(proj_x, 0, w - 1, out=proj_x)
    np.clip(proj_y, 0, h - 1, out=proj_y)

    lin = proj_y.astype(np.int64) * w + proj_x.astype(np.int64)
    order = np.lexsort((depth, lin))
    lin_s = lin[order]

    first = np.empty_like(lin_s, dtype=bool)
    if first.size:
        first[0] = True
        first[1:] = lin_s[1:] != lin_s[:-1]

    sel = order[first]
    lin_unique = lin[sel]
    y_unique = (lin_unique // w).astype(np.int32)
    x_unique = (lin_unique % w).astype(np.int32)

    proj_range = np.full((h, w), -1, np.float32)
    proj_pix = np.full((h, w, 3), -1, np.int32)
    proj_mask = np.zeros((h, w), np.int32)
    proj_range[y_unique, x_unique] = depth[sel]
    proj_pix[y_unique, x_unique] = pix_coords[sel].astype(np.int32)
    proj_mask[y_unique, x_unique] = 1
    return proj_range, proj_pix, proj_mask


def _densify_proj_pix(proj_pix: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid_mask).astype(bool)
    if valid.shape != proj_pix.shape[:2]:
        raise ValueError(
            f"valid_mask shape {valid.shape} does not match proj_pix shape {proj_pix.shape[:2]}"
        )

    if valid.all():
        return proj_pix.astype(np.int32, copy=False)
    if not valid.any():
        raise ValueError("Projection mapping has no valid pixels to densify.")

    _, inds = distance_transform_edt(~valid, return_distances=True, return_indices=True)
    i_near, j_near = inds
    return proj_pix[i_near, j_near].astype(np.int32, copy=False)


def _resolve_frame_camera_calib(info: dict, frame_key: str, camera_name: str) -> Optional[dict]:
    sensor_metas = info.get(f"sensor_metas_{frame_key}", {})
    camera_metas = sensor_metas.get("camera", {}).get("calibrated_sensor", {})
    if camera_name in camera_metas:
        return camera_metas[camera_name]
    return info.get(f"{frame_key}_cameras", {}).get(camera_name)


def _build_waymo_frame_mapping(
    info: dict,
    frame_key: str,
    depth_map_dict: dict[str, np.ndarray],
    affine_matrix: np.ndarray,
    h_r: int = WAYMO_PROJ_CACHE_HW[0],
    w_r: int = WAYMO_PROJ_CACHE_HW[1],
):
    fov_up, fov_down = _resolve_waymo_range_fov(info)
    horizontal_layout = _resolve_waymo_range_horizontal_layout(info)
    horizontal_alignment_rotation = _resolve_waymo_range_horizontal_alignment(info, frame_key)
    lidar_to_vehicle = _resolve_waymo_lidar_to_vehicle_transform(info, frame_key)

    all_points = []
    all_pix = []

    for cam_idx, cam_name in enumerate(WAYMO_CAMERAS):
        cam_calib = _resolve_frame_camera_calib(info, frame_key, cam_name)
        depth_map = depth_map_dict.get(cam_name)
        if cam_calib is None or depth_map is None:
            continue

        k_raw, dist, r_l2c, t_l2c = _build_waymo_lidar_to_camera_projection(
            cam_calib,
            lidar_to_vehicle=lidar_to_vehicle,
        )
        raw_w = int(cam_calib.get("width", WAYMO_BASE_IMAGE_SIZE[0]))
        raw_h = int(cam_calib.get("height", WAYMO_BASE_IMAGE_SIZE[1]))
        k_base = _scale_waymo_k(k_raw, (raw_w, raw_h), WAYMO_BASE_IMAGE_SIZE).astype(np.float32)
        r_c2l = r_l2c.T.astype(np.float32)
        t_c2l = (-r_c2l @ t_l2c.reshape(3, 1)).astype(np.float32)

        h_img, w_img = depth_map.shape
        pix_orig = _precompute_pix_orig(h_img, w_img, affine_matrix)
        depths = depth_map.reshape(-1).astype(np.float32)
        valid = depths > 0
        if not np.any(valid):
            continue

        pts = _geometry_cam2lidar_from_depth_with_distortion(
            k_base,
            np.asarray(dist, dtype=np.float32),
            r_c2l,
            t_c2l,
            pix_orig[:, valid],
            depths[valid],
        ).astype(np.float32)

        uu, vv = np.meshgrid(np.arange(w_img), np.arange(h_img))
        uu = uu.reshape(-1)[valid].astype(np.int32)
        vv = vv.reshape(-1)[valid].astype(np.int32)
        cam_col = np.full_like(uu, cam_idx, dtype=np.int32)
        pix = np.stack([cam_col, uu, vv], axis=1).astype(np.int32)

        all_points.append(pts)
        all_pix.append(pix)

    if not all_points:
        raise RuntimeError("Waymo on-the-fly projection fallback found no valid depth points.")

    points = np.concatenate(all_points, axis=0)
    pix = np.concatenate(all_pix, axis=0)
    _, proj_pix, proj_mask = _range_projection_with_mapping_fast(
        points,
        pix,
        h=h_r,
        w=w_r,
        fov_up=fov_up,
        fov_down=fov_down,
        horizontal_layout=horizontal_layout,
        horizontal_alignment_rotation=horizontal_alignment_rotation,
    )
    return _densify_proj_pix(proj_pix, proj_mask)


def _resize_rgb_pil(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    if img.size == target_size:
        return img
    return img.resize(target_size, Image.BILINEAR)


def _resize_depth_to_base(depth: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    if depth.shape == (target_h, target_w):
        return depth.astype(np.float32, copy=False)
    return cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _apply_affine_to_depth_map(
    depth: np.ndarray,
    affine_params: dict,
    crop_size: tuple[int, int],
) -> np.ndarray:
    depth = depth.astype(np.float32, copy=False)
    resize_w, resize_h = affine_params["resize"]
    if depth.shape != (resize_h, resize_w):
        depth = cv2.resize(depth, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    crop_x, crop_y = affine_params["crop"]
    final_h, final_w = crop_size
    depth = depth[crop_y: crop_y + final_h, crop_x: crop_x + final_w]

    if affine_params.get("flip_h", False):
        depth = np.ascontiguousarray(depth[:, ::-1])
    if affine_params.get("flip_v", False):
        depth = np.ascontiguousarray(depth[::-1, :])

    if affine_params.get("rotate", False):
        angle = float(affine_params.get("angle", 0.0))
        h, w = depth.shape
        center = (w * 0.5, h * 0.5)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        depth = cv2.warpAffine(
            depth,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    return depth.astype(np.float32, copy=False)


def _build_waymo_frame_sensor_metas(
    info: dict,
    frame_key: str,
    affine_matrix: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    sensor_metas = {}
    lidar_to_vehicle = _resolve_waymo_lidar_to_vehicle_transform(info, frame_key)
    affine = np.asarray(affine_matrix, dtype=np.float32)

    for camera_name in WAYMO_CAMERAS:
        cam_calib = _resolve_frame_camera_calib(info, frame_key, camera_name)
        if cam_calib is None:
            raise KeyError(f"Missing camera calibration for {frame_key}:{camera_name}")

        k_raw, _dist, r_l2c, t_l2c = _build_waymo_lidar_to_camera_projection(
            cam_calib,
            lidar_to_vehicle=lidar_to_vehicle,
        )
        raw_w = int(cam_calib.get("width", WAYMO_BASE_IMAGE_SIZE[0]))
        raw_h = int(cam_calib.get("height", WAYMO_BASE_IMAGE_SIZE[1]))
        k_base = _scale_waymo_k(k_raw, (raw_w, raw_h), WAYMO_BASE_IMAGE_SIZE).astype(np.float32)
        sensor_metas[camera_name] = {
            "K": k_base,
            "R_l2c": np.asarray(r_l2c, dtype=np.float32),
            "t_l2c": np.asarray(t_l2c, dtype=np.float32).reshape(3),
            "affine": affine,
        }

    return sensor_metas


def _tensorize_sensor_metas(sensor_metas: dict) -> dict:
    tensorized = {}
    for frame_key, frame_sensor_metas in sensor_metas.items():
        tensorized[frame_key] = {}
        for channel, channel_metas in frame_sensor_metas.items():
            tensorized[frame_key][channel] = {
                key: torch.from_numpy(np.asarray(value)).float()
                for key, value in channel_metas.items()
            }
    return tensorized


class Waymo_range_image(data.Dataset):
    def __init__(
        self,
        aug_params=None,
        split="training",
        train_info_path="./Datasets/waymo/2_trainval_test_infos/train",
        train_info_file="waymo_train_infos.pkl",
        require_complete_depth=False,
        max_samples=None,
        proj_cache_root=None,
        dataset_root=None,
    ):
        self.aug_params = aug_params
        self.split = split
        self.train_info_path = train_info_path
        self.train_info_file = train_info_file
        self.require_complete_depth = require_complete_depth
        self.max_samples = max_samples
        inferred_root = infer_waymo_dataset_root(self.train_info_path)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else inferred_root
        self.proj_cache_root = (
            Path(proj_cache_root)
            if proj_cache_root is not None
            else default_waymo_proj_cache_root(self.dataset_root, DEFAULT_WAYMO_PROJ_CACHE_NAME)
        )
        self.split_name = _infer_waymo_split_name(self.train_info_path, self.train_info_file)
        self.camera_channels = WAYMO_CAMERAS

        pkl_file_path = osp.join(self.train_info_path, self.train_info_file)
        if not osp.exists(pkl_file_path):
            raise FileNotFoundError(f"No such file: {pkl_file_path}")
        with open(pkl_file_path, "rb") as f:
            loaded_data = pickle.load(f)
        print(f"Loaded data from {pkl_file_path}")

        original_samples = len(loaded_data)
        complete_depth_samples = original_samples
        filtered_samples = []
        for original_index, info in enumerate(loaded_data):
            if self.require_complete_depth and not self._has_complete_depth(info):
                continue
            filtered_samples.append((original_index, info))

        if self.require_complete_depth:
            complete_depth_samples = len(filtered_samples)

        if self.max_samples is not None:
            filtered_samples = filtered_samples[:self.max_samples]

        self.samples = filtered_samples
        final_samples = len(self.samples)
        if self.require_complete_depth or self.max_samples is not None:
            print(
                "[Waymo_range_image] sample stats: "
                f"original={original_samples}, "
                f"complete_depth={complete_depth_samples}, "
                f"final={final_samples}"
            )

        if final_samples == 0:
            raise ValueError(
                "No samples available for Waymo_range_image after filtering. "
                "Check the current pkl and 4_depth_map coverage."
            )

        self.augmentor = None
        if self.aug_params is not None:
            self.augmentor = NuscRangeImageAugmentor(**self.aug_params)

        crop_size = None if self.aug_params is None else self.aug_params.get("crop_size")
        _validate_waymo_proj_cache_meta(
            self.proj_cache_root,
            crop_size,
            reference_info=self.samples[0][1] if self.samples else None,
        )
        manifest_by_original_index = _load_proj_cache_manifest(self.proj_cache_root, self.split_name)
        self.cache_entries = None
        if manifest_by_original_index is not None:
            self.cache_entries = []
            for original_index, _ in self.samples:
                if original_index not in manifest_by_original_index:
                    raise KeyError(
                        f"Missing cache entry for original_index={original_index} "
                        f"in {self.proj_cache_root / self.split_name}"
                    )
                entry = dict(manifest_by_original_index[original_index])
                cache_path = self.proj_cache_root / entry["cache_relpath"]
                if not cache_path.exists():
                    raise FileNotFoundError(f"Projection cache file {cache_path} does not exist.")
                entry["cache_path"] = cache_path
                self.cache_entries.append(entry)
            print(f"Loaded Waymo projection cache from {self.proj_cache_root / self.split_name}")
        else:
            print(
                f"[Waymo_range_image] projection cache not found under "
                f"{self.proj_cache_root / self.split_name}; "
                "falling back to on-the-fly proj mapping."
            )

    def __len__(self):
        return len(self.samples)

    def _resolve_frame_image_path(self, info: dict, frame_key: str, camera_name: str) -> Path:
        frame_cameras = info.get(f"{frame_key}_cameras", {})
        image_info = frame_cameras.get(camera_name, {})
        image_path = resolve_waymo_path(image_info.get("image_path"), self.dataset_root)
        if image_path is not None:
            return image_path

        return resolve_waymo_camera_image_path(
            camera_name,
            info["segment_id"],
            info[f"{frame_key}_frame_idx"],
            self.dataset_root,
        )

    def _resolve_frame_depth_path(self, info: dict, frame_key: str, camera_name: str) -> Path:
        return resolve_waymo_depth_pred_path(
            camera_name,
            info["segment_id"],
            info[f"{frame_key}_frame_idx"],
            self.dataset_root,
        )

    def _load_frame_images_and_depths(
        self,
        info: dict,
        frame_key: str,
    ) -> tuple[dict[str, Image.Image], dict[str, np.ndarray]]:
        imgs = {}
        depths = {}
        target_w, target_h = WAYMO_BASE_IMAGE_SIZE

        for camera_name in self.camera_channels:
            image_path = self._resolve_frame_image_path(info, frame_key, camera_name)
            depth_path = self._resolve_frame_depth_path(info, frame_key, camera_name)

            img = Image.open(image_path).convert("RGB")
            imgs[camera_name] = _resize_rgb_pil(img, (target_w, target_h))

            if depth_path.exists():
                depth = np.load(depth_path).astype(np.float32, copy=False)
            else:
                depth = np.zeros((target_h, target_w), dtype=np.float32)
            depths[camera_name] = _resize_depth_to_base(depth, (target_w, target_h))

        return imgs, depths

    def _has_complete_depth(self, info: dict) -> bool:
        for frame_key in ["prev", "curr"]:
            for camera_name in self.camera_channels:
                if not self._resolve_frame_depth_path(info, frame_key, camera_name).exists():
                    return False
        return True

    def __getitem__(self, index):
        _, info = self.samples[index]
        prev_imgs, prev_depths = self._load_frame_images_and_depths(info, "prev")
        curr_imgs, curr_depths = self._load_frame_images_and_depths(info, "curr")

        range_image_dir = resolve_waymo_path(info["gt_map_path"], self.dataset_root)
        range_image_path = Path(range_image_dir) / "range_image_curr.npy"
        if not range_image_path.exists():
            raise FileNotFoundError(f"Range image file {range_image_path} does not exist.")
        range_image = np.load(range_image_path, allow_pickle=True).item()
        gt_scale_map = np.asarray(range_image["scale"], dtype=np.float32)
        gt_risk_score_map = np.asarray(range_image["risk_score"], dtype=np.float32)
        valid_mask = range_image.get("image_pair_valid_mask")
        if valid_mask is None:
            valid_mask = (gt_scale_map > 0.3) & (gt_scale_map < 3.0)
        valid_mask = np.asarray(valid_mask, dtype=np.float32)

        if self.augmentor is not None:
            orig_size = next(iter(prev_imgs.values())).size
            affine_params = self.augmentor.sample_params(orig_size)
            affine_matrix = self.augmentor.get_affine_matrix(affine_params)
            prev_imgs, _ = self.augmentor(prev_imgs, affine_params)
            curr_imgs, _ = self.augmentor(curr_imgs, affine_params)
            prev_depths = {
                camera_name: _apply_affine_to_depth_map(
                    depth_map,
                    affine_params,
                    tuple(self.augmentor.crop_size),
                )
                for camera_name, depth_map in prev_depths.items()
            }
            curr_depths = {
                camera_name: _apply_affine_to_depth_map(
                    depth_map,
                    affine_params,
                    tuple(self.augmentor.crop_size),
                )
                for camera_name, depth_map in curr_depths.items()
            }
        else:
            affine_matrix = np.eye(3, dtype=np.float32)
            prev_imgs = {
                camera_name: np.asarray(img, dtype=np.uint8)
                for camera_name, img in prev_imgs.items()
            }
            curr_imgs = {
                camera_name: np.asarray(img, dtype=np.uint8)
                for camera_name, img in curr_imgs.items()
            }
            prev_depths = {
                camera_name: depth_map.astype(np.float32, copy=False)
                for camera_name, depth_map in prev_depths.items()
            }
            curr_depths = {
                camera_name: depth_map.astype(np.float32, copy=False)
                for camera_name, depth_map in curr_depths.items()
            }

        if self.cache_entries is not None:
            with np.load(self.cache_entries[index]["cache_path"]) as cache_npz:
                proj_pix_prev = cache_npz["proj_pix_prev"].astype(np.int32, copy=False)
                proj_pix_curr = cache_npz["proj_pix_curr"].astype(np.int32, copy=False)
                valid_mask_prev = cache_npz.get("valid_mask_prev")
                valid_mask_curr = cache_npz.get("valid_mask_curr")
            if valid_mask_prev is not None:
                proj_pix_prev = _densify_proj_pix(proj_pix_prev, valid_mask_prev)
            if valid_mask_curr is not None:
                proj_pix_curr = _densify_proj_pix(proj_pix_curr, valid_mask_curr)
        else:
            proj_pix_prev = _build_waymo_frame_mapping(
                info,
                "prev",
                prev_depths,
                affine_matrix,
                h_r=WAYMO_PROJ_CACHE_HW[0],
                w_r=WAYMO_PROJ_CACHE_HW[1],
            )
            proj_pix_curr = _build_waymo_frame_mapping(
                info,
                "curr",
                curr_depths,
                affine_matrix,
                h_r=WAYMO_PROJ_CACHE_HW[0],
                w_r=WAYMO_PROJ_CACHE_HW[1],
            )

        sensor_metas = {
            "prev": _build_waymo_frame_sensor_metas(info, "prev", affine_matrix),
            "curr": _build_waymo_frame_sensor_metas(info, "curr", affine_matrix),
        }
        sensor_metas = _tensorize_sensor_metas(sensor_metas)

        prev_imgs_tensor = torch.stack(
            [torch.from_numpy(prev_imgs[camera_name]).permute(2, 0, 1).float() for camera_name in self.camera_channels],
            dim=0,
        )
        curr_imgs_tensor = torch.stack(
            [torch.from_numpy(curr_imgs[camera_name]).permute(2, 0, 1).float() for camera_name in self.camera_channels],
            dim=0,
        )

        prev_depths_tensor = torch.stack(
            [torch.from_numpy(prev_depths[camera_name]).float() for camera_name in self.camera_channels],
            dim=0,
        ).unsqueeze(1)
        curr_depths_tensor = torch.stack(
            [torch.from_numpy(curr_depths[camera_name]).float() for camera_name in self.camera_channels],
            dim=0,
        ).unsqueeze(1)

        gt_scale_map = torch.from_numpy(gt_scale_map).float()
        gt_risk_score_map = torch.from_numpy(gt_risk_score_map).float()
        valid_mask_tensor = torch.from_numpy(valid_mask).float()
        gt_scale_map_with_mask = torch.cat(
            (gt_scale_map.unsqueeze(0), valid_mask_tensor.unsqueeze(0)),
            dim=0,
        )
        gt_risk_map_with_mask = torch.cat(
            (gt_risk_score_map.unsqueeze(0), valid_mask_tensor.unsqueeze(0)),
            dim=0,
        )

        proj_pix_prev_tensor = torch.from_numpy(proj_pix_prev.astype(np.int64))
        proj_pix_curr_tensor = torch.from_numpy(proj_pix_curr.astype(np.int64))

        return (
            prev_imgs_tensor,
            curr_imgs_tensor,
            prev_depths_tensor,
            curr_depths_tensor,
            proj_pix_prev_tensor,
            proj_pix_curr_tensor,
            gt_scale_map_with_mask,
            gt_risk_map_with_mask,
            sensor_metas,
        )

    def __rmul__(self, value):
        self.samples = value * self.samples
        if self.cache_entries is not None:
            self.cache_entries = value * self.cache_entries
        return self
