"""
Waymo range-image dataset for FP-TTC training / evaluation.

This mirrors the nuScenes range-image dataset, but keeps the projection chain
consistent with the actual Waymo camera images used by the model:

* 5 cameras (no rear camera)
* Raw side cameras are 1920x886 and are first resized to a common base canvas
  1920x1280 before any augmentation
* The camera->range mapping is built online from the per-sample depth maps and
  the exact affine used for the current sample
* Invalid range pixels remain invalid instead of being filled by nearest
  neighbours
"""
from __future__ import annotations

import copy
import os
import os.path as osp
import pickle

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

from utils.waymo_paths import (
    WAYMO_CAMERAS,
    resolve_waymo_path,
    resolve_waymo_depth_pred_path,
    resolve_waymo_camera_image_path,
    infer_waymo_dataset_root,
)


WAYMO_BASE_IMAGE_SIZE = (1920, 1280)  # (W, H)


# ---------------------------------------------------------------------------
# Waymo calibration helpers
# ---------------------------------------------------------------------------

def waymo_intrinsic_to_K_and_dist(intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Waymo intrinsic [fx,fy,cx,cy,k1,k2,p1,p2,k3] to (K, dist)."""
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    fx, fy, cx, cy = intrinsic[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = intrinsic[4:9].astype(np.float64)
    return K, dist


def scale_waymo_K(
    K: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    """Scale intrinsics from source image size to destination image size."""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    K_scaled = np.asarray(K, dtype=np.float64).copy()
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return K_scaled


# Waymo camera frame: x-forward, y-left, z-up
# Standard CV frame:  x-right, y-down, z-forward
WAYMO_CAM_TO_CV = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]], dtype=np.float64)


def build_waymo_lidar_to_camera_projection(
    cam_calib: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build vehicle(lidar)-frame -> CV-camera projection for Waymo.

    Returns
    -------
    P      : (3, 4) projection matrix
    K      : (3, 3) intrinsic matrix
    dist   : (5,) distortion coefficients [k1, k2, p1, p2, k3]
    R_l2c  : (3, 3) vehicle -> CV-camera rotation
    t_l2c  : (3,)   vehicle -> CV-camera translation
    """
    T_cam2veh = np.asarray(cam_calib["extrinsic"], dtype=np.float64)
    T_veh2cam_waymo = np.linalg.inv(T_cam2veh)

    R_veh2cam_cv = WAYMO_CAM_TO_CV @ T_veh2cam_waymo[:3, :3]
    t_veh2cam_cv = WAYMO_CAM_TO_CV @ T_veh2cam_waymo[:3, 3]

    R_l2c = R_veh2cam_cv
    t_l2c = t_veh2cam_cv

    K, dist = waymo_intrinsic_to_K_and_dist(cam_calib["intrinsic"])
    RT = np.hstack([R_l2c, t_l2c.reshape(3, 1)])
    P = K @ RT
    return P, K, dist, R_l2c, t_l2c


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _identity_affine() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


def _precompute_pix_orig(H_img: int, W_img: int, affine_matrix: np.ndarray) -> np.ndarray:
    """
    Map processed pixel coordinates back to the common pre-augmentation canvas.
    """
    us = np.arange(W_img)
    vs = np.arange(H_img)
    u_grid, v_grid = np.meshgrid(us, vs)
    ones = np.ones_like(u_grid, dtype=np.float32)
    pix_proc = np.stack([u_grid, v_grid, ones], axis=-1).reshape(-1, 3).T
    invA = np.linalg.inv(affine_matrix).astype(np.float32)
    pix_orig = invA @ pix_proc
    pix_orig /= np.maximum(pix_orig[2:3, :], 1e-8)
    return pix_orig.astype(np.float32)


def _geometry_cam2lidar_from_depth_with_distortion(
    K: np.ndarray,
    dist: np.ndarray,
    R_c2l: np.ndarray,
    t_c2l: np.ndarray,
    pix_orig: np.ndarray,
    ds: np.ndarray,
) -> np.ndarray:
    """
    Back-project distorted image pixels to vehicle frame.

    `pix_orig` should live in the common pre-augmentation canvas. The distortion
    model is handled in pixel space using OpenCV's undistortPoints.
    """
    if ds.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    pts_2d = pix_orig[:2, :].T.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts_2d, K, dist, P=None).reshape(-1, 2).T
    undist = undist.astype(np.float32)

    Xc = np.vstack([
        undist[0] * ds,
        undist[1] * ds,
        ds,
    ]).astype(np.float32)
    Xl = (R_c2l @ Xc) + t_c2l.reshape(3, 1).astype(np.float32)
    return Xl.T


def _range_projection_with_mapping_fast(
    points: np.ndarray,
    pix_coords: np.ndarray,
    H: int = 160,
    W: int = 1600,
    fov_up: float = 5.0,
    fov_down: float = -25.0,
):
    """Vectorised z-buffer: keep the nearest point per range pixel."""
    depth = np.linalg.norm(points, axis=1).astype(np.float32)
    safe = np.maximum(depth, 1e-9)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / safe, -1.0, 1.0))

    fov_up_rad = np.deg2rad(fov_up)
    fov_down_rad = np.deg2rad(fov_down)
    fov = abs(fov_down_rad) + abs(fov_up_rad)

    proj_x = np.floor(0.5 * (yaw / np.pi + 1.0) * W).astype(np.int32)
    proj_y = np.floor((1.0 - (pitch + abs(fov_down_rad)) / fov) * H).astype(np.int32)
    np.clip(proj_x, 0, W - 1, out=proj_x)
    np.clip(proj_y, 0, H - 1, out=proj_y)

    lin = proj_y.astype(np.int64) * W + proj_x.astype(np.int64)
    order = np.lexsort((depth, lin))
    lin_s = lin[order]

    first = np.empty_like(lin_s, dtype=bool)
    if first.size:
        first[0] = True
        first[1:] = lin_s[1:] != lin_s[:-1]

    sel = order[first]
    lin_unique = lin[sel]
    y_unique = (lin_unique // W).astype(np.int32)
    x_unique = (lin_unique % W).astype(np.int32)

    proj_range = np.full((H, W), -1, np.float32)
    proj_pix = np.full((H, W, 3), -1, np.int32)
    proj_mask = np.zeros((H, W), np.int32)

    proj_range[y_unique, x_unique] = depth[sel]
    proj_pix[y_unique, x_unique] = pix_coords[sel].astype(np.int32)
    proj_mask[y_unique, x_unique] = 1
    return proj_range, proj_pix, proj_mask


def build_waymo_frame_mapping(
    info: dict,
    frame_key: str,
    depth_map_dict: dict[str, np.ndarray],
    affine_matrix: np.ndarray,
    H_r: int = 40,
    W_r: int = 400,
    pixel_stride: int = 1,
    base_canvas_size: tuple[int, int] = WAYMO_BASE_IMAGE_SIZE,
):
    """
    Build camera->range mapping for one Waymo frame using the exact processed
    depth maps and the affine applied to the current sample.

    Returns
    -------
    proj_range : (H_r, W_r) float32
    proj_pix   : (H_r, W_r, 3) int32 = (cam_idx, u_proc, v_proc)
    sensor_metas_channel : dict per camera with K, dist, R_l2c, t_l2c, affine
    """
    if frame_key not in {"prev", "curr"}:
        raise ValueError(f"Unsupported frame_key={frame_key}")

    cameras_dict = info.get(f"{frame_key}_cameras", {})
    sensor_metas_channel = {}
    all_points = []
    all_pix = []

    base_w, base_h = base_canvas_size

    for cam_idx, cam_name in enumerate(WAYMO_CAMERAS):
        cam_calib = cameras_dict.get(cam_name)
        depth_map = depth_map_dict.get(cam_name)
        if cam_calib is None or depth_map is None:
            continue

        _, K_raw, dist, R_l2c, t_l2c = build_waymo_lidar_to_camera_projection(cam_calib)
        raw_w = int(cam_calib["width"])
        raw_h = int(cam_calib["height"])
        K_base = scale_waymo_K(K_raw, (raw_w, raw_h), (base_w, base_h))

        R_c2l = R_l2c.T.astype(np.float32)
        t_c2l = (-R_c2l @ t_l2c.reshape(3, 1)).astype(np.float32)

        sensor_metas_channel[cam_name] = {
            "K": K_base.astype(np.float32),
            "dist": dist.astype(np.float32),
            "R_l2c": R_l2c.astype(np.float32),
            "t_l2c": t_l2c.astype(np.float32),
            "affine": affine_matrix.astype(np.float32),
            "base_image_size": np.asarray([base_w, base_h], dtype=np.float32),
            "raw_image_size": np.asarray([raw_w, raw_h], dtype=np.float32),
        }

        H_img, W_img = depth_map.shape
        pix_orig = _precompute_pix_orig(H_img, W_img, affine_matrix)

        if pixel_stride > 1:
            us = np.arange(0, W_img, pixel_stride)
            vs = np.arange(0, H_img, pixel_stride)
            u_grid_s, v_grid_s = np.meshgrid(us, vs)
            pick_lin = (v_grid_s * W_img + u_grid_s).reshape(-1)
            pix_orig_use = pix_orig[:, pick_lin]
            ds = depth_map.reshape(-1)[pick_lin]
            uu_base = u_grid_s.reshape(-1)
            vv_base = v_grid_s.reshape(-1)
        else:
            pix_orig_use = pix_orig
            ds = depth_map.reshape(-1)
            uu_full, vv_full = np.meshgrid(np.arange(W_img), np.arange(H_img))
            uu_base = uu_full.reshape(-1)
            vv_base = vv_full.reshape(-1)

        valid = ds > 0
        if not np.any(valid):
            continue

        pts = _geometry_cam2lidar_from_depth_with_distortion(
            K_base,
            dist,
            R_c2l,
            t_c2l,
            pix_orig_use[:, valid],
            ds[valid].astype(np.float32),
        ).astype(np.float32)

        uu = uu_base[valid].astype(np.int32)
        vv = vv_base[valid].astype(np.int32)
        cam_col = np.full_like(uu, cam_idx, dtype=np.int32)
        pix = np.stack([cam_col, uu, vv], axis=1).astype(np.int32)

        all_points.append(pts)
        all_pix.append(pix)

    if not all_points:
        proj_range = np.full((H_r, W_r), -1, np.float32)
        proj_pix = np.full((H_r, W_r, 3), -1, np.int32)
        return proj_range, proj_pix, sensor_metas_channel

    points = np.concatenate(all_points, axis=0)
    pix = np.concatenate(all_pix, axis=0)
    proj_range, proj_pix, _ = _range_projection_with_mapping_fast(
        points, pix, H=H_r, W=W_r, fov_up=5.0, fov_down=-25.0
    )
    return proj_range, proj_pix, sensor_metas_channel


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
    """
    Apply the same resize/crop/flip/rotate params used by NuscRangeImageAugmentor
    to a float depth map without relying on PIL mode='F' helpers.
    """
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


class Waymo_range_image(data.Dataset):
    """
    Waymo surround-view range-image dataset, compatible with the FP-TTC
    training loop.

    Output signature matches ``nuScenes_range_image.__getitem__``.
    """

    CAMERA_CHANNELS = WAYMO_CAMERAS
    BASE_IMAGE_SIZE = WAYMO_BASE_IMAGE_SIZE

    def __init__(
        self,
        aug_params=None,
        split="training",
        train_info_path="./Datasets/waymo/2_trainval_test_infos/train",
        train_info_file="waymo_train_infos_scene_flow_key_frames_160_1600_fov_5_25.pkl",
        require_complete_depth=False,
        max_samples=None,
    ):
        self.aug_params = aug_params
        self.split = split
        self.train_info_path = train_info_path
        self.train_info_file = train_info_file
        self.require_complete_depth = require_complete_depth
        self.max_samples = max_samples
        self.dataset_root = infer_waymo_dataset_root(self.train_info_path)
        self.camera_channels = self.CAMERA_CHANNELS

        pkl_path = osp.join(self.train_info_path, self.train_info_file)
        if not osp.exists(pkl_path):
            raise FileNotFoundError(f"No such file: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} samples from {pkl_path}")

        original_samples = len(self.data)
        if self.require_complete_depth:
            self.data = [info for info in self.data if self._has_complete_depth(info)]
        if self.max_samples is not None:
            self.data = self.data[: self.max_samples]

        final_samples = len(self.data)
        if final_samples != original_samples:
            print(
                f"[Waymo_range_image] sample stats: "
                f"original={original_samples}, final={final_samples}"
            )
        if final_samples == 0:
            raise ValueError("No samples available for Waymo_range_image after filtering.")

        self.augmentor = None
        if self.aug_params is not None:
            try:
                from .utils.augmentor import NuscRangeImageAugmentor
                self.augmentor = NuscRangeImageAugmentor(**self.aug_params)
            except ImportError:
                print("[Waymo_range_image] NuscRangeImageAugmentor not available, no augmentation.")

        self.image_list = []
        self.scale_map_list = []
        self.risk_score_map_list = []
        self.depth_map_list = []

        for i in tqdm(range(len(self.data)), desc="Loading Waymo Range Image Dataset"):
            info = self.data[i]

            prev_camera_data = {}
            curr_camera_data = {}
            for cam_name in self.camera_channels:
                prev_camera_data[cam_name] = {
                    "image_path": str(resolve_waymo_camera_image_path(
                        cam_name,
                        f"segment_{info['segment_id']}",
                        info["prev_frame_idx"],
                        self.dataset_root,
                    )),
                }
                curr_camera_data[cam_name] = {
                    "image_path": str(resolve_waymo_camera_image_path(
                        cam_name,
                        f"segment_{info['segment_id']}",
                        info["curr_frame_idx"],
                        self.dataset_root,
                    )),
                }
            self.image_list.append([prev_camera_data, curr_camera_data])

            range_image_dir = resolve_waymo_path(info["gt_map_path"], self.dataset_root)
            range_image_path = osp.join(str(range_image_dir), "range_image_curr.npy")
            if not osp.exists(range_image_path):
                raise FileNotFoundError(f"Range image not found: {range_image_path}")
            range_image = np.load(range_image_path, allow_pickle=True).item()
            self.scale_map_list.append(range_image["scale"])
            self.risk_score_map_list.append(range_image["risk_score"])
            self.depth_map_list.append(range_image["depth"])

    def _has_complete_depth(self, info: dict) -> bool:
        for frame_idx_key in ["prev_frame_idx", "curr_frame_idx"]:
            for cam_name in self.camera_channels:
                depth_path = resolve_waymo_depth_pred_path(
                    cam_name,
                    f"segment_{info['segment_id']}",
                    info[frame_idx_key],
                    self.dataset_root,
                )
                if not depth_path.exists():
                    return False
        return True

    def __len__(self):
        return len(self.image_list)

    def _load_frame_images_and_depths(
        self,
        info: dict,
        frame_key: str,
    ) -> tuple[dict[str, Image.Image], dict[str, np.ndarray]]:
        target_w, target_h = self.BASE_IMAGE_SIZE
        imgs = {}
        depths = {}
        frame_idx = info[f"{frame_key}_frame_idx"]
        segment_name = f"segment_{info['segment_id']}"

        for cam_name in self.camera_channels:
            img_path = resolve_waymo_camera_image_path(
                cam_name, segment_name, frame_idx, self.dataset_root
            )
            depth_path = resolve_waymo_depth_pred_path(
                cam_name, segment_name, frame_idx, self.dataset_root
            )

            img = Image.open(str(img_path)).convert("RGB")
            img = _resize_rgb_pil(img, (target_w, target_h))
            imgs[cam_name] = img

            if depth_path.exists():
                depth = np.load(str(depth_path)).astype(np.float32)
            else:
                depth = np.zeros((target_h, target_w), dtype=np.float32)
            depth = _resize_depth_to_base(depth, (target_w, target_h))
            depths[cam_name] = depth

        return imgs, depths

    def __getitem__(self, index):
        info = self.data[index]
        prev_imgs_pil, prev_depths_raw = self._load_frame_images_and_depths(info, "prev")
        curr_imgs_pil, curr_depths_raw = self._load_frame_images_and_depths(info, "curr")

        if self.augmentor is not None:
            orig_size = next(iter(prev_imgs_pil.values())).size
            affine_params = self.augmentor.sample_params(orig_size)
            affine_matrix = self.augmentor.get_affine_matrix(affine_params)

            prev_imgs_np, _ = self.augmentor(copy.deepcopy(prev_imgs_pil), affine_params)
            curr_imgs_np, _ = self.augmentor(copy.deepcopy(curr_imgs_pil), affine_params)

            prev_depths_proc = {
                cam_name: _apply_affine_to_depth_map(
                    depth,
                    affine_params,
                    self.augmentor.crop_size,
                )
                for cam_name, depth in prev_depths_raw.items()
            }
            curr_depths_proc = {
                cam_name: _apply_affine_to_depth_map(
                    depth,
                    affine_params,
                    self.augmentor.crop_size,
                )
                for cam_name, depth in curr_depths_raw.items()
            }
        else:
            affine_matrix = _identity_affine()
            prev_imgs_np = {
                cam_name: np.asarray(img, dtype=np.uint8)
                for cam_name, img in prev_imgs_pil.items()
            }
            curr_imgs_np = {
                cam_name: np.asarray(img, dtype=np.uint8)
                for cam_name, img in curr_imgs_pil.items()
            }
            prev_depths_proc = {
                cam_name: depth.astype(np.float32, copy=False)
                for cam_name, depth in prev_depths_raw.items()
            }
            curr_depths_proc = {
                cam_name: depth.astype(np.float32, copy=False)
                for cam_name, depth in curr_depths_raw.items()
            }

        proj_range_prev, proj_pix_prev, sensor_metas_prev = build_waymo_frame_mapping(
            info,
            "prev",
            prev_depths_proc,
            affine_matrix,
            H_r=40,
            W_r=400,
        )
        proj_range_curr, proj_pix_curr, sensor_metas_curr = build_waymo_frame_mapping(
            info,
            "curr",
            curr_depths_proc,
            affine_matrix,
            H_r=40,
            W_r=400,
        )
        _ = (proj_range_prev, proj_range_curr)

        prev_imgs_tensor = torch.stack([
            torch.from_numpy(prev_imgs_np[ch]).permute(2, 0, 1).float()
            for ch in self.camera_channels
        ], dim=0)
        curr_imgs_tensor = torch.stack([
            torch.from_numpy(curr_imgs_np[ch]).permute(2, 0, 1).float()
            for ch in self.camera_channels
        ], dim=0)

        prev_depths_tensor = torch.stack([
            torch.from_numpy(prev_depths_proc[ch]).float()
            for ch in self.camera_channels
        ], dim=0).unsqueeze(1)
        curr_depths_tensor = torch.stack([
            torch.from_numpy(curr_depths_proc[ch]).float()
            for ch in self.camera_channels
        ], dim=0).unsqueeze(1)

        gt_scale_map = torch.from_numpy(self.scale_map_list[index]).float()
        gt_risk_score_map = torch.from_numpy(self.risk_score_map_list[index]).float()
        _gt_depth_map = torch.from_numpy(self.depth_map_list[index]).float()
        _ = _gt_depth_map
        mask_scale = (gt_scale_map > 0.3) & (gt_scale_map < 3.0)
        gt_scale_map_with_mask = torch.cat(
            (gt_scale_map.unsqueeze(0), mask_scale.unsqueeze(0).float()), dim=0
        )
        gt_risk_map_with_mask = torch.cat(
            (gt_risk_score_map.unsqueeze(0), mask_scale.unsqueeze(0).float()), dim=0
        )

        proj_pix_prev_tensor = torch.from_numpy(proj_pix_prev.astype(np.int64))
        proj_pix_curr_tensor = torch.from_numpy(proj_pix_curr.astype(np.int64))

        sensor_metas = {"prev": sensor_metas_prev, "curr": sensor_metas_curr}
        for frame_key in ["prev", "curr"]:
            for cam_name in self.camera_channels:
                if cam_name not in sensor_metas[frame_key]:
                    continue
                for k, v in sensor_metas[frame_key][cam_name].items():
                    sensor_metas[frame_key][cam_name][k] = torch.from_numpy(
                        np.asarray(v)
                    ).float()

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

    def __rmul__(self, v):
        self.data = v * self.data
        self.image_list = v * self.image_list
        self.scale_map_list = v * self.scale_map_list
        self.risk_score_map_list = v * self.risk_score_map_list
        self.depth_map_list = v * self.depth_map_list
        return self


def fetch_waymo_dataloader(args):
    """Create the Waymo range-image dataloader."""
    aug_params = {
        "crop_size": args.image_size,
        "do_flip": False,
        "rotate": False,
        "rotate_prob": 0.1,
        "rotate_angle": 90,
    }

    dataset = Waymo_range_image(
        aug_params=aug_params,
        train_info_file=getattr(
            args,
            "train_info_file",
            "waymo_train_infos_scene_flow_key_frames_160_1600_fov_5_25.pkl",
        ),
        train_info_path=getattr(
            args,
            "train_info_path",
            "./Datasets/waymo/2_trainval_test_infos/train",
        ),
        require_complete_depth=getattr(args, "require_complete_depth", False),
        max_samples=getattr(args, "max_train_samples", None),
        split="training",
    )
    return dataset
