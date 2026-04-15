import json
import threading
import time
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import os
import pickle
from glob import glob
import os.path as osp
from pathlib import Path
from typing import Union
from tqdm import tqdm

from .utils.augmentor import NuscRangeImageAugmentor
from dataloader.utils.geometry import get_geometry, range_projection_with_mapping
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from fpttc.scale_net.utils.spherical import build_lidar_to_camera_projection
from utils.nusc_paths import infer_nusc_dataset_root, resolve_nusc_depth_pred_path, resolve_nusc_path


CAMERA_CHANNELS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
]
DEFAULT_NUSC_PROJ_CACHE_NAME = 'nusc_150_keyframes_160x320_fov8_15_hardproj_v1'


PathLike = Union[str, Path]


def _default_nusc_proj_cache_root(dataset_root: PathLike) -> Path:
    return Path(dataset_root) / '5_proj_cache' / DEFAULT_NUSC_PROJ_CACHE_NAME


def _infer_nusc_split_name(train_info_path: PathLike, train_info_file: str) -> str:
    path_name = Path(train_info_path).name.lower()
    if path_name in {'train', 'val', 'test'}:
        return path_name

    file_name = train_info_file.lower()
    if 'nusc_train_' in file_name:
        return 'train'
    if 'nusc_val_' in file_name:
        return 'val'
    if 'nusc_test_' in file_name:
        return 'test'

    raise ValueError(
        f'Unable to infer split from train_info_path={train_info_path} '
        f'and train_info_file={train_info_file}'
    )


def _load_proj_cache_manifest(cache_root: Path, split_name: str):
    manifest_pkl = cache_root / split_name / 'manifest.pkl'
    manifest_json = cache_root / split_name / 'manifest.json'

    if manifest_pkl.exists():
        with open(manifest_pkl, 'rb') as f:
            manifest = pickle.load(f)
    elif manifest_json.exists():
        manifest = json.loads(manifest_json.read_text())
    else:
        return None

    entries = manifest['entries'] if isinstance(manifest, dict) and 'entries' in manifest else manifest
    if not isinstance(entries, list):
        raise ValueError(f'Unsupported manifest format: {manifest_pkl if manifest_pkl.exists() else manifest_json}')

    return {int(entry['original_index']): entry for entry in entries}


def _validate_proj_cache_meta(cache_root: Path, crop_size):
    meta_path = cache_root / 'cache_meta.json'
    if not meta_path.exists():
        return

    meta = json.loads(meta_path.read_text())
    expected_crop = [int(crop_size[0]), int(crop_size[1])]
    if meta.get('image_size') != expected_crop:
        raise ValueError(
            f'Projection cache image_size mismatch: cache={meta.get("image_size")} '
            f'expected={expected_crop}'
        )

    if meta.get('camera_order') != CAMERA_CHANNELS:
        raise ValueError('Projection cache camera_order mismatch.')

    if meta.get('H_r') != 40 or meta.get('W_r') != 480:
        raise ValueError('Projection cache range-view resolution mismatch.')

    if meta.get('fov_up') != 8.0 or meta.get('fov_down') != -15.0:
        raise ValueError('Projection cache FOV mismatch.')

    if meta.get('augmentor', {}).get('do_flip') is not False or meta.get('augmentor', {}).get('rotate') is not False:
        raise ValueError('Projection cache was generated with incompatible augmentation settings.')


def build_frame_sensor_metas(data, dataset_key, frame_key, affine_matrix, idx=None):
    """Build per-camera geometry metadata for one frame without changing proj mapping APIs."""
    sample_info = data if idx is None else data[idx]
    sensor_metas_channel = {}

    for channel in CAMERA_CHANNELS:
        if dataset_key == 'sjtu':
            channel_meta = sample_info[f'sensor_metas_{frame_key}'][channel]
            K_key = 'K_undist' if 'K_undist' in channel_meta else 'K'
            sensor_metas_channel[channel] = {
                'K': np.asarray(channel_meta[K_key], dtype=np.float32),
                'R_l2c': np.asarray(channel_meta['R_l2c'], dtype=np.float32),
                't_l2c': np.asarray(channel_meta['t_l2c'], dtype=np.float32).reshape(3),
                'affine': np.asarray(affine_matrix, dtype=np.float32),
            }
            continue

        if dataset_key == 'nusc':
            sensor_metas = sample_info[f'sensor_metas_{frame_key}']
            _, K, R_l2c, t_l2c = build_lidar_to_camera_projection(
                sensor_metas,
                sensor_metas['camera']['calibrated_sensor'][channel],
                sensor_metas['camera']['ego_pose'][channel]
            )
            sensor_metas_channel[channel] = {
                'K': np.asarray(K, dtype=np.float32),
                'R_l2c': np.asarray(R_l2c, dtype=np.float32),
                't_l2c': np.asarray(t_l2c, dtype=np.float32).reshape(3),
                'affine': np.asarray(affine_matrix, dtype=np.float32),
            }
            continue

        raise ValueError(f"Unsupported dataset key: {dataset_key}")

    return sensor_metas_channel


def tensorize_sensor_metas(sensor_metas):
    tensorized = {}
    for frame_key, frame_sensor_metas in sensor_metas.items():
        tensorized[frame_key] = {}
        for channel, channel_metas in frame_sensor_metas.items():
            tensorized[frame_key][channel] = {
                key: torch.from_numpy(np.asarray(value)).float()
                for key, value in channel_metas.items()
            }
    return tensorized

class nuScenes_range_image(data.Dataset):
    def __init__(self,
                 aug_params=None,
                 split='training',
                 train_info_path='./Datasets/nuscenes/2_trainval_test_infos/train',
                 train_info_file='nusc_train_infos_key_frames_160_1920_fov_8_15.pkl',
                 require_complete_depth=False,
                 max_samples=None,
                 proj_cache_root=None,
                 ):
        self.aug_params = aug_params
        self.split = split
        self.train_info_path = train_info_path
        self.train_info_file = train_info_file
        self.require_complete_depth = require_complete_depth
        self.max_samples = max_samples
        self.dataset_root = infer_nusc_dataset_root(self.train_info_path)
        self.proj_cache_root = Path(proj_cache_root) if proj_cache_root is not None else _default_nusc_proj_cache_root(self.dataset_root)
        self.split_name = _infer_nusc_split_name(self.train_info_path, self.train_info_file)
        self.camera_channels = CAMERA_CHANNELS

        pkl_file_path = osp.join(self.train_info_path, self.train_info_file)
        if osp.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as f:
                loaded_data = pickle.load(f)
            print(f"Loaded data from {pkl_file_path}")
        else:
            raise FileNotFoundError(f"No such file: {pkl_file_path}")

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
                "[nuScenes_range_image] sample stats: "
                f"original={original_samples}, "
                f"complete_depth={complete_depth_samples}, "
                f"final={final_samples}"
            )

        if final_samples == 0:
            raise ValueError(
                "No samples available for nuScenes_range_image after filtering. "
                "Check the current pkl and 4_depth_map coverage."
            )

        self.augmentor = None
        if self.aug_params is not None:
            self.augmentor = NuscRangeImageAugmentor(**self.aug_params)

        orig_size = (1600, 900)  # (W, H)
        self.affine_params = self.augmentor.sample_params(orig_size)
        self.affine_matrix = self.augmentor.get_affine_matrix(self.affine_params)

        _validate_proj_cache_meta(self.proj_cache_root, self.aug_params['crop_size'])
        manifest_by_original_index = _load_proj_cache_manifest(self.proj_cache_root, self.split_name)
        self.cache_entries = None
        if manifest_by_original_index is not None:
            self.cache_entries = []
            for original_index, _ in self.samples:
                if original_index not in manifest_by_original_index:
                    raise KeyError(
                        f'Missing cache entry for original_index={original_index} '
                        f'in {self.proj_cache_root / self.split_name}'
                    )
                entry = dict(manifest_by_original_index[original_index])
                cache_path = self.proj_cache_root / entry['cache_relpath']
                if not cache_path.exists():
                    raise FileNotFoundError(f'Projection cache file {cache_path} does not exist.')
                entry['cache_path'] = cache_path
                self.cache_entries.append(entry)
            print(f"Loaded projection cache from {self.proj_cache_root / self.split_name}")
        else:
            print(
                f"[nuScenes_range_image] projection cache not found under "
                f"{self.proj_cache_root / self.split_name}; "
                "falling back to on-the-fly proj mapping."
            )

    def __len__(self):
        return len(self.samples)

    def _has_complete_depth(self, info):
        for frame_key in ['prev_camera_data', 'curr_camera_data']:
            for channel in self.camera_channels:
                depth_path = resolve_nusc_depth_pred_path(
                    info[frame_key][channel],
                    self.dataset_root,
                )
                if depth_path is None or not depth_path.exists():
                    return False
        return True

    def __getitem__(self, index):
        _, info = self.samples[index]
        prev_surr_view_imgs = {}
        curr_surr_view_imgs = {}
        prev_surr_view_depths = {}
        curr_surr_view_depths = {}

        camera_channels = self.camera_channels
        for channel in camera_channels:
            prev_surr_view_imgs_path = resolve_nusc_path(
                info['prev_camera_data'][channel]['filename'],
                self.dataset_root,
            )
            prev_surr_view_imgs[channel] = Image.open(prev_surr_view_imgs_path)

            curr_surr_view_imgs_path = resolve_nusc_path(
                info['curr_camera_data'][channel]['filename'],
                self.dataset_root,
            )
            curr_surr_view_imgs[channel] = Image.open(curr_surr_view_imgs_path)

            prev_surr_view_depths_path = resolve_nusc_depth_pred_path(
                info['prev_camera_data'][channel],
                self.dataset_root,
            )
            prev_surr_view_depths[channel] = np.load(prev_surr_view_depths_path)

            curr_surr_view_depths_path = resolve_nusc_depth_pred_path(
                info['curr_camera_data'][channel],
                self.dataset_root,
            )
            curr_surr_view_depths[channel] = np.load(curr_surr_view_depths_path)

        range_image_dir = resolve_nusc_path(info['gt_map_path'], self.dataset_root)
        range_image_path = Path(range_image_dir) / 'range_image_curr.npy'
        if not range_image_path.exists():
            raise FileNotFoundError(f"Range image file {range_image_path} does not exist.")
        range_image = np.load(range_image_path, allow_pickle=True).item()
        gt_scale_map = range_image['scale']
        gt_risk_score_map = range_image['risk_score']

        orig_size = next(iter(prev_surr_view_imgs.values())).size  # (W, H)
        affine_params = self.augmentor.sample_params(orig_size)
        prev_surr_view_imgs, _ = self.augmentor(prev_surr_view_imgs, affine_params)
        curr_surr_view_imgs, _ = self.augmentor(curr_surr_view_imgs, affine_params)
        affine_matrix = self.augmentor.get_affine_matrix(affine_params)

        if self.cache_entries is not None:
            with np.load(self.cache_entries[index]['cache_path']) as cache_npz:
                proj_pix_prev = cache_npz['proj_pix_prev']
                proj_pix_curr = cache_npz['proj_pix_curr']
        else:
            _, proj_pix_prev = build_frame_mapping(
                info, 'nusc', 'prev', None, affine_matrix,
                idx=None, dataset_root=self.dataset_root, H_r=40, W_r=480
            )
            _, proj_pix_curr = build_frame_mapping(
                info, 'nusc', 'curr', None, affine_matrix,
                idx=None, dataset_root=self.dataset_root, H_r=40, W_r=480
            )

        sensor_metas = {
            'prev': build_frame_sensor_metas(info, 'nusc', 'prev', affine_matrix),
            'curr': build_frame_sensor_metas(info, 'nusc', 'curr', affine_matrix),
        }
        sensor_metas = tensorize_sensor_metas(sensor_metas)

        for channel in camera_channels:
            prev_surr_view_imgs[channel] = torch.from_numpy(prev_surr_view_imgs[channel]).permute(2, 0, 1).float()
            curr_surr_view_imgs[channel] = torch.from_numpy(curr_surr_view_imgs[channel]).permute(2, 0, 1).float()
            prev_surr_view_depths[channel] = torch.from_numpy(prev_surr_view_depths[channel]).float()
            curr_surr_view_depths[channel] = torch.from_numpy(curr_surr_view_depths[channel]).float()
        
        prev_surr_view_imgs_tensor = torch.stack([prev_surr_view_imgs[channel] for channel in camera_channels], dim=0)
        curr_surr_view_imgs_tensor = torch.stack([curr_surr_view_imgs[channel] for channel in camera_channels], dim=0)

        prev_surr_view_depths_tensor = torch.stack([prev_surr_view_depths[channel] for channel in camera_channels], dim=0)
        curr_surr_view_depths_tensor = torch.stack([curr_surr_view_depths[channel] for channel in camera_channels], dim=0)
        prev_surr_view_depths_tensor = prev_surr_view_depths_tensor.unsqueeze(1)
        curr_surr_view_depths_tensor = curr_surr_view_depths_tensor.unsqueeze(1)

        gt_scale_map = torch.from_numpy(gt_scale_map).float()
        gt_risk_score_map = torch.from_numpy(gt_risk_score_map).float()
        mask_scale = (gt_scale_map > 0.3) & (gt_scale_map < 3.0)
        gt_scale_map_with_mask = torch.cat((gt_scale_map.unsqueeze(0), mask_scale.unsqueeze(0).float()), dim=0)
        gt_risk_map_with_mask  = torch.cat((gt_risk_score_map.unsqueeze(0), mask_scale.unsqueeze(0).float()), dim=0)

        proj_pix_prev_tensor = torch.from_numpy(proj_pix_prev.astype(np.int64))   # (M, 3)
        proj_pix_curr_tensor = torch.from_numpy(proj_pix_curr.astype(np.int64))   # (M, 3)

        return (prev_surr_view_imgs_tensor,
                curr_surr_view_imgs_tensor,
                prev_surr_view_depths_tensor,
                curr_surr_view_depths_tensor,
                proj_pix_prev_tensor,
                proj_pix_curr_tensor,
                gt_scale_map_with_mask,
                gt_risk_map_with_mask,
                sensor_metas)

    def __rmul__(self, v):
        self.samples = v * self.samples
        if self.cache_entries is not None:
            self.cache_entries = v * self.cache_entries
        return self


def build_frame_mapping(data, dataset_key, frame_key, depth_map,
                        affine_matrix=None, idx=None, dataset_root='./Datasets/nuscenes',
                        H_r=40, W_r=480, visualize=False):
    """
    读取该帧的相机深度预测结果，以及内外参信息，将其反投影到 LiDAR 坐标系
    并进行 range projection，得到 range image 的投影坐标
    以及 (cam_idx, u, v) 到 range image (u, v) 的映射关系
    该映射关系用于后续的多视角特征融合
    """
    sample_info = data if idx is None else data[idx]
    if isinstance(depth_map, np.ndarray) and depth_map.shape == (3, 3) and isinstance(affine_matrix, (int, np.integer)):
        idx = int(affine_matrix)
        affine_matrix = depth_map
        depth_map = None
        sample_info = data[idx]

    camera_channels = CAMERA_CHANNELS
    all_points = []
    all_pix    = []

    for cam_idx, channel in enumerate(camera_channels):
        
        # 2) 将 nusc 提供的四元数内参转换为矩阵 K
        #    将 LiDAR --> Ego_LiDAR_Frame --> Global --> Ego_Camera_Frame --> Camera 的外参投影矩阵合并
        #    得到 LiDAR --> Camera 的旋转、平移矩阵
        if dataset_key == 'sjtu':
            K = sample_info[f'sensor_metas_{frame_key}'][channel]['K_undist']
            R_l2c = sample_info[f'sensor_metas_{frame_key}'][channel]['R_l2c']
            t_l2c = sample_info[f'sensor_metas_{frame_key}'][channel]['t_l2c']
            sensor_meta = {'K': K, 'R_l2c': R_l2c, 't_l2c': t_l2c}

            # 3) 根据深度图和相机内外参，将像素坐标转换为 LiDAR 坐标系下的 XYZ 坐标
            depth_pred_map = depth_map[channel]
            coords = get_geometry(depth_pred_map, sensor_meta, affine_matrix)  # (H_img, W_img, 3)
            coords_n = coords.copy() 
            # 翻转 x、y
            coords_n[:, :, 0] *= -1            # x_n = -x_local
            coords_n[:, :, 1] *= -1            # y_n = -y_local

        elif dataset_key == 'nusc':
            proj_matrix, K, R_l2c, t_l2c = build_lidar_to_camera_projection(
                sample_info[f'sensor_metas_{frame_key}'],
                sample_info[f'sensor_metas_{frame_key}']['camera']['calibrated_sensor'][channel],
                sample_info[f'sensor_metas_{frame_key}']['camera']['ego_pose'][channel]
            )
            sensor_meta = {'K': K, 'R_l2c': R_l2c, 't_l2c': t_l2c}

            if depth_map is None:
                depth_pred_path = resolve_nusc_depth_pred_path(
                    sample_info[f'{frame_key}_camera_data'][channel],
                    dataset_root,
                )
                depth_pred_map = np.load(depth_pred_path)
            else:
                depth_pred_map = depth_map[channel]

            # 3) 根据深度图和相机内外参，将像素坐标转换为 LiDAR 坐标系下的 XYZ 坐标
            coords = get_geometry(depth_pred_map, sensor_meta, affine_matrix)  # (H_img, W_img, 3)
            coords_n = coords.copy()   

        else:
            raise ValueError(f"Unsupported dataset key: {dataset_key}")

        H_img, W_img, _ = coords_n.shape
        pts = coords_n.reshape(-1, 3)

        # 4) 构造 (cam_idx, u, v)
        u_grid, v_grid = np.meshgrid(np.arange(W_img), np.arange(H_img))
        cam_idx_arr    = np.full((H_img, W_img), cam_idx, dtype=np.int32)
        pix            = np.stack([cam_idx_arr, u_grid, v_grid], axis=-1).reshape(-1, 3)

        # 5) 过滤无效点
        valid = np.linalg.norm(pts, axis=1) > 0
        pts   = pts[valid]
        pix   = pix[valid]

        all_points.append(pts)
        all_pix.append(pix)

    # 6) 合并所有相机
    points = np.vstack(all_points)  # (M, 3)
    pix    = np.vstack(all_pix)     # (M, 3)

    # 7) range 投影并保留 (idx, u_rgb, v_rgb) <-> (u_range, v_range) 映射关系
    proj_range, proj_xyz, proj_idx, proj_mask, proj_pix = \
        range_projection_with_mapping(points, pix, H=H_r, W=W_r,
                                    fov_up=8.0, fov_down=-15.0)
    
    # 8) 如果有空洞，使用最近邻填补 proj_pix 和 proj_range
    valid = proj_mask.astype(bool)
    if not valid.all():
        # distance_transform_edt on the *holes* mask, get indices of nearest valid
        # inds shape = (2, H_r, W_r): inds[0] = row indices, inds[1] = col indices
        _, inds = distance_transform_edt(~valid, return_distances=True, return_indices=True)
        i_near, j_near = inds  # each is shape (H_r, W_r)

        # fill proj_pix: for each hole (h,w) copy from (i_near[h,w], j_near[h,w])
        proj_pix = proj_pix[i_near, j_near]

        # 同理，将 proj_range 也补全：
        proj_range = proj_range[i_near, j_near]
        proj_mask[:] = 1
    
  
    # 9) 可视化
    if visualize and frame_key == 'prev':

         # —— 归一化 proj_range 到 [0,1]
        valid = proj_mask.astype(bool)
        if valid.any():
            r_min = proj_range[valid].min()
            r_max = proj_range[valid].max()
            proj_range_norm = (proj_range - r_min) / (r_max - r_min + 1e-6)
        else:
            proj_range_norm = np.zeros_like(proj_range)

        # only save the normalized prev‐frame range image
        plt.figure(figsize=(5,4))
        plt.title("Prev frame - Normalized Range")
        plt.imshow(proj_range_norm, cmap='jet', vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout()
        vis_name = idx if idx is not None else 'single'
        plt.savefig(f"./Datasets/cyberrock/scene_7/depth_vis/{frame_key}_normalized_range_{vis_name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return proj_range, proj_pix

def _precompute_pix_orig(H_img: int, W_img: int, affine_matrix: np.ndarray) -> np.ndarray:
    """
    计算处理后像素坐标经 A^{-1} 映射回“原图”像素坐标（齐次），并做归一化。
    返回 shape = (3, N)，N = H_img * W_img
    """
    us = np.arange(W_img); vs = np.arange(H_img)
    u_grid, v_grid = np.meshgrid(us, vs)                    # (H, W)
    ones = np.ones_like(u_grid, dtype=np.float32)
    pix_proc = np.stack([u_grid, v_grid, ones], axis=-1).reshape(-1, 3).T  # (3, N)
    invA = np.linalg.inv(affine_matrix).astype(np.float32)
    pix_orig = invA @ pix_proc                               # (3, N)
    pix_orig /= pix_orig[2:3, :]                             # 归一化
    return pix_orig.astype(np.float32)


_PIXEL_GRID_CACHE = {}
_PIXEL_GRID_CACHE_LOCK = threading.Lock()
_K_INV_CACHE = {}
_K_INV_CACHE_LOCK = threading.Lock()
_CAMERA_RAY_CACHE = {}
_CAMERA_RAY_CACHE_LOCK = threading.Lock()


def _affine_cache_key(H_img: int, W_img: int, pixel_stride: int, affine_matrix: np.ndarray):
    affine = np.asarray(affine_matrix, dtype=np.float32)
    return (int(H_img), int(W_img), int(pixel_stride), affine.shape, affine.tobytes())


def _get_cached_inverse_affine_grid(H_img: int, W_img: int, pixel_stride: int, affine_matrix: np.ndarray):
    cache_key = _affine_cache_key(H_img, W_img, pixel_stride, affine_matrix)
    cached = _PIXEL_GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    pix_orig_full = _precompute_pix_orig(H_img, W_img, affine_matrix)
    if pixel_stride > 1:
        us = np.arange(0, W_img, pixel_stride)
        vs = np.arange(0, H_img, pixel_stride)
        u_grid_s, v_grid_s = np.meshgrid(us, vs)
        pick_lin = (v_grid_s * W_img + u_grid_s).reshape(-1)
        cached_value = (
            pix_orig_full[:, pick_lin],
            u_grid_s.reshape(-1),
            v_grid_s.reshape(-1),
            pick_lin,
        )
    else:
        uu_full, vv_full = np.meshgrid(np.arange(W_img), np.arange(H_img))
        cached_value = (
            pix_orig_full,
            uu_full.reshape(-1),
            vv_full.reshape(-1),
            None,
        )

    with _PIXEL_GRID_CACHE_LOCK:
        existing = _PIXEL_GRID_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _PIXEL_GRID_CACHE[cache_key] = cached_value
    return cached_value


def _get_cached_k_inv(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32)
    cache_key = (K.shape, K.tobytes())
    cached = _K_INV_CACHE.get(cache_key)
    if cached is not None:
        return cached

    k_inv = np.linalg.inv(K).astype(np.float32)
    with _K_INV_CACHE_LOCK:
        existing = _K_INV_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _K_INV_CACHE[cache_key] = k_inv
    return k_inv


def _camera_signature_bytes(sample_info, dataset_key: str, frame_key: str, camera_channel: str) -> bytes:
    if dataset_key == 'sjtu':
        sensor_meta = sample_info[f'sensor_metas_{frame_key}'][camera_channel]
        parts = [
            np.asarray(sensor_meta['K'], dtype=np.float32).tobytes(),
            np.asarray(sensor_meta['R_l2c'], dtype=np.float32).tobytes(),
            np.asarray(sensor_meta['t_l2c'], dtype=np.float32).reshape(-1).tobytes(),
        ]
    else:
        sensor_metas = sample_info[f'sensor_metas_{frame_key}']
        cam_cs = sensor_metas['camera']['calibrated_sensor'][camera_channel]
        cam_pose = sensor_metas['camera']['ego_pose'][camera_channel]
        lidar_cs = sensor_metas['lidar']['calibrated_sensor']
        lidar_pose = sensor_metas['lidar']['ego_pose']
        parts = [
            np.asarray(cam_cs['camera_intrinsic'], dtype=np.float32).tobytes(),
            np.asarray(cam_cs['translation'], dtype=np.float32).tobytes(),
            np.asarray(cam_cs['rotation'], dtype=np.float32).tobytes(),
            np.asarray(cam_pose['translation'], dtype=np.float32).tobytes(),
            np.asarray(cam_pose['rotation'], dtype=np.float32).tobytes(),
            np.asarray(lidar_cs['translation'], dtype=np.float32).tobytes(),
            np.asarray(lidar_cs['rotation'], dtype=np.float32).tobytes(),
            np.asarray(lidar_pose['translation'], dtype=np.float32).tobytes(),
            np.asarray(lidar_pose['rotation'], dtype=np.float32).tobytes(),
        ]
    return b''.join(parts)


def _get_cached_camera_ray_base(
    sample_info,
    dataset_key: str,
    frame_key: str,
    camera_channel: str,
    pix_orig: np.ndarray,
    H_img: int,
    W_img: int,
    pixel_stride: int,
    affine_matrix: np.ndarray,
):
    affine_key = _affine_cache_key(H_img, W_img, pixel_stride, affine_matrix)
    camera_signature = _camera_signature_bytes(sample_info, dataset_key, frame_key, camera_channel)
    cache_key = (dataset_key, camera_channel, affine_key, camera_signature)

    cached = _CAMERA_RAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if dataset_key == 'sjtu':
        sensor_meta = sample_info[f'sensor_metas_{frame_key}'][camera_channel]
        K = np.asarray(sensor_meta['K'], dtype=np.float32)
        R_l2c = np.asarray(sensor_meta['R_l2c'], dtype=np.float32)
        t_l2c = np.asarray(sensor_meta['t_l2c'], dtype=np.float32).reshape(3, 1)
        K_inv = _get_cached_k_inv(K)
        R_c2l = R_l2c.T.astype(np.float32)
        t_c2l = (-R_c2l @ t_l2c).astype(np.float32)
        ray_lidar_base = (R_c2l @ (K_inv @ pix_orig)).astype(np.float32)
        ray_lidar_base[:2, :] *= -1.0
        t_c2l[:2, :] *= -1.0
    else:
        sensor_metas = sample_info[f'sensor_metas_{frame_key}']
        _, K, R_l2c, t_l2c = build_lidar_to_camera_projection(
            sensor_metas,
            sensor_metas['camera']['calibrated_sensor'][camera_channel],
            sensor_metas['camera']['ego_pose'][camera_channel]
        )
        K = np.asarray(K, dtype=np.float32)
        R_l2c = np.asarray(R_l2c, dtype=np.float32)
        t_l2c = np.asarray(t_l2c, dtype=np.float32).reshape(3, 1)
        K_inv = _get_cached_k_inv(K)
        R_c2l = R_l2c.T.astype(np.float32)
        t_c2l = (-R_c2l @ t_l2c).astype(np.float32)
        ray_lidar_base = (R_c2l @ (K_inv @ pix_orig)).astype(np.float32)

    cached_value = (ray_lidar_base, t_c2l.astype(np.float32))
    with _CAMERA_RAY_CACHE_LOCK:
        existing = _CAMERA_RAY_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _CAMERA_RAY_CACHE[cache_key] = cached_value
    return cached_value


def _geometry_cam2lidar_from_depth(
    depth_map: np.ndarray,
    K_inv: np.ndarray,
    R_c2l: np.ndarray,
    t_c2l: np.ndarray,
    pix_orig: np.ndarray,          # (3, N') 对应下采样后的像素
    *,
    flip_xy: bool = False,
    ds: np.ndarray = None,         # (N',) 与 pix_orig 列数严格一致
    pick_lin: np.ndarray = None    # 若 ds=None，可用 pick_lin 从 depth_map 内部索引
) -> np.ndarray:
    """
    返回 (N', 3) LiDAR 坐标。确保 pix_orig 与 ds 对齐！
    """
    if ds is None:
        flat = depth_map.reshape(-1).astype(np.float32)
        ds = flat[pick_lin] if pick_lin is not None else flat
    else:
        ds = ds.astype(np.float32)

    Xc = (K_inv @ pix_orig).astype(np.float32)               # (3, N')
    Xc *= ds[np.newaxis, :]                                  # (3, N')
    Xl = (R_c2l @ Xc) + t_c2l.reshape(3, 1).astype(np.float32)
    pts = Xl.T                                               # (N', 3)

    if flip_xy:
        pts[:, 0] *= -1.0
        pts[:, 1] *= -1.0
    return pts

def _range_projection_with_mapping_np_fast(
    points: np.ndarray,
    pix_coords: np.ndarray,
    H: int = 160, W: int = 1920,
    fov_up: float = 8.0, fov_down: float = -15.0
):
    """
    矢量化 z-buffer：对每个 (py,px) 仅保留最近点（最小 depth）。
    返回：
      proj_range: (H,W) float32
      proj_xyz:   (H,W,3) float32
      proj_idx:   (H,W)   int32
      proj_mask:  (H,W)   int32
      proj_pix:   (H,W,3) int32
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert pix_coords.ndim == 2 and pix_coords.shape[1] == 3

    depth = np.linalg.norm(points, axis=1).astype(np.float32)  # (M,)
    safe = np.maximum(depth, 1e-9)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    yaw   = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / safe, -1.0, 1.0))

    fov_up_rad   = np.deg2rad(fov_up)
    fov_down_rad = np.deg2rad(fov_down)
    fov = abs(fov_down_rad) + abs(fov_up_rad)

    proj_x = np.floor(0.5 * (yaw / np.pi + 1.0) * W).astype(np.int32)
    proj_y = np.floor((1.0 - (pitch + abs(fov_down_rad)) / fov) * H).astype(np.int32)
    np.clip(proj_x, 0, W - 1, out=proj_x)
    np.clip(proj_y, 0, H - 1, out=proj_y)

    lin = (proj_y.astype(np.int64) * W + proj_x.astype(np.int64))  # (M,)

    # 按 (lin, depth) 升序排序，每组第一个即最近点
    order = np.lexsort((depth, lin))
    lin_s = lin[order]

    first = np.empty_like(lin_s, dtype=bool)
    if first.size:
        first[0] = True
        first[1:] = lin_s[1:] != lin_s[:-1]

    sel = order[first]                            # 源点索引
    lin_unique = lin[sel]
    y_unique = (lin_unique // W).astype(np.int32)
    x_unique = (lin_unique %  W).astype(np.int32)

    proj_range = np.full((H, W), -1, np.float32)
    proj_xyz   = np.full((H, W, 3), -1, np.float32)
    proj_idx   = np.full((H, W), -1, np.int32)
    proj_mask  = np.zeros((H, W), np.int32)
    proj_pix   = np.full((H, W, 3), -1, np.int32)

    proj_range[y_unique, x_unique] = depth[sel]
    proj_xyz[y_unique, x_unique]   = points[sel].astype(np.float32)
    proj_idx[y_unique, x_unique]   = sel.astype(np.int32)
    proj_mask[y_unique, x_unique]  = 1
    proj_pix[y_unique, x_unique]   = pix_coords[sel].astype(np.int32)

    return proj_range, proj_xyz, proj_idx, proj_mask, proj_pix

def build_camera_mapping_fast(
    sample_info,
    dataset_key: str,
    frame_key: str,
    depth_map: np.ndarray,
    affine_matrix: np.ndarray,
    camera_channel: str,
    camera_index: int,
    pixel_stride: int = 1,
    return_timing: bool = False,
):
    t0 = time.perf_counter()

    depth_map = np.asarray(depth_map, dtype=np.float32)
    H_img, W_img = depth_map.shape
    pix_orig, uu_base, vv_base, pick_lin = _get_cached_inverse_affine_grid(
        H_img, W_img, pixel_stride, affine_matrix
    )
    ray_lidar_base, t_c2l = _get_cached_camera_ray_base(
        sample_info,
        dataset_key,
        frame_key,
        camera_channel,
        pix_orig,
        H_img,
        W_img,
        pixel_stride,
        affine_matrix,
    )
    t1 = time.perf_counter()

    ds = depth_map.reshape(-1)[pick_lin] if pick_lin is not None else depth_map.reshape(-1)
    valid = ds > 0
    if not np.any(valid):
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_pix = np.empty((0, 3), dtype=np.int32)
        if return_timing:
            return empty_points, empty_pix, {
                'transform_setup_ms': (t1 - t0) * 1000.0,
                'camera_backproject_ms': 0.0,
            }
        return empty_points, empty_pix

    ds_valid = ds[valid].astype(np.float32, copy=False)
    points = (ray_lidar_base[:, valid] * ds_valid[np.newaxis, :]) + t_c2l
    uu = uu_base[valid].astype(np.int32, copy=False)
    vv = vv_base[valid].astype(np.int32, copy=False)
    cam_col = np.full_like(uu, camera_index, dtype=np.int32)
    pix = np.stack([cam_col, uu, vv], axis=1).astype(np.int32, copy=False)
    t2 = time.perf_counter()

    if return_timing:
        return points.T.astype(np.float32, copy=False), pix, {
            'transform_setup_ms': (t1 - t0) * 1000.0,
            'camera_backproject_ms': (t2 - t1) * 1000.0,
        }
    return points.T.astype(np.float32, copy=False), pix


def _resolve_mapping_range_backend(mapping_range_backend: str) -> str:
    if mapping_range_backend == 'auto':
        return 'gpu' if torch.cuda.is_available() else 'cpu'
    if mapping_range_backend == 'gpu' and not torch.cuda.is_available():
        return 'cpu'
    return mapping_range_backend


def _range_projection_with_mapping_torch_fast(
    points: np.ndarray,
    pix_coords: np.ndarray,
    H: int = 160,
    W: int = 1920,
    fov_up: float = 8.0,
    fov_down: float = -15.0,
):
    device = torch.device('cuda')
    points_t = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device, non_blocking=False)
    pix_t = torch.from_numpy(np.asarray(pix_coords, dtype=np.int32)).to(device, non_blocking=False)

    depth = torch.linalg.norm(points_t, dim=1)
    safe = torch.clamp_min(depth, 1e-9)
    yaw = -torch.atan2(points_t[:, 1], points_t[:, 0])
    pitch = torch.asin(torch.clamp(points_t[:, 2] / safe, -1.0, 1.0))

    fov_up_rad = float(np.deg2rad(fov_up))
    fov_down_rad = float(np.deg2rad(fov_down))
    fov = abs(fov_down_rad) + abs(fov_up_rad)

    proj_x = torch.floor(0.5 * (yaw / np.pi + 1.0) * W).to(torch.int64)
    proj_y = torch.floor((1.0 - (pitch + abs(fov_down_rad)) / fov) * H).to(torch.int64)
    proj_x.clamp_(0, W - 1)
    proj_y.clamp_(0, H - 1)
    lin = proj_y * W + proj_x

    flat_size = H * W
    min_depth = torch.full((flat_size,), float('inf'), device=device, dtype=torch.float32)
    min_depth.scatter_reduce_(0, lin, depth, reduce='amin', include_self=True)

    candidate = depth <= (min_depth[lin] + 1e-6)
    point_idx = torch.arange(depth.shape[0], device=device, dtype=torch.int64)
    invalid_idx = torch.full_like(point_idx, depth.shape[0])
    candidate_idx = torch.where(candidate, point_idx, invalid_idx)
    chosen_idx = torch.full((flat_size,), depth.shape[0], device=device, dtype=torch.int64)
    chosen_idx.scatter_reduce_(0, lin, candidate_idx, reduce='amin', include_self=True)

    valid_flat = chosen_idx < depth.shape[0]
    flat_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
    selected_idx = chosen_idx[flat_idx]

    proj_range = torch.full((flat_size,), -1.0, device=device, dtype=torch.float32)
    proj_mask = torch.zeros((flat_size,), device=device, dtype=torch.int32)
    proj_pix = torch.full((flat_size, 3), -1, device=device, dtype=torch.int32)

    proj_range[flat_idx] = depth[selected_idx]
    proj_mask[flat_idx] = 1
    proj_pix[flat_idx] = pix_t[selected_idx]

    torch.cuda.synchronize()

    return (
        proj_range.view(H, W).cpu().numpy(),
        proj_mask.view(H, W).cpu().numpy(),
        proj_pix.view(H, W, 3).cpu().numpy(),
    )


def finalize_frame_mapping_fast(
    camera_results,
    H_r: int = 40,
    W_r: int = 480,
    return_timing: bool = False,
    mapping_range_backend: str = 'auto',
):
    t0 = time.perf_counter()
    resolved_backend = _resolve_mapping_range_backend(mapping_range_backend)

    valid_results = [(points, pix) for points, pix in camera_results if points.size > 0]
    if not valid_results:
        proj_range = np.full((H_r, W_r), -1, np.float32)
        proj_pix = np.full((H_r, W_r, 3), -1, np.int32)
        if return_timing:
            return proj_range, proj_pix, {
                'range_project_ms': 0.0,
                'hole_fill_ms': 0.0,
                'range_backend': resolved_backend,
            }
        return proj_range, proj_pix

    points = np.concatenate([points for points, _ in valid_results], axis=0)
    pix = np.concatenate([pix for _, pix in valid_results], axis=0)

    if resolved_backend == 'gpu':
        proj_range, proj_mask, proj_pix = _range_projection_with_mapping_torch_fast(
            points, pix, H=H_r, W=W_r, fov_up=8.0, fov_down=-15.0
        )
    else:
        proj_range, proj_xyz, proj_idx, proj_mask, proj_pix = _range_projection_with_mapping_np_fast(
            points, pix, H=H_r, W=W_r, fov_up=8.0, fov_down=-15.0
        )
    t1 = time.perf_counter()

    valid_im = proj_mask.astype(bool)
    hole_fill_ms = 0.0
    if not valid_im.all():
        hole_mask = ~valid_im
        invalid_count = int(hole_mask.sum())
        t_fill_0 = time.perf_counter()
        _, inds = distance_transform_edt(hole_mask, return_distances=True, return_indices=True)
        i_near, j_near = inds
        if invalid_count > 0:
            hole_i, hole_j = np.nonzero(hole_mask)
            proj_pix[hole_i, hole_j] = proj_pix[i_near[hole_i, hole_j], j_near[hole_i, hole_j]]
            proj_range[hole_i, hole_j] = proj_range[i_near[hole_i, hole_j], j_near[hole_i, hole_j]]
            proj_mask[hole_i, hole_j] = 1
        t_fill_1 = time.perf_counter()
        hole_fill_ms = (t_fill_1 - t_fill_0) * 1000.0

    if return_timing:
        return proj_range, proj_pix, {
            'range_project_ms': (t1 - t0) * 1000.0,
            'hole_fill_ms': hole_fill_ms,
            'range_backend': resolved_backend,
        }
    return proj_range, proj_pix


def build_frame_mapping_fast(
    data,
    dataset_key: str,
    frame_key: str,
    depth_map_dict: dict,
    affine_matrix: np.ndarray,
    idx: int,
    H_r: int = 40, W_r: int = 480,
    visualize: bool = False,
    pixel_stride: int = 1,
    mapping_range_backend: str = 'auto',
    return_timing: bool = False,
):
    """
    读取该帧 6 个相机的深度与标定，反投影到 LiDAR，再做 range 投影，返回：
      proj_range: (H_r, W_r) float32
      proj_pix:   (H_r, W_r, 3) int32   # (cam_idx, u, v)
    - dataset_key ∈ {'sjtu', 'nusc'}
    - depth_map_dict[channel] -> (H_img, W_img) ndarray
    - pixel_stride >= 1（>1 会对像素网格均匀下采样）
    """
    assert dataset_key in ('sjtu', 'nusc')

    sample_info = data[idx]
    camera_results = []
    timing = {
        'transform_setup_ms': 0.0,
        'camera_backproject_ms': 0.0,
        'range_project_ms': 0.0,
        'hole_fill_ms': 0.0,
        'range_backend': _resolve_mapping_range_backend(mapping_range_backend),
    }

    transform_setup_samples = []
    camera_backproject_samples = []
    for cam_idx, ch in enumerate(CAMERA_CHANNELS):
        if return_timing:
            points, pix, camera_timing = build_camera_mapping_fast(
                sample_info,
                dataset_key,
                frame_key,
                depth_map_dict[ch],
                affine_matrix,
                ch,
                cam_idx,
                pixel_stride,
                True,
            )
            transform_setup_samples.append(camera_timing['transform_setup_ms'])
            camera_backproject_samples.append(camera_timing['camera_backproject_ms'])
        else:
            points, pix = build_camera_mapping_fast(
                sample_info,
                dataset_key,
                frame_key,
                depth_map_dict[ch],
                affine_matrix,
                ch,
                cam_idx,
                pixel_stride,
                False,
            )
        camera_results.append((points, pix))

    proj_range, proj_pix, finalize_timing = finalize_frame_mapping_fast(
        camera_results,
        H_r=H_r,
        W_r=W_r,
        return_timing=True,
        mapping_range_backend=mapping_range_backend,
    )

    if return_timing:
        timing['transform_setup_ms'] = max(transform_setup_samples) if transform_setup_samples else 0.0
        timing['camera_backproject_ms'] = max(camera_backproject_samples) if camera_backproject_samples else 0.0
        timing['range_project_ms'] = finalize_timing['range_project_ms']
        timing['hole_fill_ms'] = finalize_timing['hole_fill_ms']
        timing['range_backend'] = finalize_timing['range_backend']
        return proj_range, proj_pix, timing

    return proj_range, proj_pix

def fetch_dataloader(args, TRAIN_DS='C+T+K/S'):
    """ Create the data loader for the corresponding trainign set """
    train_dataset = None

    if args.stage == 'nuscenes_range_image':
        aug_params = {'crop_size': args.image_size, 'do_flip': False, 'rotate': False,
                      'rotate_prob': 0.1, 'rotate_angle': 90,
                      'color_aug': True}
        train_info_file = args.train_info_file
        train_info_path = args.train_info_path

        nuscenes = nuScenes_range_image(aug_params,
                                        train_info_file=train_info_file,
                                        train_info_path=train_info_path,
                                        require_complete_depth=args.require_complete_depth,
                                        proj_cache_root=args.proj_cache_root,
                                        max_samples=args.max_train_samples,
                                        split='training')

        train_dataset = 1*nuscenes
    
    else:
        raise ValueError(f"Unknown args.stage: {args.stage}")

    # print('Training with %d image pairs' % len(train_dataset.image_list))
    return train_dataset


def fetch_val_dataloader(args):
    """Create the validation dataset for the current training stage."""
    if args.stage != 'nuscenes_range_image':
        raise ValueError(f"Unknown args.stage: {args.stage}")

    aug_params = {'crop_size': args.image_size, 'do_flip': False, 'rotate': False, 'rotate_prob': 0.1, 'rotate_angle': 90}

    return nuScenes_range_image(
        aug_params,
        train_info_file=args.val_info_file,
        train_info_path=args.val_info_path,
        require_complete_depth=args.require_complete_depth,
        proj_cache_root=args.val_proj_cache_root,
        max_samples=None,
        split='validation',
    )
