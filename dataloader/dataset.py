import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import os
import pickle
from glob import glob
import os.path as osp
from tqdm import tqdm

from .utils.augmentor import NuscRangeImageAugmentor
from dataloader.utils.geometry import get_geometry, range_projection_with_mapping
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from fpttc.scale_net.utils.spherical import build_lidar_to_camera_projection
from utils.nusc_paths import resolve_nusc_path

class nuScenes_range_image(data.Dataset):
    def __init__(self,
                 aug_params=None,
                 split='training',
                 train_info_path='./Datasets/nuscenes/2_trainval_test_infos',
                 train_info_file='nusc_trainval_infos_160_1920.pkl'
                 ):
        self.aug_params = aug_params
        self.split = split
        # self.root = root
        self.train_info_path = train_info_path
        self.train_info_file = train_info_file
        self.dataset_root = osp.dirname(osp.normpath(self.train_info_path))
        self.data = None  # 用于存储从 pkl 文件中加载的数据
        # self.train_location = train_location

        # 根据 split 加载对应的 pkl 文件
        pkl_file_path = osp.join(self.train_info_path, self.train_info_file)

        # 检查文件是否存在
        if osp.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded data from {pkl_file_path}")
        else:
            raise FileNotFoundError(f"No such file: {pkl_file_path}")

        # 数据增强设置
        self.augmentor = None
        if self.aug_params is not None:
            self.augmentor = NuscRangeImageAugmentor(**self.aug_params)      
        # 获取数据增强的 affine 参数
        orig_size = (1600, 900)  # (W, H)
        self.affine_params = self.augmentor.sample_params(orig_size)
        self.affine_matrix = self.augmentor.get_affine_matrix(self.affine_params)
        
        self.camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

        # pkl 文件中包含：
        # - surround view images pairs (nusc sample data format)
        # - lidar data pairs (nusc sample data format)
        # - sensor metas (including calibrated infos and ego pose infos of LiDAR and cameras)
        # - gt range images path (including scale map, depth map and risk score map)
        # - nuscenes scene flow pointcloud path
        self.image_list = [] # input images
        # self.sensor_meta_list = []
        self.scale_map_list = [] # ground truth
        self.risk_score_map_list = []
        # [TODO] 目前 depth map 是将碰撞点反投影回图像平面时使用的，仅在可视化时使用
        self.depth_map_list = []

        self.proj_list = [] # 记录 环视图 (cam_idx, u, v) 和 range image (u, v) 的映射关系

        # 在加载数据集时离线构建 spherical voxel grid
        # 结合 DepthAnything 预测的 Depth Pred Map，提前计算每一个像素坐标对应的 Range View 坐标

        for i in tqdm(range(len(self.data)), desc='Loading nuScenes Range Image Dataset'):
            
            # 1. 模型 Input
            self.image_list.append([self.data[i]['prev_camera_data'],
                                    self.data[i]['curr_camera_data']])
            
            # 2. Ground Truth Range Image —— Scale Map, Risk Score Map, Depth Map
            range_image_dir = resolve_nusc_path(self.data[i]['gt_map_path'], self.dataset_root)
            range_image_path = os.path.join(str(range_image_dir), 'range_image_curr.npy')
            if not osp.exists(range_image_path):
                raise FileNotFoundError(f"Range image file {range_image_path} does not exist.")
            range_image = np.load(range_image_path, allow_pickle=True).item()
            self.scale_map_list.append(range_image['scale'])
            self.risk_score_map_list.append(range_image['risk_score'])
            self.depth_map_list.append(range_image['depth'])

            # 3. 环视图像 (cam_idx, u, v) 与 range image (u, v) 之间的映射关系
            #    通过 DepthAnything 预测的 Depth Pred Map + 内外参 计算得到
            proj_range_prev, proj_pix_prev = build_frame_mapping(self.data, 'nusc', 'prev', self.affine_matrix, i, H_r=40, W_r=480)
            proj_range_curr, proj_pix_curr = build_frame_mapping(self.data, 'nusc', 'curr', self.affine_matrix, i, H_r=40, W_r=480)
            self.proj_list.append([proj_pix_prev, proj_pix_curr])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        prev_surr_view_imgs = {}
        curr_surr_view_imgs = {}

        prev_surr_view_depths = {}
        curr_surr_view_depths = {}

        camera_channels = self.camera_channels    
        # 1. 按照相机通道读取相邻帧的图像和深度预测结果
        for channel in camera_channels:
            # 1）读取相邻帧的图像
            prev_surr_view_imgs_path = resolve_nusc_path(
                self.image_list[index][0][channel]['filename'],
                self.dataset_root,
            )
            prev_surr_view_imgs[channel] = Image.open(prev_surr_view_imgs_path)

            curr_surr_view_imgs_path = resolve_nusc_path(
                self.image_list[index][1][channel]['filename'],
                self.dataset_root,
            )
            curr_surr_view_imgs[channel] = Image.open(curr_surr_view_imgs_path)

            # 2) 读取相邻帧的 Depth Pred Map (DepthAnythingV2 Metric)
            prev_surr_view_depths_path = resolve_nusc_path(
                self.image_list[index][0][channel]['depth_pred'],
                self.dataset_root,
            )
            prev_surr_view_depths[channel] = np.load(prev_surr_view_depths_path)

            curr_surr_view_depths_path = resolve_nusc_path(
                self.image_list[index][1][channel]['depth_pred'],
                self.dataset_root,
            )
            curr_surr_view_depths[channel] = np.load(curr_surr_view_depths_path)

        # 2. 获取 ground truth 的 scale map、risk score map 和 depth map
        gt_scale_map      = self.scale_map_list[index]
        gt_risk_score_map = self.risk_score_map_list[index]
        gt_depth_map      = self.depth_map_list[index]

        # 3. 获取 (cam_idx, u, v) 到 range image (u, v) 的映射关系
        proj_pix_prev, proj_pix_curr = self.proj_list[index]

        # 4. 对 input 图像进行数据增强
        orig_size = next(iter(prev_surr_view_imgs.values())).size  # (W, H)
        affine_params = self.augmentor.sample_params(orig_size)
        prev_surr_view_imgs, _ = self.augmentor(prev_surr_view_imgs, affine_params)
        curr_surr_view_imgs, _ = self.augmentor(curr_surr_view_imgs, affine_params)
        affine_matrix = self.augmentor.get_affine_matrix(affine_params)
        
        # 5. 将上述收集的信息转换为 Tensor
        # 1）将 input 图像和 depth pred map 转换为 Tensor
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

        # 2）将 ground truth 的 scale map、risk score map 和 depth map 转换为 Tensor
        gt_scale_map = torch.from_numpy(gt_scale_map).float()
        gt_risk_score_map = torch.from_numpy(gt_risk_score_map).float()
        gt_depth_map = torch.from_numpy(gt_depth_map).float()
        mask_scale = (gt_scale_map > 0.3) & (gt_scale_map < 3.0)
        gt_scale_map_with_mask = torch.cat((gt_scale_map.unsqueeze(0), mask_scale.unsqueeze(0).float()), dim=0)
        gt_risk_map_with_mask  = torch.cat((gt_risk_score_map.unsqueeze(0), mask_scale.unsqueeze(0).float()), dim=0)

        # 3）将 (cam_idx, u, v) 到 range image (u, v) 的映射关系转换为 Tensor
        proj_pix_prev_tensor = torch.from_numpy(proj_pix_prev.astype(np.int64))   # (M, 3)
        proj_pix_curr_tensor = torch.from_numpy(proj_pix_curr.astype(np.int64))   # (M, 3)

        # 4) 将图像增强的仿射矩阵转换为 Tensor
        affine_matrix = torch.from_numpy(affine_matrix)

        return (prev_surr_view_imgs_tensor,
                curr_surr_view_imgs_tensor,
                prev_surr_view_depths_tensor,
                curr_surr_view_depths_tensor,
                proj_pix_prev_tensor,
                proj_pix_curr_tensor,
                gt_scale_map_with_mask,
                gt_risk_map_with_mask)

    def __rmul__(self, v):
        self.image_list          = v * self.image_list
        self.scale_map_list      = v * self.scale_map_list
        self.risk_score_map_list = v * self.risk_score_map_list
        self.depth_map_list      = v * self.depth_map_list
        self.proj_list           = v * self.proj_list
        return self


def build_frame_mapping(data, dataset_key, frame_key, depth_map,
                        affine_matrix, idx, H_r=40, W_r=480, visualize=False):
    """
    读取该帧的相机深度预测结果，以及内外参信息，将其反投影到 LiDAR 坐标系
    并进行 range projection，得到 range image 的投影坐标
    以及 (cam_idx, u, v) 到 range image (u, v) 的映射关系
    该映射关系用于后续的多视角特征融合
    """
    camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                       'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
    all_points = []
    all_pix    = []

    for cam_idx, channel in enumerate(camera_channels):
        
        # 2) 将 nusc 提供的四元数内参转换为矩阵 K
        #    将 LiDAR --> Ego_LiDAR_Frame --> Global --> Ego_Camera_Frame --> Camera 的外参投影矩阵合并
        #    得到 LiDAR --> Camera 的旋转、平移矩阵
        if dataset_key == 'sjtu':
            K = data[idx][f'sensor_metas_{frame_key}'][channel]['K_undist']
            R_l2c = data[idx][f'sensor_metas_{frame_key}'][channel]['R_l2c']
            t_l2c = data[idx][f'sensor_metas_{frame_key}'][channel]['t_l2c']
            sensor_meta = {'K': K, 'R_l2c': R_l2c, 't_l2c': t_l2c}

            # 3) 根据深度图和相机内外参，将像素坐标转换为 LiDAR 坐标系下的 XYZ 坐标
            depth_pred_map = depth_map[channel]
            coords = get_geometry(depth_pred_map, sensor_meta, affine_matrix)  # (H_img, W_img, 3)
            coords_n = coords.copy() 
            # 翻转 x、y
            coords_n[:, :, 0] *= -1            # x_n = -x_local
            coords_n[:, :, 1] *= -1            # y_n = -y_local

        elif dataset_key == 'nusc':
            # # 读取 idx 帧、cam_idx 相机的 depth pred map
            # depth_pred_path = data[idx][f'{frame_key}_camera_data'][channel]['depth_pred']
            # depth_pred_map  = np.load(depth_pred_path)  # (H_img, W_img)
            proj_matrix, K, R_l2c, t_l2c = build_lidar_to_camera_projection(
                data[idx][f'sensor_metas_{frame_key}'],
                data[idx][f'sensor_metas_{frame_key}']['camera']['calibrated_sensor'][channel],
                data[idx][f'sensor_metas_{frame_key}']['camera']['ego_pose'][channel]
            )
            sensor_meta = {'K': K, 'R_l2c': R_l2c, 't_l2c': t_l2c}

            # 3) 根据深度图和相机内外参，将像素坐标转换为 LiDAR 坐标系下的 XYZ 坐标
            coords = get_geometry(depth_map, sensor_meta, affine_matrix)  # (H_img, W_img, 3)
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
        plt.savefig(f"./Datasets/cyberrock/scene_7/depth_vis/{frame_key}_normalized_range_{idx}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return proj_range, proj_pix



CAMERA_CHANNELS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
]

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

def build_frame_mapping_fast(
    data,
    dataset_key: str,
    frame_key: str,
    depth_map_dict: dict,
    affine_matrix: np.ndarray,
    idx: int,
    H_r: int = 40, W_r: int = 480,
    visualize: bool = False,
    pixel_stride: int = 1
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
    camera_channels = CAMERA_CHANNELS

    # 1) 帧级像素网格逆仿射（一次计算，所有相机复用）
    any_ch = camera_channels[0]
    H_img, W_img = depth_map_dict[any_ch].shape
    pix_orig_full = _precompute_pix_orig(H_img, W_img, affine_matrix)          # (3, N)

    if pixel_stride > 1:
        us = np.arange(0, W_img, pixel_stride)
        vs = np.arange(0, H_img, pixel_stride)
        u_grid_s, v_grid_s = np.meshgrid(us, vs)
        pick_lin = (v_grid_s * W_img + u_grid_s).reshape(-1)                   # (N')
        pix_orig = pix_orig_full[:, pick_lin]                                   # (3, N')
        # 下采样后的 (u,v) 基网格（后续各相机按有效掩膜过滤）
        uu_sub = u_grid_s.reshape(-1)
        vv_sub = v_grid_s.reshape(-1)
    else:
        pick_lin = None
        pix_orig = pix_orig_full                                               # (3, N)
        # 全量 (u,v)
        uu_full, vv_full = np.meshgrid(np.arange(W_img), np.arange(H_img))
        uu_full = uu_full.reshape(-1)
        vv_full = vv_full.reshape(-1)

    # 2) 相机级缓存（K_inv/R_c2l/t_c2l/flip_xy）
    cam_cache = {}
    if dataset_key == 'sjtu':
        for ch in camera_channels:
            K     = np.asarray(data[idx][f'sensor_metas_{frame_key}'][ch]['K'], dtype=np.float32)
            R_l2c = np.asarray(data[idx][f'sensor_metas_{frame_key}'][ch]['R_l2c'], dtype=np.float32)
            t_l2c = np.asarray(data[idx][f'sensor_metas_{frame_key}'][ch]['t_l2c'], dtype=np.float32).reshape(3, 1)

            K_inv = np.linalg.inv(K).astype(np.float32)
            R_c2l = R_l2c.T.astype(np.float32)
            t_c2l = (-R_c2l @ t_l2c).astype(np.float32)
            cam_cache[ch] = (K_inv, R_c2l, t_c2l, True)  # SJTU: flip_xy=True

    else:  # 'nusc'
        # 采用 build_lidar_to_camera_projection 来得到 K, R_l2c, t_l2c
        

        sensor_metas = data[idx][f'sensor_metas_{frame_key}']
        for ch in camera_channels:
            proj_matrix, K, R_l2c, t_l2c = build_lidar_to_camera_projection(
                sensor_metas,
                sensor_metas['camera']['calibrated_sensor'][ch],
                sensor_metas['camera']['ego_pose'][ch]
            )
            K     = np.asarray(K, dtype=np.float32)
            R_l2c = np.asarray(R_l2c, dtype=np.float32)
            t_l2c = np.asarray(t_l2c, dtype=np.float32).reshape(3, 1)

            K_inv = np.linalg.inv(K).astype(np.float32)
            R_c2l = R_l2c.T.astype(np.float32)
            t_c2l = (-R_c2l @ t_l2c).astype(np.float32)
            cam_cache[ch] = (K_inv, R_c2l, t_c2l, False)  # NuScenes: flip_xy=False

    # 3) 各相机反投影 + 像素坐标收集
    all_points = []
    all_pix    = []

    for cam_idx, ch in enumerate(camera_channels):
        depth_map = np.asarray(depth_map_dict[ch], dtype=np.float32)

        # 与 pix_orig 对齐的深度向量 ds
        if pixel_stride > 1:
            ds = depth_map.reshape(-1)[pick_lin]            # (N',)
            uu_base, vv_base = uu_sub, vv_sub              # 下采样 (u,v) 基
        else:
            ds = depth_map.reshape(-1)                      # (N,)
            uu_base, vv_base = uu_full, vv_full            # 全量 (u,v)

        valid = ds > 0
        if not np.any(valid):
            continue

        # 对应地裁切 pix_orig 与 ds
        pix_orig_use = pix_orig[:, valid]                  # (3, N_valid)
        ds_use = ds[valid]                                 # (N_valid,)

        K_inv, R_c2l, t_c2l, flip_xy = cam_cache[ch]
        pts = _geometry_cam2lidar_from_depth(
            depth_map, K_inv, R_c2l, t_c2l, pix_orig_use,
            flip_xy=flip_xy, ds=ds_use
        ).astype(np.float32)                               # (N_valid, 3)

        uu = uu_base[valid].astype(np.int32)
        vv = vv_base[valid].astype(np.int32)
        cam_col = np.full_like(uu, cam_idx, dtype=np.int32)
        pix = np.stack([cam_col, uu, vv], axis=1).astype(np.int32)  # (N_valid, 3)

        all_points.append(pts)
        all_pix.append(pix)

    # 若所有相机都无有效点，返回空图
    if not all_points:
        proj_range = np.full((H_r, W_r), -1, np.float32)
        proj_pix   = np.full((H_r, W_r, 3), -1, np.int32)
        return proj_range, proj_pix

    points = np.concatenate(all_points, axis=0)
    pix    = np.concatenate(all_pix,    axis=0)

    # 4) 矢量化 range 投影 + 最近点选择
    proj_range, proj_xyz, proj_idx, proj_mask, proj_pix = _range_projection_with_mapping_np_fast(
        points, pix, H=H_r, W=W_r, fov_up=8.0, fov_down=-15.0
    )

    # 5) 小图补洞（40×480 开销很低）
    valid_im = proj_mask.astype(bool)
    if not valid_im.all():
        _, inds = distance_transform_edt(~valid_im, return_distances=True, return_indices=True)
        i_near, j_near = inds
        proj_pix   = proj_pix[i_near, j_near]
        proj_range = proj_range[i_near, j_near]
        proj_mask[:] = 1

    return proj_range, proj_pix

def fetch_dataloader(args, TRAIN_DS='C+T+K/S'):
    """ Create the data loader for the corresponding trainign set """
    train_dataset = None

    if args.stage == 'nuscenes_range_image':
        aug_params = {'crop_size': args.image_size, 'do_flip': False, 'rotate': False, 'rotate_prob': 0.1, 'rotate_angle': 90}
        train_info_file = 'nusc_trainval_infos_160_1920_fov_8_15_dpt.pkl'
        train_info_path = './Datasets/nuscenes/2_trainval_test_infos'

        nuscenes = nuScenes_range_image(aug_params,
                                        train_info_file=train_info_file,
                                        train_info_path=train_info_path,
                                        split='training')

        train_dataset = 1*nuscenes
    
    else:
        raise ValueError(f"Unknown args.stage: {args.stage}")

    # print('Training with %d image pairs' % len(train_dataset.image_list))
    return train_dataset
