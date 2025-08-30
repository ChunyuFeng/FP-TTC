import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import os
import pickle
import math
import random
from glob import glob
import os.path as osp
import re
from tqdm import tqdm
from PIL import ImageDraw

from .utils.rectangle_noise import retangle
from .utils import frame_utils
import  cv2
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentorm, NuscAugmentor, NuscRangeImageAugmentor
from dataloader.utils.geometry import get_geometry, range_projection_with_mapping
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from fpttc.scale_net.utils.spherical import build_spherical_voxels, project_voxel_to_camera, build_lidar_to_camera_projection, make_homog
from pyquaternion import Quaternion

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

        for i in tqdm(range(len(self.data)-1790), desc='Loading nuScenes Range Image Dataset'):

            if self.data[i]['scene_indice'] == '10':
                continue
            
            # 1. 模型 Input
            self.image_list.append([self.data[i]['prev_camera_data'],
                                    self.data[i]['curr_camera_data']])
            
            # 2. Ground Truth Range Image —— Scale Map, Risk Score Map, Depth Map
            range_image_path = os.path.join(self.data[i]['gt_map_path'], 'range_image_curr.npy')
            if not osp.exists(range_image_path):
                raise FileNotFoundError(f"Range image file {range_image_path} does not exist.")
            range_image = np.load(range_image_path, allow_pickle=True).item()
            self.scale_map_list.append(range_image['scale'])
            self.risk_score_map_list.append(range_image['risk_score'])
            self.depth_map_list.append(range_image['depth'])

            # 3. 环视图像 (cam_idx, u, v) 与 range image (u, v) 之间的映射关系
            #    通过 DepthAnything 预测的 Depth Pred Map + 内外参 计算得到
            # proj_range_prev, proj_pix_prev = build_frame_mapping(self.data, 'nusc', 'prev', None,
            #                                                      self.affine_matrix, i, H_r=40, W_r=480)
            # proj_range_curr, proj_pix_curr = build_frame_mapping(self.data, 'nusc', 'curr', None, 
            #                                                      self.affine_matrix, i, H_r=40, W_r=480)
            proj_pix_prev = np.zeros((40, 480, 3), dtype=np.int32)
            proj_pix_curr = np.zeros((40, 480, 3), dtype=np.int32)
            self.proj_list.append([proj_pix_prev, proj_pix_curr])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        prev_surr_view_imgs = {}
        curr_surr_view_imgs = {}

        prev_surr_view_depths = {}
        curr_surr_view_depths = {}

        camera_channels = self.camera_channels    
        path_prefix = './Datasets/nuscenes/'

        # 1. 按照相机通道读取相邻帧的图像和深度预测结果
        for channel in camera_channels:
            # 1）读取相邻帧的图像
            prev_surr_view_imgs_path     = os.path.join(path_prefix, self.image_list[index][0][channel]['filename'])
            prev_surr_view_imgs[channel] = Image.open(prev_surr_view_imgs_path)

            curr_surr_view_imgs_path     = os.path.join(path_prefix, self.image_list[index][1][channel]['filename'])
            curr_surr_view_imgs[channel] = Image.open(curr_surr_view_imgs_path)

            # 2) 读取相邻帧的 Depth Pred Map (DepthAnythingV2 Metric)
            prev_surr_view_depths_path     = self.image_list[index][0][channel]['depth_pred']
            prev_surr_view_depths[channel] = np.load(prev_surr_view_depths_path)

            curr_surr_view_depths_path     = self.image_list[index][1][channel]['depth_pred']
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

        # 5. 打包相机内外参，以及跨帧车体相对位姿
        sensor_metas_prev = self.data[index]['sensor_metas_prev']
        sensor_metas_curr = self.data[index]['sensor_metas_curr']
        K_curr, T_E_from_C_curr, T_Ecurr_from_Eprev = pack_geocalib_tensors_per_cam_to_lidar(
            sensor_metas_prev = sensor_metas_prev, 
            sensor_metas_curr = sensor_metas_curr, 
            camera_channels   = self.camera_channels)
        
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
        affine_matrix = affine_matrix.unsqueeze(0).expand(6, -1, -1).contiguous()  # (6, 3, 3)
        # K_curr = affine_matrix @ K_curr


        return (prev_surr_view_imgs_tensor,
                curr_surr_view_imgs_tensor,
                prev_surr_view_depths_tensor,
                curr_surr_view_depths_tensor,
                proj_pix_prev_tensor,
                proj_pix_curr_tensor,
                gt_scale_map_with_mask,
                gt_risk_map_with_mask,
                gt_depth_map,
                K_curr,
                affine_matrix,
                T_E_from_C_curr,
                T_Ecurr_from_Eprev)

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
            depth_pred_path = data[idx][f'{frame_key}_camera_data'][channel]['depth_pred']
            depth_pred_map  = np.load(depth_pred_path)  # (H_img, W_img)
            proj_matrix, K, R_l2c, t_l2c = build_lidar_to_camera_projection(
                data[idx][f'sensor_metas_{frame_key}'],
                data[idx][f'sensor_metas_{frame_key}']['camera']['calibrated_sensor'][channel],
                data[idx][f'sensor_metas_{frame_key}']['camera']['ego_pose'][channel]
            )
            sensor_meta = {'K': K, 'R_l2c': R_l2c, 't_l2c': t_l2c}

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
        plt.savefig(f"./Datasets/cyberrock/scene_7/depth_vis/{frame_key}_normalized_range_{idx}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return proj_range, proj_pix

def lidar2cam_to_cam2lidar(R_l2c: np.ndarray, t_l2c: np.ndarray):
    """
    输入:
      R_l2c: [3,3] 或 [...,3,3]
      t_l2c: [3]   或 [...,3]
    返回:
      T_c2l: [4,4] 或 [...,4,4]
    """
    Rt = np.swapaxes(R_l2c, -1, -2)              # R^T
    t  = - Rt @ t_l2c[..., None]                 # -R^T t
    # 组 4x4
    T = np.zeros(R_l2c.shape[:-2] + (4, 4), dtype=R_l2c.dtype)
    T[..., :3, :3] = Rt
    T[..., :3,  3] = t[..., 0]
    T[...,  3,  3] = 1.0
    return T

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

    # print('Training with %d image pairs' % len(train_dataset.image_list))
    return train_dataset


def _Rt_from_calib(cs: dict):
    # calibrated_sensor: sensor -> ego
    R = Quaternion(cs['rotation']).rotation_matrix.astype(np.float64)
    t = np.array(cs['translation'], dtype=np.float64)
    return R, t

def _Rt_from_egopose(ep: dict):
    # ego_pose: ego -> global
    R = Quaternion(ep['rotation']).rotation_matrix.astype(np.float64)
    t = np.array(ep['translation'], dtype=np.float64)
    return R, t


def _stack_torch(mats, dtype=torch.float32):
    """
    接受由 np.ndarray 或 torch.Tensor 混合组成的列表，统一转成 dtype，再 stack。
    """
    tlist = []
    for x in mats:
        if isinstance(x, torch.Tensor):
            tlist.append(x.to(dtype))
        else:
            # 允许 list/tuple，统一转 np 后再 as_tensor
            tlist.append(torch.as_tensor(np.asarray(x), dtype=dtype))
    return torch.stack(tlist, dim=0)

def pack_geocalib_tensors_per_cam_to_lidar(
    sensor_metas_prev: dict,
    sensor_metas_curr: dict,
    camera_channels: list
):
    """
    直接在 LiDAR 坐标系（x右, y前, z上）下输出：
      K_curr              [V,3,3]
      T_L_from_C_curr     [V,4,4]   （L <- C, 当前帧）
      T_Lcurr_from_Lprev  [4,4]     （L_curr <- L_prev）

    链路（均按 nuScenes 原生定义，无任何轴变换）：
      G <- E   来自 ego_pose
      E <- S   来自 calibrated_sensor
      L <- G   = (G <- L)^-1，其中 G <- L = (G <- E_lidar) @ (E_lidar <- L)
      L <- C   = (L <- G_curr) @ (G <- C_curr)
    """
    V = len(camera_channels)

    # -------- LiDAR（prev / curr）--------
    # 当前帧 LiDAR
    lidar_cs_curr   = sensor_metas_curr['lidar']['calibrated_sensor']  # L -> E_lidar(curr)
    lidar_pose_curr = sensor_metas_curr['lidar']['ego_pose']           # E_lidar(curr) -> G
    R_E_from_Lc, t_E_from_Lc = _Rt_from_calib(lidar_cs_curr)
    R_G_from_Ec, t_G_from_Ec = _Rt_from_egopose(lidar_pose_curr)
    T_E_from_Lc = make_homog(R_E_from_Lc, t_E_from_Lc)  # E <- L
    T_G_from_Ec = make_homog(R_G_from_Ec, t_G_from_Ec)  # G <- E

    # 上一帧 LiDAR
    lidar_cs_prev   = sensor_metas_prev['lidar']['calibrated_sensor']  # L -> E_lidar(prev)
    lidar_pose_prev = sensor_metas_prev['lidar']['ego_pose']           # E_lidar(prev) -> G
    R_E_from_Lp, t_E_from_Lp = _Rt_from_calib(lidar_cs_prev)
    R_G_from_Ep, t_G_from_Ep = _Rt_from_egopose(lidar_pose_prev)
    T_E_from_Lp = make_homog(R_E_from_Lp, t_E_from_Lp)
    T_G_from_Ep = make_homog(R_G_from_Ep, t_G_from_Ep)

    # G <- L_prev
    T_G_from_Lprev = T_G_from_Ep @ T_E_from_Lp
    # L_curr <- G
    T_L_from_Gcurr = np.linalg.inv(T_G_from_Ec @ T_E_from_Lc)
    # L_curr <- L_prev
    T_Lcurr_from_Lprev = (T_L_from_Gcurr @ T_G_from_Lprev).astype(np.float32)

    # -------- 每个相机：K 与 T_L_from_C_curr --------
    K_list = []
    TL_from_C_list = []
    for cam in camera_channels:
        cam_cs   = sensor_metas_curr['camera']['calibrated_sensor'][cam]  # C -> E_cam
        cam_pose = sensor_metas_curr['camera']['ego_pose'][cam]           # E_cam -> G

        # K
        K = np.array(cam_cs['camera_intrinsic'], dtype=np.float32)
        K_list.append(torch.from_numpy(K))

        # G <- C_curr
        T_E_from_C   = make_homog(*_Rt_from_calib(cam_cs))       # E_cam <- C
        T_G_from_Ecm = make_homog(*_Rt_from_egopose(cam_pose))   # G <- E_cam
        T_G_from_C   = T_G_from_Ecm @ T_E_from_C                 # G <- C

        # L_curr <- C_curr
        T_L_from_C = (T_L_from_Gcurr @ T_G_from_C).astype(np.float32)
        TL_from_C_list.append(torch.from_numpy(T_L_from_C))

    K_curr             = _stack_torch(K_list)                       # [V,3,3]
    T_L_from_C_curr    = torch.stack(TL_from_C_list, dim=0)         # [V,4,4]
    T_Lcurr_from_Lprev = torch.from_numpy(T_Lcurr_from_Lprev)       # [4,4]

    return K_curr, T_L_from_C_curr, T_Lcurr_from_Lprev
