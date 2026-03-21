import argparse
from tqdm import tqdm
import os
import numpy as np
import pickle
from pathlib import Path
from nuscenes.utils.geometry_utils import view_points
import argparse
from pyquaternion import Quaternion
import time
from utils.nusc_paths import infer_nusc_dataset_root, resolve_nusc_path

def project_spherical_voxels_to_cameras(sensor_metas: dict,
                                        xyz: np.ndarray,
                                        cam_channels: list,
                                        min_dist: float = 1.0):
    """
    Args:
      sensor_metas: dict，包含 'lidar' 和 'camera' 的外参/内参
      xyz: (H, W, R, 3) 体素中心在 LiDAR 传感器坐标系下的坐标
      cam_channels: ['CAM_FRONT', ...]
      min_dist: 深度阈值

    Returns:
      uv_map:       (H, W, R, 2) 每个体素的 (u,v)，无效处填 0
      cam_idx_map:  (H, W, R)    每个体素属于哪个相机（枚举索引），无效处填 -1
      valid_mask:   (H, W, R)    投影是否有效
    """
    H, W, R, _ = xyz.shape
    N = H * W * R
    # 扁平化
    pts = xyz.reshape(-1, 3)  # (N,3)

    # 预计算 LiDAR -> Global
    lidar_cs   = sensor_metas['lidar']['calibrated_sensor']
    lidar_pose = sensor_metas['lidar']['ego_pose']
    R_s2e = Quaternion(lidar_cs['rotation']).rotation_matrix
    t_s2e = np.array(lidar_cs['translation'])
    R_e2g = Quaternion(lidar_pose['rotation']).rotation_matrix
    t_e2g = np.array(lidar_pose['translation'])

    # 先把所有点从 sensor -> global
    pts = (R_s2e @ pts.T).T + t_s2e
    pts = (R_e2g @ pts.T).T + t_e2g

    uv_map      = np.zeros((N, 2), dtype=np.float32)
    cam_idx_map = -np.ones((N,),   dtype=np.int32)
    valid_mask  = np.zeros((N,),   dtype=bool)

    # 对每个相机投影
    for cam_i, channel in enumerate(cam_channels):
        cam_cs   = sensor_metas['camera']['calibrated_sensor'][channel]
        cam_pose = sensor_metas['camera']['ego_pose'][channel]

        # Global -> Ego_cam
        t_g2e_cam = -np.array(cam_pose['translation'])
        R_g2e_cam = Quaternion(cam_pose['rotation']).rotation_matrix.T
        pts_cam_ego = (R_g2e_cam @ (pts + t_g2e_cam).T).T

        # Ego_cam -> Camera_sensor
        t_e2c = -np.array(cam_cs['translation'])
        R_e2c = Quaternion(cam_cs['rotation']).rotation_matrix.T
        pts_cam = (R_e2c @ (pts_cam_ego + t_e2c).T).T  # (N,3)

        depths = pts_cam[:, 2]
        intrinsic = np.array(cam_cs['camera_intrinsic'])
        uv = view_points(pts_cam.T, intrinsic, normalize=True)  # (2, N)

        # 过滤
        image_width, image_height = 1600, 900  # or 从 sensor_metas 中读
        mask = (
            (depths > min_dist) &
            (uv[0, :] >= 0) & (uv[0, :] < image_width) &
            (uv[1, :] >= 0) & (uv[1, :] < image_height)
        )
        idxs = np.nonzero(mask)[0]

        # 填入结果
        uv_map[idxs, 0]      = uv[0, idxs]
        uv_map[idxs, 1]      = uv[1, idxs]
        cam_idx_map[idxs]    = cam_i
        valid_mask[idxs]     = True

    # reshape 回 (H,W,R,...)
    uv_map      = uv_map.reshape(H, W, R, 2)
    cam_idx_map = cam_idx_map.reshape(H, W, R)
    valid_mask  = valid_mask.reshape(H, W, R)

    return uv_map, cam_idx_map, valid_mask


def build_spherical_voxels(H, W, R,
                           r_min=3.0, r_max=48.0,
                           fov_up_deg=8.0, fov_down_deg=-15.0):
    """
    在球坐标系下构建不均匀体素：
      yaw phi ∈ [-π, π] 切分 H 段
      pitch theta ∈ [fov_down, fov_up] 切分 W 段
      r ∈ [r_min, r_max] 取 R 个中心点
    
    返回：
      psi_centers: (H,)    —— 每个 yaw 体素的中心角
      theta_centers: (W,)    —— 每个 pitch 体素的中心角
      r_centers: (R,)    —— 每个径向体素的中心半径
      xyz      : (H,W,R,3) —— 对应于每个体素中心的 (x,y,z)
    """
    # 1. 角度转弧度
    fov_up   = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    
    # 2. 计算各维中心点
    # yaw: edges 分 H+1, center 在两端中点
    yaw_edges = np.linspace(-np.pi, np.pi, H+1)
    psi_centers = 0.5 * (yaw_edges[:-1] + yaw_edges[1:])
    
    # pitch: edges 分 W+1
    pitch_edges = np.linspace(fov_down, fov_up, W+1)
    theta_centers   = 0.5 * (pitch_edges[:-1] + pitch_edges[1:])
    
    # r: 这里假设线性取 R 个中心
    r_edges    = np.linspace(r_min, r_max, R+1)
    r_centers  = 0.5 * (r_edges[:-1] + r_edges[1:])
    
    # 3. 网格化所有体素中心
    #    psi_centers.shape=(H,), theta_centers=(W,), r_centers=(R,)
    phi, theta, Rr = np.meshgrid(psi_centers,
                            theta_centers,
                            r_centers,
                            indexing='ij')  # → (H, W, R)
    
    # 4. 转回笛卡尔坐标
    X = Rr * np.cos(theta) * np.cos(phi)
    Y = Rr * np.cos(theta) * np.sin(phi)
    Z = Rr * np.sin(theta)
    
    xyz = np.stack([X, Y, Z], axis=-1)  # (H, W, R, 3)
    sph = np.stack([phi, theta, Rr], axis=-1)

    return sph, xyz

def main(args):
    dataset_root = infer_nusc_dataset_root(args.pkl_path)

    try:
        with open(args.pkl_path, "rb") as f:
            # 尝试加载 pkl 文件
            trainval_test_info = pickle.load(f)
        print("pkl 文件加载成功!")
    except Exception as e:
        print("加载 pkl 文件失败:")
        print(e)

    for idx, info in enumerate(tqdm(trainval_test_info, desc="Processing info")):
        start = time.time()
        gt_map_path = resolve_nusc_path(info['gt_map_path'], dataset_root) / 'range_image.npy'
        gt_map = np.load(gt_map_path,
                         allow_pickle=True).item()
        H, W, R = 160, 1920, 16
        sph, xyz = build_spherical_voxels(
            H=H, W=W, R=R,
            r_min=5.0, r_max=50.0,
            fov_up_deg=8, fov_down_deg=-15)
        
        cam_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_RIGHT', 'CAM_BACK',  'CAM_BACK_LEFT']
        
        uv, cam_idx, valid_mask = project_spherical_voxels_to_cameras(
            sensor_metas=info['sensor_metas_curr'],
            xyz=xyz,
            cam_channels=cam_channels,
            min_dist=1.0
        )

        projected = {
            'spherical_voxels': sph,  # (H, W, R, 3) 每个体素的球坐标 (phi, theta, r)
            'xyz': xyz,               # (H, W, R, 3) 每个体素的笛卡尔坐标 (x,y,z)
            'uv': uv,                 # (H, W, R, 2) 每个体素投影到相机图像的 (u,v)
            'cam_idx': cam_idx,       # (H, W, R) 每个体素对应的相机索引
            'valid_mask': valid_mask  # (H, W, R) 每个体素投影是否有效
        }

        end = time.time()
        print(f"Projection took {end - start:.2f} seconds")
        # [TODO] 计算每一帧的时间 
        print(f"Processing {idx+1}/{len(trainval_test_info)}: {info['sample_token']}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", help="Path to the pkl file")
    # parser.add_argument("--save_path", help="Path to save the output RGB range image")
    args = parser.parse_args()
    main(args)


'''
import torch

def transform_uv_grid_torch(uv, A):
    """
    uv: Tensor, shape=(H, W, R, 2)
    A:  Tensor, shape=(3, 3)
    """
    H, W, R, _ = uv.shape
    # Flatten
    uv_flat = uv.view(-1, 2)                # (N,2)
    ones    = uv_flat.new_ones((uv_flat.shape[0], 1))  # (N,1)
    hom     = torch.cat([uv_flat, ones], dim=1)        # (N,3)

    # Transform
    hom_t   = (A @ hom.t()).t()             # (N,3)
    uv_t_flat = hom_t[:, :2]                # (N,2)

    # Reshape back
    return uv_t_flat.view(H, W, R, 2)
'''
