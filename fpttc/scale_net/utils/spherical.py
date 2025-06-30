import argparse
from tqdm import tqdm
import os
import numpy as np
import pickle
from nuscenes.utils.geometry_utils import view_points
import argparse
from pyquaternion import Quaternion
import time

def make_homog(R: np.ndarray, t: np.ndarray):
    """构造 4×4 齐次变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


def build_lidar_to_camera_projection(
    sensor_meta: dict,
    cam_cs: dict,
    cam_pose: dict
):
    """
    构造 LiDAR(S) -> Camera(C) 的投影矩阵 P（3x4），以及 R_tot(3x3) 和 t_tot(3,)。
    """
    # 1) LiDAR -> Ego
    lidar_cs   = sensor_meta['lidar']['calibrated_sensor']
    lidar_pose = sensor_meta['lidar']['ego_pose']
    R_s2e = Quaternion(lidar_cs['rotation']).rotation_matrix
    t_s2e = np.array(lidar_cs['translation'])
    T_s2e = make_homog(R_s2e, t_s2e)

    # 2) Ego -> Global
    R_e2g = Quaternion(lidar_pose['rotation']).rotation_matrix
    t_e2g = np.array(lidar_pose['translation'])
    T_e2g = make_homog(R_e2g, t_e2g)

    # 3) Global -> Ego_Cam (inverse)
    R_e2g_cam = Quaternion(cam_pose['rotation']).rotation_matrix
    t_e2g_cam = np.array(cam_pose['translation'])
    T_g2ecam  = np.linalg.inv(make_homog(R_e2g_cam, t_e2g_cam))

    # 4) Ego_Cam -> Cam_Sensor (inverse)
    R_c2e = Quaternion(cam_cs['rotation']).rotation_matrix
    t_c2e = np.array(cam_cs['translation'])
    T_ecam2c = np.linalg.inv(make_homog(R_c2e, t_c2e))

    # 5) 合并变换 S -> C
    T_s2c = T_ecam2c @ T_g2ecam @ T_e2g @ T_s2e
    R_tot = T_s2c[:3, :3]
    t_tot = T_s2c[:3,  3]

    # 6) 加内参得到 P
    K = np.array(cam_cs['camera_intrinsic'])
    P = K @ T_s2c[:3, :]
    return P, R_tot, t_tot


def project_voxel_to_camera(
    xyz,
    sensor_metas,
    camera_channels,
    raw_img_size,
    # img_size,
    # affine,
    min_dist):
    """
    对一批 LiDAR 体素点 (H,W,R,3) 进行投影，输出 (H,W,R,3) 数组，其中每个 voxel 对应的三个值为:
      [cam_idx, u, v]，若无效则均为 -1。

    :param xyz: shape=(H, W, R, 3)
    :param sensor_metas: length B batch 的 sensor_meta
    :param raw_img_size: 原始图像大小 (H, W)
    :param img_size: 预处理后的图像大小 (H, W)
    :param camera_channels: 相机顺序列表
    :param affine: optional，仿射变换矩阵，表示输入图像在预处理时经过的变换
    :param min_dist: 投影深度阈值
    :return: (cam_idx, u, v) shape=(H, W, R, 3) float32，
    """
    H, W, R, _ = xyz.shape
    # 初始化输出为 -1
    out = -np.ones((H, W, R, 3), dtype=np.float32)

    # 将体素 coords 展平
    coords = xyz.reshape(-1, 3)  # (H*W*R, 3)
    ones   = np.ones((coords.shape[0], 1), dtype=np.float32)
    pts_h  = np.concatenate([coords, ones], axis=1).T  # (4, H*W*R)

    for cam_idx, cam in enumerate(camera_channels):
        # h_cam, w_cam = img_size[:2]

        cam_cs = sensor_metas['camera']['calibrated_sensor'][cam]
        cam_pose = sensor_metas['camera']['ego_pose'][cam]
        P, R_tot, t_tot = build_lidar_to_camera_projection(sensor_metas, cam_cs, cam_pose)

        # # 应用仿射
        # if affine:
        #     P = affine[bs] @ P
        #     h_cam, w_cam = img_size[:2]
        # else:
        #     h_cam, w_cam = raw_img_size[:2]

        # 计算深度
        pts_cam = R_tot @ coords.T + t_tot[:, None]
        depths = pts_cam[2, :]

        # 齐次投影
        uvw = P @ pts_h
        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]

        h_cam, w_cam = raw_img_size[:2]
        # 有效 mask
        valid = (depths > min_dist) & (u >= 0) & (u < w_cam) & (v >= 0) & (v < h_cam)

        # 将有效投影写入 out, 最早写入保留
        sel = np.nonzero(valid)[0]
        # sel 对应 coords 索引 -> 还原到 (h,w,r)
        h_idx = sel // (W * R)
        rem = sel % (W * R)
        w_idx = rem // R
        r_idx = rem % R
        out[h_idx, w_idx, r_idx, 0] = float(cam_idx)
        out[h_idx, w_idx, r_idx, 1] = u[sel]
        out[h_idx, w_idx, r_idx, 2] = v[sel]
        
    return out

def build_spherical_voxels(H, W, R,
                           r_min=5.0, r_max=50.0,
                           fov_up_deg=8.0, fov_down_deg=-15.0):
    """
    在球坐标系下构建不均匀体素：
      H 为垂直方向（pitch）的切分数
      W 为水平方向（yaw）的切分数
      r 为径向的采样数 R

    返回：
      sph: (H, W, R, 3) —— 每个体素的 (yaw, pitch, r)
      xyz: (H, W, R, 3) —— 对应于每个体素中心的 (x, y, z)
    """
    # 1. 角度转弧度
    fov_up   = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    
    # 2. 计算各维中心点
    # pitch (垂直) 切分 H 段
    pitch_edges   = np.linspace(fov_down, fov_up, H + 1)
    theta_centers = 0.5 * (pitch_edges[:-1] + pitch_edges[1:])  # (H,)

    # yaw (水平) 切分 W 段
    yaw_edges     = np.linspace(-np.pi, np.pi, W + 1)
    psi_centers   = 0.5 * (yaw_edges[:-1] + yaw_edges[1:])      # (W,)

    # 径向 r 切分 R 段
    r_edges   = np.linspace(r_min, r_max, R + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])              # (R,)

    # 3. 网格化所有体素中心
    #   传入 (theta_centers, psi_centers, r_centers) 并使用 indexing='ij'
    theta_mesh, psi_mesh, Rr = np.meshgrid(
        theta_centers,
        psi_centers,
        r_centers,
        indexing='ij'
    )  # -> (H, W, R) 各自对应 axes (0,1,2)

    # 4. 转回笛卡尔坐标，坐标系定义：
    # 原点在球心，x 轴指向右侧，y 轴指向正前方，z 轴指向上方
    X = Rr * np.cos(theta_mesh) * np.sin(psi_mesh)
    Y = Rr * np.cos(theta_mesh) * np.cos(psi_mesh)
    Z = Rr * np.sin(theta_mesh)

    xyz = np.stack([X, Y, Z], axis=-1)         # -> (H, W, R, 3)
    sph = np.stack([psi_mesh, theta_mesh, Rr], axis=-1)  # -> (H, W, R, 3)

    return sph, xyz
