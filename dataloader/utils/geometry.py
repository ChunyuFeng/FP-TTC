import argparse
from tqdm import tqdm
import os
import numpy as np
import pickle
from nuscenes.utils.geometry_utils import view_points
import argparse
from pyquaternion import Quaternion
import time

import numpy as np
from math import atan2, sqrt
from pyquaternion import Quaternion

def pixel_to_elevation_angles(R_s2c, K, image_size):
    """
    计算图像顶部 (v=0) 与底部 (v=H-1) 对应的激光雷达仰角（°）。

    参数:
      R_s2c       : (3, 3) 雷达系到相机系的旋转矩阵
      K           : (3, 3) 相机内参矩阵
      image_size  : (H, W) 原始图像高宽

    返回:
      theta_top_deg, theta_bottom_deg （单位：度）
    """
    

    # 2) 反求相机->LiDAR 的旋转（忽略平移）
    R_c2s = R_s2c.T   # 相机系到雷达系的旋转矩阵

    # 3) 提取内参
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    H, W = image_size

    def elevation(u, v):
        # 相机系归一化射线
        dc = np.array([(u - cx)/fx, (v - cy)/fy, 1.0], dtype=float)
        # 转到雷达系（只旋转，不加平移）
        dl = R_c2s.dot(dc)
        # 计算仰角：上正，下负
        return atan2(dl[2], sqrt(dl[0]**2 + dl[1]**2))

    # 4) 选取图像中心列像素点
    u_center = cx

    theta_top    = elevation(u_center, 0)
    theta_bottom = elevation(u_center, H-1)

    # 转成度
    theta_top_deg    = np.degrees(theta_top)
    theta_bottom_deg = np.degrees(theta_bottom)
    return theta_top_deg, theta_bottom_deg

# 示例调用
# 假设你已经有了 sensor_meta、cam_cs、cam_pose：
# cam = 'CAM_FRONT'
# cam_cs   = sensor_meta['camera']['calibrated_sensor'][cam]
# cam_pose = sensor_meta['camera']['ego_pose'][cam]
# H, W = cam_cs['height'], cam_cs['width']
# top_deg, bot_deg = pixel_to_elevation_angles(sensor_meta, cam_cs, cam_pose, (H, W))
# print("上边缘仰角:", top_deg, "°；下边缘仰角:", bot_deg, "°")

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
    return P, K, R_tot, t_tot


def project_voxel_to_camera(
    xyz,
    sensor_metas,
    camera_channels,
    raw_img_size,
    min_dist,
    sjtu=False):
    """
    对一批 LiDAR 体素点 (H,W,R,3) 进行投影，输出 (H,W,R,3) 数组，其中每个 voxel 对应的三个值为:
      [cam_idx, u, v]，若无效则均为 -1。

    :param xyz: shape=(H, W, R, 3)
    :param sensor_metas: length B batch 的 sensor_meta
    :param raw_img_size: 原始图像大小 (H, W)
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
        if sjtu:
            # SJTU 数据集的相机元数据结构
            # R_tot, t_tot = sensor_metas[cam]['R'], sensor_metas[cam]['t']
            K_undist = sensor_metas[cam]['K_undist']
            # RT = np.hstack([R_tot, t_tot.reshape(3, 1)])  # (3, 4)
            RT = sensor_metas[cam]['old_ext']
            R_tot = RT[:3, :3]  # (3, 3)
            t_tot = RT[:3, 3]  # (3,)
            P = K_undist.dot(RT)
            # top_deg, bot_deg = pixel_to_elevation_angles(R_tot, K_undist, raw_img_size)
            # print(f"Camera {cam} top: {top_deg:.2f}°, bottom: {bot_deg:.2f}°")
            # t_tot = t_tot.reshape(3, 1)  # (3, 1)
            # P = K_undist @ RT
        else:
            # NuScenes 数据集的相机元数据结构
            cam_cs = sensor_metas['camera']['calibrated_sensor'][cam]
            cam_pose = sensor_metas['camera']['ego_pose'][cam]
            P, K, R_tot, t_tot = build_lidar_to_camera_projection(sensor_metas, cam_cs, cam_pose)
            # top_deg, bot_deg = pixel_to_elevation_angles(R_tot, K, raw_img_size)
            # print(f"Camera {cam} top: {top_deg:.2f}°, bottom: {bot_deg:.2f}°")
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
                           fov_up_deg=8.0, fov_down_deg=-15.0,
                           sjtu=False):
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




def get_geometry(depth_map: np.ndarray,
                 sensor_meta: dict,
                 affine_matrix: np.ndarray) -> np.ndarray:
    """
    根据深度图和相机内外参，将像素坐标转换为 LiDAR 坐标系下的 XYZ 坐标。
    :param depth_map:    (H, W) 预处理后图像上的深度
    :param sensor_meta:  包含内参和 LiDAR→Camera 外参的字典，格式如下:
                         - 'K'    : 3×3 相机内参
                         - 'R_l2c': 3×3 LiDAR→Camera 旋转
                         - 't_l2c': 3×1 LiDAR→Camera 平移
    :param affine_matrix:3×3 预处理仿射矩阵（原图→处理后图）
    :return:             (H, W, 3) 在 LiDAR 坐标系下的 XYZ
    """
    H, W = depth_map.shape

    # 1) 生成像素网格并扁平化
    us = np.arange(W); vs = np.arange(H)
    u_grid, v_grid = np.meshgrid(us, vs)
    ones = np.ones_like(u_grid)
    pix_proc = np.stack([u_grid, v_grid, ones], axis=-1).reshape(-1, 3).T  # (3, H*W)

    # 2) 反仿射：原图像素 = A^{-1} * 处理后像素
    invA = np.linalg.inv(affine_matrix)
    pix_orig = invA @ pix_proc
    pix_orig /= pix_orig[2:3, :]

    # 3) 深度扁平化
    ds = depth_map.reshape(-1)  # (H*W,)

    # 4) 像素→相机系
    K_inv = np.linalg.inv(sensor_meta['K'])
    Xc = (K_inv @ pix_orig) * ds  # (3, H*W)

    # 5) 准备逆外参：Camera→LiDAR
    R_l2c = sensor_meta['R_l2c']          # LiDAR→Cam
    t_l2c = sensor_meta['t_l2c'].reshape(3,1)
    R_c2l = R_l2c.T                       # 转置
    t_c2l = - R_c2l @ t_l2c               #  -Rᵀ * t

    # 6) 相机→LiDAR
    Xl = R_c2l @ Xc + t_c2l               # (3, H*W)

    # 7) reshape 回 (H, W, 3)
    XYZ = Xl.T.reshape(H, W, 3)
    return XYZ

def range_projection_with_mapping(
    points: np.ndarray,
    pix_coords: np.ndarray,
    H=160, W=1920,
    fov_up=8.0, fov_down=-15.0
):
    """
    :param points:     (m,3) array of (x,y,z)
    :param pix_coords: (m,3) array of (cam_idx, u, v) 对应同一顺序的原始像素
    :return:
      proj_range: (H,W)
      proj_xyz:   (H,W,3)
      proj_idx:   (H,W)
      proj_mask:  (H,W)
      proj_pix:   (H,W,3)  原始像素映射
    """
    assert points.ndim==2 and points.shape[1]==3
    assert pix_coords.ndim==2 and pix_coords.shape[1]==3
    m = points.shape[0]
    # 1. 初始化输出
    proj_range = np.full((H, W), -1, dtype=np.float32)
    proj_xyz   = np.full((H, W, 3), -1, dtype=np.float32)
    proj_idx   = np.full((H, W), -1, dtype=np.int32)
    proj_mask  = np.zeros((H, W), dtype=np.int32)
    proj_pix   = np.full((H, W, 3), -1, dtype=np.int32)  # 新增：存 cam_idx, u, v
    # 连续球面坐标，浮点类型，不取整
    proj_xy_float = np.full((H, W, 2), -1.0, dtype=np.float32)

    # 2. 角度计算
    fov_up_rad   = fov_up   /180.0 * np.pi
    fov_down_rad = fov_down /180.0 * np.pi
    fov          = abs(fov_down_rad) + abs(fov_up_rad)

    # 3. Depth、yaw、pitch
    depth = np.linalg.norm(points, axis=1)         # (m,)
    x, y, z = points[:,0], points[:,1], points[:,2]
    yaw   = -np.arctan2(y, x)                     # (m,)
    pitch = np.arcsin(z / depth)                  # (m,)

    # 4. 归一化到 [0,1] 并映射到像素
    proj_xf = 0.5*( yaw/np.pi + 1.0 ) * W          # (m,)
    proj_yf = (1.0 - (pitch + abs(fov_down_rad))/fov) * H

    # 5. 整数化 & clip
    proj_x = np.floor(proj_xf).astype(np.int32)
    proj_x = np.clip(proj_x, 0, W-1)
    proj_y = np.floor(proj_yf).astype(np.int32)
    proj_y = np.clip(proj_y, 0, H-1)

    # 6. 按深度近到远排序
    order = np.argsort(depth)
    depth_s = depth[order]
    pts_s   = points[order]
    x_s     = proj_x[order]
    y_s     = proj_y[order]
    pix_s   = pix_coords[order]   # 对齐排序
    # proj_xf, proj_yf 也按 order 排序
    xf_s    = proj_xf[order]
    yf_s    = proj_yf[order]

    # 7. 赋值
    for d, (py, px, pt, pi, xff, yff) in enumerate(zip(y_s, x_s, pts_s, pix_s, xf_s, yf_s)):
        # 如果该像素第一次被写入（idx=-1），就写入
        if proj_idx[py,px] == -1:
            proj_range[py,px] = depth_s[d]
            proj_xyz[py,px]   = pt
            proj_idx[py,px]   = order[d]
            proj_mask[py,px]  = 1
            proj_pix[py,px]   = pi.astype(np.int32)
            proj_xy_float[py,px,0] = xff
            proj_xy_float[py,px,1] = yff

    return proj_range, proj_xyz, proj_idx, proj_mask, proj_pix, proj_xy_float
