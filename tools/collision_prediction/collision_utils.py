import os
import cv2
import numpy as np
import math
import time
from sklearn.cluster import DBSCAN
import torch


from PIL import Image
import os.path as osp
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points



# ==================== collision_utils 部分函数 ====================

def get_rect_ttc_mask(scale, length=35):
    """
    构造 Scale 数据的掩码，将图像左右下角区域置零，排除边缘干扰
    """
    mask = np.ones_like(scale)
    h, w = mask.shape
    mask[h-length:h, 0:length] = 0
    mask[h-length:h, w-length:w] = 0
    return mask

def get_second_grad(data, stride=1):
    """
    计算输入数据的二阶梯度，返回一阶梯度 (gy, gx) 及二阶梯度 (gyy, gxx)
    """
    gy = data[stride:, :] - data[:-stride, :]
    gx = data[:, stride:] - data[:, :-stride]
    # 计算二阶梯度：对 gy 再做一次差分
    gyy = gy[stride:, :] - gy[:-stride, :]
    # 这里采用 np.gradient 计算 gx 的梯度作为 gxx
    _, gxx = np.gradient(gx)
    return gy, gx, gyy, gxx

def get_ttc_var(scale, grid_len=5):
    """
    将 Scale 数据按 grid_len 划分网格，计算每个网格内的方差
    """
    h, w = scale.shape
    h_var = math.ceil(h / grid_len) - 1
    w_var = math.ceil(w / grid_len) - 1
    var = np.zeros([h_var, w_var])
    for i in range(h_var):
        for j in range(w_var):
            var[i, j] = np.var(scale[i*grid_len:(i+1)*grid_len, j*grid_len:(j+1)*grid_len])
    return var

def get_grid_ttc(scale, scale_var, grid_len=5):
    """
    根据每个网格内的方差筛选局部均匀区域，返回网格掩码和二值图
    """
    var_mean = 0.3 * np.mean(scale_var)
    valid_grid = np.zeros([scale.shape[0], scale.shape[1]])
    valid_map = np.zeros([scale_var.shape[0], scale_var.shape[1]])
    h, w = scale_var.shape
    for i in range(h):
        for j in range(w):
            if scale_var[i, j] < var_mean:
                valid_grid[i*grid_len:(i+1)*grid_len, j*grid_len:(j+1)*grid_len] = 1
                valid_map[i, j] = 1
    return valid_map, valid_grid

def get_chess_board(template):
    """
    生成棋盘状掩码，用于过滤部分噪声
    """
    h, w = template.shape
    mask1 = np.ones((h, w), dtype=bool)
    mask2 = np.ones((h, w), dtype=bool)
    for i in range(h):
        if i % 2 != 0:
            mask1[i, :] = 0
    for j in range(w):
        if j % 2 == 0:
            mask2[:, j] = 0
    return np.bitwise_xor(mask1, mask2)

def get_turning_rate(expansion):
    """
    返回一组固定参数（兼容调用），此处直接返回固定值
    """
    grid_len = 15
    turning_rate = 1
    drawing_rate = 1
    drawing_rate_y = 1
    ttc_rate = 1
    return grid_len, turning_rate, drawing_rate, drawing_rate_y, ttc_rate, expansion

# ==================== 可视化函数 ====================

def visual_scale_map_range_image(scale_map, valid_mask, colormap_name='seismic'):
    """
    对 Scale 数据进行可视化：
      1. 将 valid_mask 区域内的 Scale 值裁剪到 [0.5, 1.5] 范围；
      2. 计算 deviations = scale - 1；
      3. 对正负部分分别归一化，得到范围在 [-1,1] 内的 normalized_display。
    返回 normalized_display。
    """
    scale_display = np.copy(scale_map)
    scale_display[valid_mask] = np.clip(scale_display[valid_mask], 0.5, 1.5)
    deviations = np.zeros_like(scale_display, dtype=np.float32)
    deviations[valid_mask] = scale_display[valid_mask] - 1.0
    pos_mask = deviations > 0
    neg_mask = deviations < 0
    normalized_display = np.zeros_like(deviations, dtype=np.float32)
    if np.any(pos_mask):
        pos_devs = deviations[pos_mask]
        pos_max = pos_devs.max()
        pos_min = pos_devs.min()
        if pos_max >= pos_min >= 0:
            normalized_display[pos_mask] = (pos_devs - pos_min) / (pos_max - pos_min)
        else:
            normalized_display[pos_mask] = 0.0
    if np.any(neg_mask):
        neg_devs = deviations[neg_mask]
        neg_max = neg_devs.max()
        neg_min = neg_devs.min()
        if neg_min <= neg_max <= 0:
            normalized_display[neg_mask] = (neg_devs - neg_min) / (neg_max - neg_min) - 1.0
        else:
            normalized_display[neg_mask] = 0.0
    normalized_display[~valid_mask] = 0.0
    return normalized_display

def inverse_range_projection(u, v, depth, H=160, W=1920, fov_up=10.0, fov_down=-30.0):
    """
    将 range image 上的像素坐标 (u, v) 及对应的深度值 depth 反投影回 3D 空间坐标。

    参数:
    - u (int 或 np.ndarray): range image 上的水平坐标（列）。
    - v (int 或 np.ndarray): range image 上的垂直坐标（行）。
    - depth (float 或 np.ndarray): 对应像素的深度值。
    - H (int): range image 的高度。默认 64。
    - W (int): range image 的宽度。默认 1024。
    - fov_up (float): 向上的视场角（度）。默认 3.0。
    - fov_down (float): 向下的视场角（度）。默认 -25.0。

    返回:
    - np.ndarray: 对应的 3D 坐标 [x, y, z]，如果 u, v, depth 为数组，则返回形状一致的数组。
    """
    # 视场角转换为弧度
    fov_up_rad = np.radians(fov_up)
    fov_down_rad = np.radians(fov_down)
    fov = abs(fov_down_rad) + abs(fov_up_rad)

    # 根据 u 计算水平角 yaw，注意 u 归一化到 [0,1]
    yaw = (2 * (u / W) - 1) * np.pi

    # 根据 v 计算垂直角 pitch，注意 v 归一化到 [0,1]
    pitch = (1 - v / H) * fov - abs(fov_down_rad)

    # 反投影到 3D 空间
    x = depth * np.cos(pitch) * np.cos(yaw)
    y = -depth * np.cos(pitch) * np.sin(yaw)
    z = depth * np.sin(pitch)

    return np.array([x, y, z])

def project_lidar_to_surround_view_img(nusc,
                                       lidar_data,
                                       lidar_token,
                                       surround_view_img_token,
                                       min_dist=1.0):
    '''
    将 LiDAR 点云投影到相机视图上
    Args:
        nusc: nuScenes 数据集对象
        lidar_data: lidar 点云数据
        lidar_token: lidar sample_data token
        surround_view_img_token: lidar 帧对应的环视图像token
        min_dist: 用于过滤点云的最小距离

    Returns:
        projected_points: 投影到相机视图上的2D点云
        点云在相机坐标系下的深度值和序号
        点云对应的原始图像

    '''

    projected_points = {}

    # 获取 LiDAR 样本的基本信息，包括传感器标定信息、位姿信息等
    lidar_sample_data = nusc.get('sample_data', lidar_token)

    for channel in surround_view_img_token:
        cam_sample_data = nusc.get('sample_data', surround_view_img_token[channel])

        # 获取 LiDAR 和 Camera 的标定数据以及位姿信息
        lidar_cs_record = nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
        cam_cs_record = nusc.get('calibrated_sensor', cam_sample_data['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', lidar_sample_data['ego_pose_token'])
        cam_pose = nusc.get('ego_pose', cam_sample_data['ego_pose_token'])

        # 加载相机图像
        im = Image.open(osp.join(nusc.dataroot, cam_sample_data['filename']))

        # 1. 从传感器坐标系到车辆坐标系
        rotation_matrix_sensor_to_ego = Quaternion(lidar_cs_record['rotation']).rotation_matrix
        translation_sensor_to_ego = np.array(lidar_cs_record['translation'])
        pc = lidar_data.T  # 转置点云矩阵
        pc = np.dot(rotation_matrix_sensor_to_ego, pc).T  # 旋转
        pc += translation_sensor_to_ego  # 平移

        # 2. 从车辆坐标系到全局坐标系
        rotation_matrix_ego_to_global = Quaternion(lidar_pose['rotation']).rotation_matrix
        translation_ego_to_global = np.array(lidar_pose['translation'])
        pc = pc.T
        pc = np.dot(rotation_matrix_ego_to_global, pc).T
        pc += translation_ego_to_global

        # 3. 从全局坐标系到相机帧的车辆坐标系
        translation_global_to_ego_cam = -np.array(cam_pose['translation'])
        rotation_matrix_global_to_ego_cam = Quaternion(cam_pose['rotation']).rotation_matrix.T
        pc += translation_global_to_ego_cam
        pc = np.dot(rotation_matrix_global_to_ego_cam, pc.T).T

        # 4. 从车辆坐标系转换到相机坐标系
        translation_ego_cam_to_camera = -np.array(cam_cs_record['translation'])
        rotation_matrix_ego_cam_to_camera = Quaternion(cam_cs_record['rotation']).rotation_matrix.T
        pc += translation_ego_cam_to_camera
        pc = np.dot(rotation_matrix_ego_cam_to_camera, pc.T).T

        # 获取点的深度信息（z轴）
        depths = pc[:, 2]

        # 将点云投影到相机视图上
        points_2d = view_points(pc.T[:3, :], np.array(cam_cs_record['camera_intrinsic']), normalize=True)

        # 过滤点云，将不在相机视野范围内的点剔除
        mask = (depths > min_dist) & \
               (points_2d[0, :] > 1) & (points_2d[0, :] < im.size[0] - 1) & \
               (points_2d[1, :] > 1) & (points_2d[1, :] < im.size[1] - 1)

        # 通过 mask 过滤点云、深度和序号
        points_2d = points_2d[:, mask]
        depths = depths[mask]

        # 将点云在激光雷达坐标系下的坐标也保存下来
        pc_lidar_coord = lidar_data[mask]

        # 将结果存储在字典中
        projected_points[channel] = {
            'points': points_2d,
            'points_lidar_coord': pc_lidar_coord,
            'depths': depths,
            'original_img': im
        }

    return projected_points

def view_points_gpu(points: torch.Tensor, view: torch.Tensor, normalize: bool) -> torch.Tensor:
    """
    使用 GPU 上的 torch tensor 将 3D 点投影到 2D 平面上。

    Args:
        points: 形状为 (3, n) 的 tensor，每个点为 (x, y, z)
        view: 投影矩阵，形状可以是 (3, 3) 或 (3, 4)
        normalize: 是否归一化第三个坐标（透视投影时设为 True）

    Returns:
        投影后的点，形状为 (3, n)。当 normalize=True 时，第三个坐标被归一化。
    """
    # 确保 view 的维度不超过 4x4，points 为 (3, n)
    assert view.shape[0] <= 4 and view.shape[1] <= 4, "view 矩阵维度应不超过4x4"
    assert points.shape[0] == 3, "points 应为形状 (3, n) 的 tensor"

    # 获取 points 所在设备与数据类型
    device = points.device
    dtype = points.dtype

    # 构造 4x4 的单位矩阵，并将 view 矩阵填充到对应的子矩阵中
    viewpad = torch.eye(4, device=device, dtype=dtype)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # 将 points 扩展为齐次坐标形式，形状 (4, n)
    ones = torch.ones((1, nbr_points), device=device, dtype=dtype)
    points_hom = torch.cat([points, ones], dim=0)

    # 投影：矩阵乘法
    points_proj = torch.matmul(viewpad, points_hom)  # 形状 (4, n)
    points_proj = points_proj[:3, :]  # 取前3行

    if normalize:
        # 防止除以 0，加上一个微小值
        eps = 1e-6
        points_proj = points_proj / (points_proj[2:3, :] + eps).expand_as(points_proj)

    return points_proj