from tqdm import tqdm
import os
import numpy as np
import pickle
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from utils.nusc_paths import make_nusc_relative_path

CAMERA_CHANNELS = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
                   'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT']

def compute_percentage_in_range(data_frame, t):
    """
    计算 data_frame 中位于 [1-t, 1+t] 区间的数值占总数的百分比。

    参数:
      data_frame: np.ndarray - 当前帧数据，可以是任意形状的数组
      t: float - 区间偏移量

    返回:
      percentage: float - 百分比
    """
    # 计算满足条件的元素（返回一个布尔数组）
    mask = (data_frame >= (1 - t)) & (data_frame <= (1 + t))
    # 计算满足条件的元素数量
    count_in_range = np.sum(mask)
    # 计算总数（所有元素数量）
    total_count = data_frame.size
    
    # 计算百分比，如果 total_count 为0, 避免除以0
    percentage = count_in_range / total_count * 100 if total_count > 0 else 0
    return percentage

def find_matching_camera_sweep_in_nusc(nusc, target_timestamp_range, camera_channel):
    """
    找到在指定时间戳范围内指定相机通道的相机数据
    :param nusc: NuScenes 数据集对象
    :param target_timestamp_range: 目标时间戳范围，单位为毫秒。例如：[timestamp_start - 50, timestamp_end + 50]
    :param camera_channel: 相机通道名称，例如 'CAM_FRONT'
    :return: 匹配到的所有相机数据 list，如果没有找到则返回 None
    """
    sample_data_list = []
    # 遍历所有 sample_data 找到指定时间戳范围内的 sample_data
    for sd in nusc.sample_data:
        # 指定传感器模态为 camera，且通道为指定的相机通道
        if sd['sensor_modality'] == 'camera' and sd['channel'] == camera_channel:
            if (((target_timestamp_range[0] - 50 * 1e3) <=
                    sd['timestamp'] <= (target_timestamp_range[1] + 50 * 1e3))):
                    # and sd['is_key_frame'] == False): # 前后各扩展50ms的时间范围
                sample_data_list.append(sd)

    # 如果找到了匹配的相机数据，则返回
    if len(sample_data_list) > 0:
        return sample_data_list
    else:
        return None

def find_matching_sf_sweep_in_lut(scene_flow_lut, target_timestamp_range):
    """
    找到在指定时间戳范围内的 scene flow 数据
    :param scene_flow_lut: scene flow 数据字典，包含文件夹名和时间戳
    :param target_timestamp_range: 目标时间戳范围，单位为微秒。例如：[timestamp_start, timestamp_end]
    :return: 匹配到的所有 scene flow 数据 list，如果没有找到则返回 None

    Args:
        scene_flow_lut:
    """
    sf_list = []
    # 遍历 scene_flow_lut 找到指定时间戳范围内的 数据
    for sf_data in scene_flow_lut:
        timestamp = sf_data['timestamp']
        if target_timestamp_range[0] <= timestamp <= target_timestamp_range[1]:
            sf_list.append(sf_data)

    if len(sf_list) == 1:
        return sf_list[0]
    elif len(sf_list) > 1:
        print("Warning: Multiple matching scene flow data found.")
        return sf_list[0]
    else:
        return None

# 提取时间戳的函数
def extract_timestamp(folder_name):
    """
    从文件夹名称中提取时间戳，时间戳通常是文件夹名称中的最后一部分
    例如：n015-2018-07-18-11-18-34+0800__LIDAR_TOP__1531884294899270
    """
    try:
        # 文件夹名称是以时间戳结尾的，分割后提取最后的时间戳
        timestamp = folder_name.split('__')[-1]
        return int(timestamp)
    except Exception as e:
        print(f"无法从 {folder_name} 中提取时间戳: {e}")
        return None

def infer_output_variant(scene_flow_path):
    scene_flow_dir = os.path.basename(os.path.normpath(scene_flow_path))
    if scene_flow_dir.startswith('scene_flow_'):
        return scene_flow_dir[len('scene_flow_'):]
    if 'key_frame' in scene_flow_dir:
        return 'key_frames'
    if 'all_frame' in scene_flow_dir:
        return 'all_frames'
    return scene_flow_dir or 'scene_flow'

def load_scene_names(scene_list_file):
    scene_names = []
    with open(scene_list_file, 'r') as file:
        for line in file:
            scene_name = line.strip()
            if not scene_name or scene_name.startswith('#'):
                continue
            scene_names.append(scene_name)
    return scene_names

def resolve_scene_indices(nusc, scene_names):
    scene_index_by_name = {scene['name']: idx for idx, scene in enumerate(nusc.scene)}

    missing_scene_names = [scene_name for scene_name in scene_names if scene_name not in scene_index_by_name]
    if missing_scene_names:
        missing_str = ', '.join(missing_scene_names[:10])
        if len(missing_scene_names) > 10:
            missing_str += ', ...'
        raise ValueError(f"Unknown scene names in scene list: {missing_str}")

    selected_scene_indices = []
    seen_scene_names = set()
    duplicate_scene_names = []
    for scene_name in scene_names:
        if scene_name in seen_scene_names:
            duplicate_scene_names.append(scene_name)
            continue
        seen_scene_names.add(scene_name)
        selected_scene_indices.append(scene_index_by_name[scene_name])

    if duplicate_scene_names:
        duplicate_scene_names = sorted(set(duplicate_scene_names))
        duplicate_str = ', '.join(duplicate_scene_names[:10])
        if len(duplicate_scene_names) > 10:
            duplicate_str += ', ...'
        print(f"Ignoring duplicate scene names from scene list: {duplicate_str}")

    return selected_scene_indices


def resolve_target_scene_indices(nusc, args):
    if args.scene_list_file:
        scene_names = load_scene_names(args.scene_list_file)
        if not scene_names:
            raise ValueError(f"No valid scene names found in {args.scene_list_file}")
        target_scene_indices = {str(scene_idx) for scene_idx in resolve_scene_indices(nusc, scene_names)}
        if args.scene_indices:
            print(
                f"Using custom scene list from {args.scene_list_file}; ignoring --scene_indices={args.scene_indices!r}."
            )
        return target_scene_indices

    if args.scene_indices:
        return {str(scene_idx) for scene_idx in args.scene_indices}

    return None


def collect_scene_flow_folders(scene_flow_path, target_scene_indices=None):
    folders_with_timestamps = []
    for folder_name in os.listdir(scene_flow_path):
        full_path = os.path.join(scene_flow_path, folder_name)
        if not os.path.isdir(full_path):
            continue

        folder = folder_name[:-4] if folder_name.endswith('.pcd') else folder_name
        try:
            scene_indice = folder.split('_')[1]
        except IndexError:
            continue

        if target_scene_indices is not None and scene_indice not in target_scene_indices:
            continue

        timestamp = extract_timestamp(folder)
        if timestamp is None:
            continue

        folders_with_timestamps.append((folder, timestamp, scene_indice))

    folders_with_timestamps.sort(key=lambda x: x[1])
    return folders_with_timestamps


def iter_scene_samples(nusc, scene_index):
    scene = nusc.scene[scene_index]
    sample = nusc.get('sample', scene['first_sample_token'])
    samples = []
    while True:
        samples.append(sample)
        if sample['next'] == '':
            break
        sample = nusc.get('sample', sample['next'])
    return samples


def build_scene_flow_lut_global(nusc, folders_with_timestamps):
    lidar_sweep_not_found_in_nusc_count = 0
    scene_flow_lut = []
    for folder, timestamp, scene_indice in folders_with_timestamps:
        lidar_sample_data = find_lidar_top_sample_data(nusc, timestamp)
        if lidar_sample_data:
            lidar_record = dict(lidar_sample_data)
            lidar_record['folder_name'] = folder
            lidar_record['scene_indice'] = scene_indice
            scene_flow_lut.append(lidar_record)
        else:
            lidar_sweep_not_found_in_nusc_count += 1
    return scene_flow_lut, lidar_sweep_not_found_in_nusc_count


def build_scene_flow_lut_single_scene(nusc, folders_with_timestamps, scene_index):
    scene_samples = iter_scene_samples(nusc, scene_index)
    lidar_by_timestamp = {}
    for sample in scene_samples:
        lidar_sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_by_timestamp[lidar_sample_data['timestamp']] = lidar_sample_data

    lidar_sweep_not_found_in_nusc_count = 0
    scene_flow_lut = []
    for folder, timestamp, scene_indice in folders_with_timestamps:
        lidar_sample_data = lidar_by_timestamp.get(timestamp)
        if lidar_sample_data:
            lidar_record = dict(lidar_sample_data)
            lidar_record['folder_name'] = folder
            lidar_record['scene_indice'] = scene_indice
            scene_flow_lut.append(lidar_record)
        else:
            lidar_sweep_not_found_in_nusc_count += 1

    return scene_flow_lut, lidar_sweep_not_found_in_nusc_count, scene_samples


def build_matched_frame_records_single_scene(nusc, scene_samples, scene_flow_lut):
    scene_flow_by_timestamp = {record['timestamp']: record for record in scene_flow_lut}
    matched_frame_records = []

    for sample in scene_samples:
        lidar_sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_record = scene_flow_by_timestamp.get(lidar_sample_data['timestamp'])
        if lidar_record is None:
            continue

        camera_data_combined = {}
        is_complete_surround_view = True
        for camera_channel in CAMERA_CHANNELS:
            camera_token = sample['data'].get(camera_channel)
            if not camera_token:
                is_complete_surround_view = False
                break
            camera_data_combined[camera_channel] = nusc.get('sample_data', camera_token)

        if not is_complete_surround_view:
            continue

        matched_frame_records.append({
            'camera': camera_data_combined,
            'lidar': lidar_record
        })

    matched_frame_records.sort(key=lambda x: x['camera']['CAM_BACK_LEFT']['timestamp'])
    return matched_frame_records

# 获取 LIDAR_TOP 对应的 sample data
def find_lidar_top_sample_data(nusc, timestamp):
    """
    根据时间戳在 NuScenes 数据集中查找对应的 LIDAR_TOP 的 sample data
    """
    for sd in nusc.sample_data:
        if sd['sensor_modality'] == 'lidar' and sd['timestamp'] == timestamp:
            return sd
    return None

def find_surround_view_images_for_same_frame(camera_data_dict, target_timestamp, delta_timestamp):
    for cam_data in camera_data_dict:
        timestamp = cam_data['timestamp']
        if 0 < (target_timestamp - timestamp) <= delta_timestamp:
            return cam_data
    return None

def range_projection(points, scales, risk_score, H=160, W=1920, fov_up=10.0, fov_down=-30.0):
    """
    将 3D 点云投影到 2D 球面范围图像。

    参数:
    - points (np.ndarray): 点云数据，形状为 [m, 3]，每行表示 (x, y, z)。
    - scales (np.ndarray): 尺度数据，形状为 [m,]，每个值表示对应点云处的 scale。
    - risk_score (np.ndarray): 风险系数，形状为 [m,]，每个值表示相邻帧的运动矢量指向原点的分量,
                               正值表示物体朝向原点运动，负值表示远离原点运动。
    - H (int): 投影图像的高度（像素）。默认值为 160。
    - W (int): 投影图像的宽度（像素）。默认值为 1920。
    - fov_up (float): 向上的视场角（度）。默认值为 8.0。
    - fov_down (float): 向下的视场角（度）。默认值为 -15.0。

    返回:
    - proj_range (np.ndarray): 投影的深度图像，形状为 [H, W]。
    - proj_scale (np.ndarray): 投影的尺度图像，形状为 [H, W]。
    - proj_risk_score (np.ndarray): 风险系数，形状为 [H, W]。
    - proj_xyz (np.ndarray): 投影的 3D 坐标图像，形状为 [H, W, 3]。
    - proj_idx (np.ndarray): 投影的点索引图像，形状为 [H, W]。
    - proj_mask (np.ndarray): 掩码，形状为 [H, W]，标记哪些像素包含有效点。
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("Points should be a numpy array.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points should have shape [m, 3].")

    # 初始化投影图像
    proj_range = np.full((H, W), -1, dtype=np.float32)       # 深度图
    proj_scale = np.full((H, W), -1, dtype=np.float32)       # 尺度图
    proj_risk_score = np.full((H, W), 0.0, dtype=np.float32)     # 风险系数图
    proj_xyz = np.full((H, W, 3), -1, dtype=np.float32)     # 3D 坐标图
    proj_idx = np.full((H, W), -1, dtype=np.int32)          # 点索引图
    proj_mask = np.zeros((H, W), dtype=np.int32)            # 掩码

    # 将视场角转换为弧度
    fov_up_rad = fov_up / 180.0 * np.pi      # 向上视场角（弧度）
    fov_down_rad = fov_down / 180.0 * np.pi  # 向下视场角（弧度）
    fov = abs(fov_down_rad) + abs(fov_up_rad)  # 总垂直视场角（弧度）

    # 计算每个点的深度（距离）
    depth = np.linalg.norm(points, axis=1)

    # 分离点云的各个坐标分量
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # nuscenes 的坐标系定义: y 前、x 右、z 上 
    # 这里使用的坐标系定义： x 前、y 左、z 上 （由 NuScenes 坐标系顺时针旋转得到）
    # 因此投影得到的range image中，车辆正前方不在图像中心，而是在图像的左侧1/4处
    # 这样 range image 从左到右，可以依次与环视图像的 左前、前、右前、右后、后、左后 对齐

    # 计算每个点的水平角（yaw）和垂直角（pitch）
    yaw = -np.arctan2(scan_y, scan_x)                  # 水平角
    pitch = np.arcsin(scan_z / depth)                  # 垂直角

    # 将角度归一化到 [0, 1] 范围
    proj_x = 0.5 * (yaw / np.pi + 1.0)                 # [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down_rad)) / fov  # [0.0, 1.0]

    # 缩放到图像尺寸
    proj_x *= W                                         # [0, W]
    proj_y *= H                                         # [0, H]

    # 向下取整并限制在图像边界内
    proj_x = np.floor(proj_x).astype(np.int32)
    proj_x = np.clip(proj_x, 0, W - 1)
    proj_y = np.floor(proj_y).astype(np.int32)
    proj_y = np.clip(proj_y, 0, H - 1)

    # 按深度从远到近排序（确保近的点覆盖远的点）
    order = np.argsort(depth)[::-1]
    depth_sorted = depth[order]
    scale_sorted = scales[order]
    risk_score_sorted = risk_score[order]
    points_sorted = points[order]
    proj_x_sorted = proj_x[order]
    proj_y_sorted = proj_y[order]
    indices_sorted = order

    # 赋值到投影图像
    proj_range[proj_y_sorted, proj_x_sorted] = depth_sorted
    proj_scale[proj_y_sorted, proj_x_sorted] = scale_sorted
    proj_risk_score[proj_y_sorted, proj_x_sorted] = risk_score_sorted
    proj_xyz[proj_y_sorted, proj_x_sorted] = points_sorted
    proj_idx[proj_y_sorted, proj_x_sorted] = indices_sorted

    # 创建掩码
    proj_mask = (proj_idx >= 0).astype(np.int32)

    return proj_range, proj_scale, proj_risk_score, proj_xyz, proj_idx, proj_mask

def compute_radical_angle(points_prev,
                          points_curr,
                          motion_thresh=1e-3):
    """
    计算每个点在 x-y 平面上，从 points_prev 到 points_curr 的运动矢量
    与从 points_prev 到原点连线方向之间的夹角（单位为弧度）。

    对于运动幅度小于 motion_thresh 的点，视作“静止”，其角度设为 np.nan。

    参数:
      points_prev: np.ndarray, 形状 (n, 3)
      points_curr: np.ndarray, 形状 (n, 3)
      motion_thresh: float
          运动矢量长度的最小阈值，小于此阈值认为该点静止，不计算角度。

    返回:
      angles: np.ndarray, 形状 (n,)
          每个点的夹角（单位为弧度），取值范围 [0, π]。静止点为 np.nan。
    """
    # 提取 x-y 分量
    p_prev_xy = points_prev[:, :2]  # (n, 2)
    p_curr_xy = points_curr[:, :2]

    # 运动矢量
    motion_xy = p_curr_xy - p_prev_xy  # (n, 2)
    motion_mag = np.linalg.norm(motion_xy, axis=1)  # (n,)

    # 标记有效运动点
    valid = motion_mag >= motion_thresh
    n = points_prev.shape[0]
    angles = np.full(n, np.pi/2, dtype=np.float32)

    if not np.any(valid):
        return angles  # 全部静止

    # 只在有效点上计算
    to_origin = -p_prev_xy[valid]  # (m, 2), m = sum(valid)

    dot = np.einsum('ij,ij->i', motion_xy[valid], to_origin)  # (m,)
    norm_motion = motion_mag[valid]
    norm_to_origin = np.linalg.norm(to_origin, axis=1)

    # 防止除零
    eps = 1e-6
    norm_to_origin = np.maximum(norm_to_origin, eps)

    cos_theta = dot / (norm_motion * norm_to_origin)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles_valid = np.arccos(cos_theta)  # (m,)

    angles[valid] = angles_valid
    return angles

def compute_radical_component(points_prev, points_curr):
    """
    计算每个点在 x-y 平面上，从 points_prev 到 points_curr 的运动矢量
    在从 points_prev 到原点连线方向上的分量。
    
    参数:
      points_prev: np.ndarray, 形状 (n, 3)
      points_curr: np.ndarray, 形状 (n, 3)

    返回:
      radial_components: np.ndarray, 形状 (n,)
          每个点的径向分量 (x-y 平面)，正值表示物体朝向原点运动，负值表示远离原点运动。
    """
    # 提取 x-y 分量
    p_prev_xy = points_prev[:, :2]  # 形状 (n,2)
    p_curr_xy = points_curr[:, :2]

    # 计算 x-y 平面上的运动矢量
    motion_xy = p_curr_xy - p_prev_xy  # (n,2)

    # 计算 p_prev_xy 到原点的单位向量
    distances = np.linalg.norm(p_prev_xy, axis=1, keepdims=True)
    eps = 1e-6
    distances = np.where(distances < eps, eps, distances)
    # 从 p_prev_xy 到原点的单位向量
    unit_vectors = -p_prev_xy / distances

    # 计算运动矢量在该单位向量方向上的分量
    radical_components = np.einsum('ij,ij->i', motion_xy, unit_vectors)
    return radical_components

def main(args):

    split_display_name = {
        'train': 'Train',
        'val': 'Val',
        'test': 'Test',
    }.get(args.trainval_test_split)
    if split_display_name is None:
        raise ValueError("Invalid trainval_test_split value. Must be 'train', 'val' or 'test'.")
    print(f"Creating {split_display_name} Info ...")

    output_variant = infer_output_variant(args.scene_flow_path)
    range_image_width = args.image_size[1] * 6
    range_image_dir_name = (
        f'range_image_{output_variant}_{args.image_size[0]}_{range_image_width}_fov_{args.fov[0]}_{args.fov[1]}'
    )
    pkl_file_name = (
        f'nusc_{args.trainval_test_split}_infos_{output_variant}_{args.image_size[0]}_{range_image_width}_fov_'
        f'{args.fov[0]}_{args.fov[1]}.pkl'
    )

    target_scene_indices = resolve_target_scene_indices(nusc, args)
    if target_scene_indices is not None:
        print(f"目标场景过滤集合: {sorted(target_scene_indices)}")

    ################################################ 创建场景流 LUT ################################################
    folders_with_timestamps = collect_scene_flow_folders(args.scene_flow_path, target_scene_indices)
    if not folders_with_timestamps:
        raise RuntimeError(
            f"No scene flow folders matched the requested scenes under {args.scene_flow_path}"
        )

    single_scene_fast_path = (
        target_scene_indices is not None
        and len(target_scene_indices) == 1
        and output_variant == 'key_frames'
    )
    scene_samples = None
    if single_scene_fast_path:
        target_scene_index = int(next(iter(target_scene_indices)))
        print(
            f"Using single-scene fast path for scene index {target_scene_index} "
            f"({nusc.scene[target_scene_index]['name']})."
        )
        scene_flow_lut, lidar_sweep_not_found_in_nusc_count, scene_samples = build_scene_flow_lut_single_scene(
            nusc, folders_with_timestamps, target_scene_index
        )
    else:
        scene_flow_lut, lidar_sweep_not_found_in_nusc_count = build_scene_flow_lut_global(
            nusc, folders_with_timestamps
        )

    print(f"场景流 LUT 已创建完成，已按照时间戳排序，共有 {len(scene_flow_lut)} 条数据")

    if not scene_flow_lut:
        raise RuntimeError(f"No scene flow data found under {args.scene_flow_path}")

    if lidar_sweep_not_found_in_nusc_count > 0:
        print(f"在 NuScenes 数据集中找不到的 LIDAR sweeps 数量: {lidar_sweep_not_found_in_nusc_count}")

    ################################################ 创建环视相机数据 LUT ################################################
    # 组合并筛选相机数据
    # 1. 将 6 channel 的相机数据按照 ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
    #                    'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT'] 的顺序组合；
    #    CAM_BACK_LEFT的时间戳最接近LIDAR_TOP的时间戳，可以以此为基准进行组合；
    #    CAM_BACK_LEFT的时间戳最大，CAM_FRONT_LEFT的时间戳最小；
    # 2. 剔除其中不足 6 帧的数据

    # 提取时间戳信息
    lidar_timestamps = [record['timestamp'] for record in scene_flow_lut]

    if single_scene_fast_path:
        matched_frame_records = build_matched_frame_records_single_scene(nusc, scene_samples, scene_flow_lut)
    else:
        # 获取 LIDAR_TOP 时间戳范围内的相机数据
        camera_data_within_timestamp_range = {}
        timestamp_range = [lidar_timestamps[0], lidar_timestamps[-1]]
        for camera_channel in CAMERA_CHANNELS:
            camera_data_within_timestamp_range[camera_channel] = \
                find_matching_camera_sweep_in_nusc(nusc, timestamp_range, camera_channel)

        matched_frame_records = []

        # 对6个相机通道的数据按照时间戳进行排序
        for camera_channel in CAMERA_CHANNELS:
            camera_data = camera_data_within_timestamp_range[camera_channel] or []
            camera_data.sort(key=lambda x: x['timestamp'])
            camera_data_within_timestamp_range[camera_channel] = camera_data

        # 以 CAM_BACK_LEFT 为基准，向前查找属于同一组数据的 6 帧图像
        for camera_data_bl in camera_data_within_timestamp_range['CAM_BACK_LEFT']:
            # 检查 CAM_BACK_LEFT 时间戳前后 25ms 内是否有 LIDAR_TOP 数据
            # 如果没有，说明这一组图像没有对应的 LIDAR_TOP 数据，跳过
            timestamp_range = [camera_data_bl['timestamp'] - 25 * 1e3, camera_data_bl['timestamp'] + 25 * 1e3]
            lidar_data = find_matching_sf_sweep_in_lut(scene_flow_lut, timestamp_range)
            if not lidar_data:
                continue

            camera_data_combined = {}
            camera_data_combined['CAM_BACK_LEFT'] = camera_data_bl
            target_timestamp = camera_data_bl['timestamp']

            # 查找 CAM_BACK_LEFT 前 50ms 内的相机数据
            for camera_channel in ['CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT']:
                # 同一组数据的 6 帧图像时间戳差值在 50 ms 以内
                matched_cam = find_surround_view_images_for_same_frame(
                    camera_data_within_timestamp_range[camera_channel], target_timestamp, delta_timestamp=50*1e3)
                if matched_cam:
                    camera_data_combined[camera_channel] = matched_cam

            # 检查 camera_data_combined 的 6 个相机通道中是否都有数据
            is_complete_surround_view = (len(camera_data_combined) == 6)
            if is_complete_surround_view:
                matched_frame_records.append({
                    'camera': camera_data_combined,
                    'lidar': lidar_data
                })

    # 保持相机组与对应的场景流样本一一绑定后再排序，避免两侧分别排序造成错位。
    matched_frame_records.sort(key=lambda x: x['camera']['CAM_BACK_LEFT']['timestamp'])

    if not matched_frame_records:
        raise RuntimeError("No complete surround-view camera groups matched to scene-flow data.")

    # 相邻两组数据进行组合，打包成可以直接用于训练的格式
    trainval_test_infos = []
    for i in tqdm(range(1, len(matched_frame_records)), desc="Combining camera and scene flow data"):
        # 取出连续两组数据
        previous_frame_record = matched_frame_records[i - 1]
        current_frame_record = matched_frame_records[i]
        previous_surround_view_data = previous_frame_record['camera']
        current_surround_view_data = current_frame_record['camera']

        # 取出时间戳信息
        camera_timestamp_1 = previous_surround_view_data['CAM_BACK_LEFT']['timestamp']
        camera_timestamp_2 = current_surround_view_data['CAM_BACK_LEFT']['timestamp']

        # 计算时间戳差值
        time_diff_cam = camera_timestamp_2 - camera_timestamp_1

        # 取出 scene flow 数据
        previous_sf_record = previous_frame_record['lidar']
        current_sf_record = current_frame_record['lidar']

        # 禁止跨 scene 组合相邻样本。
        if previous_sf_record['scene_indice'] != current_sf_record['scene_indice']:
            continue

        time_diff_sf = current_sf_record['timestamp'] - previous_sf_record['timestamp']

        if time_diff_sf <= 0:
            continue

        # 只保留与相邻 scene-flow 时间差一致的图像对。
        if abs(time_diff_cam - time_diff_sf) > 50 * 1e3:
            continue

        time_diff_ratio = time_diff_cam / time_diff_sf

        scene_indice = current_sf_record['scene_indice']

        # if scene_indice not in ['20', '23', '24']:
        #     continue

        # 读取 scene flow 数据
        current_scene_flow_path = os.path.join(args.scene_flow_path, current_sf_record['folder_name'])
        points_prev_path = os.path.join(current_scene_flow_path, 'pc_prev.npy')
        points_curr_path = os.path.join(current_scene_flow_path, 'pc_curr.npy')
        if (not os.path.exists(points_prev_path)
                or not os.path.exists(points_curr_path)):
            continue

        points_prev = np.load(points_prev_path)
        points_curr = np.load(points_curr_path)
        # 对 scene flow 数据进行时间插值/外推，使其对齐当前图像对的时间间隔。
        points_prev_aligned = points_curr - time_diff_ratio * (points_curr - points_prev)

        # 计算激光雷达坐标系下的尺度
        # depth 为 x-y 平面上距原点的距离
        depth_xy_prev = np.linalg.norm(points_prev_aligned[:, :2], axis=1)
        depth_xy_curr = np.linalg.norm(points_curr[:, :2], axis=1)

        # 避免除以零
        depth_xy_prev[depth_xy_prev == 0] = 1e-6
        scales = depth_xy_curr / depth_xy_prev

        # 用径向速度作为风险系数
        # risk_score = compute_radical_component(points_prev, points_curr)
        # 用运动矢量和径向之间的角度作为风险系数
        risk_score = compute_radical_angle(points_prev_aligned, points_curr, motion_thresh=0.005)

        # Range Projection，将 scale 和 risk_score 投影到 Range Image 上
        proj_range, proj_scale, proj_risk_score, proj_xyz, proj_idx, proj_mask = range_projection(
            points_prev_aligned,
            scales,
            risk_score,
            H=args.image_size[0],
            W=range_image_width,
            fov_up=args.fov[0], # nuscenes 使用的 LiDAR fov
            fov_down=-args.fov[1]  # nuscenes 使用的 LiDAR fov
        )

        proj_range_, proj_scale_, proj_risk_score_, proj_xyz_, proj_idx_, proj_mask_ = range_projection(
            points_curr,
            scales,
            risk_score,
            H=args.image_size[0],
            W=range_image_width,
            fov_up=args.fov[0], # nuscenes 使用的 LiDAR fov
            fov_down=-args.fov[1]  # nuscenes 使用的 LiDAR fov
        )

        range_image_prev = {
            'depth': proj_range,
            'scale': proj_scale,
            'risk_score': proj_risk_score,
            'xyz': proj_xyz,
            'idx': proj_idx,
            'mask': proj_mask
        }

        range_image_curr = {
            'depth': proj_range_,
            'scale': proj_scale_,
            'risk_score': proj_risk_score_,
            'xyz': proj_xyz_,
            'idx': proj_idx_,
            'mask': proj_mask_
        }

        range_image_save_path = os.path.join(args.gt_map_save_path,
                                             range_image_dir_name,
                                             current_sf_record['folder_name'])
        os.makedirs(range_image_save_path, exist_ok=True)
        
        # 保存 range image 为 npy 文件
        np.save(os.path.join(range_image_save_path, 'range_image_prev.npy'), range_image_prev)
        np.save(os.path.join(range_image_save_path, 'range_image_curr.npy'), range_image_curr)

        # 读取 LiDAR 和 Camera 对应的标定信息，作为 sensor_metas 传入网络
        prev_camera_calib = {}
        prev_camera_pose = {}
        # prev 帧的标定信息
        for camera_channel in CAMERA_CHANNELS:
            camera_data = previous_surround_view_data[camera_channel]
            prev_camera_calib[camera_channel] = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
            prev_camera_pose[camera_channel] = nusc.get('ego_pose', camera_data['ego_pose_token'])
        lidar_calib = nusc.get('calibrated_sensor', previous_sf_record['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', previous_sf_record['ego_pose_token'])
        sensor_metas_prev = {
            'lidar':{
                'calibrated_sensor': lidar_calib,
                'ego_pose': lidar_pose,
            },
            'camera':{
                'calibrated_sensor': prev_camera_calib,
                'ego_pose': prev_camera_pose,
            }
        }

        # curr 帧的标定信息
        curr_camera_calib = {}
        curr_camera_pose = {}
        for camera_channel in CAMERA_CHANNELS:
            camera_data = current_surround_view_data[camera_channel]
            curr_camera_calib[camera_channel] = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
            curr_camera_pose[camera_channel] = nusc.get('ego_pose', camera_data['ego_pose_token'])

        lidar_calib = nusc.get('calibrated_sensor', current_sf_record['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', current_sf_record['ego_pose_token'])
        sensor_metas_curr = {
            'lidar':{
                'calibrated_sensor': lidar_calib,
                'ego_pose': lidar_pose,
            },
            'camera':{
                'calibrated_sensor': curr_camera_calib,
                'ego_pose': curr_camera_pose,
            }
        }

        info = {
            'prev_camera_data': previous_surround_view_data,
            'curr_camera_data': current_surround_view_data,
            'prev_lidar_data': previous_sf_record,
            'curr_lidar_data': current_sf_record,
            'sensor_metas_prev': sensor_metas_prev,
            'sensor_metas_curr': sensor_metas_curr,
            'gt_map_path': make_nusc_relative_path(range_image_save_path),
            'scene_flow_path': make_nusc_relative_path(current_scene_flow_path),
            'scene_indice': scene_indice,
            'time_diff_cam_us': time_diff_cam,
            'time_diff_sf_us': time_diff_sf,
            'time_diff_ratio': time_diff_ratio,
        }
        
        trainval_test_infos.append(info)

        ###################################### 可视化 ######################################
        
        if args.risk_score_map_vis:
            risk_score_display = np.copy(proj_risk_score) - np.pi/2
            risk_score_display[proj_mask == 0] = 0.0  # 将无效像素设为0
            
            risk_score_vis_save_path = os.path.join(args.vis_dir, 'risk_score_map',
                                                f"{output_variant}_{args.image_size[0]}_{range_image_width}_fov_{args.fov[0]}_{args.fov[1]}_theta")
            
            os.makedirs(risk_score_vis_save_path, exist_ok=True)

            risk_score_vis_name = f"scene_{scene_indice}_risk_score_map_{i}.png"

            plt.imsave(os.path.join(risk_score_vis_save_path, risk_score_vis_name),
                       -risk_score_display, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)
            
        if args.scale_map_vis:
            scale_display = np.copy(proj_scale)

            valid_mask = scale_display >= 0

            # 将用于可视化的尺度值裁切到 [0.5, 1.5] 范围
            scale_display[valid_mask] = np.clip(scale_display[valid_mask], 0.5, 1.5)

            # 将尺度的分界线从 1 移至 0（scale - 1）
            deviations = np.zeros_like(scale_display, dtype=np.float32)
            deviations[valid_mask] = scale_display[valid_mask] - 1.0

            # 分别处理大于0和小于0的部分
            pos_mask = deviations > 0
            neg_mask = deviations < 0

            # 初始化归一化后的显示数组
            normalized_display = np.zeros_like(deviations, dtype=np.float32)

            # 处理大于1的尺度
            if np.any(pos_mask):
                pos_devs = deviations[pos_mask]
                pos_max = pos_devs.max()
                pos_min = pos_devs.min()
                if pos_max > pos_min >= 0:
                    normalized_display[pos_mask] = (pos_devs - pos_min) / (pos_max - pos_min)  # 归一化到 [0,1]
                else:
                    normalized_display[pos_mask] = 0.0  # 如果没有变化，设为0

            # 处理小于1的尺度
            if np.any(neg_mask):
                neg_devs = deviations[neg_mask]
                neg_max = neg_devs.max()
                neg_min = neg_devs.min()
                if neg_min < neg_max <= 0:
                    normalized_display[neg_mask] = (neg_devs - neg_min) / (neg_max - neg_min) - 1.0  # 归一化到 [-1,0]
                    # normalized_display[neg_mask] = neg_devs / abs(neg_min)  # 归一化到 [-1,0]
                else:
                    normalized_display[neg_mask] = 0.0  # 如果没有变化，设为0

            # # 对无效像素赋值为0
            normalized_display[~valid_mask] = 0.0
            
            scale_map_vis_save_path = os.path.join(args.vis_dir, 'scale_map',
                                                f"{output_variant}_{args.image_size[0]}_{range_image_width}_fov_{args.fov[0]}_{args.fov[1]}")
            os.makedirs(scale_map_vis_save_path, exist_ok=True)

            out_name = f"scene_{scene_indice}_scale_map_{i}.png"
            out_path = os.path.join(scale_map_vis_save_path, out_name)

            plt.imsave(out_path, -normalized_display, cmap='seismic', vmin=-1, vmax=1)

        # TODO: 处理 depth map 的可视化:
        if args.depth_map_vis:
            raise NotImplementedError
        ###################################### 可视化 ######################################
    
    os.makedirs(args.pkl_save_path, exist_ok=True)
    print(f"Succesfully created (Surround View Images Pair & Lidar Sample Data Pair & Scene Flow & Scale Map & Depth Map & Risk Score Map). \
          Sorted by timestamp. {len(trainval_test_infos)} items in total.")

    with open(os.path.join(args.pkl_save_path, pkl_file_name), 'wb') as f:
        pickle.dump(trainval_test_infos, f)
    print(f"Saved {pkl_file_name} to {args.pkl_save_path}")

if __name__ == "__main__":
    # NuScenes 数据集路径
    nusc = NuScenes(version='v1.0-trainval', dataroot='./Datasets/nuscenes',
                    verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_flow_path', type=str, default='./Datasets/nuscenes/0_scene_flow/scene_flow_key_frames',
                       help='Path to scene flow data')
    parser.add_argument('--gt_map_save_path', type=str, default='./Datasets/nuscenes/1_gt_map',
                       help='Path to save ground truth, including scale map, depth map and risk score map')
    parser.add_argument('--pkl_save_path', type=str, default='./Datasets/nuscenes/2_trainval_test_infos',
                       help='Path to save train/val/test info pickle, including camera sample data, lidar sample data, sensor metas, gt map path, etc.')
    parser.add_argument('--vis_dir', type=str, default='./Datasets/nuscenes/3_visualization',
                       help='Path to save visualization outputs')
    parser.add_argument('--image_size', default=[160, 320], type=int, nargs='+',
                       help='image size for training')
    parser.add_argument('--fov', default=[10, 20], type=int, nargs='+',
                       help='LiDAR fov, [fov_up, fov_down], in degree')
    parser.add_argument('--trainval_test_split', default='train', type=str,
                       help='Output split name, can be train, val or test')
    parser.add_argument('--scene_list_file', default=None, type=str,
                       help='Optional text file containing one scene name per line, e.g. scene-0001.')
    parser.add_argument('--scene_indices', default=None, type=str, nargs='+',
                       help='Optional scene indices to filter existing scene_flow_key_frames, e.g. --scene_indices 0 1')
    # parser.add_argument('--save_gt', default=False, type=bool,
    #                    help='Flag to control whether to save ground truth map (including scale map, depth map and risk score map)')
    parser.add_argument('--scale_map_vis', action='store_true',
                       help='Flag to control whether to save the scale map visualization results')
    parser.add_argument('--depth_map_vis', action='store_true',
                       help='Flag to control whether to save the depth map visualization results')
    parser.add_argument('--risk_score_map_vis', action='store_true',
                       help='Flag to control whether to save the risk score map visualization results')
    
    args = parser.parse_args()

    main(args=args)
