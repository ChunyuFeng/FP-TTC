from tqdm import tqdm
import os
import numpy as np
import pickle
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import argparse
from PIL import Image

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

    if args.trainval_test_split == 'trainval':
        print("Creating TrainVal Info ...")
    elif args.trainval_test_split == 'test':
        print("Creating Test Info ...")
    else:
        raise ValueError("Invalid trainval_test_split value. Must be 'trainval' or 'test'.")

    ################################################ 创建场景流 LUT ################################################
    # folder name 包含了时间戳信息，读取所有的文件夹名称
    folder_names = [folder_name[:-4] if folder_name.endswith('.pcd') else folder_name
                    for folder_name in os.listdir(args.scene_flow_path)
                    if os.path.isdir(os.path.join(args.scene_flow_path, folder_name))]

    # 提取folder name中的时间戳信息，并将timestamp和folder name绑定在一起
    folders_with_timestamps = [(folder_name, extract_timestamp(folder_name)) for folder_name in folder_names if
                               extract_timestamp(folder_name) is not None]

    # 按照时间戳进行排序
    folders_with_timestamps.sort(key=lambda x: x[1])

    # 提取排序后的文件夹名称和对应的 LIDAR_TOP token
    # token信息暂时没有使用
    lidar_sweep_not_found_in_nusc = []
    lidar_sweep_not_found_in_nusc_count = 0
    scene_flow_lut = []
    # count = 0
    for folder, timestamp in folders_with_timestamps:
        lidar_sample_data = find_lidar_top_sample_data(nusc, timestamp)
        scene_indice = folder.split('_')[1]
        if lidar_sample_data:
            lidar_sample_data['folder_name'] = folder
            lidar_sample_data['scene_indice'] = scene_indice
            scene_flow_lut.append(lidar_sample_data)
        else:
            # lidar_sweep_not_found_in_nusc.append(folder)
            lidar_sweep_not_found_in_nusc_count += 1
    print(f"场景流 LUT 已创建完成，已按照时间戳排序，共有 {len(scene_flow_lut)} 条数据")

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

    # 定义6个相机通道
    camera_channels = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
                       'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT']

    # 获取 LIDAR_TOP 时间戳范围内的相机数据
    camera_data_within_timestamp_range = {}
    timestamp_range = [lidar_timestamps[0], lidar_timestamps[-1]]
    for camera_channel in camera_channels:
        camera_data_within_timestamp_range[camera_channel] = \
            find_matching_camera_sweep_in_nusc(nusc, timestamp_range, camera_channel)

    camera_lut = []
    # nuscenes 数据集中，激光雷达数据的频率为20hz，相机数据的频率为12hz
    # 保留与图像数据对应的激光雷达数据
    lidar_data_matched_with_images = []

    # camera_data_dict = {}
    # for camera_channel, camera_data in camera_data_within_timestamp_range.items():
    #     camera_data_dict[camera_channel] = []
    #     for data in camera_data:
    #         camera_data_dict[camera_channel].append(data)

    # 对6个相机通道的数据按照时间戳进行排序
    for camera_channel in camera_channels:
        camera_data_within_timestamp_range[camera_channel].sort(key=lambda x: x['timestamp'])

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
            camera_lut.append(camera_data_combined)
            lidar_data_matched_with_images.append(lidar_data)

    # 对组合好的6通道相机数据按照时间戳进行排序
    camera_lut = sorted(camera_lut, key=lambda x: x['CAM_BACK_LEFT']['timestamp'])
    lidar_data_matched_with_images = sorted(lidar_data_matched_with_images, key=lambda x: x['timestamp'])

    # 相邻两组数据进行组合，打包成可以直接用于训练的格式
    trainval_test_infos = []
    for i in tqdm(range(1, len(camera_lut)), desc="Combining camera and scene flow data"):
        # 取出连续两组数据
        previous_surround_view_data = camera_lut[i-1]
        current_surround_view_data = camera_lut[i]

        # 取出时间戳信息
        camera_timestamp_1 = previous_surround_view_data['CAM_BACK_LEFT']['timestamp']
        camera_timestamp_2 = current_surround_view_data['CAM_BACK_LEFT']['timestamp']

        # 计算时间戳差值
        time_diff_cam = camera_timestamp_2 - camera_timestamp_1

        # 每秒内的数据分布:
        # LiDAR  20hz: | x x x x x x x x x x x x x x x x x x x x |
        # Camera 12hz: | x   x   x x   x   x x   x   x x   x   x |  
        #              | x<->x<->x x<->x<->x x<->x<->x x<->x<->x |  只选择100ms间隔的图像对
        # 如果时间戳差值小于 75 毫秒或者大于 125 毫秒，则跳过
        # 50 ms 为激光雷达的采样间隔，25 ms 为余量
        if time_diff_cam > (50*2+25) * 1e3 or time_diff_cam < (50*1+25) * 1e3:
            continue

        # 相邻帧图像数据的时间戳差值为 n*50 ms
        time_diff_ratio = round(time_diff_cam / (50*1e3))

        # 取出 scene flow 数据
        previous_sf_record = lidar_data_matched_with_images[i-1]
        current_sf_record = lidar_data_matched_with_images[i]

        scene_indice = current_sf_record['scene_indice']

        if scene_indice not in ['11', '12', '13', '14', '15']:
            continue

        # 读取 scene flow 数据
        if (not os.path.exists(os.path.join(args.scene_flow_path, current_sf_record['folder_name'], 'pc_prev.npy'))
                or not os.path.exists(os.path.join(args.scene_flow_path, current_sf_record['folder_name'], 'pc_curr.npy'))):
            i = i + 1
            continue
        points_prev = np.load(os.path.join(args.scene_flow_path, current_sf_record['folder_name'], 'pc_prev.npy'))
        points_curr = np.load(os.path.join(args.scene_flow_path, current_sf_record['folder_name'], 'pc_curr.npy'))
        # 对 scene flow 数据进行时间插值
        # sf = pc3 - pc1; sf' = time_diff_ratio * sf; pc3' = pc1 + sf'
        # => pc3' = pc1 + time_diff_ratio * (pc3 - pc1)
        # => pc3' = time_diff_ratio * pc3 + (1 - time_diff_ratio) * pc1
        points_prev = points_curr - time_diff_ratio * (points_curr - points_prev)
        # points_curr = time_diff_ratio * points_curr + (1 - time_diff_ratio) * points_prev

        # 计算激光雷达坐标系下的尺度
        # depth 为 x-y 平面上距原点的距离
        depth_xy_prev = np.linalg.norm(points_prev[:, :2], axis=1)
        depth_xy_curr = np.linalg.norm(points_curr[:, :2], axis=1)

        # 避免除以零
        depth_xy_prev[depth_xy_prev == 0] = 1e-6
        scales = depth_xy_curr / depth_xy_prev

        risk_score = compute_radical_component(points_prev, points_curr)

        # Range Projection，将 scale 和 risk_score 投影到 Range Image 上
        proj_range, proj_scale, proj_risk_score, proj_xyz, proj_idx, proj_mask = range_projection(
            points_prev,
            scales,
            risk_score,
            H=args.image_size[0],
            W=args.image_size[1]*6,
            fov_up=args.fov[0], # nuscenes 使用的 LiDAR fov
            fov_down=-args.fov[1]  # nuscenes 使用的 LiDAR fov
        )

        proj_range_, proj_scale_, proj_risk_score_, proj_xyz_, proj_idx_, proj_mask_ = range_projection(
            points_curr,
            scales,
            risk_score,
            H=args.image_size[0],
            W=args.image_size[1]*6,
            fov_up=args.fov[0], # nuscenes 使用的 LiDAR fov
            fov_down=-args.fov[1]  # nuscenes 使用的 LiDAR fov
        )

        # # [TODO] ONEBEV 的 stitch 函数，仍然有重叠，且没有与 range image 对齐，需要改进
        # image_list = []
        # cam_info_list = []
        # camera_channel_stitch = [
        #     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT',
        #     'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT'
        # ]
        # for camera_channel in camera_channel_stitch:
        #     filename = current_surround_view_data[camera_channel]['filename']
        #     image_path = os.path.join(nusc.dataroot, filename)
        #     image_list.append(cv2.imread(image_path))

        #     cs_record = nusc.get('calibrated_sensor', 
        #                          current_surround_view_data[camera_channel]['calibrated_sensor_token'])
            
        #     cam_info_list.append({
        #         "translation": 
        #         cs_record['translation'],
        #         "rotation":
        #         cs_record['rotation'],
        #         "camera_intrinsic":
        #         cs_record['camera_intrinsic'],
        #     })

        # pano_image_rgb = stitch(
        #     image_list,
        #     cam_info_list
        # )

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
                                             f'range_image_all_frames_{args.image_size[0]}_{args.image_size[1]*6}_fov_{args.fov[0]}_{args.fov[1]}',
                                             current_sf_record['folder_name'])
        if not os.path.exists(range_image_save_path):
            os.makedirs(range_image_save_path)
        
        # 保存 range image 为 npy 文件
        np.save(os.path.join(range_image_save_path, 'range_image_prev.npy'), range_image_prev)
        np.save(os.path.join(range_image_save_path, 'range_image_curr.npy'), range_image_curr)

        # 读取 LiDAR 和 Camera 对应的标定信息，作为 sensor_metas 传入网络
        camera_calib = {}
        camera_pose = {}
        # prev 帧的标定信息
        for camera_channel in camera_channels:
            camera_data = previous_surround_view_data[camera_channel]
            camera_calib[camera_channel] = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
            camera_pose[camera_channel] = nusc.get('ego_pose', camera_data['ego_pose_token'])
        lidar_calib = nusc.get('calibrated_sensor', previous_sf_record['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', previous_sf_record['ego_pose_token'])
        sensor_metas_prev = {
            'lidar':{
                'calibrated_sensor': lidar_calib,
                'ego_pose': lidar_pose,
            },
            'camera':{
                'calibrated_sensor': camera_calib,
                'ego_pose': camera_pose,
            }
        }

        # curr 帧的标定信息
        for camera_channel in camera_channels:
            camera_data = current_surround_view_data[camera_channel]
            camera_calib[camera_channel] = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
            camera_pose[camera_channel] = nusc.get('ego_pose', camera_data['ego_pose_token'])

        lidar_calib = nusc.get('calibrated_sensor', current_sf_record['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', current_sf_record['ego_pose_token'])
        sensor_metas_curr = {
            'lidar':{
                'calibrated_sensor': lidar_calib,
                'ego_pose': lidar_pose,
            },
            'camera':{
                'calibrated_sensor': camera_calib,
                'ego_pose': camera_pose,
            }
        }

        info = {
            'prev_camera_data': previous_surround_view_data,
            'curr_camera_data': current_surround_view_data,
            'prev_lidar_data': lidar_data_matched_with_images[i-1], 
            'curr_lidar_data': lidar_data_matched_with_images[i],
            'sensor_metas_prev': sensor_metas_prev,
            'sensor_metas_curr': sensor_metas_curr,
            'gt_map_path': range_image_save_path,
            'scene_flow_path': os.path.join(args.scene_flow_path, current_sf_record['folder_name']),
            'scene_indice': scene_indice
        }
        
        trainval_test_infos.append(info)

        ###################################### 可视化 ######################################
        if args.risk_score_map_vis:
            risk_score_display = np.copy(proj_risk_score)

            pos_mask = risk_score_display > 0
            neg_mask = risk_score_display < 0

            normalized_risk_score = np.zeros_like(risk_score_display, dtype=np.float32)

            # 分段归一化，大于 0 表示朝向自车运动，小于 0 表示远离自车运动
            # 处理大于0的部分
            if np.any(pos_mask):
                pos_risk = risk_score_display[pos_mask]
                pos_max = pos_risk.max()
                pos_min = pos_risk.min()
                if pos_max > pos_min >= 0:
                    normalized_risk_score[pos_mask] = (pos_risk - pos_min) / (pos_max - pos_min)
                else:
                    normalized_risk_score[pos_mask] = 0.0
            # 处理小于0的部分
            if np.any(neg_mask):
                neg_risk = risk_score_display[neg_mask]
                neg_max = neg_risk.max()
                neg_min = neg_risk.min()
                if neg_min < neg_max <= 0:
                    normalized_risk_score[neg_mask] = (neg_risk - neg_min) / (neg_max - neg_min) - 1.0
                else:
                    normalized_risk_score[neg_mask] = 0.0
            
            risk_score_vis_save_path = os.path.join(args.vis_dir, 'risk_score_map',
                                                f"{args.image_size[0]}_{args.image_size[1]*6}_fov_{args.fov[0]}_{args.fov[1]}")
            
            if not os.path.exists(risk_score_vis_save_path):
                os.makedirs(risk_score_vis_save_path)

            risk_score_vis_name = f"scene_{scene_indice}_risk_score_map_{i}.png"

            plt.imsave(os.path.join(risk_score_vis_save_path, risk_score_vis_name),
                       normalized_risk_score, cmap='seismic', vmin=-1, vmax=1)
             
        if args.scale_map_vis:
            scale_display = np.copy(proj_scale)

            valid_mask = scale_display >= 0

            # 将用于可视化的尺度值裁切到 [0.5, 1.5] 范围
            scale_display[valid_mask] = np.clip(scale_display[valid_mask], 0.85, 1.15)

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
                                                f"{args.image_size[0]}_{args.image_size[1]*6}_fov_{args.fov[0]}_{args.fov[1]}")
            if not os.path.exists(scale_map_vis_save_path):
                os.makedirs(scale_map_vis_save_path)

            out_name = f"scene_{scene_indice}_scale_map_{i}.png"
            out_path = os.path.join(scale_map_vis_save_path, out_name)

            plt.imsave(out_path, -normalized_display, cmap='seismic', vmin=-1, vmax=1)

        # TODO: 处理 depth map 的可视化:
        if args.depth_map_vis:
            raise NotImplementedError
        ###################################### 可视化 ######################################
    
    # 对 camera_lut 和 lidar_data_matched_with_images 
    print(f"Succesfully created (Surround View Images Pair & Lidar Sample Data Pair & Scene Flow & Scale Map & Depth Map & Risk Score Map). \
          Sorted by timestamp. {len(trainval_test_infos)} items in total.")

    with open(os.path.join(args.pkl_save_path,
                           f"nusc_{args.trainval_test_split}_infos_{args.image_size[0]}_{args.image_size[1]*6}_fov_{args.fov[0]}_{args.fov[1]}_.pkl"),'wb') as f:
        pickle.dump(trainval_test_infos, f)
    print(f"Saved nusc_trainval_infos.pkl to {args.pkl_save_path}")

if __name__ == "__main__":
    # NuScenes 数据集路径
    nusc = NuScenes(version='v1.0-trainval', dataroot='./Datasets/nuscenes',
                    verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_flow_path', type=str, default='./Datasets/nuscenes/0_scene_flow/scene_flow_all_frames',
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
    parser.add_argument('--trainval_test_split', default='trainval', type=str,
                       help='trainval and test split, can be trainval or test')
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