from tqdm import tqdm
from PIL import Image
import os.path as osp
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from collections import defaultdict
from tools.generate_ttc_nuscenes.utils import design_depth_map

def parse_sf_lut(sf_info):
    """
    解析信息字符串，提取文件夹名、时间戳和 token。

    参数:
    - sf_info: 输入的字符串，格式为 'folder_name__timestamp token'

    返回:
    - folder_name: 提取的文件夹名称
    - timestamp: 提取的时间戳（整数形式）
    - token: 提取的 token
    """
    try:
        # 按空格分割字符串
        folder_name, token = sf_info.split(' ')

        # 提取时间戳：假设文件夹名的格式为 '...__timestamp'
        timestamp_str = folder_name.split('__')[-1]
        timestamp = int(timestamp_str)

        return folder_name, timestamp, token
    except ValueError as e:
        print(f"解析字符串失败: {e}")
        return None, None, None

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

def find_matching_camera_sweep_in_lut(camera_lut, target_timestamp_range, camera_channel):
    """
    找到在指定时间戳范围内指定相机通道的相机数据
    :param camera_lut: 相机数据字典
    :param target_timestamp_range: 目标时间戳范围，单位为微秒。例如：[timestamp_start - 50 * 1e3, timestamp_end + 50 * 1e3]
    :param camera_channel: 相机通道名称，例如 'CAM_FRONT'
    :return: 匹配到的所有相机数据 timestamp list，如果没有找到则返回 None
    """
    camera_timestamp_list = []
    # 遍历 camera_lut 找到指定时间戳范围内的 数据
    for cam_data in camera_lut:
        timestamp = cam_data[camera_channel]['timestamp']
        # key = next(iter(cam_data[camera_channel].keys()))
        # timestamp = int(key)
        if ((target_timestamp_range[0] - 50 * 1e3) <= timestamp
                <= (target_timestamp_range[1] + 50 * 1e3)):
            camera_timestamp_list.append(timestamp)

    # 如果找到了匹配的相机数据，则返回
    if len(camera_timestamp_list) > 0:
        return camera_timestamp_list
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

def find_matching_filename(camera_data_dict, target_timestamp, delta):
    """查找与目标时间戳匹配的文件名"""
    for timestamp in camera_data_dict:
        if 0 < (target_timestamp - timestamp) <= delta:
            return {timestamp: camera_data_dict[timestamp]}
    return None

def project_lidar_to_surround_view_img(nusc,
                                       lidar_data,
                                       lidar_indices,
                                       lidar_token,
                                       surround_view_img,
                                       min_dist=1.0):
    '''
    将 LiDAR 点云投影到相机视图上
    Args:
        nusc: nuScenes 数据集对象
        lidar_data: lidar 点云数据
        lidar_indices: lidar 点云序号
        lidar_token: lidar sample_data token
        surround_view_img: lidar 帧对应的环视图像
        min_dist: 用于过滤点云的最小距离

    Returns:
        projected_points: 投影到相机视图上的2D点云
        点云在相机坐标系下的深度值和序号
        点云对应的原始图像

    '''

    projected_points = {}

    # 获取 LiDAR 样本的基本信息，包括传感器标定信息、位姿信息等
    lidar_sample_data = nusc.get('sample_data', lidar_token)

    for channel in surround_view_img:
        cam_sample_data = nusc.get('sample_data', surround_view_img[channel]['token'])

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
        filtered_indices = lidar_indices[mask]

        # 将点云在激光雷达坐标系下的坐标也保存下来
        pc_lidar_coord = lidar_data[mask]

        # 将结果存储在字典中
        projected_points[channel] = {
            'points': points_2d,
            'points_lidar_coord': pc_lidar_coord,
            'depths': depths,
            'indices': filtered_indices,
            'original_img': im
        }

    return projected_points


def get_scale_map(nusc,
                  previous_surround_view_data,
                  current_surround_view_data,
                  previous_lidar_data,
                  current_lidar_data,
                  previous_lidar_token,
                  current_lidar_token):
    '''
    获取连续帧 LiDAR 数据在环视图像上的投影，并计算深度比值
    Args:
        nusc: nuScenes 数据集对象
        previous_surround_view_data: 第一帧 LiDAR 数据对应的环视图像
        current_surround_view_data: 第二帧 LiDAR 数据对应的环视图像
        previous_lidar_data: 第一帧 LiDAR 数据，包含点云和序号（手动添加的 int 类型序号）
        current_lidar_data: 第二帧 LiDAR 数据，包含点云和序号 （手动添加的 int 类型序号）
        previous_lidar_token: 第一帧 LiDAR 数据的 sample_data token
        current_lidar_token: 第二帧 LiDAR 数据的 sample_data token

    Returns:
        scale_map: 包含
        1）第一帧lidar点云投影到环视图像上的2D点云
        2）两帧点云在相机坐标系下的深度信息（PV视角下的深度值）
        3）两帧点云深度比值
        的字典
    '''

    # 从 previous_lidar_data 和 current_lidar_data 中提取点云和序号
    previous_pc = previous_lidar_data[:, :3]
    previous_pc_indices = previous_lidar_data[:, 3]
    current_pc = current_lidar_data[:, :3]
    current_pc_indices = current_lidar_data[:, 3]

    # 调用函数投影 previous_lidar_data 和 current_lidar_data
    previous_projected_points = project_lidar_to_surround_view_img(nusc,
                                                                   previous_pc,
                                                                   previous_pc_indices,
                                                                   previous_lidar_token,
                                                                   previous_surround_view_data)
    current_projected_points = project_lidar_to_surround_view_img(nusc,
                                                                  current_pc,
                                                                  current_pc_indices,
                                                                  current_lidar_token,
                                                                  current_surround_view_data)
    scale_map = {}
    for channel in previous_projected_points:
        # 由于投影时通过 mask 进行了筛选，previous 和 current此时的点云数量可能不一致
        # 因此需要通过 indices 进一步筛选，只保留两者都存在的点云
        common_indices = np.intersect1d(previous_projected_points[channel]['indices'],
                                        current_projected_points[channel]['indices'])

        # 通过 indices 筛选点云和深度信息
        prev_mask = np.isin(previous_projected_points[channel]['indices'], common_indices)
        curr_mask = np.isin(current_projected_points[channel]['indices'], common_indices)

        previous_points = previous_projected_points[channel]['points'][:, prev_mask]
        previous_depths = previous_projected_points[channel]['depths'][prev_mask]
        current_points = current_projected_points[channel]['points'][:, curr_mask]
        current_depths = current_projected_points[channel]['depths'][curr_mask]

        previous_xyz = previous_projected_points[channel]['points_lidar_coord'][prev_mask]
        current_xyz = current_projected_points[channel]['points_lidar_coord'][curr_mask]

        ########################### 去掉重叠的点云 ###########################
        previous_points_round = np.round(previous_points).astype(int)
        previous_points_wo_overlap = np.full((1600, 900, 2), np.inf)
        previous_xyz_wo_overlap = np.full((1600, 900, 3), np.inf)

        for i in range(len(previous_depths)):
            u, v = previous_points_round[0, i], previous_points_round[1, i]
            if previous_points_wo_overlap[u, v, 0] > previous_depths[i]:
                # 更新深度信息，在图像上重叠的点保留近处的点云 - 相邻帧深度信息作比值即为scale
                previous_points_wo_overlap[u, v, 0] = previous_depths[i]
                previous_points_wo_overlap[u, v, 1] = current_depths[i]
                # previous_points_wo_overlap[u, v, 2] = scale[i]

                # 更新3D坐标，保留点云在激光雷达坐标系下的坐标，用于制作range image
                previous_xyz_wo_overlap[u, v, :] = previous_xyz[i]

        previous_depth_wo_overlap = previous_points_wo_overlap[:, :, 0]
        current_depth_wo_overlap = previous_points_wo_overlap[:, :, 1]
        # scale_wo_overlap = previous_points_wo_overlap[:, :, 2]

        u, v = np.meshgrid(np.arange(previous_points_wo_overlap.shape[1]),
                           np.arange(previous_points_wo_overlap.shape[0]))

        # np.inf -> 0
        previous_depth_wo_overlap = np.where(np.isinf(previous_depth_wo_overlap), 0, previous_depth_wo_overlap)
        current_depth_wo_overlap = np.where(np.isinf(current_depth_wo_overlap), 0, current_depth_wo_overlap)

        previous_dense_depth_out, _ = design_depth_map.create_map(previous_projected_points[channel]['original_img'],
                                                      previous_depth_wo_overlap,
                                                      max_depth=50.0,
                                                      dilation_kernel_far=design_depth_map.kernels.diamond_kernel_7(),
                                                      dilation_kernel_med=design_depth_map.kernels.diamond_kernel_7(),
                                                      dilation_kernel_near=design_depth_map.kernels.diamond_kernel_7())

        current_dense_depth_out, _ = design_depth_map.create_map(previous_projected_points[channel]['original_img'],
                                                               current_depth_wo_overlap,
                                                               max_depth=50.0,
                                                               dilation_kernel_far=design_depth_map.kernels.diamond_kernel_7(),
                                                               dilation_kernel_med=design_depth_map.kernels.diamond_kernel_7(),
                                                               dilation_kernel_near=design_depth_map.kernels.diamond_kernel_7())

        valid_mask = (previous_depth_wo_overlap != 0) & (current_depth_wo_overlap != 0)
        # valid_mask = ~np.isinf(depth_wo_overlap) & ~np.isinf(scale_wo_overlap)
        u = u[valid_mask]
        v = v[valid_mask]

        previous_depth_wo_overlap = previous_dense_depth_out[valid_mask]
        current_depth_wo_overlap = current_dense_depth_out[valid_mask]
        scale_wo_overlap = current_depth_wo_overlap / previous_depth_wo_overlap

        previous_points_wo_overlap_ = np.stack([v, u], axis=0)
        previous_xyz_wo_overlap = previous_xyz_wo_overlap[valid_mask]

        scale_map[channel] = {
            'points': previous_points_wo_overlap_,
            'points_lidar_coord': previous_xyz_wo_overlap,
            'depth_1': previous_depth_wo_overlap,
            'depth_3': current_depth_wo_overlap,
            'scale': scale_wo_overlap,
            'original_img': previous_projected_points[channel]['original_img']
        }

        ########################### 去掉重叠的点云 ###########################

        # scale = current_depths / previous_depths
        #
        # scale_map[channel] = {
        #     'points': previous_points,
        #     'depth_1': previous_depths,
        #     'depth_3': current_depths,
        #     'scale': scale,
        #     'original_img': previous_projected_points[channel]['original_img']
        # }

    return scale_map

