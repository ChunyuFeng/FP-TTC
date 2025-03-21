import os
import cv2
import pickle
import numpy as np
import os.path as osp

from PIL import Image
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from tools.generate_ttc_nuscenes.utils.nusc_lidar_cam_match import (find_matching_camera_sweep_in_nusc,
                                                                       get_scale_map, find_matching_sf_sweep_in_lut, depth_estimator)
from utils.draw import gt_scale_2_rgb

import matplotlib.pyplot as plt

from depthanything.depth_anything_v2.dpt import DepthAnythingV2

# IN 相邻两帧激光点云（多帧叠加）数据路径，点云已经实现了一一对应，直接作差即可得到 scene flow
scene_flow_data_path = '/mnt/fpttc_data/scene_flow/multi_frame/31_scene_flow'
# OUT
gt_save_path = '/mnt/fpttc_data/scale_map/31_scale_map'
train_pkl_save_path = '/mnt/fpttc_data/TVT_infos'
gt_vis_save_path = '/mnt/fpttc_data/output_vis/31_scale_map'
gt_with_img_vis_save_path = '/mnt/fpttc_data/output_vis/31_scale_map_with_img'

# NuScenes 数据集路径
nusc = NuScenes(version='v1.0-trainval', dataroot='./Datasets/nuscenes',
                verbose=True)

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

# 查找对应 LIDAR_TOP 的 token
def find_lidar_top_token(nusc, timestamp):
    """
    根据时间戳在 NuScenes 数据集中查找对应的 LIDAR_TOP 的 token
    """
    for sd in nusc.sample_data:
        if sd['sensor_modality'] == 'lidar' and sd['timestamp'] == timestamp:
            return sd['token']
    return None

def find_surround_view_images_for_same_frame(camera_data_dict, target_timestamp, delta_timestamp):
    for cam_data in camera_data_dict:
        timestamp = cam_data['timestamp']
        if 0 < (target_timestamp - timestamp) <= delta_timestamp:
            return cam_data
    return None

################################################ 创建场景流 LUT ################################################
# folder name 包含了时间戳信息，读取所有的文件夹名称
folder_names = [folder_name[:-4] if folder_name.endswith('.pcd') else folder_name
                for folder_name in os.listdir(scene_flow_data_path)
                if os.path.isdir(os.path.join(scene_flow_data_path, folder_name))]

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
count = 0
for folder, timestamp in folders_with_timestamps:
    token = find_lidar_top_token(nusc, timestamp)
    if token:
        scene_flow_lut.append({
        'token': token,
        'folder_name': folder,
        'timestamp': timestamp
    })
        count += 1
    else:
        lidar_sweep_not_found_in_nusc.append(folder)
        lidar_sweep_not_found_in_nusc_count += 1
print(f"场景流 LUT 已创建完成，已按照时间戳排序，共有 {len(scene_flow_lut)} 条数据")

if lidar_sweep_not_found_in_nusc_count > 0:
    print(f"在 NuScenes 数据集中找不到的 LIDAR sweeps 数量: {lidar_sweep_not_found_in_nusc_count}")

################################################ 创建环视相机数据 LUT ################################################

# 提取时间戳信息
lidar_timestamps = [record['timestamp'] for record in scene_flow_lut]

# 定义6个相机通道
camera_channels = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
                   'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT']

# 获取 LIDAR_TOP 时间戳范围内的相机数据
camera_data_within_timestamp_range = {}
timestamp_range = [lidar_timestamps[0], lidar_timestamps[-1]]
for camera_channel in camera_channels:
    camera_data_within_timestamp_range[camera_channel] = find_matching_camera_sweep_in_nusc(nusc, timestamp_range, camera_channel)

# 组合并筛选相机数据
# 1. 将 6 channel 的相机数据按照 ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
#                    'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT'] 的顺序组合；
#    CAM_BACK_LEFT的时间戳最接近LIDAR_TOP的时间戳，可以以此为基准进行组合；
#    CAM_BACK_LEFT的时间戳最大，CAM_FRONT_LEFT的时间戳最小；
# 2. 剔除其中不足 6 帧的数据

camera_lut = []
# nuscenes 数据集中，激光雷达数据的频率为20hz，相机数据的频率为12hz
# scene_flow_lut_12hz 中保留与图像数据对应的激光雷达数据
scene_flow_lut_12hz = []

camera_data_dict = {}
for camera_channel, camera_data in camera_data_within_timestamp_range.items():
    camera_data_dict[camera_channel] = []
    for data in camera_data:
        camera_data_dict[camera_channel].append(
            {
            'token': data['token'],
            'timestamp': data['timestamp'],
            'filename': data['filename']
            }
        )

# 对6个相机通道的数据按照时间戳进行排序
for camera_channel in camera_channels:
    camera_data_dict[camera_channel].sort(key=lambda x: x['timestamp'])

# 以 CAM_BACK_LEFT 为基准，向前查找属于同一组数据的 6 帧图像
for camera_data_bl in camera_data_dict['CAM_BACK_LEFT']:
    # 检查 CAM_BACK_LEFT 时间戳前后 25ms 内是否有 LIDAR_TOP 数据
    # 如果没有，说明这一组图像没有对应的 LIDAR_TOP 数据，跳过
    timestamp_range = [camera_data_bl['timestamp'] - 25 * 1e3, camera_data_bl['timestamp'] + 25 * 1e3]
    lidar_data = find_matching_sf_sweep_in_lut(scene_flow_lut, timestamp_range)
    if not lidar_data:
        continue

    camera_data_combined = {}
    camera_data_combined['CAM_BACK_LEFT'] = {
        'token': camera_data_bl['token'],
        'timestamp': camera_data_bl['timestamp'],
        'filename': camera_data_bl['filename']
    }
    # camera_data_combined = {'CAM_BACK_LEFT': {camera_data_bl['timestamp']: camera_data_bl['filename']}}
    target_timestamp = camera_data_bl['timestamp']

    # 查找 CAM_BACK_LEFT 前 50ms 内的相机数据
    for camera_channel in ['CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT']:
        # 同一组数据的 6 帧图像时间戳差值在 50 ms 以内
        matched_cam = find_surround_view_images_for_same_frame(
            camera_data_dict[camera_channel], target_timestamp, delta_timestamp=50*1e3)
        if matched_cam:
            camera_data_combined[camera_channel] = matched_cam

    # 检查 camera_data_combined 的 6 个相机通道中是否都有数据
    is_complete_surround_view = (len(camera_data_combined) == 6)
    if is_complete_surround_view:
        camera_lut.append(camera_data_combined)
        scene_flow_lut_12hz.append(lidar_data)

# 对组合好的6通道相机数据按照时间戳进行排序
camera_lut = sorted(camera_lut, key=lambda x: x['CAM_BACK_LEFT']['timestamp'])
print(f"环视相机数据 LUT 已创建完成，已按照时间戳排序，共有 {len(camera_lut)} 条数据")

################################################ 将激光点云图像投影到环视相机图像上 ################################################
# 将 scene flow 点云数据投影到 6 个相机图像上；
# 投影结果 (u, v, scale = depth_pc1 / depth_pc3) 保存为 tiff 图像文件，其中 u 和 v 是像素坐标，depth_pc1 和 depth_pc3 是点云的深度信息。
GENERATE_GT = False
VISUALIZATION = True
VISUALIZATION_ON_IMAGE = False
train_infos = []
matching_count = 0

# for i in tqdm(range(30), desc="copy images:"):
#     surr_view_img = camera_lut[i]
#     # 依次取出环视相机图像数据
#     for camera_channel, camera_data in surr_view_img.items():
#         filename = camera_data['filename']
#         # 根据文件名将图像拷贝到指定目录
#         src_path = os.path.join(nusc.dataroot, camera_data['filename'])
#         dst_path_prefix = os.path.join('/home/chunyu/WorkSpace/BugStudio/FP-TTC/depthanything/nusc_imgs', camera_channel)
#         if not os.path.exists(dst_path_prefix):
#             os.makedirs(dst_path_prefix)
#         dst_path = os.path.join(dst_path_prefix, f'{i}.jpg')
#         os.system(f"cp {src_path} {dst_path}")

# 循环依次取出 camera_lut 中连续两组数据（每组数据包含 6 channel 图像数据）
# 用于匹配 scene flow 和 nuScenes 数据集中的 camera 图像
for i in tqdm(range(len(camera_lut) - 1), desc="Creating ground truth images and pkl file"):
    # 取出连续两组数据
    previous_surround_view_data = camera_lut[i]
    current_surround_view_data = camera_lut[i + 1]

    # 取出时间戳信息
    camera_timestamp_1 = previous_surround_view_data['CAM_BACK_LEFT']['timestamp']
    camera_timestamp_2 = current_surround_view_data['CAM_BACK_LEFT']['timestamp']

    # 计算时间戳差值
    time_diff_cam = camera_timestamp_2 - camera_timestamp_1

    # 如果时间戳差值小于 25 毫秒或者大于 125 毫秒，则跳过
    # 50 ms 为激光雷达的采样间隔，25 ms 为余量
    if time_diff_cam > (50 * 2 + 25) * 1e3 or time_diff_cam < 25 * 1e3:
        continue

    # 假设：pc3 的时间戳比 pc1 的时间戳大 50ms；实际数据中绝大多数数据符合这个假设
    time_diff_ratio = round(time_diff_cam / (50 * 1e3))

    # 取出 scene flow 数据
    previous_sf_record = scene_flow_lut_12hz[i]
    current_sf_record = scene_flow_lut_12hz[i + 1]

    # 将 scene flow 数据投影到环视 6 相机图像上
    # 1. 读取 scene flow 数据
    if (not os.path.exists(os.path.join(scene_flow_data_path, previous_sf_record['folder_name'], 'pc1.npy'))
            or not os.path.exists(os.path.join(scene_flow_data_path, previous_sf_record['folder_name'], 'pc3.npy'))):
        i = i+1
        continue
    pc_1 = np.load(os.path.join(scene_flow_data_path, previous_sf_record['folder_name'], 'pc1.npy'))
    pc_3 = np.load(os.path.join(scene_flow_data_path, previous_sf_record['folder_name'], 'pc3.npy'))
    # 对 scene flow 数据进行时间插值
    # sf = pc3 - pc1; sf' = time_diff_ratio * sf; pc3' = pc1 + sf'
    # => pc3' = pc1 + time_diff_ratio * (pc3 - pc1)
    # => pc3' = time_diff_ratio * pc3 + (1 - time_diff_ratio) * pc1
    pc_3 = time_diff_ratio * pc_3 + (1 - time_diff_ratio) * pc_1

    # 为每个点云增加序号（从0开始），并添加为第四列
    pc1_with_indices = np.hstack([pc_1, np.arange(pc_1.shape[0]).reshape(-1, 1)])
    pc3_with_indices = np.hstack([pc_3, np.arange(pc_3.shape[0]).reshape(-1, 1)])

    idx = i
    prev_pc_fake, curr_pc_fake = depth_estimator(nusc,
                                                previous_surround_view_data,
                                                current_surround_view_data,
                                                pc1_with_indices,
                                                pc3_with_indices,
                                                previous_lidar_token=previous_sf_record['token'],
                                                current_lidar_token=current_sf_record['token'],
                                                use_metric_depth_estimator=True,
                                                data_idx=idx)

    # 将 point_cloud_fake 保存为 npy 文件
    prev_pc_fake_save_path = os.path.join('/mnt/fpttc_data/fake_pc',
                                     previous_sf_record['folder_name'])
    if not os.path.exists(prev_pc_fake_save_path):
        os.makedirs(prev_pc_fake_save_path)
    np.save(os.path.join(prev_pc_fake_save_path, 'fake_pc.npy'), prev_pc_fake)

    curr_pc_fake_save_path = os.path.join('/mnt/fpttc_data/fake_pc',
                                     current_sf_record['folder_name'])
    if not os.path.exists(curr_pc_fake_save_path):
        os.makedirs(curr_pc_fake_save_path)
    np.save(os.path.join(curr_pc_fake_save_path, 'fake_pc.npy'), curr_pc_fake)

