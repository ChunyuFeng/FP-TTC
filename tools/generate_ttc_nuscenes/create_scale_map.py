import os
import cv2
import pickle
import numpy as np
import os.path as osp

from PIL import Image
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from tools.generate_ttc_nuscenes.utils.nusc_lidar_cam_match import (find_matching_camera_sweep_in_nusc,
                                                                       get_scale_map, find_matching_sf_sweep_in_lut)
from utils.draw import gt_scale_2_rgb
from range_projection import range_projection

import matplotlib.pyplot as plt

scene_flow_data_path = '/mnt/fpttc_data/scene_flow/multi_frame/31_scene_flow'
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
VISUALIZATION = False
VISUALIZATION_ON_IMAGE = False
train_infos = []
matching_count = 0
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

    # timestamp_range_1 = [camera_timestamp_1 - 25 * 1e3, camera_timestamp_1 + 25 * 1e3]
    # timestamp_range_2 = [camera_timestamp_2 - 25 * 1e3, camera_timestamp_2 + 25 * 1e3]
    #
    # # 从 scene_flow_lut 中找到与 camera_timestamp_1 和 camera_timestamp_2 最接近的时间戳
    # matching_sf_sweep_1 = find_matching_sf_sweep_in_lut(scene_flow_lut, timestamp_range_1)
    # matching_sf_sweep_2 = find_matching_sf_sweep_in_lut(scene_flow_lut, timestamp_range_2)
    #
    # if matching_sf_sweep_1 and matching_sf_sweep_2:
    #     # 寻找 matching_sf_sweep_1 在 scene_flow_lut 中的索引
    #     idx_1 = [record['timestamp'] for record in scene_flow_lut].index(matching_sf_sweep_1['timestamp'])
    #     idx_2 = [record['timestamp'] for record in scene_flow_lut].index(matching_sf_sweep_2['timestamp'])
    #     time_diff_sf = scene_flow_lut[idx_1 + 1]['timestamp'] - matching_sf_sweep_1['timestamp']
    #     # 如果 scene flow 数据时间戳差值大于 125ms 或者小于 25ms，则舍弃这一帧数据
    #     if time_diff_sf > (50 * 2 + 25) * 1e3 or time_diff_sf < 25 * 1e3:
    #         continue
    #     previous_sf_record = scene_flow_lut[idx_1]
    #     current_sf_record = scene_flow_lut[idx_2]
    #     matching_count += 1
    # else:
    #     continue
    #
    # if matching_count == TRAIN_INFO_ITEM_NUM:
    #     break

    # time_diff_ratio = int(round(time_diff_cam / time_diff_sf))

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

    # pc_1_new[:, 0] = pc_1[:, 1]
    # pc_1_new[:, 1] = pc_1[:, 0]
    # pc_1_new[:, 2] = -pc_1[:, 2]
    #
    # pc3_new

    # 将 y 轴反转以匹配 nuScenes 坐标系
    # pc_1[:, 1] = -pc_1[:, 1]
    # pc_3[:, 1] = -pc_3[:, 1]

    ############### 处理带地面的点云数据 ################
    # # 取消反转 y 轴
    # pc_1[:, 1] = -pc_1[:, 1]
    # pc_3[:, 1] = -pc_3[:, 1]

    # pc_1[:, 2] += 1.5
    # pc_3[:, 2] += 1.5
    #
    # pc_1_new = np.zeros_like(pc_1)
    # pc_3_new = np.zeros_like(pc_3)
    #
    # pc_1_new[:, 0] = -pc_1[:, 1]
    # pc_1_new[:, 1] = pc_1[:, 2]
    # pc_1_new[:, 2] = -pc_1[:, 0]
    #
    # pc_3_new[:, 0] = -pc_3[:, 1]
    # pc_3_new[:, 1] = pc_3[:, 2]
    # pc_3_new[:, 2] = -pc_3[:, 0]
    #
    # pc_1 = pc_1_new
    # pc_3 = pc_3_new
    ############### 处理带地面的点云数据 ################

    # 为每个点云增加序号（从0开始），并添加为第四列
    pc1_with_indices = np.hstack([pc_1, np.arange(pc_1.shape[0]).reshape(-1, 1)])
    pc3_with_indices = np.hstack([pc_3, np.arange(pc_3.shape[0]).reshape(-1, 1)])

    # 将相邻两帧点云数据分别投影到对应的 6 个相机图像上
    # 输入：previous_surround_view_data, current_surround_view_data, pc1_with_indices, pc3_with_indices
    # 返回值：scale_map: dict
    # key: camera_channel,
    # value: {'points': previous_points_2d,
    # 'depth_1': previous_depths,
    # 'depth_3': current_depths,
    # 'scale': depth_3 / depth_1, scale > 1 表示该点在远离；scale < 1 表示该点在靠近；
    # 'original_img': previous_img}

    scale_map = get_scale_map(nusc,
                              previous_surround_view_data,
                              current_surround_view_data,
                              pc1_with_indices,
                              pc3_with_indices,
                              previous_lidar_token=previous_sf_record['token'],
                              current_lidar_token=current_sf_record['token'])

    original_img_path = {}
    gt_scale_path = {}
    # 用于生成 range image
    points_lidar_coord_list = []
    scales_list = []

    # 保存投影结果，制作真值供网络训练使用；以及用于可视化
    # 真值的格式为 tiff 图像，其中 u 和 v 是像素坐标，scale 是深度比值
    # 可视化的格式为 png 图像，其中 u 和 v 是像素坐标，scale 是深度比值
    for camera_channel, data in scale_map.items():
        u_coords = data['points'][0, :].astype(np.int32)
        v_coords = data['points'][1, :].astype(np.int32)
        scales = data['scale']
        img_path = previous_surround_view_data[camera_channel]['filename']
        original_img_filename = img_path.split('/')[-1]
        tiff_filename = f"{original_img_filename}.tiff"
        gt_scale_path[camera_channel] = osp.join(gt_save_path, previous_sf_record['folder_name'], tiff_filename)
        original_img_path[camera_channel] = img_path

        height = data['original_img'].height
        width = data['original_img'].width

        ################################### 拼接每个视图对应的原始点云 ###################################
        # 拼接每个视图对应的 激光雷达坐标系下的点云 以及 scales，用于生成 range image
        points_xyz = data['points_lidar_coord']
        points_lidar_coord_list.append(points_xyz)
        scales_list.append(scales)

        point_size = 1  # visualization 激光点的尺寸

        if GENERATE_GT:
            # 生成 TIFF 图像保存真值，分辨率为 1600x900，使用单通道 float32 类型
            scale_tiff = np.ones((height, width), dtype=np.float32) * np.nan  # 初始设置为 NaN 表示无效区域
            # 将 scale 值批量写入指定的 u, v 坐标位置
            # 在 gt_save_path 目录下建立一个子目录，子目录名为 previous_sf_record['folder_name']，用于保存真值图像
            # 真值图像的文件名为 camera_filename.tiff
            scale_tiff[v_coords, u_coords] = scales
            tiff_image = Image.fromarray(scale_tiff, mode='F')  # mode 'F' 为 32 位浮点数格式
            subdir_path = osp.join(gt_save_path, previous_sf_record['folder_name'])
            os.makedirs(subdir_path, exist_ok=True)
            tiff_image.save(osp.join(subdir_path, tiff_filename))

        if VISUALIZATION:
            background = np.zeros((height, width))

            for u, v, scale in zip(u_coords, v_coords, scales):
                # 为每个点填充一个正方形区域，大小为 (2 * point_size + 1) x (2 * point_size + 1)
                background[max(0, v - point_size):min(height, v + point_size + 1),
                max(0, u - point_size):min(width, u + point_size + 1)] = (scale - 0.5) / 1.0

            # 使用 disp2rgb 生成彩色可视化图像
            scale_gt_image = gt_scale_2_rgb(np.clip(background, 0.0, 1.0))
            scale_gt_image = (scale_gt_image * 255.0).astype(np.uint8)

            # 保存可视化结果
            vis_subdir_path = osp.join(gt_vis_save_path, previous_sf_record['folder_name'])
            os.makedirs(vis_subdir_path, exist_ok=True)
            vis_filename = f"{original_img_filename}.jpg"
            cv2.imwrite(os.path.join(vis_subdir_path, vis_filename), scale_gt_image)

        if VISUALIZATION_ON_IMAGE:
            # 加载原始图像作为背景
            background_img = np.array(data['original_img'].convert("RGB"))

            # 创建单通道 scale_layer 图像，用于绘制 scale 点
            scale_layer = np.zeros((height, width))
            for u, v, scale in zip(u_coords, v_coords, scales):
                scale_layer[max(0, v - point_size):min(height, v + point_size + 1),
                max(0, u - point_size):min(width, u + point_size + 1)] = (scale - 0.5) / 1.0

            # 使用 disp2rgb 生成彩色的 scale 散点图
            scale_rgb_layer = gt_scale_2_rgb(np.clip(scale_layer, 0.0, 1.0))
            scale_rgb_layer = (scale_rgb_layer * 255.0).astype(np.uint8)

            # 创建掩膜，标记 scale_rgb_layer 中的有效区域
            mask = np.any(scale_rgb_layer != [255, 255, 255], axis=-1)

            # 将 scale_rgb_layer 的有效区域覆盖到 background_img 上
            combined_image = background_img.copy()
            combined_image[mask] = scale_rgb_layer[mask]

            # 保存叠加后的可视化结果
            vis_subdir_path = osp.join(gt_with_img_vis_save_path, previous_sf_record['folder_name'])
            os.makedirs(vis_subdir_path, exist_ok=True)
            vis_filename = f"{original_img_filename}.jpg"
            cv2.imwrite(os.path.join(vis_subdir_path, vis_filename), combined_image)


    ################################### 生成 range image ###################################
    # 生成 range image
    points_lidar_coord_list = np.concatenate(points_lidar_coord_list, axis=0)
    scales_list = np.concatenate(scales_list, axis=0)
    proj_range, proj_scale, proj_xyz, proj_idx, proj_mask = range_projection(
        points_lidar_coord_list,
        scales_list,
        H=480,
        W=5120,
        fov_up=15.0,
        fov_down=-25.0
    )

    # 打印一些信息
    print("投影深度图形状:", proj_range.shape)
    print("投影3D坐标图形状:", proj_xyz.shape)
    print("投影索引图形状:", proj_idx.shape)
    print("投影掩码形状:", proj_mask.shape)

    ################################### 可视化 scale ###################################
    plt.figure(figsize=(55, 6))
    plt.title("Scale Projection (Normalized)")

    # 1) 复制一份，以免修改原数据
    scale_display = np.copy(proj_scale)

    # 2) 创建一个有效掩码 valid_mask，标记非 -1 的有效像素
    valid_mask = (scale_display != -1)

    # 3) 取出有效像素值并计算最小值和最大值，用于归一化
    valid_values = scale_display[valid_mask]
    if len(valid_values) > 0:
        min_val = valid_values.min()
        max_val = valid_values.max()
        # 防止出现 max_val == min_val 的情况
        if max_val > min_val:
            # 4) 对有效像素做归一化 [0,1]
            scale_display[valid_mask] = (scale_display[valid_mask] - min_val) / (max_val - min_val)
        else:
            # 所有有效像素都是同一个值，直接赋 0.5 做可视化
            scale_display[valid_mask] = 0.5

    # 5) 对无效像素赋 0
    scale_display[~valid_mask] = 0

    # 6) 显示归一化后的图像
    plt.imshow(scale_display, cmap='plasma', vmin=0, vmax=1)

    # 如果想加 colorbar
    # plt.colorbar(label='Normalized Scale')

    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.tight_layout()
    plt.show()

    info = {
        'timestamp': previous_sf_record['timestamp'],
        'imgs_path': original_img_path,
        'gt_path': gt_scale_path
    }

    train_infos.append(info)

    ################################### 可视化 depth ###################################
    # 可视化深度图
    plt.figure(figsize=(55, 6))  # 调整图像大小
    plt.title("Depth Projection")
    depth_display = np.copy(proj_range)
    depth_display[depth_display == -1] = 0  # 将无数据点设为0以便可视化
    plt.imshow(depth_display, cmap='plasma')

    # plt.colorbar(label='Depth (m)')
    plt.xlabel('Width')
    plt.ylabel('Height')

    plt.tight_layout()
    plt.show()

    # ################################### 可视化 scale ###################################
    # plt.figure(figsize=(55, 6))
    # plt.title("Scale Projection (Normalized)")
    #
    # # 1) 复制一份，以免修改原数据
    # scale_display = np.copy(proj_scale)
    #
    # # 2) 创建一个有效掩码 valid_mask，标记非 -1 的有效像素
    # valid_mask = (scale_display != -1)
    #
    # # 3) 取出有效像素值并计算最小值和最大值，用于归一化
    # valid_values = scale_display[valid_mask]
    # if len(valid_values) > 0:
    #     min_val = valid_values.min()
    #     max_val = valid_values.max()
    #     # 防止出现 max_val == min_val 的情况
    #     if max_val > min_val:
    #         # 4) 对有效像素做归一化 [0,1]
    #         scale_display[valid_mask] = (scale_display[valid_mask] - min_val) / (max_val - min_val)
    #     else:
    #         # 所有有效像素都是同一个值，直接赋 0.5 做可视化
    #         scale_display[valid_mask] = 0.5
    #
    # # 5) 对无效像素赋 0
    # scale_display[~valid_mask] = 0
    #
    # # 6) 显示归一化后的图像
    # plt.imshow(scale_display, cmap='plasma', vmin=0, vmax=1)
    # # plt.colorbar(label='Normalized Scale')  # 如果需要，可以取消注释
    #
    # plt.xlabel('Width')
    # plt.ylabel('Height')
    # plt.tight_layout()
    #
    # # 保存到本地文件，不显示
    # plt.savefig(f"/mnt/fpttc_data/output_vis/31_range_img/scale_{i}.png", dpi=100)  # 可根据需求设置图像分辨率 dpi
    # plt.close()  # 关闭图像，释放内存
    #
    # info = {
    #     'timestamp': previous_sf_record['timestamp'],
    #     'imgs_path': original_img_path,
    #     'gt_path': gt_scale_path
    # }
    #
    # train_infos.append(info)
    #
    # ################################### 可视化 depth ###################################
    # plt.figure(figsize=(55, 6))  # 调整图像大小
    # plt.title("Depth Projection")
    #
    # depth_display = np.copy(proj_range)
    # depth_display[depth_display == -1] = 0  # 将无数据点设为0以便可视化
    #
    # plt.imshow(depth_display, cmap='plasma')
    # # plt.colorbar(label='Depth (m)')  # 如果需要，可以取消注释
    #
    # plt.xlabel('Width')
    # plt.ylabel('Height')
    # plt.tight_layout()
    #
    # # 保存到本地文件，不显示
    # plt.savefig(f"/mnt/fpttc_data/output_vis/31_range_img/depth_{i}.png", dpi=100)
    # plt.close()

# 保存 train_infos 和 metadata 到 pkl 文件
version = 'v1.0-trainval'
metadata = dict(version=version)
pkl_data = dict(infos=train_infos, metadata=metadata)

train_pkl_save_path = osp.join(train_pkl_save_path, 'nusc_infos_train.pkl')

with open(train_pkl_save_path, 'wb') as f:
    pickle.dump(pkl_data, f)