import glob
from tqdm import tqdm
import os
import numpy as np
import pickle
from nuscenes.nuscenes import NuScenes
from tools.generate_ttc_nuscenes.utils.nusc_lidar_cam_match import (find_matching_camera_sweep_in_nusc,
                                                                       get_scale_map, find_matching_sf_sweep_in_lut)
import matplotlib.pyplot as plt

scene_flow_data_path = '/mnt/fpttc_data/scene_flow/multi_frame/31_scene_flow'
gt_save_path = '/mnt/fpttc_data/scale_map/31_scale_map'
train_pkl_save_path = '/mnt/fpttc_data/TVT_infos'
gt_vis_save_path = '/mnt/fpttc_data/output_vis/31_scale_map'

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

def range_projection(points, scales, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    """
    将 3D 点云投影到 2D 球面范围图像。

    参数:
    - points (np.ndarray): 点云数据，形状为 [m, 3]，每行表示 (x, y, z)。
    - scales (np.ndarray): 尺度数据，形状为 [m,]，每个值表示对应点云处的 scale。
    - H (int): 投影图像的高度（像素）。默认值为 64。
    - W (int): 投影图像的宽度（像素）。默认值为 1024。
    - fov_up (float): 向上的视场角（度）。默认值为 3.0。
    - fov_down (float): 向下的视场角（度）。默认值为 -25.0。

    返回:
    - proj_range (np.ndarray): 投影的深度图像，形状为 [H, W]。
    - proj_scale (np.ndarray): 投影的尺度图像，形状为 [H, W]。
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
    points_sorted = points[order]
    proj_x_sorted = proj_x[order]
    proj_y_sorted = proj_y[order]
    indices_sorted = order

    # 赋值到投影图像
    proj_range[proj_y_sorted, proj_x_sorted] = depth_sorted
    proj_scale[proj_y_sorted, proj_x_sorted] = scale_sorted
    proj_xyz[proj_y_sorted, proj_x_sorted] = points_sorted
    proj_idx[proj_y_sorted, proj_x_sorted] = indices_sorted

    # 创建掩码
    proj_mask = (proj_idx >= 0).astype(np.int32)

    return proj_range, proj_scale, proj_xyz, proj_idx, proj_mask


def main():
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
    VISUALIZATION = 0
    vis_output_dir = '/mnt/fpttc_data/output_vis/31_range_img'

    GENERATE_GT = 1
    train_infos = []

    # 循环依次取出 camera_lut 中连续两组数据（每组数据包含 6 channel 图像数据）
    # 用于匹配 scene flow 和 nuScenes 数据集中的 camera 图像
    for i in tqdm(range(len(camera_lut) - 1), desc="Creating scale map ground truth"):
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

        # 读取 scene flow 数据
        if (not os.path.exists(os.path.join(scene_flow_data_path, previous_sf_record['folder_name'], 'pc1.npy'))
                or not os.path.exists(os.path.join(scene_flow_data_path, previous_sf_record['folder_name'], 'pc3.npy'))):
            i = i + 1
            continue
        points_prev = np.load(os.path.join(scene_flow_data_path, previous_sf_record['folder_name'], 'pc1.npy'))
        points_curr = np.load(os.path.join(scene_flow_data_path, previous_sf_record['folder_name'], 'pc3.npy'))
        # 对 scene flow 数据进行时间插值
        # sf = pc3 - pc1; sf' = time_diff_ratio * sf; pc3' = pc1 + sf'
        # => pc3' = pc1 + time_diff_ratio * (pc3 - pc1)
        # => pc3' = time_diff_ratio * pc3 + (1 - time_diff_ratio) * pc1
        points_curr = time_diff_ratio * points_curr + (1 - time_diff_ratio) * points_prev

        # 计算激光雷达坐标系下的尺度
        # depth 为 x-y 平面上距原点的距离
        depth_xy_prev = np.linalg.norm(points_prev[:, :2], axis=1)
        depth_xy_curr = np.linalg.norm(points_curr[:, :2], axis=1)

        # 避免除以零
        depth_xy_prev[depth_xy_prev == 0] = 1e-6
        scales = depth_xy_curr / depth_xy_prev

        # Range Projection
        proj_range, proj_scale, proj_xyz, proj_idx, proj_mask = range_projection(
            points_prev,
            scales,
            H=480,
            W=5120,
            fov_up=10.0,
            fov_down=-30.0
        )

        ###################################### 可视化 ######################################
        if VISUALIZATION:
            # 分段归一化；使用 bwr 色表
            scale_display = np.copy(proj_scale)
            valid_mask = (scale_display != -1)

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
                if pos_max >= pos_min >= 0:
                    normalized_display[pos_mask] = (pos_devs - pos_min) / (pos_max - pos_min)  # 归一化到 [0,1]
                else:
                    normalized_display[pos_mask] = 0.0  # 如果没有变化，设为0

            # 处理小于1的尺度
            if np.any(neg_mask):
                neg_devs = deviations[neg_mask]
                neg_max = neg_devs.max()
                neg_min = neg_devs.min()
                if neg_min <= neg_max <= 0:
                    normalized_display[neg_mask] = (neg_devs - neg_min) / (neg_max - neg_min) - 1.0  # 归一化到 [-1,0]
                    # normalized_display[neg_mask] = neg_devs / abs(neg_min)  # 归一化到 [-1,0]
                else:
                    normalized_display[neg_mask] = 0.0  # 如果没有变化，设为0

            # 设置等于1的尺度为0
            # 由于 deviations = scale -1，等于1的scale对应 deviations = 0，已经在 normalized_display 中为0

            # 对无效像素赋值为0
            normalized_display[~valid_mask] = 0.0

            # # 打印当前尺度的范围
            # pos_range = deviations[pos_mask] if np.any(pos_mask) else np.array([])
            # neg_range = deviations[neg_mask] if np.any(neg_mask) else np.array([])
            #
            # if pos_range.size > 0:
            #     print(f"Scale >1 range: [{pos_range.min()}, {pos_range.max()}]")
            # if neg_range.size > 0:
            #     print(f"Scale <1 range: [{neg_range.min()}, {neg_range.max()}]")

            # 保存归一化后的 scale 映射图像，使用 bwr 色表
            out_name = f"seismic_clip_range_{i}.png"
            out_path = os.path.join(vis_output_dir, out_name)

            plt.imsave(out_path, normalized_display, cmap='seismic', vmin=-1, vmax=1)
            # print(f"Saved range image: {out_path}")

        ###################################### 保存 GT 数据 ######################################
        if GENERATE_GT:
            # 保存scale map - proj_scale
            scale_map = proj_scale
            sub_folder_name = previous_sf_record['folder_name']
            os.makedirs(os.path.join(gt_save_path, sub_folder_name), exist_ok=True)
            scale_map_filename = "scale_map.npy"
            scale_map_path = os.path.join(gt_save_path, sub_folder_name, scale_map_filename)
            np.save(scale_map_path, scale_map)

            original_img_path = {}
            for camera_channel, data in previous_surround_view_data.items():
                original_img_path[camera_channel] = previous_surround_view_data[camera_channel]['filename']

            # GT: timestamp, original_imgs_path, scale_map_path
            info = {
                'timestamp': previous_sf_record['timestamp'],
                'original_imgs_path': original_img_path,
                'scale_map_path': scale_map_path
            }

            train_infos.append(info)

    if GENERATE_GT:
        # 保存 train_infos.pkl
        with open(os.path.join(train_pkl_save_path, 'nusc_range_image_train_infos.pkl'), 'wb') as f:
            pickle.dump(train_infos, f)
        print(f"Saved nusc_range_image_train_infos.pkl to {train_pkl_save_path}")

if __name__ == "__main__":
    main()






# # ------------------- 主逻辑：遍历子文件夹，读取 pc1.npy 和 pc3.npy，投影并保存 -------------------
# def main():
#     # 根目录，包含多个类似 n015-2018-xx__LIDAR_TOP__1531883538048176 的子文件夹
#     base_dir = "/mnt/fpttc_data/scene_flow/multi_frame/31_scene_flow"
#     # 输出图像目录
#     output_dir = "/mnt/fpttc_data/output_vis/31_range_img"
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 找到所有子文件夹
#     # 子文件夹名形如: n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883538048176
#     subfolders = glob.glob(os.path.join(base_dir, "n015-*__LIDAR_TOP__*"))
#     if not subfolders:
#         print("No matching subfolders found.")
#         return
#
#     # 根据时间戳排序
#     # folder_name 类似: ".../n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883538048176"
#     # 取最后一个 '__' 后面的UNIX时间戳做排序
#     def extract_timestamp(folder_path):
#         folder_basename = os.path.basename(folder_path)
#         # "n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883538048176"
#         # split('__')[-1] = '1531883538048176'
#         ts_str = folder_basename.split('__')[-1]
#         return int(ts_str)  # 转 int 便于按数值排序
#
#     subfolders_sorted = sorted(subfolders, key=extract_timestamp)
#
#     # # 计算相邻时间戳差值，并处理成 50ms 的倍数
#     # timestamp_differences = []
#     # for i in range(1, len(subfolders_sorted)):
#     #     ts_current = extract_timestamp(subfolders_sorted[i])
#     #     ts_previous = extract_timestamp(subfolders_sorted[i - 1])
#     #
#     #     # 计算时间戳差值（单位为毫秒）
#     #     time_diff_ms = (ts_current - ts_previous) * 1e-3
#     #
#     #     # 计算50ms的倍数并取整
#     #     time_diff_50ms = round(time_diff_ms / 50)
#     #     timestamp_differences.append(time_diff_50ms)
#
#     # 依序处理
#     for i, folder_path in enumerate(subfolders_sorted):
#         # pc1.npy 和 pc3.npy 路径
#         pc1_path = os.path.join(folder_path, "pc1.npy")
#         pc3_path = os.path.join(folder_path, "pc3.npy")
#         if not os.path.isfile(pc1_path) or not os.path.isfile(pc3_path):
#             print(f"Skipping {folder_path}, pc1.npy or pc3.npy not found.")
#             continue
#
#         print(f"[{i}] Processing folder: {folder_path}")
#
#         # 读取点云
#         points_prev = np.load(pc1_path)  # shape (m, 3)
#         points_curr = np.load(pc3_path)  # shape (m, 3)
#
#         # 计算激光雷达坐标系下的尺度
#         # depth 为 x-y 平面上距原点的距离
#         depth_xy_prev = np.linalg.norm(points_prev[:, :2], axis=1)
#         depth_xy_curr = np.linalg.norm(points_curr[:, :2], axis=1)
#
#         # 避免除以零
#         depth_xy_prev[depth_xy_prev == 0] = 1e-6
#         scales = depth_xy_curr / depth_xy_prev
#
#         # Range Projection
#         proj_range, proj_scale, proj_xyz, proj_idx, proj_mask = range_projection(
#             points_prev,
#             scales,
#             H=480,
#             W=5120,
#             fov_up=10.0,
#             fov_down=-30.0
#         )
#
#         # 分段归一化；使用 bwr 色表
#         scale_display = np.copy(proj_scale)
#         valid_mask = (scale_display != -1)
#
#         # 将用于可视化的尺度值裁切到 [0.5, 1.5] 范围
#         scale_display[valid_mask] = np.clip(scale_display[valid_mask], 0.9, 1.1)
#
#         # 将尺度的分界线从 1 移至 0（scale - 1）
#         deviations = np.zeros_like(scale_display, dtype=np.float32)
#         deviations[valid_mask] = scale_display[valid_mask] - 1.0
#
#         # 分别处理大于0和小于0的部分
#         pos_mask = deviations > 0
#         neg_mask = deviations < 0
#
#         # 初始化归一化后的显示数组
#         normalized_display = np.zeros_like(deviations, dtype=np.float32)
#
#         # 处理大于1的尺度
#         if np.any(pos_mask):
#             pos_devs = deviations[pos_mask]
#             pos_max = pos_devs.max()
#             pos_min = pos_devs.min()
#             if pos_max >= pos_min >= 0:
#                 normalized_display[pos_mask] = (pos_devs - pos_min) / (pos_max - pos_min)  # 归一化到 [0,1]
#             else:
#                 normalized_display[pos_mask] = 0.0  # 如果没有变化，设为0
#
#         # 处理小于1的尺度
#         if np.any(neg_mask):
#             neg_devs = deviations[neg_mask]
#             neg_max = neg_devs.max()
#             neg_min = neg_devs.min()
#             if neg_min <= neg_max <= 0:
#                 normalized_display[neg_mask] = (neg_devs - neg_min) / (neg_max - neg_min) - 1.0  # 归一化到 [-1,0]
#                 # normalized_display[neg_mask] = neg_devs / abs(neg_min)  # 归一化到 [-1,0]
#             else:
#                 normalized_display[neg_mask] = 0.0  # 如果没有变化，设为0
#
#         # 设置等于1的尺度为0
#         # 由于 deviations = scale -1，等于1的scale对应 deviations = 0，已经在 normalized_display 中为0
#
#         # 对无效像素赋值为0
#         normalized_display[~valid_mask] = 0.0
#
#         # 打印当前尺度的范围
#         pos_range = deviations[pos_mask] if np.any(pos_mask) else np.array([])
#         neg_range = deviations[neg_mask] if np.any(neg_mask) else np.array([])
#
#         if pos_range.size > 0:
#             print(f"Scale >1 range: [{pos_range.min()}, {pos_range.max()}]")
#         if neg_range.size > 0:
#             print(f"Scale <1 range: [{neg_range.min()}, {neg_range.max()}]")
#
#         # 保存归一化后的 scale 映射图像，使用 bwr 色表
#         out_name = f"seismic_clip_range_{i}.png"
#         out_path = os.path.join(output_dir, out_name)
#
#         plt.imsave(out_path, normalized_display, cmap='seismic', vmin=-1, vmax=1)
#         print(f"Saved range image: {out_path}")
#
#     print("All done.")
#
# if __name__ == "__main__":
#     main()
