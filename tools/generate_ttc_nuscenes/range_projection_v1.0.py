import os
import glob
import numpy as np
import matplotlib.pyplot as plt

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

# ------------------- 主逻辑：遍历子文件夹，读取 pc1.npy 和 pc3.npy，投影并保存 -------------------
def main():
    # 根目录，包含多个类似 n015-2018-xx__LIDAR_TOP__1531883538048176 的子文件夹
    base_dir = "/mnt/fpttc_data/scene_flow/multi_frame/31_scene_flow"
    # 输出图像目录
    output_dir = "/mnt/fpttc_data/output_vis/31_range_img"
    os.makedirs(output_dir, exist_ok=True)

    # 找到所有子文件夹
    # 子文件夹名形如: n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883538048176
    subfolders = glob.glob(os.path.join(base_dir, "n015-*__LIDAR_TOP__*"))
    if not subfolders:
        print("No matching subfolders found.")
        return

    # 根据时间戳排序
    # folder_name 类似: ".../n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883538048176"
    # 取最后一个 '__' 后面的UNIX时间戳做排序
    def extract_timestamp(folder_path):
        folder_basename = os.path.basename(folder_path)
        # "n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883538048176"
        # split('__')[-1] = '1531883538048176'
        ts_str = folder_basename.split('__')[-1]
        return int(ts_str)  # 转 int 便于按数值排序

    subfolders_sorted = sorted(subfolders, key=extract_timestamp)

    # # 计算相邻时间戳差值，并处理成 50ms 的倍数
    # timestamp_differences = []
    # for i in range(1, len(subfolders_sorted)):
    #     ts_current = extract_timestamp(subfolders_sorted[i])
    #     ts_previous = extract_timestamp(subfolders_sorted[i - 1])
    #
    #     # 计算时间戳差值（单位为毫秒）
    #     time_diff_ms = (ts_current - ts_previous) * 1e-3
    #
    #     # 计算50ms的倍数并取整
    #     time_diff_50ms = round(time_diff_ms / 50)
    #     timestamp_differences.append(time_diff_50ms)

    # 依序处理
    for i, folder_path in enumerate(subfolders_sorted):
        # pc1.npy 和 pc3.npy 路径
        pc1_path = os.path.join(folder_path, "pc1.npy")
        pc3_path = os.path.join(folder_path, "pc3.npy")
        if not os.path.isfile(pc1_path) or not os.path.isfile(pc3_path):
            print(f"Skipping {folder_path}, pc1.npy or pc3.npy not found.")
            continue

        print(f"[{i}] Processing folder: {folder_path}")

        # 读取点云
        points_prev = np.load(pc1_path)  # shape (m, 3)
        points_curr = np.load(pc3_path)  # shape (m, 3)

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

        # 分段归一化；使用 bwr 色表
        scale_display = np.copy(proj_scale)
        valid_mask = (scale_display != -1)

        # 将用于可视化的尺度值裁切到 [0.5, 1.5] 范围
        scale_display[valid_mask] = np.clip(scale_display[valid_mask], 0.9, 1.1)

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

        # 打印当前尺度的范围
        pos_range = deviations[pos_mask] if np.any(pos_mask) else np.array([])
        neg_range = deviations[neg_mask] if np.any(neg_mask) else np.array([])

        if pos_range.size > 0:
            print(f"Scale >1 range: [{pos_range.min()}, {pos_range.max()}]")
        if neg_range.size > 0:
            print(f"Scale <1 range: [{neg_range.min()}, {neg_range.max()}]")

        # 保存归一化后的 scale 映射图像，使用 bwr 色表
        out_name = f"seismic_clip_range_{i}.png"
        out_path = os.path.join(output_dir, out_name)

        plt.imsave(out_path, normalized_display, cmap='seismic', vmin=-1, vmax=1)
        print(f"Saved range image: {out_path}")

    print("All done.")

if __name__ == "__main__":
    main()
