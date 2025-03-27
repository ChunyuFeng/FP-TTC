import os
import cv2
import numpy as np
import math
import time
from sklearn.cluster import DBSCAN
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import pickle
from nuscenes.nuscenes import NuScenes

from collision_utils import (get_second_grad,
                             get_ttc_var,
                             get_grid_ttc,
                             visual_scale_map_range_image,
                             inverse_range_projection,
                             project_lidar_to_surround_view_img)

nusc = NuScenes(version='v1.0-trainval', dataroot='./Datasets/nuscenes',
                verbose=True)

# ==================== 主处理代码 ====================

# 参数设置
fps = 12
delta_t = 1.0 / fps
approach_threshold = 1.0
risk_time_threshold = 3  # 设定风险阈值（单位与 delta_t 相关，可根据需要调整）
grid_size = 3
risk_rate = 0.28
kp_detector_threshold = 2
grad_step = 2
vertical_risk_line = 0.5

# 设置交互和保存模式：KEYBOARD==True时逐帧等待键盘输入；SAVE==True时保存结果图而不等待
KEYBOARD = False
SAVE = True

# 输入目录，目录下存储 pred{i}.npy 文件
pred_scale_path = '/mnt/fpttc_data/output_vis/collision_detect/25_03_22-11_53_44_selfcon_ttc'
gt_scale_path = '/mnt/fpttc_data/output_vis/collision_detect/25_03_24-21_14_41_selfcon_ttc'
depth_range_image_path = '/mnt/fpttc_data/output_vis/collision_detect/25_03_26-20_52_26_selfcon_ttc'
infos4proj_pkl_path = '/mnt/fpttc_data/output_vis/collision_detect/infos_for_projection.pkl'
output_dir = '/mnt/fpttc_data/output_vis/collision_detect/collision_points_vis'
rgb_output_dir = '/mnt/fpttc_data/output_vis/collision_detect/collision_points_on_rgb'
image_list_prefix = '/home/chunyu/WorkSpace/BugStudio/FP-TTC/Datasets/nuscenes'
pred_file_list = sorted(
    [f for f in os.listdir(pred_scale_path) if f.endswith('.npy')],
    key=lambda x: int(x[4:-4])
)
gt_file_list = sorted(
    [f for f in os.listdir(gt_scale_path) if f.endswith('.npy')],
    key=lambda x: int(x[2:-4])
)
depth_file_list = sorted(
    [f for f in os.listdir(depth_range_image_path) if f.endswith('.npy')],
    key=lambda x: int(x[5:-4])
)
# 保存所有文件的碰撞风险检测结果（可选）
collision_risk_results = {}

# 采用 matplotlib 的 seismic 色图
cmap = mpl_cm.get_cmap('seismic')

# 初始化 matplotlib 图像
plt.ion()
fig, ax = plt.subplots()
ax.axis('off')

# 读取 image_list.pkl
with open(infos4proj_pkl_path, 'rb') as f:
    infos_for_proj = pickle.load(f)

file_idx = 0

for filename in pred_file_list:

    file_path = os.path.join(pred_scale_path, filename)
    print("Processing file:", filename)
    # 加载 Scale 数据，尺寸 (160,1920)
    scale_data = np.load(file_path)
    H, W = scale_data.shape
    proc_h = (math.ceil(H / grid_size) - 1) * grid_size
    proc_w = (math.ceil(W / grid_size) - 1) * grid_size
    scale_data = scale_data[:proc_h, :proc_w]
    
    # 此处将所有像素视为有效区域
    scale_mask = scale_data > 0
    
    # 内部归一化（仅用于候选点提取），将 Scale 数据归一化到 0~255
    norm_scale = (scale_data - np.min(scale_data)) / (approach_threshold - np.min(scale_data)) * 255.0
    
    # 计算网格内方差
    scale_variance = get_ttc_var(scale_data, grid_len=grid_size)
    
    # 计算二阶梯度
    grad_y, grad_x, grad_yy, grad_xx = get_second_grad(scale_data, stride=grad_step)
    
    # 构造梯度有效掩码：选择梯度较小且 Scale 值低于 approach_threshold 的区域
    valid_grad_mask = np.logical_and(np.abs(grad_y) < risk_rate * np.mean(np.abs(grad_y)),
                                     scale_data[grad_step:, :] < approach_threshold)
    valid_grad_mask = np.logical_and(valid_grad_mask, scale_data[1:-1, :] < approach_threshold)
    
    # 网格划分及有效区域筛选
    grid_valid_map, grid_valid_mask = get_grid_ttc(scale_data, scale_variance, grid_len=grid_size)
    grid_valid_mask = np.logical_and(grid_valid_mask, scale_data < approach_threshold).astype(np.uint8)
    grid_valid_mask[-(grid_size*2):, :] = 0
    grid_valid_map[-2:, :] = 0
    
    # -------------------- FAST 关键点检测 --------------------
    scale_vis_gray = np.asarray(norm_scale, dtype=np.uint8)
    fast_detector = cv2.FastFeatureDetector_create(6400)
    fast_detector.setThreshold(kp_detector_threshold)
    keypoints = fast_detector.detect(scale_vis_gray, None)
    
    # -------------------- 候选点提取 --------------------
    candidate_points = []
    dense_kp_map = np.zeros_like(valid_grad_mask, dtype=np.uint8)
    invalid_candidates = []
    for kp in keypoints:
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        # 只关注图像中部区域
        if y < proc_h * 0.4 or y > proc_h * 0.9:
            continue
        # 限制横向范围，避免两侧噪声
        if x < proc_w * 0.2 or x > proc_w * 0.8:
            continue
        area_size = 9
        kp_area_size = 20
        y0 = max(y - area_size // 2, 0)
        y1_area = min(y + area_size // 2 + 1, proc_h)
        x0 = max(x - area_size // 2, 0)
        x1_area = min(x + area_size // 2 + 1, proc_w)
        area = valid_grad_mask[y0:y1_area, x0:x1_area]
        dense_kp_map[y0:y1_area, x0:x1_area] = 1
        if area.sum() > 3:
            area_norm = norm_scale[y0:y1_area, x0:x1_area]
            min_scale_in_area = np.min(area_norm)
            area_orig = scale_data[y0:y1_area, x0:x1_area]
            local_min = np.min(area_orig)
            local_max = np.max(area_orig)
            if local_min < approach_threshold and local_max < 1.04:
                candidate_points.append([x, y, min_scale_in_area])
        else:
            invalid_candidates.append([x, y])
    
    for pt in invalid_candidates:
        x = int(pt[0])
        y = int(pt[1])
        area_size = 7
        if dense_kp_map[y, x]:
            y0 = max(y - area_size // 2, 0)
            y1_area = min(y + area_size // 2 + 1, proc_h)
            x0 = max(x - area_size // 2, 0)
            x1_area = min(x + area_size // 2 + 1, proc_w)
            area_norm = norm_scale[y0:y1_area, x0:x1_area]
            min_scale_in_area = np.min(area_norm)
            area_orig = scale_data[y0:y1_area, x0:x1_area]
            local_min = np.min(area_orig)
            local_max = np.max(area_orig)
            if local_min < approach_threshold and local_max < 1.04:
                candidate_points.append([x, y, min_scale_in_area])
    
    for i in range(grid_valid_map.shape[0]):
        for j in range(grid_valid_map.shape[1]):
            cell = scale_data[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
            cell_mean = np.mean(cell)
            center_x = int((2*j+1)*grid_size/2)
            center_y = int((2*i+1)*grid_size/2)
            if grid_valid_map[i, j] and valid_grad_mask[center_y, center_x] and cell_mean < approach_threshold and center_y > proc_h * 0.4:
                candidate_points.append([center_x, center_y, np.mean(norm_scale[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size])])
    
    candidate_points = np.array(candidate_points)
    
    # -------------------- 候选点聚类与碰撞风险计算 --------------------
    collision_points = []  # 存储检测到的碰撞风险点 [x, y, collision_time, 原始scale值]
    if len(candidate_points) > 0:
        clustering1 = DBSCAN(eps=50, min_samples=5).fit(candidate_points)
        clusters_initial = {}
        for idx, label in enumerate(clustering1.labels_):
            if label == -1:
                continue
            clusters_initial.setdefault(label, []).append(candidate_points[idx])
        
        secondary_centers = []
        secondary_regions = []
        for label, pts in clusters_initial.items():
            pts_array = np.array(pts)[:, :2].astype(int)
            (cx, cy), radius = cv2.minEnclosingCircle(pts_array)
            # 计算碰撞时刻： collision_time = delta_t/(1 - scale + 1e-5)
            collision_time = delta_t / (1 - scale_data[int(cy), int(cx)] + 1e-5)
            if radius < 6 and pts_array.shape[0] < 4:
                continue
            secondary_centers.append([cx, cy, collision_time * 200])
            secondary_regions.append(pts_array)
        
        secondary_centers = np.array(secondary_centers)
        if len(secondary_centers) > 0:
            clustering2 = DBSCAN(eps=100, min_samples=3).fit(secondary_centers)
            clusters_secondary = {}
            region_points = {}
            for idx, label in enumerate(clustering2.labels_):
                clusters_secondary.setdefault(label, []).append(secondary_centers[idx])
                region_points.setdefault(label, []).append(secondary_regions[idx])
            for label, centers in clusters_secondary.items():
                if label == -1:
                    for region in region_points[label]:
                        pts_array = region
                        (cx, cy), radius = cv2.minEnclosingCircle(pts_array)
                        collision_time = delta_t / (1 - scale_data[int(cy), int(cx)] + 1e-5)
                        if collision_time > 0 and collision_time < risk_time_threshold:
                            collision_points.append([int(cx), int(cy), collision_time, scale_data[int(cy), int(cx)]])
                else:
                    merged_pts = np.concatenate(region_points[label])
                    (cx, cy), radius = cv2.minEnclosingCircle(merged_pts)
                    collision_times = []
                    for pt in merged_pts:
                        collision_times.append(delta_t / (1 - scale_data[int(pt[1]), int(pt[0])] + 1e-5))
                    collision_time = np.min(np.array(collision_times))
                    if collision_time > 0 and collision_time < risk_time_threshold:
                        collision_points.append([int(cx), int(cy), collision_time, scale_data[int(cy), int(cx)]])
    
    collision_risk_results[filename] = collision_points
    print("File:", filename, "Collision risk points:", collision_points)
    
    # -------------------- 可视化 --------------------
    # 使用 visual_scale_map_range_image 生成归一化显示图（范围 [-1,1]）
    scale_data_full = np.load(file_path)
    pred_valid_mask = scale_data_full > 0
    norm_display = visual_scale_map_range_image(scale_data_full, pred_valid_mask)
    # 在当前坐标轴上显示图像，使用 seismic colormap，vmin=-1, vmax=1
    ax.imshow(norm_display, cmap='seismic', vmin=-1, vmax=1)
    # 在图上叠加碰撞风险点及其碰撞时间
    for pt in collision_points:
        x, y, ct, scale_value = pt
        circle = plt.Circle((x, y), 5, color='red', fill=True)
        ax.add_patch(circle)
        ax.text(x + 5, y - 5, f"{ct:.2f}", color='black', fontsize=16)
    
    if KEYBOARD:
        plt.title(f"{filename}")
        plt.draw()
        plt.waitforbuttonpress()  # 逐帧等待键盘输入
        ax.clear()
    elif SAVE:
        # -------------- 保存碰撞点在 Pred 图上的可视化结果 --------------
        dpi = 100
        fig.set_size_inches(scale_data_full.shape[1] / dpi, scale_data_full.shape[0] / dpi)
        # 不显示坐标轴和标题
        ax.axis('off')
        # 保存时去掉边框
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{filename[:-4]}.png")
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
        ax.clear()

        # -------------- 保存碰撞点在 GT 图上的可视化结果 --------------
        # 对应的 gt 文件名将 "pred" 替换为 "gt"
        gt_filename = filename.replace("pred", "gt")
        gt_file_path = os.path.join(gt_scale_path, gt_filename)
        if os.path.exists(gt_file_path):
            gt_scale = np.load(gt_file_path)  # gt_scale 尺寸应为 (160,1920)
            gt_mask = gt_scale > 0
            norm_display_gt = visual_scale_map_range_image(gt_scale, gt_mask)
            ax.imshow(norm_display_gt, cmap='seismic', vmin=-1, vmax=1)
            # 叠加与预测相同的碰撞风险点
            for pt in collision_points:
                x, y, ct, scale_value = pt
                circle = plt.Circle((x, y), 5, color='red', fill=True)
                ax.add_patch(circle)
                ax.text(x + 5, y - 5, f"{ct:.2f}", color='black', fontsize=16)
            fig.set_size_inches(gt_scale.shape[1] / dpi, gt_scale.shape[0] / dpi)
            ax.axis('off')
            output_file_gt = os.path.join(output_dir, f"{gt_filename[:-4]}.png")
            plt.savefig(output_file_gt, dpi=dpi, bbox_inches='tight', pad_inches=0)
            ax.clear()

        # -------------- 保存碰撞点在环视 RGB 图上可视化结果 --------------
        # 对应的 depth 文件名将 "pred" 替换为 "depth"
        depth_filename = filename.replace("pred", "depth")
        depth_file_path = os.path.join(depth_range_image_path, depth_filename)
        if os.path.exists(depth_file_path):
            depth_data = np.load(depth_file_path)

        lidar_token = infos_for_proj['lidar_tokens'][file_idx]
        surround_view_img_token = infos_for_proj['image_tokens'][file_idx]

        if len(collision_points) > 0:
            # 有碰撞风险点，逐个处理
            for pt in collision_points:
                x, y, ct, _ = pt
                depth = depth_data[y, x]
                if depth < 0.0:
                    continue
                # 将 2D 碰撞点反投影到 3D
                x3d, y3d, z3d = inverse_range_projection(x, y, depth, H=160, W=1920, fov_up=10.0, fov_down=-30.0)
                points_3d = np.array([[x3d, y3d, z3d]])
                
                proj_pts = project_lidar_to_surround_view_img(nusc,
                                                            points_3d,
                                                            lidar_token,
                                                            surround_view_img_token,
                                                            min_dist=1.0)
                for channel in proj_pts:
                    # 取出当前通道的数据，并提取2D坐标（只取前两行）
                    data = proj_pts[channel]
                    pts2d = data['points'][:2, :]

                    # 获取原始图像并确保为 numpy 数组
                    original_img = data['original_img']
                    if not isinstance(original_img, np.ndarray):
                        original_img = np.array(original_img)
                    original_img = original_img.copy()
                    
                    if pts2d.shape[1] > 0:
                        x_proj, y_proj = int(pts2d[0, 0]), int(pts2d[1, 0])
                        # 调大圆形标记（半径10）和加粗绘制
                        cv2.circle(original_img, (x_proj, y_proj), 10, (0, 0, 255), thickness=-1)
                        # 添加较大的文字显示碰撞时间，字体比例 1，厚度 2
                        cv2.putText(original_img, f"{ct:.2f}", (x_proj + 15, y_proj - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
                    
                    # 因为输入为 RGB，保存前转换为 BGR 格式
                    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                    output_filename = os.path.join(rgb_output_dir, f"{channel}{file_idx}.jpg")
                    cv2.imwrite(output_filename, original_img_bgr)
        else:
            # 当前帧无碰撞风险点，仍需保存原始图像
            # 调用投影函数，传入空的 3D 点数组，获取原图
            points_3d = np.empty((0, 3))
            proj_pts = project_lidar_to_surround_view_img(nusc,
                                                        points_3d,
                                                        lidar_token,
                                                        surround_view_img_token,
                                                        min_dist=1.0)
            for channel in proj_pts:
                data = proj_pts[channel]
                original_img = data['original_img']
                if not isinstance(original_img, np.ndarray):
                    original_img = np.array(original_img)
                original_img = original_img.copy()
                original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                output_filename = os.path.join(rgb_output_dir, f"{channel}{file_idx}.jpg")
                cv2.imwrite(output_filename, original_img_bgr)


        # # -------------- 保存碰撞点在环视 RGB 图上可视化结果 --------------
        # # 对应的 depth 文件名将 "pred" 替换为 "depth"
        # depth_filename = filename.replace("pred", "depth")
        # depth_file_path = os.path.join(depth_range_image_path, depth_filename)
        # if os.path.exists(depth_file_path):
        #     depth_data = np.load(depth_file_path)
        # for pt in collision_points:
        #     x, y, ct, _ = pt
        #     depth = depth_data[y, x]
        #     if depth < 0.0:
        #         continue
        #     x3d, y3d, z3d = inverse_range_projection(x, y, depth, H=160, W=1920, fov_up=10.0, fov_down=-30.0)
        #     points_3d = np.array([[x3d, y3d, z3d]])
        #     lidar_token = infos_for_proj['lidar_tokens'][file_idx]
        #     surround_view_img_token = infos_for_proj['image_tokens'][file_idx]        
        #     proj_pts = project_lidar_to_surround_view_img(nusc,
        #                                                   points_3d,
        #                                                   lidar_token,
        #                                                   surround_view_img_token,
        #                                                   min_dist=1.0)
        #     for channel in proj_pts:
        #         # 取出当前通道的数据，并提取2D坐标（只取前两行）
        #         data = proj_pts[channel]
        #         pts2d = data['points'][:2, :]

        #         # 确保 original_img 为 numpy 数组，如果不是，则转换
        #         original_img = data['original_img']
        #         if not isinstance(original_img, np.ndarray):
        #             original_img = np.array(original_img)
        #         original_img = original_img.copy()
                
        #         # 如果该通道存在投影点（假设只有一个点）
        #         if pts2d.shape[1] > 0:
        #             x, y = int(pts2d[0, 0]), int(pts2d[1, 0])
        #             # 绘制红色实心圆标记碰撞风险点
        #             cv2.circle(original_img, (x, y), 10, (0, 0, 255), thickness=-1)
                    
        #             # 假设碰撞时间已由变量 collision_time（例如：collision_time = ct）给出
        #             collision_time = ct  # 请确保此变量在当前作用域中已定义
        #             # 添加文本显示碰撞时间（保留两位小数）
        #             cv2.putText(original_img, f"{collision_time:.2f}", (x + 15, y - 15),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
                    
        #         original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        #         # 保存图像，格式为：channel{file_idx}.jpg，保存路径为 rgb_output_dir
        #         output_filename = os.path.join(rgb_output_dir, f"{channel}{file_idx}.jpg")
        #         cv2.imwrite(output_filename, original_img_bgr)

    else:
        # 如果既不交互也不保存，则仅刷新显示
        plt.draw()
        ax.clear()

    file_idx += 1

plt.ioff()
print("Processing complete. Collision risk results:")
for fn, risks in collision_risk_results.items():
    print(fn, ":", risks)
