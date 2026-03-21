#!/usr/bin/env python3
import os
import cv2
import numpy as np
import math
import argparse
import pickle
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import mplcursors
import random

from dataloader.utils.augmentor import NuscRangeImageAugmentor

from utils.draw import (
    visual_scale_map_range_image,
    visual_risk_score_map_range_image
)
from utils.nusc_paths import resolve_nusc_path
from collision_utils import (
    get_second_grad,
    get_ttc_var,
    get_grid_ttc,
    inverse_range_projection,
    project_lidar_to_surround_view_img
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collision risk detection from scale predictions"
    )
    # Dataset settings
    parser.add_argument("--version", type=str, default="v1.0-trainval",
                        help="NuScenes dataset version")
    parser.add_argument("--dataroot", type=str, default="./Datasets/nuscenes",
                        help="Path to NuScenes dataroot")
    # Timing and thresholds
    parser.add_argument("--fps", type=int, default=12,
                        help="Frame rate for delta_t calculation")
    parser.add_argument("--approach_threshold", type=float, default=1.0,
                        help="Scale threshold for valid region")
    parser.add_argument("--risk_time_threshold", type=float, default=3.0,
                        help="Max collision time to consider risk (seconds)")
    parser.add_argument("--risk_pred_threshold", type=float, default=0.0,
                        help="Only keep collision points whose risk > this threshold")
    # Feature extraction parameters
    parser.add_argument("--grid_size", type=int, default=3,
                        help="Size of grid cell for variance and grid TTC")
    parser.add_argument("--risk_rate", type=float, default=0.28,
                        help="Proportion of mean gradient for valid_grad_mask")
    parser.add_argument("--kp_detector_threshold", type=int, default=2,
                        help="FAST keypoint detector threshold")
    parser.add_argument("--grad_step", type=int, default=2,
                        help="Stride for second-order gradient computation")
    parser.add_argument('--image_size', default=[160, 320], type=int, nargs='+',
                        help='[height, width] for resize operations')
    # Modes
    parser.add_argument("--keyboard", action="store_true",
                        help="Wait for keyboard input each frame")
    parser.add_argument("--save", action="store_true",
                        help="Save results instead of interactive display")
    # Paths
    parser.add_argument("--pred_npy_dir", type=str, required=True,
                        help="Directory of predicted .npy files, including pred scale and risk")
    parser.add_argument('--test_info_path', default='./Datasets/nuscenes/2_trainval_test_infos/nusc_trainval_infos_160_1920.pkl',
                        type=str,
                        help='Path to test info file (e.g., nusc_trainval_infos_160_1920.pkl)')
    parser.add_argument("--vis_dir", type=str, required=True,
                        help="Output directory for RGB overlays")
    parser.add_argument("--sjtu_test", action="store_true",
                        help="Use SJTU test set, which has no ground truth data")
    parser.add_argument("--sjtu_surround_view_path", type=str, default='./Datasets/sjtu_surround_view',
                        help="Path to SJTU surround view images")
    parser.add_argument("--sjtu_scene_indice", type=int, default=0,
                        help="Scene index for SJTU test set, default is 0")
    return parser.parse_args()

def inspect_scale_risk(scale_map, risk_map):
    """
    弹出一个 1×2 的窗口：
     - 左图显示原始 scale_map，鼠标 hover 即可查看 scale 值
     - 右图显示原始 risk_map，鼠标 hover 即可查看 risk 值
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(scale_map, cmap='viridis')
    axes[0].set_title('Scale Map')
    axes[0].axis('off')
    im1 = axes[1].imshow(risk_map, cmap='inferno')
    axes[1].set_title('Risk Map')
    axes[1].axis('off')
    plt.tight_layout()

    # scale hover
    cursor0 = mplcursors.cursor(im0, hover=True)
    @cursor0.connect("add")
    def _(sel):
        x, y = int(sel.target[0]+0.5), int(sel.target[1]+0.5)
        sel.annotation.set_text(f"({x},{y})  scale={scale_map[y,x]:.3f}")

    # risk hover
    cursor1 = mplcursors.cursor(im1, hover=True)
    @cursor1.connect("add")
    def _(sel):
        x, y = int(sel.target[0]+0.5), int(sel.target[1]+0.5)
        sel.annotation.set_text(f"({x},{y})  risk ={risk_map[y,x]:.3f}")

    plt.show()

def compute_and_save_masks(scale_map, risk_map, delta_t, ttc_thresh, risk_thresh, save_dir, idx):
    """
    计算两个 mask 后，将它们以二值图形式保存到磁盘：
      mask_ttc： ttc = delta_t/(1-scale_map) < ttc_thresh
      mask_risk：risk_map > risk_thresh
    save_dir:  输出目录
    idx:       用于文件命名的帧索引
    返回:
      mask_ttc, mask_risk （布尔数组）
    """
    eps = 1e-5
    # 1) 计算 ttc 和两个 mask
    ttc       = delta_t / (1.0 - scale_map + eps)
    mask_ttc  = (ttc       < ttc_thresh)
    mask_risk = (risk_map  > risk_thresh)
    # 2) 转为 0/255 的 uint8 图
    ttc_img  = (mask_ttc .astype(np.uint8) * 255)
    risk_img = (mask_risk.astype(np.uint8) * 255)
    # 3) 确保目录存在并保存
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"mask_ttc_{idx}.png"),  ttc_img)
    cv2.imwrite(os.path.join(save_dir, f"mask_risk_{idx}.png"), risk_img)
    return mask_ttc, mask_risk


def compute_and_save_percentile_masks_and_compose(
    scale_map, save_dir, idx
):
    """
    1) 在 scale_map 的子区域（纵向 30%~90%，横向 10%~90%）中，
       计算最小 thr_percentile% 阈值 thr_scale。
    2) 在全图上生成 mask_scale_low：子区域内 scale ≤ thr_scale。
    3) 生成章节 5（sector）掩码 valid_mask：
       - 横向角度范围 [-112.5°, -67.5°]
       - 纵向行数范围 [30%H, 80%H]
    4) 纵向拼接 4 幅图并保存：
       [0] 全量 scale
       [1] scale ≤ thr_scale
       [2] sector scale map (angle & row mask)
       [3] 二者交集：scale ≤ thr_scale ∧ sector mask
    """
    thr_percentile = 5
    H, W = scale_map.shape

    # 1) 子区域 ROI & 计算 thr_scale
    y1, y2 = int(0.3*H), int(0.9*H)
    x1, x2 = int(0.1*W), int(0.9*W)
    sub_scale = scale_map[y1:y2, x1:x2]
    thr_scale = np.percentile(sub_scale, thr_percentile)

    # 2) mask_scale_low
    mask_scale_low = np.zeros_like(scale_map, dtype=bool)
    mask_scale_low[y1:y2, x1:x2] = (sub_scale <= thr_scale)

    # 3) sector 掩码 valid_mask
    angles = np.linspace(-180.0, 180.0, W, endpoint=False)
    h_mask = (angles >= -112.5) & (angles <= -67.5)
    y1_v, y2_v = int(0.3 * H), int(0.8 * H)
    v_mask = np.zeros(H, dtype=bool)
    v_mask[y1_v:y2_v] = True
    valid_mask = np.outer(v_mask, h_mask)

    # 4) 生成 disp 图
    disp0 = -visual_scale_map_range_image(scale_map, (scale_map>0.3)&(scale_map<3.0))
    disp1 = -visual_scale_map_range_image(scale_map, mask_scale_low)
    disp5 = -visual_scale_map_range_image(scale_map, valid_mask)
    # 新增：mask_scale_low ∧ valid_mask
    mask_both = mask_scale_low & valid_mask
    disp_both = -visual_scale_map_range_image(scale_map, mask_both)

    # 5) 拼接并保存
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1,
                             figsize=(W/100, 4*H/100),
                             constrained_layout=True)
    for ax in axes:
        ax.axis('off')

    fig.text(
        0.01, 0.99,
        f"Threshold: Scale ≤ {thr_scale:.3f}",
        color='black', fontsize=12, va='top', ha='left'
    )

    axes[0].imshow(disp0, cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title('Full Scale Map', color='white', pad=4)

    axes[1].imshow(disp1, cmap='seismic', vmin=-1, vmax=1)
    axes[1].set_title(f'Scale ≤ {thr_scale:.3f}', color='white', pad=4)

    axes[2].imshow(disp5, cmap='seismic', vmin=-1, vmax=1)
    axes[2].set_title('Sector Scale Map (Angle & Row Mask)', color='white', pad=4)

    axes[3].imshow(disp_both, cmap='seismic', vmin=-1, vmax=1)
    axes[3].set_title('Scale ≤ thr_scale ∧ Sector Mask', color='white', pad=4)

    out_path = os.path.join(save_dir, f"composed_{idx}.png")
    fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return mask_scale_low, valid_mask, mask_both

def extract_small_regions(args, scale_map, risk_map, delta_t, ttc_thresh):
    """
    把所有 scale<阈值 的连通域都当碰撞点，
    返回 [x, y, ttc, scale, risk]
    """
    thresh_mask = (scale_map < args.approach_threshold).astype(np.uint8)  # 0/1
    # 可选：先膨胀一轮，填补极窄通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_closed = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)

    # 连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_closed, connectivity=8
    )
    out = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        cx, cy = int(cx), int(cy)
        ttc = delta_t / (1 - scale_map[cy, cx] + 1e-5)
        if 0 < ttc < ttc_thresh:
            out.append([cx, cy, ttc,
                        scale_map[cy, cx],
                        risk_map[cy, cx]])
    return out

def init_visualization(save_mode, keyboard_mode):
    """
    初始化 Matplotlib 可视化窗口
    """
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis('off')
    return fig, ax


def load_test_infos(test_info_path):
    """
    加载 test_infos 数据
    """
    with open(test_info_path, 'rb') as f:
        return pickle.load(f)


def get_pred_npy_files(pred_dir):
    """
    获取并排序预测 .npy 文件列表，文件名格式为 xxx_<idx>.npy
    """
    files = [f for f in os.listdir(pred_dir) if f.endswith('.npy')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    return files


def load_prediction(pred_path):
    """
    加载单个预测文件，返回 scale 和 risk map
    """
    data = np.load(pred_path, allow_pickle=True).item()
    return data['scale_pred'], data['risk_pred']


def preprocess_scale_map(scale_map, grid_size, approach_threshold):
    """
    裁剪和归一化 scale_map
    返回裁剪后的 scale_map 和归一化结果 norm_scale
    """
    H, W = scale_map.shape
    proc_h = (math.ceil(H / grid_size) - 1) * grid_size
    proc_w = (math.ceil(W / grid_size) - 1) * grid_size
    cropped = scale_map[:proc_h, :proc_w]
    min_s = np.min(cropped)
    norm_scale = (cropped - min_s) / (approach_threshold - min_s) * 255.0
    return cropped, norm_scale.astype(np.uint8)


def compute_valid_masks(scale_map, grid_size, grad_step, risk_rate, approach_threshold):
    """
    计算有效梯度 mask 和网格级有效 map，返回：
      - valid_grad: 与 scale_map[grad_step:-1, :] 对齐的布尔数组（像素级）
      - grid_map: 与 scale_map 形状缩小 grid_size 倍后的网格级布尔数组
    """
    # 1. 计算局部方差（供网格TTC使用）
    variance = get_ttc_var(scale_map, grid_len=grid_size)

    # 2. 计算二阶梯度（与 scale_map[grad_step:, :] 对齐）
    gy, gx, gy2, gx2 = get_second_grad(scale_map, stride=grad_step)

    # 3. 梯度 & 阈值筛选（保持和原脚本对齐的切片）
    mask_grad    = np.abs(gy) < (risk_rate * np.mean(np.abs(gy)))
    mask_thresh1 = scale_map[grad_step:, :] < approach_threshold
    mask_thresh2 = scale_map[1:-1, :]   < approach_threshold

    valid_grad = np.logical_and(mask_grad, mask_thresh1)
    valid_grad = np.logical_and(valid_grad, mask_thresh2)

    # 4. 网格级TTC——获得 (grid_valid_map, grid_valid_mask)
    grid_map, _ = get_grid_ttc(scale_map, variance, grid_len=grid_size)

    # 5. 只保留布尔类型，后面按网格迭代使用
    grid_map = grid_map.astype(bool)

    return valid_grad, grid_map


def extract_candidate_points(scale_map, norm_scale, valid_grad_mask, grid_map, grid_size, grad_step, thresh, kp_thresh):
    """
    提取 FAST 关键点并生成候选碰撞点
    返回候选点列表 (x, y, norm_scale_value)
    """
    # FAST 检测
    fast = cv2.FastFeatureDetector_create(6400)
    fast.setThreshold(kp_thresh)
    kps = fast.detect(norm_scale, None)

    H, W = scale_map.shape
    proc_h, proc_w = H, W
    candidates = []
    dense_map = np.zeros_like(valid_grad_mask, dtype=np.uint8)
    invalid = []

    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if not (0.3*proc_h < y < 0.9*proc_h and 0.1*proc_w < x < 0.9*proc_w):
            continue
        area = 9
        y0, y1 = max(y-area//2,0), min(y+area//2+1,proc_h)
        x0, x1 = max(x-area//2,0), min(x+area//2+1,proc_w)
        window = valid_grad_mask[y0:y1, x0:x1]
        dense_map[y0:y1, x0:x1] = 1
        if window.sum()>3:
            orig = scale_map[y0:y1, x0:x1]
            if orig.min()<thresh and orig.max()<1.04:
                candidates.append([x, y, norm_scale[y0:y1, x0:x1].min()])
        else:
            invalid.append([x,y])

    # 重新检查 invalid
    for x,y in invalid:
        if dense_map[y,x]==0:
            continue
        area=7
        y0,y1 = max(y-area//2,0), min(y+area//2+1,proc_h)
        x0,x1 = max(x-area//2,0), min(x+area//2+1,proc_w)
        orig = scale_map[y0:y1, x0:x1]
        if orig.min()<thresh and orig.max()<1.04:
            candidates.append([x,y,np.min(norm_scale[y0:y1, x0:x1])])

    # 网格候选点
    rows, cols = grid_map.shape
    for i in range(rows):
        for j in range(cols):
            if grid_map[i,j]:
                cy = int((2*i+1)*grid_size/2)
                cx = int((2*j+1)*grid_size/2)
                if not valid_grad_mask[cy, cx] or cy<0.4*proc_h:
                    continue
                if scale_map[cy, cx]<thresh:
                    local_norm = np.mean(
                        norm_scale[i*grid_size:(i+1)*grid_size,
                                   j*grid_size:(j+1)*grid_size]
                    )
                    candidates.append([cx,cy,local_norm])
    return np.array(candidates)


def cluster_and_compute(scale_map, risk_map, candidates, delta_t, risk_time_threshold):
    """
    对候选点进行两阶段 DBSCAN 聚类并计算碰撞时间
    返回 collision_points 列表 [x, y, time_to_collision, scale]
    """
    collisions = []
    if candidates.size==0:
        return collisions

    # 第一阶段
    clu1 = DBSCAN(eps=50, min_samples=5).fit(candidates)
    clusters1 = {}
    for idx,label in enumerate(clu1.labels_):
        if label<0: continue
        clusters1.setdefault(label,[]).append(candidates[idx])

    sec_centers, sec_regions = [], []
    for pts in clusters1.values():
        arr = np.array(pts)[:,:2].astype(int)
        (cx,cy), r = cv2.minEnclosingCircle(arr)
        ct = delta_t/(1-scale_map[int(cy),int(cx)]+1e-5)
        if r<6 and arr.shape[0]<4:
            continue
        sec_centers.append([cx,cy,ct*200])
        sec_regions.append(arr)
    sec_centers = np.array(sec_centers)
    if sec_centers.size==0:
        return collisions

    # 第二阶段
    clu2 = DBSCAN(eps=100, min_samples=3).fit(sec_centers)
    regions = {}
    for idx,label in enumerate(clu2.labels_):
        regions.setdefault(label,[]).append(sec_regions[idx])

    for label, regs in regions.items():
        if label<0:
            for reg in regs:
                (cx,cy),_ = cv2.minEnclosingCircle(reg)
                ct = delta_t/(1-scale_map[int(cy),int(cx)]+1e-5)
                if 0<ct<risk_time_threshold:
                    collisions.append([int(cx),int(cy),ct,scale_map[int(cy),int(cx)],risk_map[int(cy),int(cx)]])
        else:
            merged = np.vstack(regs)
            cts = []
            for p in merged:
                y,x = int(p[1]),int(p[0])
                cts.append(delta_t/(1-scale_map[y,x]+1e-5))
            ctmin = min(cts)
            (cx,cy),_ = cv2.minEnclosingCircle(merged)
            if 0<ctmin<risk_time_threshold:
                collisions.append([int(cx),int(cy),ctmin,scale_map[int(cy),int(cx)],risk_map[int(cy),int(cx)]])
    return collisions

# def cluster_and_compute(scale_map, risk_map, candidates,
#                         delta_t, risk_time_threshold,
#                         save_dir=None, idx=None, visualize=False):
#     collisions = []
#     if candidates.size == 0:
#         return collisions

#     # ———— 第一阶段聚类 ————
#     clu1 = DBSCAN(eps=50, min_samples=5).fit(candidates)
#     clusters1 = {}
#     for i, label in enumerate(clu1.labels_):
#         if label < 0: continue
#         clusters1.setdefault(label, []).append(candidates[i])

#     # 如果需要可视化并保存第一阶段结果
#     if visualize and save_dir and idx is not None:
#         fig1, ax1 = plt.subplots(figsize=(6,6))
#         ax1.set_title(f'Stage1 Clusters (frame {idx})')
#         ax1.invert_yaxis()
#         for label, pts in clusters1.items():
#             arr = np.array(pts)
#             color = (random.random(), random.random(), random.random())
#             ax1.scatter(arr[:,0], arr[:,1], s=10, c=[color], label=f'C{label}')
#             (cx,cy), r = cv2.minEnclosingCircle(arr[:,:2].astype(np.int32))
#             circle = plt.Circle((cx,cy), r, fill=False, color=color, linewidth=2)
#             ax1.add_patch(circle)
#         ax1.legend(loc='upper right', fontsize='small')
#         os.makedirs(save_dir, exist_ok=True)
#         fig1.savefig(os.path.join(save_dir, f'stage1_clusters_{idx}.png'),
#                      dpi=150, bbox_inches='tight')
#         plt.close(fig1)

#     # 准备第二阶段的临时中心和区域
#     sec_centers, sec_regions = [], []
#     for pts in clusters1.values():
#         arr = np.array(pts)[:,:2].astype(int)
#         (cx,cy), r = cv2.minEnclosingCircle(arr)
#         ct = delta_t/(1-scale_map[int(cy),int(cx)]+1e-5)
#         if r<6 and arr.shape[0]<4:
#             continue
#         sec_centers.append([cx,cy,ct*200])
#         sec_regions.append(arr)
#     sec_centers = np.array(sec_centers)
#     if sec_centers.size == 0:
#         return collisions

#     # ———— 第二阶段聚类 ————
#     clu2 = DBSCAN(eps=100, min_samples=3).fit(sec_centers)
#     regions = {}
#     for i, label in enumerate(clu2.labels_):
#         regions.setdefault(label, []).append(sec_regions[i])

#     # 如果需要可视化并保存第二阶段结果
#     if visualize and save_dir and idx is not None:
#         fig2, ax2 = plt.subplots(figsize=(6,6))
#         ax2.set_title(f'Stage2 Clusters (frame {idx})')
#         ax2.invert_yaxis()
#         for label, regs in regions.items():
#             merged = np.vstack(regs)
#             color = (random.random(), random.random(), random.random())
#             ax2.scatter(merged[:,0], merged[:,1], s=10, c=[color], label=f'C{label}')
#             (cx,cy), r = cv2.minEnclosingCircle(merged)
#             circle = plt.Circle((cx,cy), r, fill=False, color=color, linewidth=2)
#             ax2.add_patch(circle)
#         ax2.legend(loc='upper right', fontsize='small')
#         os.makedirs(save_dir, exist_ok=True)
#         fig2.savefig(os.path.join(save_dir, f'stage2_clusters_{idx}.png'),
#                      dpi=150, bbox_inches='tight')
#         plt.close(fig2)

    # ———— 后续原有的 collision 过滤逻辑 ————
    for label, regs in regions.items():
        if label < 0:
            for reg in regs:
                (cx,cy),_ = cv2.minEnclosingCircle(reg)
                ct = delta_t/(1-scale_map[int(cy),int(cx)]+1e-5)
                if 0<ct<risk_time_threshold:
                    collisions.append([int(cx),int(cy),ct,
                                       scale_map[int(cy),int(cx)],
                                       risk_map[int(cy),int(cx)]])
        else:
            merged = np.vstack(regs)
            cts = [delta_t/(1-scale_map[int(p[1]),int(p[0])]+1e-5) 
                   for p in merged]
            ctmin = min(cts)
            (cx,cy),_ = cv2.minEnclosingCircle(merged)
            if 0<ctmin<risk_time_threshold:
                collisions.append([int(cx),int(cy),ctmin,
                                   scale_map[int(cy),int(cx)],
                                   risk_map[int(cy),int(cx)]])

    return collisions

def filter_by_risk(collisions, risk_pred_threshold):
    """
    根据 risk_pred_map 过滤碰撞点
    """
    return [pt for pt in collisions if pt[-1] > risk_pred_threshold]


def visualize_range_image(ax, map_to_vis, collisions, type='scale'):
    """
    在 range image 上绘制碰撞点
    - map_to_vis: 要可视化的 range image，与 type 对应
    - collisions: 碰撞点列表 [x, y, time_to_collision, scale, risk]
    - type: 'scale' 或 'risk'，决定使用的可视化方法
    """
    if type == 'scale':
        mask = (map_to_vis > 0.3) & (map_to_vis < 3.0) 
        disp = visual_scale_map_range_image(map_to_vis, mask)
        ax.imshow(-disp, cmap='seismic', vmin=-1, vmax=1)
    elif type == 'risk':
        disp = visual_risk_score_map_range_image(map_to_vis)
        ax.imshow(disp, cmap='seismic', vmin=-1, vmax=1)
    else:
        raise ValueError("Invalid type. Use 'scale' or 'risk'.")
    # 绘制碰撞点，并添加文本标签：ttc 和 risk
    for x,y,ttc,_,risk in collisions:
        circ = plt.Circle((x,y),5,color='black',fill=True)
        ax.add_patch(circ)
        ax.text(x+6, y-6,
                f"t={ttc:.2f}\nr={risk:.2f}",
                color='black', fontsize=12,
                va='top', ha='left')
    # # 仅在图像中心（纵向 10%~90%，横向 10%~90%）区域选取最小/最大值位置并标注其数值
    # H, W = map_to_vis.shape
    # y1, y2 = int(0.1 * H), int(0.9 * H)
    # x1, x2 = int(0.1 * W), int(0.9 * W)
    # sub_map = map_to_vis[y1:y2, x1:x2]

    # if type == 'scale':
    #     val = np.min(sub_map)
    #     idx0 = np.argmin(sub_map)
    # else:  # type == 'risk'
    #     val = np.max(sub_map)
    #     idx0 = np.argmax(sub_map)

    # # 将局部索引转换为全图坐标
    # y_rel, x_rel = np.unravel_index(idx0, sub_map.shape)
    # y0, x0 = y1 + y_rel, x1 + x_rel

    # # 绘制标记和文本
    # circ = plt.Circle((x0, y0), 8, edgecolor='yellow', facecolor='none', linewidth=2)
    # ax.add_patch(circ)
    # ax.text(
    #     x0 + 10, y0,
    #     f"{val:.2f}",
    #     color='yellow',
    #     fontsize=12,
    #     va='center', ha='left',
    #     bbox=dict(facecolor='black', alpha=0.6, pad=2)
    # )

def save_range_image(fig, ax, out_path, scale_map):
    """
    保存 range image 可视化结果
    """
    dpi = 100
    fig.set_size_inches(scale_map.shape[1]/dpi, scale_map.shape[0]/dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    ax.clear()

def overlay_surround_views_sjtu(collisions, test_info, args, idx):
    """
    在 SJTU 环视图上叠加碰撞点、裁剪、拼接并保存；
    即使 collisions 为空，也会输出原始环视图。
    [WARN] 由于目前 SJTU 测试集没有 ground truth 数据，
    这里直接将碰撞点在 range image 上的坐标点同步到环视图上。
    """
    surround_view_img_files = [f for f in os.listdir(args.sjtu_surround_view_path) if f.endswith('.png')] 
    # 1. 过滤出当前 SJTU 场景和帧对应的拼接环视图文件
    scene_idx = args.sjtu_scene_indice
    target_filename = f"scene_{scene_idx}_concat_prev_{idx}.png"
    if target_filename not in surround_view_img_files:
        print(f"Warning: {target_filename} not found in {args.sjtu_surround_view_path}. Skipping overlay.")
        return
    img_path = os.path.join(args.sjtu_surround_view_path, target_filename)

    # 2. 读取拼接图，并在上面叠加碰撞点和文字
    cv_img = cv2.imread(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_h = int(30 * font_scale)

    for x, y, ttc, _, risk in collisions:
        cv2.circle(cv_img, (x, y), 6, (255, 0, 0), -1)
        cv2.putText(cv_img,
                    f"t={ttc:.2f}",
                    (x + 6, y - 6),
                    font, font_scale,
                    (255, 0, 0), thickness,
                    lineType=cv2.LINE_AA)
        cv2.putText(cv_img,
                    f"r={risk:.2f}",
                    (x + 6, y - 6 + line_h),
                    font, font_scale,
                    (255, 0, 0), thickness,
                    lineType=cv2.LINE_AA)

    # 3. 保存结果到 vis_dir
    out_path = os.path.join(args.vis_dir, f"surround_sjtu_{idx}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv_img)
    

def overlay_surround_views(collisions, test_info, args, idx):
    """
    在环视图上叠加碰撞点、裁剪、拼接并保存；
    即使 collisions 为空，也会输出原始环视图。
    """
    # 读取深度真值和相机元数据
    gt = np.load(
        resolve_nusc_path(test_info['gt_map_path'], args.dataroot) / 'range_image_prev.npy',
        allow_pickle=True
    ).item()
    depth_map = gt['depth']
    metas     = test_info['sensor_metas_prev']
    prev_cam  = test_info['prev_camera_data']

    # 1. 先获取所有原始环视图（没有任何点），并转为 PIL.Image
    empty_pts = np.zeros((0, 3))
    proj_all = project_lidar_to_surround_view_img(
        empty_pts, prev_cam, metas, min_dist=1.0
    )
    surround_imgs = {}
    for ch, data in proj_all.items():
        img_np = np.array(data['original_img'])
        surround_imgs[ch] = Image.fromarray(img_np)

    # 2. 如果有碰撞点，则在对应视图上叠加红圈和文字
    if collisions and depth_map is not None:
        for x, y, ttc, _, risk in collisions:
            d = depth_map[y, x]
            if d < 0:
                continue
            xyz = inverse_range_projection(
                x, y, d,
                H=depth_map.shape[0],
                W=depth_map.shape[1],
                fov_up=8.0, fov_down=-15.0
            )
            proj_pts = project_lidar_to_surround_view_img(
                np.array([xyz]), prev_cam, metas, min_dist=1.0
            )
            for ch, data in proj_pts.items():
                pts2d = data['points'][:2, :].astype(int)
                if pts2d.shape[1] == 0:
                    continue

                # 从 PIL 转为 OpenCV 数组，叠加标记，再转回 PIL
                pil_img = surround_imgs[ch]
                cv_img  = np.array(pil_img)

                xp, yp = pts2d[:, 0]
                cv2.circle(cv_img, (xp, yp), 30, (255, 0, 0), -1)
                font      = cv2.FONT_HERSHEY_SIMPLEX
                font_scale= 3
                thickness = 5
                line_h    = int(30 * font_scale)  # 行高，根据 font_scale 调整

                # 第一行：t
                cv2.putText(
                    cv_img,
                    f"t={ttc:.2f}",
                    (xp + 30, yp - 30),
                    font, font_scale,
                    (255, 0, 0), thickness,
                    lineType=cv2.LINE_AA
                )
                # 第二行：r，y 坐标往下移动一个行高
                cv2.putText(
                    cv_img,
                    f"r={risk:.2f}",
                    (xp + 30, yp - 30 + line_h),
                    font, font_scale,
                    (255, 0, 0), thickness,
                    lineType=cv2.LINE_AA
                )

                surround_imgs[ch] = Image.fromarray(cv_img)

    # 3. 裁剪、拼接并保存最终合图
    out_path = os.path.join(args.vis_dir, f"surround_{idx}.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resize_param = [args.image_size[0], args.image_size[1]*6]
    combine_surround_views(surround_imgs, resize_param, out_path)


def combine_surround_views(surround_imgs, resize_param, out_path):
    """
    surround_imgs: dict[str, PIL.Image]，6 路原始环视图
    augmentor: 带有 resize_and_crop(self, img) 方法的实例
    out_path: 最终拼接图保存路径
    """

    camera_channels = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
    ]

    img_width_raw = surround_imgs[camera_channels[0]].size[0]
    range_image_width = resize_param[1]

    fx = (range_image_width / 6.0) / img_width_raw
    fy = fx

    # 1. 对每一路做 resize
    resized_list = []
    for ch in camera_channels:
        img = surround_imgs[ch]
        arr = np.array(img)
        resized_arr = cv2.resize(arr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        resized_img = Image.fromarray(resized_arr)
        resized_list.append(resized_img)

    # 2. 计算拼接后画布大小
    widths, heights = zip(*(im.size for im in resized_list))
    total_w = sum(widths)
    max_h = max(heights)

    # 3. 新建一个 RGB 画布，把每张图贴上去
    canvas = Image.new('RGB', (total_w, max_h))
    x_offset = 0
    for im in resized_list:
        canvas.paste(im, (x_offset, 0))
        x_offset += im.width

    # 4. 保存
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)

def main():
    args = parse_args()

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.vis_dir = os.path.join(args.vis_dir, f'collision_{ts}_ttc{args.risk_time_threshold}_risk{args.risk_pred_threshold}')
    os.makedirs(args.vis_dir, exist_ok=True)

    delta_t = 1.0 / args.fps
    test_infos = load_test_infos(args.test_info_path)

    files = get_pred_npy_files(args.pred_npy_dir)
    # 如果 init_visualization 返回的是 (fig, ax)，你可以继续沿用它；
    # 下面示例直接创建一个新的 fig, ax
    fig, ax = plt.subplots(figsize=(10, 10))

    idx_offset = int(files[0].split('_')[-1].split('.')[0]) if files else 0
    for idx, fn in enumerate(tqdm(files)):

        idx = idx + idx_offset
        pred_path = os.path.join(args.pred_npy_dir, fn)
        pred_data = np.load(pred_path, allow_pickle=True).item()
        scale_map = pred_data['scale_pred']
        risk_map  = pred_data['risk_pred']

        # ——— 1) 生成 valid_mask_region ———
        H, W = scale_map.shape
        angles = np.linspace(-180.0, 180.0, W, endpoint=False)
        h_mask = (angles >= (-90.0 - 22.5)) & (angles <= (-90.0 + 22.5))
        y1, y2 = int(0.3 * H), int(0.5 * H)
        v_mask = np.zeros(H, dtype=bool)
        v_mask[y1:y2] = True
        valid_mask_region = np.outer(v_mask, h_mask)

        # ——— 2) 生成 mask_scale_low ———
        thr_percentile = 2
        x1, x2 = int(0.1 * W), int(0.9 * W)
        sub_scale = scale_map[y1:y2, x1:x2]
        thr_scale = np.percentile(sub_scale, thr_percentile)
        mask_scale_low = np.zeros_like(scale_map, dtype=bool)
        mask_scale_low[y1:y2, x1:x2] = (sub_scale <= thr_scale)

        # ——— 3) 可选叠加得到 final valid_mask ———
        valid_mask = valid_mask_region & mask_scale_low

        # ——— 4) 用 DBSCAN 找簇心（只保留足够大的簇） ———
        ttc_thresh = args.risk_time_threshold
        ys, xs = np.where(valid_mask)
        cluster_centers = []
        if xs.size > 0:
            scales = scale_map[ys, xs]
            pts = np.stack([xs, ys, scales * 50.0], axis=1)
            db = DBSCAN(eps=5, min_samples=20).fit(pts)

            cluster_size_threshold = 300  # 最小点数阈值
            eps = 1e-5
            for lbl in set(db.labels_):
                if lbl < 0:
                    continue
                idxs = np.where(db.labels_ == lbl)[0]
                size = idxs.size
                if size < cluster_size_threshold:
                    continue

                cx = int(xs[idxs].mean())
                cy = int(ys[idxs].mean())
                ttc = delta_t / (1.0 - scale_map[cy, cx] + eps)
                # 记录簇心坐标、TTC 和 簇大小
                if ttc < ttc_thresh:
                    cluster_centers.append((cx, cy, ttc, size))

        # ——— 5) 生成 4 张 disp，并纵向拼接 ———
        full_mask = np.ones_like(scale_map, dtype=bool)
        disp0 = -visual_scale_map_range_image(scale_map, full_mask)
        disp1 = -visual_scale_map_range_image(scale_map, valid_mask_region)
        disp2 = -visual_scale_map_range_image(scale_map, mask_scale_low)
        disp3 = -visual_scale_map_range_image(scale_map, valid_mask)
        disp4 = visual_risk_score_map_range_image(risk_map)
        composite = np.vstack([disp0, disp1, disp2, disp3])

        # ——— 6) 根据 args.keyboard / args.save 做不同处理 ———
        if args.keyboard:
            # 交互模式：显示拼接图，同时在 disp0（顶部区域）和 disp3（底部区域）标注簇信息
            ax.clear()
            ax.imshow(composite, cmap='seismic', vmin=-1, vmax=1)
            ax.axis('off')
            for cx, cy, ttc, size in cluster_centers:
                # 在 disp0 区域标注（无需偏移）
                ax.scatter(cx, cy, s=50, c='black', marker='x')
                ax.text(
                    cx + 2, cy + 2,
                    f"TTC:{ttc:.2f}\nN:{size}",
                    color='black', fontsize=8
                )
                # 在 disp3 区域标注，需要在 y 方向加上 3*H 的偏移
                y3 = cy + 3 * H
                ax.scatter(cx, y3, s=50, c='black', marker='x')
                ax.text(
                    cx + 2, y3 + 2,
                    f"TTC:{ttc:.2f}\nN:{size}",
                    color='black', fontsize=8
                )
                fig.canvas.draw()
                plt.waitforbuttonpress()  # 等待按键翻帧

        # elif args.save:
        #     # 保存模式：只保存第一张 scale_map，并在其上标注簇信息
        #     fig2, ax2 = plt.subplots(figsize=(W/100, H/100))
        #     ax2.imshow(disp0, cmap='seismic', vmin=-1, vmax=1)
        #     ax2.axis('off')
        #     for cx, cy, ttc, size in cluster_centers:
        #         ax2.scatter(cx, cy, s=50, c='black', marker='x')
        #         ax2.text(
        #             cx + 2, cy + 2,
        #             f"TTC:{ttc:.2f}",
        #             color='black', fontsize=8
        #         )
        #     os.makedirs(args.vis_dir, exist_ok=True)
        #     save_path0 = os.path.join(args.vis_dir, f'colli_scale_pred_{idx}.png')
        #     fig2.savefig(save_path0, dpi=100, bbox_inches='tight', pad_inches=0)
        #     plt.close(fig2)

        #     # 同时保存 disp4（risk map），并在其上标注簇信息
        #     fig3, ax3 = plt.subplots(figsize=(W/100, H/100))
        #     ax3.imshow(disp4, cmap='seismic', vmin=-1, vmax=1)
        #     ax3.axis('off')
        #     for cx, cy, ttc, size in cluster_centers:
        #         ax3.scatter(cx, cy, s=50, c='black', marker='x')
        #         ax3.text(
        #             cx + 2, cy + 2,
        #             f"TTC:{ttc:.2f}",
        #             color='black', fontsize=8
        #         )
        #     save_path4 = os.path.join(args.vis_dir, f'colli_risk_pred_{idx}.png')
        #     fig3.savefig(save_path4, dpi=100, bbox_inches='tight', pad_inches=0)
        #     plt.close(fig3)

        elif args.save:
            os.makedirs(args.vis_dir, exist_ok=True)

            # —— 先画带标注的 scale map —— #
            fig2, ax2 = plt.subplots(figsize=(W/100, H/100), dpi=100)
            ax2.imshow(disp0, cmap='seismic', vmin=-1, vmax=1)
            ax2.axis('off')
            # 去掉四周空白
            fig2.subplots_adjust(0, 0, 1, 1)

            for cx, cy, ttc, size in cluster_centers:
                ax2.scatter(cx, cy, s=50, c='black', marker='x')
                ax2.text(
                    cx + 2, cy + 2,
                    f"TTC:{ttc:.2f}\nN:{size}",
                    color='black', fontsize=8
                )

            # 强制渲染到 canvas
            fig2.canvas.draw()
            # 从 canvas 读取 RGB 数据
            w, h = fig2.canvas.get_width_height()
            buf = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape(h, w, 3)

            # 用 imsave 保存，不会改变分辨率，也无白边
            save_path0 = os.path.join(args.vis_dir, f'colli_scale_pred_{idx}.png')
            plt.imsave(save_path0, img)

            plt.close(fig2)

            # —— 同理再画一次 risk map —— #
            fig3, ax3 = plt.subplots(figsize=(W/100, H/100), dpi=100)
            ax3.imshow(disp4, cmap='seismic', vmin=-1, vmax=1)
            ax3.axis('off')
            fig3.subplots_adjust(0, 0, 1, 1)

            for cx, cy, ttc, size in cluster_centers:
                ax3.scatter(cx, cy, s=50, c='black', marker='x')
                ax3.text(
                    cx + 2, cy + 2,
                    f"TTC:{ttc:.2f}\nN:{size}",
                    color='black', fontsize=8
                )

            fig3.canvas.draw()
            w, h = fig3.canvas.get_width_height()
            buf = np.frombuffer(fig3.canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape(h, w, 3)

            save_path4 = os.path.join(args.vis_dir, f'colli_risk_pred_{idx}.png')
            plt.imsave(save_path4, img)

            plt.close(fig3)

        else:
            # 自动播放模式：短暂停留
            ax.clear()
            ax.imshow(composite, cmap='seismic', vmin=-1, vmax=1)
            ax.axis('off')
            fig.canvas.draw()
            plt.pause(0.001)




if __name__ == '__main__':
    main()
