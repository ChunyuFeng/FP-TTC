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
try:
    import mplcursors
except ImportError:
    mplcursors = None
import random

from dataloader.utils.augmentor import NuscRangeImageAugmentor

from utils.draw import (
    make_risk_analysis_rgb,
    make_scale_analysis_rgb,
    visual_scale_map_range_image,
    visual_risk_score_map_range_image,
    scale2rgb,
    orientation2rgb
)
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
    parser.add_argument("--save_cluster_debug_panel", action="store_true",
                        help="Save a 3x3 per-frame debug panel for cluster and region debugging")
    parser.add_argument("--cluster_debug_prefix", type=str, default="rule_debug",
                        help="Prefix for saved debug panel filenames")
    return parser.parse_args()

def inspect_scale_risk(scale_map, risk_map):
    """
    弹出一个 1×2 的窗口：
     - 左图显示原始 scale_map，鼠标 hover 即可查看 scale 值
     - 右图显示原始 risk_map，鼠标 hover 即可查看 risk 值
    """
    if mplcursors is None:
        raise ImportError("mplcursors is required for inspect_scale_risk() interactive hover mode")
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
    scale_map, risk_map,
    save_dir, idx
):
    """
    1) 在 scale_map 的子区域（纵向 30%~90%，横向 10%~90%）中，
       分别计算最小 3% 阈值 thr_scale 和最大 3% 阈值 thr_risk。
    2) 只在该子区域内生成 mask_scale_low3 与 mask_risk_high3，
       子区域外均为 False。并计算二者交集 mask_intersect。
    3) 纵向拼接 5 幅图并保存，并在左上角标注 thr_scale 和 thr_risk：
       [全量 scale]
       [scale ≤ thr_scale]
       [scale ∧ risk 交集]
       [全量 risk]
       [risk ≥ thr_risk]
    """
    H, W = scale_map.shape
    # 子区域 ROI
    y1, y2 = int(0.3*H), int(0.9*H)
    x1, x2 = int(0.1*W), int(0.9*W)
    sub_scale = scale_map[y1:y2, x1:x2]
    sub_risk  = risk_map [y1:y2, x1:x2]

    # 1) 子区域百分位阈值
    thr_scale = np.percentile(sub_scale, 10)   # 最小 3%
    thr_risk  = np.percentile(sub_risk,  70) # 最大 3%
    # thr_scale = 0.990
    # thr_risk = 0.03

    # 2) 在全图上生成 mask
    mask_scale_low3 = np.zeros_like(scale_map, dtype=bool)
    mask_risk_high3 = np.zeros_like(risk_map,  dtype=bool)
    mask_scale_low3[y1:y2, x1:x2] = (sub_scale <= thr_scale)
    mask_risk_high3[y1:y2, x1:x2] = (sub_risk  >= thr_risk)
    mask_intersect  = mask_scale_low3 & mask_risk_high3

    # 3) 准备 5 幅 disp 图
    default_mask = (scale_map > 0.3) & (scale_map < 3.0)
    disp0 = -visual_scale_map_range_image(scale_map, default_mask)
    disp1 = -visual_scale_map_range_image(scale_map, mask_scale_low3)
    disp2 = -visual_scale_map_range_image(scale_map, mask_intersect)
    disp3 = -visual_risk_score_map_range_image(risk_map, None)
    tmp   = risk_map.copy()
    tmp[~mask_risk_high3] = 0.0
    disp4 =  visual_risk_score_map_range_image(tmp)

    # 4) 纵向拼接并保存
    fig, axes = plt.subplots(5, 1, figsize=(W/100, 5*H/100), constrained_layout=True)
    for ax in axes:
        ax.axis('off')

    # 在左上角添加阈值注释（相对于整张 figure）
    fig.text(
        0.01, 0.99,
        f"Thresholds: Scale ≤ {thr_scale:.3f}, Risk ≥ {thr_risk:.3f}",
        color='black',
        fontsize=12,
        va='top',
        ha='left'
    )

    axes[0].imshow(disp0, cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title('Full Scale Map', color='white', pad=4)

    axes[1].imshow(disp1, cmap='seismic', vmin=-1, vmax=1)
    axes[1].set_title(f'Scale ≤ {thr_scale:.3f}', color='white', pad=4)

    axes[2].imshow(disp2, cmap='seismic', vmin=-1, vmax=1)
    axes[2].set_title('Scale ∧ Risk Intersection', color='white', pad=4)

    axes[3].imshow(disp3, cmap='seismic', vmin=-1, vmax=1)
    axes[3].set_title('Full Risk Map', color='white', pad=4)

    axes[4].imshow(disp4, cmap='seismic', vmin=-1, vmax=1)
    axes[4].set_title(f'Risk ≥ {thr_risk:.3f}', color='white', pad=4)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"composed_{idx}.png")
    fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return mask_scale_low3, mask_risk_high3, mask_intersect

def extract_small_regions(args, scale_map, risk_map, delta_t, ttc_thresh, return_debug=False):
    """
    把所有 scale<阈值 的连通域都当碰撞点，
    返回 [x, y, ttc, scale, risk]
    """
    thresh_mask = (scale_map < args.approach_threshold).astype(np.uint8)  # 0/1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_closed = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_closed, connectivity=8
    )
    out = []
    all_centroids = []
    kept_centroids = []
    kept_label_ids = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        cx, cy = int(cx), int(cy)
        all_centroids.append([cx, cy])
        ttc = delta_t / (1 - scale_map[cy, cx] + 1e-5)
        if 0 < ttc < ttc_thresh:
            out.append([cx, cy, ttc,
                        scale_map[cy, cx],
                        risk_map[cy, cx]])
            kept_centroids.append([cx, cy])
            kept_label_ids.append(i)
    if not return_debug:
        return out
    debug = {
        "mask_closed": mask_closed.astype(bool),
        "labels": labels.astype(np.int32),
        "num_components": int(max(num_labels - 1, 0)),
        "all_centroids": np.asarray(all_centroids, dtype=np.int32).reshape(-1, 2) if all_centroids else np.zeros((0, 2), dtype=np.int32),
        "kept_centroids": np.asarray(kept_centroids, dtype=np.int32).reshape(-1, 2) if kept_centroids else np.zeros((0, 2), dtype=np.int32),
        "kept_label_ids": np.asarray(kept_label_ids, dtype=np.int32),
        "kept_count": int(len(kept_label_ids)),
    }
    return out, debug


def ensure_uint8_rgb(image):
    arr = np.asarray(image)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype == np.uint8:
        return arr.copy()
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size == 0:
            return arr.astype(np.uint8)
        if float(arr.max()) <= 1.0 + 1e-6 and float(arr.min()) >= -1e-6:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def annotate_panel_tile(rgb_image, title):
    canvas = ensure_uint8_rgb(rgb_image)
    cv2.putText(
        canvas,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_mask_overlay(rgb_image, mask, color, alpha=0.35):
    rgb = ensure_uint8_rgb(rgb_image)
    out = rgb.copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return out
    color_arr = np.asarray(color, dtype=np.float32)
    base = out[mask_bool].astype(np.float32)
    out[mask_bool] = np.clip((1.0 - alpha) * base + alpha * color_arr, 0, 255).astype(np.uint8)
    return out


def make_binary_mask_rgb(mask):
    mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)


def align_valid_grad_mask(valid_grad_mask, target_shape, grad_step):
    aligned = np.zeros(target_shape, dtype=bool)
    h = min(valid_grad_mask.shape[0], max(target_shape[0] - grad_step, 0))
    w = min(valid_grad_mask.shape[1], target_shape[1])
    if h > 0 and w > 0:
        aligned[grad_step:grad_step + h, :w] = valid_grad_mask[:h, :w]
    return aligned


def upsample_grid_map(grid_map, grid_size, target_shape):
    if grid_map.size == 0:
        return np.zeros(target_shape, dtype=bool)
    upsampled = np.kron(grid_map.astype(np.uint8), np.ones((grid_size, grid_size), dtype=np.uint8))
    out = np.zeros(target_shape, dtype=bool)
    h = min(target_shape[0], upsampled.shape[0])
    w = min(target_shape[1], upsampled.shape[1])
    out[:h, :w] = upsampled[:h, :w].astype(bool)
    return out


def points_to_mask(shape, points, radius=0):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(points)
    if pts.size == 0:
        return mask.astype(bool)
    pts = pts.reshape(-1, 2)
    h, w = shape
    xs = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, h - 1)
    mask[ys, xs] = 1
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def render_points_overlay(rgb_image, points, color, radius=1):
    mask = points_to_mask(rgb_image.shape[:2], points, radius=radius)
    out = ensure_uint8_rgb(rgb_image)
    out[mask] = np.asarray(color, dtype=np.uint8)
    return out


def build_debug_palette(num_items):
    if num_items <= 0:
        return []
    cmap = mpl_cm.get_cmap('tab20')
    palette = []
    for idx in range(num_items):
        rgb = np.asarray(cmap((idx % 20) / 19.0)[:3]) * 255.0
        palette.append(tuple(rgb.astype(np.uint8).tolist()))
    return palette


def render_point_groups_canvas(shape, groups, point_key='points', noise_groups=None):
    canvas = np.full((shape[0], shape[1], 3), 255, dtype=np.uint8)
    palette = build_debug_palette(len(groups))
    for idx, group in enumerate(groups):
        points = group.get(point_key, np.zeros((0, 2), dtype=np.int32))
        mask = points_to_mask(shape, points, radius=0)
        canvas[mask] = np.asarray(palette[idx], dtype=np.uint8)
    if noise_groups:
        for points in noise_groups:
            mask = points_to_mask(shape, points, radius=0)
            canvas[mask] = np.asarray((160, 160, 160), dtype=np.uint8)
    return canvas


def render_small_region_canvas(shape, debug):
    canvas = np.full((shape[0], shape[1], 3), 255, dtype=np.uint8)
    canvas = render_mask_overlay(canvas, debug["mask_closed"], color=(220, 220, 220), alpha=0.9)
    if debug["kept_label_ids"].size > 0:
        kept_mask = np.isin(debug["labels"], debug["kept_label_ids"])
        canvas = render_mask_overlay(canvas, kept_mask, color=(255, 140, 0), alpha=0.85)
    canvas = render_points_overlay(canvas, debug["all_centroids"], color=(60, 60, 60), radius=1)
    canvas = render_points_overlay(canvas, debug["kept_centroids"], color=(220, 0, 0), radius=2)
    return canvas


def save_cluster_debug_panel(
    args,
    sample_idx,
    scale_map,
    crop_scale,
    risk_map,
    valid_grad_mask,
    grid_map,
    cluster_debug,
    small_region_debug,
    final_collisions,
    out_path,
):
    panel_h, panel_w = crop_scale.shape
    risk_crop = risk_map[:panel_h, :panel_w]
    valid_grad_aligned = align_valid_grad_mask(valid_grad_mask, (panel_h, panel_w), args.grad_step)
    grid_map_aligned = upsample_grid_map(grid_map, args.grid_size, (panel_h, panel_w))
    orientation_threshold_mask = risk_crop < args.risk_pred_threshold

    filtered_scale_panel = ensure_uint8_rgb(make_scale_analysis_rgb(crop_scale))
    valid_grad_panel = make_binary_mask_rgb(valid_grad_aligned)
    grid_panel = make_binary_mask_rgb(grid_map_aligned)
    candidate_panel = render_points_overlay(
        filtered_scale_panel,
        cluster_debug["candidates"][:, :2] if cluster_debug["candidates"].size > 0 else np.zeros((0, 2), dtype=np.int32),
        color=(0, 0, 0),
        radius=1,
    )
    stage1_panel = render_point_groups_canvas(
        (panel_h, panel_w),
        cluster_debug["stage1_clusters"],
        point_key='points',
        noise_groups=cluster_debug["stage1_noise_groups"],
    )
    stage2_panel = render_point_groups_canvas(
        (panel_h, panel_w),
        cluster_debug["stage2_regions"],
        point_key='merged_points',
        noise_groups=cluster_debug["stage2_noise_regions"],
    )
    small_regions_panel = render_small_region_canvas((panel_h, panel_w), small_region_debug)
    orientation_panel = ensure_uint8_rgb(make_risk_analysis_rgb(risk_crop))
    orientation_panel = render_mask_overlay(
        orientation_panel,
        orientation_threshold_mask,
        color=(40, 200, 40),
        alpha=0.42,
    )
    final_scale_panel = ensure_uint8_rgb(make_scale_analysis_rgb(scale_map[:panel_h, :panel_w]))
    final_points = np.asarray([[pt[0], pt[1]] for pt in final_collisions], dtype=np.int32) if final_collisions else np.zeros((0, 2), dtype=np.int32)
    final_panel = render_points_overlay(final_scale_panel, final_points, color=(0, 0, 0), radius=2)

    tiles = [
        annotate_panel_tile(filtered_scale_panel, f"Filtered Scale ttc<{args.risk_time_threshold:.2f}"),
        annotate_panel_tile(valid_grad_panel, f"Valid Grad area={int(valid_grad_aligned.sum())}"),
        annotate_panel_tile(grid_panel, f"Grid Map area={int(grid_map_aligned.sum())}"),
        annotate_panel_tile(candidate_panel, f"Candidates n={int(cluster_debug['candidates'].shape[0])}"),
        annotate_panel_tile(
            stage1_panel,
            f"Stage1 clusters={len(cluster_debug['stage1_clusters'])} kept={cluster_debug['stage1_kept_count']}",
        ),
        annotate_panel_tile(
            stage2_panel,
            f"Stage2 groups={len(cluster_debug['stage2_regions'])} noise={len(cluster_debug['stage2_noise_regions'])}",
        ),
        annotate_panel_tile(
            small_regions_panel,
            f"Small Regions raw={small_region_debug['num_components']} kept={small_region_debug['kept_count']}",
        ),
        annotate_panel_tile(
            orientation_panel,
            f"Orientation Filtered thr={args.risk_pred_threshold:.2f} pass_area={int(orientation_threshold_mask.sum())}",
        ),
        annotate_panel_tile(final_panel, f"Final Collisions n={len(final_collisions)}"),
    ]
    row1 = np.concatenate(tiles[0:3], axis=1)
    row2 = np.concatenate(tiles[3:6], axis=1)
    row3 = np.concatenate(tiles[6:9], axis=1)
    panel = np.concatenate([row1, row2, row3], axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

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
        if not (0.3*proc_h < y < 0.8*proc_h and 0.1*proc_w < x < 0.9*proc_w):
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


def cluster_and_compute(scale_map, risk_map, candidates, delta_t, risk_time_threshold, return_debug=False):
    """
    对候选点进行两阶段 DBSCAN 聚类并计算碰撞时间
    返回 collision_points 列表 [x, y, time_to_collision, scale]
    """
    collisions = []
    candidates = np.asarray(candidates)
    if candidates.ndim == 1 and candidates.size > 0:
        candidates = candidates[None, :]
    debug = {
        "candidates": candidates.copy() if candidates.size > 0 else np.zeros((0, 3), dtype=np.float32),
        "stage1_clusters": [],
        "stage1_noise_groups": [],
        "stage1_kept_count": 0,
        "stage2_regions": [],
        "stage2_noise_regions": [],
        "final_cluster_collisions": [],
    }
    if candidates.size == 0:
        if return_debug:
            return collisions, debug
        return collisions

    clu1 = DBSCAN(eps=50, min_samples=5).fit(candidates)
    labels1 = clu1.labels_
    clusters1 = {}
    unique_labels1 = sorted(set(labels1.tolist()))
    for label in unique_labels1:
        pts = candidates[labels1 == label, :2].astype(np.int32)
        if label < 0:
            if pts.size > 0:
                debug["stage1_noise_groups"].append(pts)
            continue
        debug["stage1_clusters"].append({
            "label": int(label),
            "points": pts,
        })
        clusters1[label] = candidates[labels1 == label]

    sec_centers, sec_regions = [], []
    for label in sorted(clusters1.keys()):
        pts = clusters1[label]
        arr = np.asarray(pts)[:, :2].astype(np.int32)
        (cx, cy), r = cv2.minEnclosingCircle(arr)
        ct = delta_t / (1 - scale_map[int(cy), int(cx)] + 1e-5)
        if r < 6 and arr.shape[0] < 4:
            continue
        sec_centers.append([cx, cy, ct * 200])
        sec_regions.append(arr)
        debug["stage1_kept_count"] += 1
    sec_centers = np.asarray(sec_centers, dtype=np.float32)
    if sec_centers.size == 0:
        debug["final_cluster_collisions"] = []
        if return_debug:
            return collisions, debug
        return collisions

    clu2 = DBSCAN(eps=100, min_samples=3).fit(sec_centers)
    labels2 = clu2.labels_
    regions = {}
    for idx, label in enumerate(labels2):
        regions.setdefault(label, []).append(sec_regions[idx])

    for label in sorted(regions.keys()):
        regs = regions[label]
        if label < 0:
            for reg in regs:
                debug["stage2_noise_regions"].append(reg.astype(np.int32))
                (cx, cy), _ = cv2.minEnclosingCircle(reg)
                ct = delta_t / (1 - scale_map[int(cy), int(cx)] + 1e-5)
                if 0 < ct < risk_time_threshold:
                    collisions.append([int(cx), int(cy), ct, scale_map[int(cy), int(cx)], risk_map[int(cy), int(cx)]])
            continue
        merged = np.vstack(regs).astype(np.int32)
        debug["stage2_regions"].append({
            "label": int(label),
            "merged_points": merged,
        })
        cts = []
        for p in merged:
            y, x = int(p[1]), int(p[0])
            cts.append(delta_t / (1 - scale_map[y, x] + 1e-5))
        ctmin = min(cts)
        (cx, cy), _ = cv2.minEnclosingCircle(merged)
        if 0 < ctmin < risk_time_threshold:
            collisions.append([int(cx), int(cy), ctmin, scale_map[int(cy), int(cx)], risk_map[int(cy), int(cx)]])
    debug["final_cluster_collisions"] = [list(pt) for pt in collisions]
    if return_debug:
        return collisions, debug
    return collisions


def filter_by_risk(collisions, risk_pred_threshold):
    """
    根据 risk_pred_map 过滤碰撞点
    """
    return [pt for pt in collisions if pt[-1] < risk_pred_threshold]


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
        disp = visual_risk_score_map_range_image(map_to_vis, None)
        ax.imshow(-disp, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)
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
        os.path.join(test_info['gt_map_path'], 'range_image_prev.npy'),
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

    # Instantiate augmentor using --image_size as [height, width]
    resize_height, resize_width = args.image_size  # height, width from CLI
    # crop_size expects (crop_h, crop_w)
    augmentor = NuscRangeImageAugmentor(crop_size=(resize_height, resize_width),
                                        do_flip=False,
                                        rotate=False)

    delta_t = 1.0 / args.fps
    test_infos = load_test_infos(args.test_info_path)
    files = get_pred_npy_files(args.pred_npy_dir)
    # results = {}
    fig, ax = init_visualization(args.save, args.keyboard)
    cluster_debug_dir = None
    if args.save_cluster_debug_panel:
        cluster_debug_dir = os.path.join(args.vis_dir, "cluster_debug")
        os.makedirs(cluster_debug_dir, exist_ok=True)

    idx_offset = int(files[0].split('_')[-1].split('.')[0]) if files else 0
    for idx, fn in enumerate(tqdm(files)):
        # if idx < 1351:
        #     continue
        idx = idx + idx_offset
        pred_path = os.path.join(args.pred_npy_dir, fn)
        scale_map, risk_map = load_prediction(pred_path)
        scale_thresh = 1 - delta_t / args.risk_time_threshold
        # 保留小于阈值的值，其余置为0
        scale_map_filtered = np.where(scale_map < scale_thresh, scale_map, 0.999)
        # scale_map_filtered = np.where(risk_map < args.risk_pred_threshold, scale_map_filtered, 0.999)
        crop_scale, norm_scale = preprocess_scale_map(
            scale_map_filtered, args.grid_size, args.approach_threshold
        )
        valid_grad, grid_map = compute_valid_masks(
            crop_scale, args.grid_size, args.grad_step,
            args.risk_rate, args.approach_threshold
        )
        # orientation_mask = np.where(risk_map < args.risk_pred_threshold, risk_map, np.pi)
        candidates = extract_candidate_points(
            crop_scale, norm_scale, valid_grad, grid_map,
            args.grid_size, args.grad_step,
            args.approach_threshold, args.kp_detector_threshold
        )
        if args.save_cluster_debug_panel:
            collisions, cluster_debug = cluster_and_compute(
                crop_scale, risk_map, candidates, delta_t, args.risk_time_threshold, return_debug=True
            )
            regions, small_region_debug = extract_small_regions(
                args, crop_scale, risk_map, delta_t, args.risk_time_threshold, return_debug=True
            )
        else:
            collisions = cluster_and_compute(
                crop_scale, risk_map, candidates, delta_t, args.risk_time_threshold
            )
            regions = extract_small_regions(
                args, crop_scale, risk_map, delta_t, args.risk_time_threshold
            )
            cluster_debug = None
            small_region_debug = None
        # 把这些小区域也加入最终碰撞列表
        for r in regions:
            collisions.append(r)

        collisions = filter_by_risk(collisions, args.risk_pred_threshold)
        if args.save_cluster_debug_panel:
            debug_panel_path = os.path.join(cluster_debug_dir, f"{args.cluster_debug_prefix}_{idx}.png")
            save_cluster_debug_panel(
                args,
                idx,
                scale_map,
                crop_scale,
                risk_map,
                valid_grad,
                grid_map,
                cluster_debug,
                small_region_debug,
                collisions,
                debug_panel_path,
            )


        # 可视化 range image
        ax.clear()
        visualize_range_image(ax, scale_map, collisions, type='scale')
        # 交互式显示
        if args.keyboard:
            plt.title(fn)
            plt.draw()
            plt.waitforbuttonpress()
            ax.clear()
            inspect_scale_risk(crop_scale, risk_map)
        # 保存可视化结果
        elif args.save:
            if not args.sjtu_test:
                # 1. 保存 pred range image 可视化结果
                out_colli_scale_pred = os.path.join(args.vis_dir, f'colli_scale_pred_{idx}.png')
                save_range_image(fig, ax, out_colli_scale_pred, scale_map)
                # 2. 保存环视图叠加结果
                overlay_surround_views(collisions, test_infos[idx], args, idx)
                # 3. 保存 gt scale range image 可视化结果
                out_colli_scale_gt = os.path.join(args.vis_dir, f'colli_scale_gt_{idx}.png')
                gt = np.load(
                    os.path.join(test_infos[idx]['gt_map_path'], 'range_image_prev.npy'),
                    allow_pickle=True
                ).item()
                gt_scale_map = gt['scale']
                ax.clear()
                visualize_range_image(ax, gt_scale_map, collisions, type='scale')
                save_range_image(fig, ax, out_colli_scale_gt, gt_scale_map)
                # 4. 保存 gt risk range image 可视化结果
                out_colli_risk_gt = os.path.join(args.vis_dir, f'colli_risk_gt_{idx}.png')
                gt = np.load(
                    os.path.join(test_infos[idx]['gt_map_path'], 'range_image_prev.npy'),
                    allow_pickle=True
                ).item()
                gt_risk_map = gt['risk_score']
                ax.clear()
                visualize_range_image(ax, gt_risk_map, collisions, type='risk')
                save_range_image(fig, ax, out_colli_risk_gt, gt_risk_map)
                # 5. 保存 pred risk range image 可视化结果
                out_colli_risk_pred = os.path.join(args.vis_dir, f'colli_risk_pred_{idx}.png')
                ax.clear()
                visualize_range_image(ax, risk_map, collisions, type='risk')
                save_range_image(fig, ax, out_colli_risk_pred, risk_map) 
            else:
                # 1. 保存 pred range image 可视化结果
                out_colli_scale_pred = os.path.join(args.vis_dir, f'colli_scale_pred_{idx}.png')
                save_range_image(fig, ax, out_colli_scale_pred, scale_map)
                # 2. 保存环视图叠加结果
                # overlay_surround_views_sjtu(collisions, test_infos[idx], args, idx)
                # 3. 保存 pred risk range image 可视化结果
                out_colli_risk_pred = os.path.join(args.vis_dir, f'colli_risk_pred_{idx}.png')
                ax.clear()
                visualize_range_image(ax, risk_map, collisions, type='risk')
                save_range_image(fig, ax, out_colli_risk_pred, risk_map) 

        # 逐帧自动显示结果
        else:
            plt.draw()
            plt.pause(0.01)
            ax.clear()


if __name__ == '__main__':
    main()
