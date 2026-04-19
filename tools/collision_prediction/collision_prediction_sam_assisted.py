#!/usr/bin/env python3
import csv
import os
import cv2
import numpy as np
import math
import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from PIL import Image
import datetime
from typing import Dict, List, Optional

from utils.draw import (
    visual_scale_map_range_image,        # 若不再需要，可移除
    visual_risk_score_map_range_image,   # 若不再需要，可移除
    scale2rgb,
    orientation2rgb,
    make_scale_analysis_rgb,
    make_risk_analysis_rgb,
)
from tools.cyberrock.sjtu_pipeline_utils import CAMERA_CHANNELS
from collision_utils import (
    get_second_grad,
    get_ttc_var,
    get_grid_ttc,
    inverse_range_projection,
    project_lidar_to_surround_view_img
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collision risk detection from scale predictions (OpenCV-only visualization)"
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
    parser.add_argument("--risk_time_threshold", type=float, default=5.0,
                        help="Max collision time to consider risk (seconds)")
    parser.add_argument("--risk_pred_threshold", type=float, default=0.1,
                        help="Only keep collision points whose risk/orientation < this threshold")
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
                        help="Use keyboard mode (cv2.imshow + waitKey)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to disk")
    parser.add_argument("--save_reprojection_debug", action="store_true",
                        help="Save extra processed-space reprojection debug images and metrics")
    # Paths
    parser.add_argument("--pred_npy_dir", type=str, required=True,
                        help="Directory of predicted .npy files, including pred scale and risk")
    parser.add_argument('--test_info_path', default='./Datasets/nuscenes/2_trainval_test_infos/nusc_trainval_infos_160_1920.pkl',
                        type=str,
                        help='Path to test info file (e.g., nusc_trainval_infos_160_1920.pkl)')
    parser.add_argument("--vis_dir", type=str, required=True,
                        help="Output directory for overlays")
    parser.add_argument(
        "--proposal_source",
        type=str,
        default="scale_contour",
        choices=["scale_contour", "sam2_rgb_fusion"],
        help="Proposal source for collision prediction",
    )
    parser.add_argument(
        "--sam2_cache_dir",
        type=str,
        default="",
        help="Directory of per-sample SAM2 mask caches (.npz), required for proposal_source=sam2_rgb_fusion",
    )
    parser.add_argument(
        "--sam2_camera_channel",
        type=str,
        default="CAM_FRONT",
        choices=CAMERA_CHANNELS,
        help="Single camera channel used by SAM2 proposal generation and fusion",
    )
    parser.add_argument(
        "--save_front_sam2_rv_debug",
        action="store_true",
        help="Save a single-frame single-camera SAM2 mask projection debug panel back to RV",
    )
    parser.add_argument(
        "--front_sam2_rv_sample_idx",
        type=int,
        default=72,
        help="Sample index used by --save_front_sam2_rv_debug",
    )
    parser.add_argument("--sjtu_test", action="store_true",
                        help="Use SJTU test set, which has no ground truth data")
    parser.add_argument("--sjtu_surround_view_path", type=str, default='./Datasets/sjtu_surround_view',
                        help="Path to SJTU surround view images")
    parser.add_argument("--sjtu_scene_indice", type=int, default=0,
                        help="Scene index for SJTU test set, default is 0")
    return parser.parse_args()


def compute_and_save_masks(scale_map, risk_map, delta_t, ttc_thresh, risk_thresh, save_dir, idx):
    """
    计算两个 mask 后，将它们以二值图形式保存到磁盘（OpenCV）：
      mask_ttc： ttc = delta_t/(1-scale_map) < ttc_thresh
      mask_risk：risk_map > risk_thresh
    """
    eps = 1e-5
    ttc       = delta_t / (1.0 - scale_map + eps)
    mask_ttc  = (ttc       < ttc_thresh)
    mask_risk = (risk_map  > risk_thresh)

    ttc_img  = (mask_ttc .astype(np.uint8) * 255)
    risk_img = (mask_risk.astype(np.uint8) * 255)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"mask_ttc_{idx}.png"),  ttc_img)
    cv2.imwrite(os.path.join(save_dir, f"mask_risk_{idx}.png"), risk_img)
    return mask_ttc, mask_risk


def extract_small_regions(args, scale_map, risk_map, delta_t, ttc_thresh):
    """
    把所有 scale<阈值 的连通域都当碰撞点，返回 [x, y, ttc, scale, risk]
    """
    thresh_mask = (scale_map < args.approach_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_closed = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_closed, connectivity=8
    )
    out = []
    for i in range(1, num_labels):
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        region_xy = np.column_stack([xs, ys])
        best_point = select_best_collision_point(
            scale_map, risk_map, region_xy, delta_t, ttc_thresh
        )
        if best_point is not None:
            out.append(best_point)
    return out


def select_best_collision_point(scale_map, risk_map, region_xy, delta_t, risk_time_threshold):
    """
    在一个区域内部选择真实像素点作为最终 collision point。
    优先级：
      1. TTC 最小
      2. risk/orientation 最小
      3. 距区域质心最近
    返回 [x, y, ttc, scale, risk] 或 None
    """
    region_xy = np.asarray(region_xy, dtype=np.int32)
    if region_xy.size == 0:
        return None

    region_xy = np.unique(region_xy, axis=0)
    centroid = region_xy.astype(np.float32).mean(axis=0)

    best_point = None
    best_key = None
    for x, y in region_xy:
        if y < 0 or y >= scale_map.shape[0] or x < 0 or x >= scale_map.shape[1]:
            continue

        scale_val = float(scale_map[y, x])
        ttc = float(delta_t / (1 - scale_val + 1e-5))
        if not (0 < ttc < risk_time_threshold):
            continue

        risk_val = float(risk_map[y, x])
        dist2 = float((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2)
        key = (ttc, risk_val, dist2)
        if best_key is None or key < best_key:
            best_key = key
            best_point = [int(x), int(y), ttc, scale_val, risk_val]

    return best_point


def compute_ttc_map(scale_map, delta_t):
    scale_map = np.asarray(scale_map, dtype=np.float32)
    return delta_t / (1.0 - scale_map + 1e-5)


def get_vertical_valid_band(height):
    valid_y_min = int(math.ceil(0.30 * float(height)))
    valid_y_max = int(math.floor(0.80 * float(height)))
    valid_y_min = max(0, min(valid_y_min, height))
    valid_y_max = max(valid_y_min, min(valid_y_max, height))
    return valid_y_min, valid_y_max


def build_orientation_support_mask(ttc_map, orientation_map, risk_time_threshold, orientation_threshold):
    support_mask = np.isfinite(ttc_map)
    support_mask &= (ttc_map > 0.0)
    support_mask &= (ttc_map < float(risk_time_threshold))
    support_mask &= np.isfinite(orientation_map)
    support_mask &= (orientation_map < float(orientation_threshold))
    return support_mask.astype(np.uint8)


def build_scale_edge_barrier(scale_map):
    scale_map = np.asarray(scale_map, dtype=np.float32)
    smooth_scale = cv2.bilateralFilter(scale_map, d=9, sigmaColor=0.03, sigmaSpace=7)
    grad_x = cv2.Scharr(smooth_scale, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(smooth_scale, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    finite_mask = np.isfinite(grad_mag)
    finite_vals = grad_mag[finite_mask]
    if finite_vals.size == 0:
        edge_barrier = np.zeros_like(scale_map, dtype=np.uint8)
        return smooth_scale, grad_mag, edge_barrier

    barrier_threshold = float(np.percentile(finite_vals, 85))
    edge_barrier = (grad_mag >= barrier_threshold) & finite_mask
    edge_barrier_u8 = (edge_barrier.astype(np.uint8) * 255)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    edge_barrier_u8 = cv2.dilate(edge_barrier_u8, dilate_kernel)
    edge_barrier_u8 = cv2.morphologyEx(edge_barrier_u8, cv2.MORPH_CLOSE, close_kernel)
    return smooth_scale, grad_mag, (edge_barrier_u8 > 0).astype(np.uint8)


def build_dense_danger_mask(scale_map, orientation_map, delta_t, risk_time_threshold, orientation_threshold):
    ttc_map = compute_ttc_map(scale_map, delta_t)
    valid_mask = np.isfinite(ttc_map)
    valid_mask &= (ttc_map > 0.0)
    valid_mask &= (ttc_map < float(risk_time_threshold))
    valid_mask &= np.isfinite(orientation_map)
    valid_mask &= (orientation_map < float(orientation_threshold))
    return ttc_map, valid_mask.astype(np.uint8)


def postprocess_danger_mask(danger_mask):
    mask_u8 = (danger_mask.astype(np.uint8) * 255)
    mask_u8 = cv2.medianBlur(mask_u8, 5)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
    return (mask_u8 > 0).astype(np.uint8)


def build_region_from_coords(region_xy, ttc_map, orientation_map, valid_y_min, valid_y_max):
    region_xy = np.asarray(region_xy, dtype=np.int32)
    region_xy = np.unique(region_xy, axis=0)
    xs_all = region_xy[:, 0]
    ys_all = region_xy[:, 1]
    bbox_all = (
        int(xs_all.min()),
        int(ys_all.min()),
        int(xs_all.max()),
        int(ys_all.max()),
    )

    valid_mask = (ys_all >= valid_y_min) & (ys_all < valid_y_max)
    coords_valid = region_xy[valid_mask]
    area_valid = int(coords_valid.shape[0])
    if area_valid > 0:
        xs_valid = coords_valid[:, 0]
        ys_valid = coords_valid[:, 1]
        bbox_valid = (
            int(xs_valid.min()),
            int(ys_valid.min()),
            int(xs_valid.max()),
            int(ys_valid.max()),
        )
        ttc_vals = ttc_map[ys_valid, xs_valid]
        orientation_vals = orientation_map[ys_valid, xs_valid]
        min_ttc = float(np.min(ttc_vals))
        min_orientation = float(np.min(orientation_vals))
    else:
        bbox_valid = None
        min_ttc = float("inf")
        min_orientation = float("inf")

    return {
        "coords_all": region_xy,
        "coords_valid": coords_valid,
        "bbox_all": bbox_all,
        "bbox_valid": bbox_valid,
        "area_all": int(region_xy.shape[0]),
        "area_valid": area_valid,
        "min_ttc": min_ttc,
        "min_orientation": min_orientation,
        "dropped_by_band": area_valid == 0,
        "bbox_width": 0,
        "bbox_height": 0,
        "bbox_area_valid": 0,
        "grad_boundary_p70": 0.0,
        "scale_inside_p50": float("inf"),
        "scale_outside_ring_p50": float("inf"),
        "scale_contrast": 0.0,
        "contrast_threshold_used": 0.0,
        "fill_ratio": 0.0,
        "ttc_region_p20": float("inf"),
        "orientation_region_p20": float("inf"),
        "quality_passed": False,
        "quality_drop_reason": "",
        "support_coords": np.empty((0, 2), dtype=np.int32),
        "support_count": 0,
        "support_fraction": 0.0,
        "ttc_p10": float("inf"),
        "orientation_p20": float("inf"),
        "support_cc_max_area": 0,
        "support_cc_max_fraction": 0.0,
        "passed_support": False,
        "support_drop_reason": "",
        "anchor_boundary_dist": 0.0,
        "aspect_ratio": 0.0,
        "bottom_gap": float("inf"),
        "horizontal_span_ratio": 0.0,
        "vertical_span_ratio": 0.0,
        "centroid_y_valid": float("inf"),
        "ground_like_passed": True,
    }


def safe_percentile(values, q, default_value):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float(default_value)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default_value)
    return float(np.percentile(values, q))


def bbox_dims(bbox):
    if bbox is None:
        return 0, 0, 0
    x1, y1, x2, y2 = bbox
    width = int(max(0, x2 - x1 + 1))
    height = int(max(0, y2 - y1 + 1))
    return width, height, int(width * height)


def compute_contrast_threshold(grad_boundary_p70, grad_band_p60, grad_band_p75):
    if float(grad_boundary_p70) >= float(grad_band_p75):
        return 0.002
    return 0.004


def compute_region_quality_stats(region, scale_map, grad_mag, ttc_map, orientation_map):
    bbox = region["bbox_valid"]
    coords = region["coords_valid"]
    if bbox is None or coords.shape[0] == 0:
        return region

    x1, y1, x2, y2 = bbox
    width, height, bbox_area = bbox_dims(bbox)
    region["bbox_width"] = width
    region["bbox_height"] = height
    region["bbox_area_valid"] = bbox_area
    region["fill_ratio"] = float(region["area_valid"]) / float(max(bbox_area, 1))

    xs = coords[:, 0]
    ys = coords[:, 1]
    scale_inside = scale_map[ys, xs]
    ttc_inside = ttc_map[ys, xs]
    orientation_inside = orientation_map[ys, xs]
    region["scale_inside_p50"] = safe_percentile(scale_inside, 50, float("inf"))
    region["ttc_region_p20"] = safe_percentile(ttc_inside, 20, float("inf"))
    region["orientation_region_p20"] = safe_percentile(orientation_inside, 20, float("inf"))

    pad = 4
    ex1 = max(0, x1 - pad)
    ey1 = max(0, y1 - pad)
    ex2 = min(scale_map.shape[1] - 1, x2 + pad)
    ey2 = min(scale_map.shape[0] - 1, y2 + pad)

    local_h = ey2 - ey1 + 1
    local_w = ex2 - ex1 + 1
    region_mask = np.zeros((local_h, local_w), dtype=np.uint8)
    region_mask[ys - ey1, xs - ex1] = 255

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary_mask = cv2.subtract(region_mask, cv2.erode(region_mask, erode_kernel))
    boundary_vals = grad_mag[ey1 : ey2 + 1, ex1 : ex2 + 1][boundary_mask > 0]
    region["grad_boundary_p70"] = safe_percentile(boundary_vals, 70, 0.0)

    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
    ring_mask = cv2.dilate(region_mask, ring_kernel)
    ring_mask = cv2.subtract(ring_mask, region_mask)
    ring_vals = scale_map[ey1 : ey2 + 1, ex1 : ex2 + 1][ring_mask > 0]
    region["scale_outside_ring_p50"] = safe_percentile(ring_vals, 50, region["scale_inside_p50"])
    if np.isfinite(region["scale_inside_p50"]) and np.isfinite(region["scale_outside_ring_p50"]):
        region["scale_contrast"] = float(region["scale_outside_ring_p50"] - region["scale_inside_p50"])
    else:
        region["scale_contrast"] = 0.0
    return region


def apply_region_quality_filters(
    regions,
    scale_map,
    grad_mag,
    ttc_map,
    orientation_map,
    grad_band_p60,
    grad_band_p75,
    min_area_valid=80,
    min_bbox_width=8,
    min_bbox_height=8,
    min_fill_ratio=0.12,
):
    kept = []
    annotated = []
    for region in regions:
        compute_region_quality_stats(region, scale_map, grad_mag, ttc_map, orientation_map)
        region["contrast_threshold_used"] = compute_contrast_threshold(
            region["grad_boundary_p70"],
            grad_band_p60,
            grad_band_p75,
        )
        drop_reason = ""
        if region["area_valid"] < min_area_valid or region["bbox_width"] < min_bbox_width or region["bbox_height"] < min_bbox_height:
            drop_reason = "drop_small"
        elif region["grad_boundary_p70"] < float(grad_band_p60):
            drop_reason = "drop_weak_edge"
        elif region["scale_contrast"] < float(region["contrast_threshold_used"]):
            drop_reason = "drop_low_contrast"
        elif region["fill_ratio"] < float(min_fill_ratio):
            drop_reason = "drop_sparse_fill"
        region["quality_drop_reason"] = drop_reason
        region["quality_passed"] = (drop_reason == "")
        annotated.append(region)
        if region["quality_passed"]:
            kept.append(region)
    return kept, annotated


def extract_contour_filled_regions(
    edge_barrier,
    ttc_map,
    orientation_map,
    valid_y_min,
    valid_y_max,
    scale_for_foreground,
    foreground_scale_threshold=0.9,
    contour_area_threshold=80.0,
    min_area=60,
):
    band_mask = np.zeros_like(edge_barrier, dtype=np.uint8)
    band_mask[valid_y_min:valid_y_max, :] = 1
    foreground_mask = np.isfinite(scale_for_foreground)
    foreground_mask &= (scale_for_foreground < float(foreground_scale_threshold))
    foreground_mask &= np.isfinite(ttc_map)
    foreground_mask_u8 = foreground_mask.astype(np.uint8)

    closed_edges = (edge_barrier.astype(np.uint8) * 255)
    closed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    closed_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_CLOSE, closed_kernel)
    closed_edges = cv2.dilate(closed_edges, dilate_kernel)
    closed_edges = cv2.bitwise_and(closed_edges, closed_edges, mask=(band_mask * 255))

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(edge_barrier, dtype=np.uint8)
    kept_contours = []
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue
        if float(cv2.contourArea(contour)) < float(contour_area_threshold):
            continue
        kept_contours.append(contour)

    if kept_contours:
        cv2.fillPoly(filled_mask, kept_contours, 1)

    proposal_mask = filled_mask.astype(bool)
    proposal_mask &= foreground_mask_u8.astype(bool)
    proposal_mask &= band_mask.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(proposal_mask.astype(np.uint8), connectivity=8)
    regions = []
    for label in range(1, num_labels):
        area_all = int(stats[label, cv2.CC_STAT_AREA])
        if area_all < min_area:
            continue
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        region_xy = np.column_stack([xs, ys])
        region = build_region_from_coords(
            region_xy,
            ttc_map,
            orientation_map,
            valid_y_min,
            valid_y_max,
        )
        region["component_id"] = len(regions)
        regions.append(region)
    return regions, {
        "band_mask": band_mask,
        "foreground_mask": foreground_mask_u8,
        "closed_contour_mask": (closed_edges > 0).astype(np.uint8),
        "filled_proposal_mask": proposal_mask.astype(np.uint8),
    }


def enrich_region_with_support(region, ttc_map, orientation_map, risk_time_threshold, orientation_threshold):
    coords = region["coords_valid"]
    if coords.shape[0] == 0:
        return region

    xs = coords[:, 0]
    ys = coords[:, 1]
    region_ttc = ttc_map[ys, xs]
    region_orientation = orientation_map[ys, xs]

    support_mask = np.isfinite(region_ttc)
    support_mask &= (region_ttc > 0.0)
    support_mask &= (region_ttc < float(risk_time_threshold))
    support_mask &= np.isfinite(region_orientation)
    support_mask &= (region_orientation < float(orientation_threshold))

    support_coords = coords[support_mask]
    support_count = int(support_coords.shape[0])
    support_fraction = float(support_count) / float(max(region["area_valid"], 1))

    region["support_coords"] = support_coords
    region["support_count"] = support_count
    region["support_fraction"] = support_fraction

    if support_count == 0:
        region["ttc_p10"] = float("inf")
        region["orientation_p20"] = float("inf")
        region["passed_support"] = False
        region["support_cc_max_area"] = 0
        region["support_cc_max_fraction"] = 0.0
        region["support_drop_reason"] = "drop_weak_support"
        return region

    support_ttc = region_ttc[support_mask]
    support_orientation = region_orientation[support_mask]
    region["ttc_p10"] = float(np.percentile(support_ttc, 10))
    region["orientation_p20"] = float(np.percentile(support_orientation, 20))

    x1, y1, x2, y2 = region["bbox_valid"]
    local_support_mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    local_support_mask[support_coords[:, 1] - y1, support_coords[:, 0] - x1] = 1
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(local_support_mask, connectivity=8)
    if num_labels > 1:
        support_cc_max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    else:
        support_cc_max_area = 0
    support_cc_max_fraction = float(support_cc_max_area) / float(max(support_count, 1))
    region["support_cc_max_area"] = support_cc_max_area
    region["support_cc_max_fraction"] = support_cc_max_fraction

    if support_count < 25 or support_fraction < 0.01:
        region["passed_support"] = False
        region["support_drop_reason"] = "drop_weak_support"
    elif support_cc_max_area < 12 or support_cc_max_fraction < 0.35:
        region["passed_support"] = False
        region["support_drop_reason"] = "drop_sparse_support"
    else:
        region["passed_support"] = True
        region["support_drop_reason"] = ""
    return region


def extract_dense_regions(mask, ttc_map, orientation_map, valid_y_min, valid_y_max, min_area=40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for label in range(1, num_labels):
        area_all = int(stats[label, cv2.CC_STAT_AREA])
        if area_all < min_area:
            continue
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        region_xy = np.column_stack([xs, ys])
        region = build_region_from_coords(
            region_xy,
            ttc_map,
            orientation_map,
            valid_y_min,
            valid_y_max,
        )
        region["component_id"] = len(regions)
        regions.append(region)
    return regions


def bbox_min_gap(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    gap_x = max(0, max(bx1 - ax2, ax1 - bx2))
    gap_y = max(0, max(by1 - ay2, ay1 - by2))
    return float(min(gap_x, gap_y))


def compute_region_interface_grad_mean(region_a, region_b, grad_mag):
    bbox_a = region_a["bbox_valid"]
    bbox_b = region_b["bbox_valid"]
    if bbox_a is None or bbox_b is None:
        return float("inf")

    gap = bbox_min_gap(bbox_a, bbox_b)
    bridge_w = max(5, int(math.ceil(gap)) * 2 + 3)
    if bridge_w % 2 == 0:
        bridge_w += 1
    bridge_h = max(3, min(11, bridge_w))
    if bridge_h % 2 == 0:
        bridge_h += 1
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_w, bridge_h))

    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ex1 = max(0, min(ax1, bx1) - bridge_w)
    ey1 = max(0, min(ay1, by1) - bridge_h)
    ex2 = min(grad_mag.shape[1] - 1, max(ax2, bx2) + bridge_w)
    ey2 = min(grad_mag.shape[0] - 1, max(ay2, by2) + bridge_h)

    mask_a = np.zeros((ey2 - ey1 + 1, ex2 - ex1 + 1), dtype=np.uint8)
    mask_b = np.zeros_like(mask_a)
    coords_a = region_a["coords_valid"]
    coords_b = region_b["coords_valid"]
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return float("inf")
    mask_a[coords_a[:, 1] - ey1, coords_a[:, 0] - ex1] = 255
    mask_b[coords_b[:, 1] - ey1, coords_b[:, 0] - ex1] = 255

    dil_a = cv2.dilate(mask_a, bridge_kernel)
    dil_b = cv2.dilate(mask_b, bridge_kernel)
    interface_mask = (dil_a > 0) & (dil_b > 0)
    if not np.any(interface_mask):
        return float("inf")
    interface_vals = grad_mag[ey1 : ey2 + 1, ex1 : ex2 + 1][interface_mask]
    return safe_percentile(interface_vals, 50, float("inf"))


def merge_close_regions(
    regions,
    scale_map,
    ttc_map,
    orientation_map,
    grad_mag,
    valid_y_min,
    valid_y_max,
    grad_band_p60,
    grad_band_p75,
    max_bbox_gap=16.0,
    max_scale_diff=0.03,
    max_ttc_proxy_diff=0.5,
    max_boundary_grad=float("inf"),
):
    if not regions:
        return []

    parent = list(range(len(regions)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            region_a = regions[i]
            region_b = regions[j]
            if bbox_min_gap(region_a["bbox_valid"], region_b["bbox_valid"]) > max_bbox_gap:
                continue
            if abs(region_a["scale_inside_p50"] - region_b["scale_inside_p50"]) > max_scale_diff:
                continue
            if abs(region_a["ttc_region_p20"] - region_b["ttc_region_p20"]) > max_ttc_proxy_diff:
                continue
            if compute_region_interface_grad_mean(region_a, region_b, grad_mag) > max_boundary_grad:
                continue
            union(i, j)

    grouped = {}
    for idx, region in enumerate(regions):
        grouped.setdefault(find(idx), []).append(region)

    merged_regions = []
    for merged_id, members in enumerate(grouped.values()):
        merged_xy = np.vstack([member["coords_all"] for member in members])
        merged_xy = np.unique(merged_xy, axis=0)
        merged_region = build_region_from_coords(
            merged_xy,
            ttc_map,
            orientation_map,
            valid_y_min,
            valid_y_max,
        )
        compute_region_quality_stats(merged_region, scale_map, grad_mag, ttc_map, orientation_map)
        merged_region["contrast_threshold_used"] = compute_contrast_threshold(
            merged_region["grad_boundary_p70"],
            grad_band_p60,
            grad_band_p75,
        )
        merged_region["merged_region_id"] = merged_id
        merged_region["component_id"] = merged_id
        merged_regions.append(merged_region)
    return merged_regions


def select_region_anchor(scale_map, orientation_map, ttc_map, region):
    region_xy = region["coords_valid"]
    if region_xy.shape[0] == 0:
        return None
    xs = region_xy[:, 0]
    ys = region_xy[:, 1]
    region_ttc = ttc_map[ys, xs]
    region_orientation = orientation_map[ys, xs]

    ttc_thr = np.percentile(region_ttc, 15)
    orientation_thr = np.percentile(region_orientation, 30)
    core_mask = (region_ttc <= ttc_thr) & (region_orientation <= orientation_thr)
    core_xy = region_xy[core_mask]

    if core_xy.shape[0] < 10:
        order = np.lexsort((region_orientation, region_ttc))
        topk = min(20, region_xy.shape[0])
        core_xy = region_xy[order[:topk]]

    core_xy = np.unique(core_xy, axis=0)
    if core_xy.shape[0] == 0:
        return None

    if core_xy.shape[0] == 1:
        medoid_xy = core_xy[0]
    else:
        pts = core_xy.astype(np.float32)
        dists = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        medoid_xy = core_xy[int(np.argmin(dists.sum(axis=1)))]

    x = int(medoid_xy[0])
    y = int(medoid_xy[1])
    return [x, y, float(ttc_map[y, x]), float(scale_map[y, x]), float(orientation_map[y, x])]


def select_scale_region_anchor(scale_map, orientation_map, ttc_map, region):
    support_xy = region["support_coords"]
    if support_xy.shape[0] == 0:
        return None, 0.0

    xs = support_xy[:, 0]
    ys = support_xy[:, 1]
    support_ttc = ttc_map[ys, xs]
    support_orientation = orientation_map[ys, xs]

    ttc_thr = np.percentile(support_ttc, 25)
    orientation_thr = np.percentile(support_orientation, 35)
    core_mask = (support_ttc <= ttc_thr) & (support_orientation <= orientation_thr)
    core_xy = support_xy[core_mask]
    core_ttc = support_ttc[core_mask]
    core_orientation = support_orientation[core_mask]

    if core_xy.shape[0] < 10:
        order = np.lexsort((support_orientation, support_ttc))
        topk = min(30, support_xy.shape[0])
        chosen = order[:topk]
        core_xy = support_xy[chosen]
        core_ttc = support_ttc[chosen]
        core_orientation = support_orientation[chosen]

    core_xy = np.unique(core_xy, axis=0)
    if core_xy.shape[0] == 0:
        return None, 0.0

    x1, y1, x2, y2 = region["bbox_valid"]
    region_mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    coords = region["coords_valid"]
    region_mask[coords[:, 1] - y1, coords[:, 0] - x1] = 255
    dist_map = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)

    best_point = None
    best_key = None
    best_boundary_dist = 0.0
    for idx, (x, y) in enumerate(core_xy):
        local_x = int(x) - x1
        local_y = int(y) - y1
        dist_to_boundary = float(dist_map[local_y, local_x])
        key = (-dist_to_boundary, float(core_ttc[idx]), float(core_orientation[idx]))
        if best_key is None or key < best_key:
            best_key = key
            best_boundary_dist = dist_to_boundary
            best_point = [
                int(x),
                int(y),
                float(ttc_map[int(y), int(x)]),
                float(scale_map[int(y), int(x)]),
                float(orientation_map[int(y), int(x)]),
            ]
    return best_point, float(best_boundary_dist)


def select_region_center_anchor(scale_map, orientation_map, ttc_map, region):
    coords = np.asarray(region.get("coords_valid", []), dtype=np.int32)
    if coords.shape[0] == 0:
        return None, 0.0

    bbox = region.get("bbox_valid")
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (float(x1) + float(x2))
        cy = 0.5 * (float(y1) + float(y2))
    else:
        cx = float(coords[:, 0].mean())
        cy = float(coords[:, 1].mean())

    deltas = coords.astype(np.float32) - np.array([cx, cy], dtype=np.float32)
    nearest_idx = int(np.argmin((deltas ** 2).sum(axis=1)))
    x = int(coords[nearest_idx, 0])
    y = int(coords[nearest_idx, 1])

    boundary_dist = 0.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        region_mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
        region_mask[coords[:, 1] - y1, coords[:, 0] - x1] = 255
        dist_map = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)
        boundary_dist = float(dist_map[y - y1, x - x1])

    anchor = [x, y, float(ttc_map[y, x]), float(scale_map[y, x]), float(orientation_map[y, x])]
    return anchor, boundary_dist


def annotate_invalid_bands(image_bgr, valid_y_min, valid_y_max):
    out = image_bgr.copy()
    overlay = out.copy()
    h = out.shape[0]
    if valid_y_min > 0:
        cv2.rectangle(overlay, (0, 0), (out.shape[1] - 1, valid_y_min - 1), (96, 96, 96), -1)
    if valid_y_max < h:
        cv2.rectangle(overlay, (0, valid_y_max), (out.shape[1] - 1, h - 1), (96, 96, 96), -1)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    cv2.line(out, (0, valid_y_min), (out.shape[1] - 1, valid_y_min), (0, 0, 0), 1)
    cv2.line(out, (0, valid_y_max - 1), (out.shape[1] - 1, valid_y_max - 1), (0, 0, 0), 1)
    return out


def make_binary_mask_bgr(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def render_gradient_barrier_panel(grad_mag, edge_barrier):
    grad_mag = np.nan_to_num(np.asarray(grad_mag, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(grad_mag > 0):
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        grad_norm = np.zeros_like(grad_mag, dtype=np.float32)
    grad_u8 = grad_norm.astype(np.uint8)
    panel = cv2.applyColorMap(grad_u8, cv2.COLORMAP_TURBO)
    overlay = panel.copy()
    overlay[edge_barrier > 0] = (0, 0, 255)
    return cv2.addWeighted(overlay, 0.45, panel, 0.55, 0)


def color_for_index(idx):
    palette = [
        (255, 128, 0),
        (0, 200, 255),
        (0, 180, 0),
        (255, 0, 180),
        (180, 0, 255),
        (0, 255, 120),
        (255, 180, 0),
        (80, 160, 255),
    ]
    return palette[idx % len(palette)]


def render_component_canvas(shape_hw, regions, use_valid_coords=False):
    h, w = shape_hw
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, region in enumerate(regions):
        coords = region["coords_valid"] if use_valid_coords else region["coords_all"]
        if coords.shape[0] == 0:
            continue
        color = color_for_index(idx)
        xs = coords[:, 0]
        ys = coords[:, 1]
        canvas[ys, xs] = color
    return canvas


def draw_region_box_and_label(img, bbox, text, color, thickness=2):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img,
        text,
        (x1 + 4, max(16, y1 + 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        lineType=cv2.LINE_AA,
    )


def save_cluster_region_panel(
    args,
    sample_idx,
    scale_map_full,
    grad_mag,
    edge_barrier,
    closed_contour_mask,
    filled_proposal_mask,
    proposal_regions,
    merged_regions,
    collisions,
    valid_y_min,
    valid_y_max,
    selected_region_id,
):
    panel_scale = render_range_image_bgr(scale_map_full, [], kind="scale")
    panel_scale = annotate_invalid_bands(panel_scale, valid_y_min, valid_y_max)

    panel_grad = render_gradient_barrier_panel(grad_mag, edge_barrier)
    panel_grad = annotate_invalid_bands(panel_grad, valid_y_min, valid_y_max)

    panel_components = render_component_canvas(scale_map_full.shape, proposal_regions, use_valid_coords=True)
    panel_components = annotate_invalid_bands(panel_components, valid_y_min, valid_y_max)
    contour_overlay = panel_components.copy()
    contour_overlay[closed_contour_mask > 0] = (255, 255, 255)
    contour_overlay[(filled_proposal_mask > 0) & np.all(panel_components == 0, axis=2)] = (80, 80, 80)
    panel_components = cv2.addWeighted(contour_overlay, 0.55, panel_components, 0.45, 0)
    for region in proposal_regions:
        if region["quality_passed"]:
            color = color_for_index(region["component_id"])
            text = f"r{region['component_id']} a={region['area_valid']}"
            draw_region_box_and_label(panel_components, region["bbox_valid"], text, color, thickness=2)
        else:
            reason = region["quality_drop_reason"] or "drop"
            if reason == "drop_low_contrast":
                text = (
                    f"r{region['component_id']} {reason} "
                    f"c={region['scale_contrast']:.3f} "
                    f"ct={region['contrast_threshold_used']:.3f}"
                )
            else:
                text = f"r{region['component_id']} {reason}"
            draw_region_box_and_label(panel_components, region["bbox_valid"], text, (0, 0, 255), thickness=2)

    panel_support = render_range_image_bgr(scale_map_full, [], kind="scale")
    panel_support = annotate_invalid_bands(panel_support, valid_y_min, valid_y_max)
    for region in merged_regions:
        bbox = region["bbox_valid"]
        if bbox is None:
            continue
        if region["component_id"] == selected_region_id:
            color = (0, 255, 0)
            thickness = 3
        elif region["passed_support"]:
            color = (0, 220, 255)
            thickness = 2
        else:
            color = (0, 0, 255)
            thickness = 1
        text = (
            f"r{region['component_id']} s={region['support_count']} "
            f"f={region['support_fraction']:.2f} "
            f"t={region['ttc_p10']:.2f} o={region['orientation_p20']:.2f} "
            f"g={region['grad_boundary_p70']:.3f} "
            f"c={region['scale_contrast']:.3f} "
            f"ct={region['contrast_threshold_used']:.3f} "
            f"m={region['support_cc_max_fraction']:.2f} "
            f"d={region['anchor_boundary_dist']:.1f}"
        )
        draw_region_box_and_label(panel_support, bbox, text, color, thickness=thickness)

    if collisions:
        for x, y, _, _, _ in collisions:
            cv2.circle(panel_support, (int(x), int(y)), 6, (0, 0, 0), -1)

    top_row = np.hstack([panel_scale, panel_grad])
    bottom_row = np.hstack([panel_components, panel_support])
    panel = np.vstack([top_row, bottom_row])
    out_path = os.path.join(args.vis_dir, f"cluster_regions_{sample_idx}.png")
    cv2.imwrite(out_path, panel)
    return out_path


def save_sam2_cluster_region_panel(
    args,
    sample_idx,
    sample_info,
    scale_map_full,
    sam2_sample_cache,
    projection_debug,
    proposal_regions,
    merged_regions,
    passed_regions,
    collisions,
    collision_details,
    valid_y_min,
    valid_y_max,
):
    camera_channel = args.sam2_camera_channel
    panel_scale = render_range_image_bgr(scale_map_full, [], kind="scale")
    panel_scale = annotate_invalid_bands(panel_scale, valid_y_min, valid_y_max)
    panel_scale = annotate_panel_tile(panel_scale, "Scale + Valid Band")

    raw_curr_rgb = load_raw_curr_camera_rgbs(sample_info)
    rgb_panel = np.asarray(raw_curr_rgb[camera_channel], dtype=np.uint8).copy()
    camera_cache = decode_sam2_camera_cache(sam2_sample_cache, camera_channel)
    for mask_id in range(camera_cache["mask_count"]):
        color = color_for_index(mask_id)
        rgb_panel = render_mask_overlay(rgb_panel, camera_cache["masks"][mask_id], color, alpha=0.35)
        x1, y1, x2, y2 = camera_cache["bbox_xyxy"][mask_id].tolist()
        cv2.rectangle(rgb_panel, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            rgb_panel,
            f"m{mask_id}",
            (x1 + 4, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    panel_masks = fit_rgb_to_canvas(rgb_panel, panel_scale.shape[:2], bg_color=(255, 255, 255))
    panel_masks = annotate_panel_tile(panel_masks, f"{camera_channel} RGB Masks")

    panel_projected = render_projected_rv_masks_only(projection_debug)
    panel_projected = annotate_invalid_bands(panel_projected, valid_y_min, valid_y_max)
    for region in merged_regions:
        bbox = region.get("bbox_valid")
        if bbox is None:
            continue
        if region.get("quality_drop_reason") == "drop_ground_like":
            text = (
                f"r{region['component_id']} drop_ground_like "
                f"w={region['bbox_width']} h={region['bbox_height']} "
                f"ar={region['aspect_ratio']:.1f} "
                f"bg={region['bottom_gap']:.0f} "
                f"fs={region['fill_ratio']:.2f}"
            )
            color = (0, 0, 255)
            thickness = 2
        elif region.get("quality_drop_reason") == "drop_high_orientation":
            text = (
                f"r{region['component_id']} drop_high_r "
                f"hc={region.get('high_orientation_count', 0)} "
                f"hf={region.get('high_orientation_fraction', 0.0):.2f}"
            )
            color = (0, 0, 255)
            thickness = 2
        elif region.get("quality_drop_reason") == "drop_small_box":
            text = (
                f"r{region['component_id']} drop_small_box "
                f"w={region['bbox_width']} h={region['bbox_height']}"
            )
            color = (0, 0, 255)
            thickness = 2
        else:
            text = (
                f"r{region['component_id']} src={region['source_mask_count']} "
                f"a={region['rv_mask_area_valid']} "
                f"w={region['bbox_width']} h={region['bbox_height']}"
            )
            color = (0, 0, 0)
            thickness = 2
        draw_region_box_and_label(panel_projected, bbox, text, color, thickness=thickness)
    panel_projected = annotate_panel_tile(panel_projected, "Projected RV Masks")

    selected_region_id = collision_details[0].get("region_id") if collision_details else None
    panel_support = render_range_image_bgr(scale_map_full, [], kind="scale")
    panel_support = annotate_invalid_bands(panel_support, valid_y_min, valid_y_max)
    for region in merged_regions:
        if not region.get("ground_like_passed", True):
            continue
        bbox = region.get("bbox_valid")
        if bbox is None:
            continue
        if region["component_id"] == selected_region_id:
            color = (0, 255, 0)
            thickness = 3
        elif region["passed_support"]:
            color = (0, 220, 255)
            thickness = 2
        else:
            color = (0, 0, 255)
            thickness = 1
        text = (
            f"r{region['component_id']} src={region['source_mask_count']} "
            f"s={region['support_count']} "
            f"f={region['support_fraction']:.2f} "
            f"t={region['ttc_p10']:.2f} "
            f"o={region['orientation_p20']:.2f} "
            f"m={region['support_cc_max_fraction']:.2f} "
            f"d={region['anchor_boundary_dist']:.1f}"
        )
        draw_region_box_and_label(panel_support, bbox, text, color, thickness=thickness)
    if collisions:
        for x, y, _, _, _ in collisions:
            cv2.circle(panel_support, (int(x), int(y)), 6, (0, 0, 0), -1)
    panel_support = annotate_panel_tile(panel_support, "Projected RV Mask Support + Final Anchor")

    top_row = np.hstack([panel_scale, panel_masks])
    bottom_row = np.hstack([panel_projected, panel_support])
    panel = np.vstack([top_row, bottom_row])
    out_path = os.path.join(args.vis_dir, f"cluster_regions_{sample_idx}.png")
    cv2.imwrite(out_path, panel)
    return out_path


def detect_dense_collision_regions(scale_map, orientation_map, delta_t, risk_time_threshold, orientation_threshold):
    ttc_map = compute_ttc_map(scale_map, delta_t)
    support_mask = build_orientation_support_mask(
        ttc_map,
        orientation_map,
        risk_time_threshold,
        orientation_threshold,
    )
    valid_y_min, valid_y_max = get_vertical_valid_band(scale_map.shape[0])
    smooth_scale, grad_mag, edge_barrier = build_scale_edge_barrier(scale_map)
    raw_regions, proposal_debug = extract_contour_filled_regions(
        edge_barrier=edge_barrier,
        ttc_map=ttc_map,
        orientation_map=orientation_map,
        valid_y_min=valid_y_min,
        valid_y_max=valid_y_max,
        scale_for_foreground=smooth_scale,
        foreground_scale_threshold=0.9,
        min_area=60,
    )

    band_mask = np.zeros_like(scale_map, dtype=bool)
    band_mask[valid_y_min:valid_y_max, :] = True
    grad_band_vals = grad_mag[np.isfinite(grad_mag) & band_mask]
    grad_band_p60 = safe_percentile(grad_band_vals, 60, 0.0)
    grad_band_p75 = safe_percentile(grad_band_vals, 75, grad_band_p60)
    merge_boundary_threshold = safe_percentile(grad_band_vals, 50, float("inf"))

    quality_regions, annotated_regions = apply_region_quality_filters(
        raw_regions,
        scale_map=smooth_scale,
        grad_mag=grad_mag,
        ttc_map=ttc_map,
        orientation_map=orientation_map,
        grad_band_p60=grad_band_p60,
        grad_band_p75=grad_band_p75,
    )
    merged_regions = merge_close_regions(
        quality_regions,
        scale_map=smooth_scale,
        ttc_map=ttc_map,
        orientation_map=orientation_map,
        grad_mag=grad_mag,
        valid_y_min=valid_y_min,
        valid_y_max=valid_y_max,
        grad_band_p60=grad_band_p60,
        grad_band_p75=grad_band_p75,
        max_bbox_gap=16.0,
        max_scale_diff=0.03,
        max_ttc_proxy_diff=0.5,
        max_boundary_grad=merge_boundary_threshold,
    )

    for region in merged_regions:
        enrich_region_with_support(
            region,
            ttc_map,
            orientation_map,
            risk_time_threshold,
            orientation_threshold,
        )

    passed_regions = [region for region in merged_regions if region["passed_support"]]

    ranked = []
    for region in passed_regions:
        anchor, anchor_boundary_dist = select_scale_region_anchor(scale_map, orientation_map, ttc_map, region)
        if anchor is None:
            continue
        region["anchor_boundary_dist"] = anchor_boundary_dist
        ranked.append(
            {
                "region": region,
                "anchor": anchor,
                "ttc_p10": region["ttc_p10"],
                "orientation_p20": region["orientation_p20"],
                "support_count": region["support_count"],
                "support_fraction": region["support_fraction"],
            }
        )

    ranked.sort(
        key=lambda item: (
            item["ttc_p10"],
            item["orientation_p20"],
            -item["support_count"],
            -item["support_fraction"],
        )
    )
    selected_region_id = ranked[0]["region"]["component_id"] if ranked else None
    quality_drop_counts = {}
    contrast_thresholds_used = {}
    for region in annotated_regions:
        reason = region.get("quality_drop_reason", "")
        if reason:
            quality_drop_counts[reason] = quality_drop_counts.get(reason, 0) + 1
        contrast_thresholds_used[int(region["component_id"])] = float(region.get("contrast_threshold_used", 0.0))
    debug = {
        "ttc_map": ttc_map,
        "raw_mask": support_mask,
        "grad_mag": grad_mag,
        "edge_barrier": edge_barrier,
        "closed_contour_mask": proposal_debug["closed_contour_mask"],
        "filled_proposal_mask": proposal_debug["filled_proposal_mask"],
        "proposal_regions": annotated_regions,
        "merged_regions": merged_regions,
        "passed_regions": passed_regions,
        "valid_y_min": valid_y_min,
        "valid_y_max": valid_y_max,
        "selected_region_id": selected_region_id,
        "proposal_count_raw": len(raw_regions),
        "proposal_count_quality": len(quality_regions),
        "proposal_count_merged": len(merged_regions),
        "quality_drop_counts": quality_drop_counts,
        "contrast_thresholds_used": contrast_thresholds_used,
        "grad_band_p60": grad_band_p60,
        "grad_band_p75": grad_band_p75,
    }
    return [item["anchor"] for item in ranked[:1]], debug


def load_test_infos(test_info_path):
    with open(test_info_path, 'rb') as f:
        return pickle.load(f)


def get_pred_npy_files(pred_dir):
    files = [f for f in os.listdir(pred_dir) if f.endswith('.npy')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    return files


def load_prediction(pred_path):
    return np.load(pred_path, allow_pickle=True).item()


def get_sample_info(test_infos, sample_idx):
    if sample_idx < 0 or sample_idx >= len(test_infos):
        raise IndexError(
            f"Prediction sample_idx={sample_idx} is out of range for test_infos with length={len(test_infos)}"
        )
    return test_infos[sample_idx]


def get_effective_sample_meta(sample_info, pred_data):
    scene_indice = pred_data.get("scene_indice", sample_info.get("scene_indice", ""))
    ros_msg_seq_prev = pred_data.get("ros_msg_seq_prev", sample_info.get("ros_msg_seq_prev", ""))
    ros_msg_seq_curr = pred_data.get("ros_msg_seq_curr", sample_info.get("ros_msg_seq_curr", ""))
    time_diff_cam_us = pred_data.get("time_diff_cam_us", sample_info.get("time_diff_cam_us"))
    bag_name = scene_indice if isinstance(scene_indice, str) and scene_indice.endswith(".bag") else ""
    return {
        "scene_indice": scene_indice,
        "bag_name": bag_name,
        "ros_msg_seq_prev": ros_msg_seq_prev,
        "ros_msg_seq_curr": ros_msg_seq_curr,
        "time_diff_cam_us": time_diff_cam_us,
    }


def resolve_delta_t_seconds(args, sample_meta):
    time_diff_cam_us = sample_meta.get("time_diff_cam_us")
    if time_diff_cam_us is not None and float(time_diff_cam_us) > 0:
        return float(time_diff_cam_us) / 1e6
    return 1.0 / args.fps


def derive_sjtu_prepared_eval_root(test_info_path: str) -> Path:
    test_info_path = Path(test_info_path).resolve()
    for candidate in [test_info_path.parent] + list(test_info_path.parents):
        stitched_manifest = candidate / "annotations" / "manifests" / "stitched_frames_manifest.csv"
        if stitched_manifest.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not derive prepared_eval_data root from test_info_path={test_info_path}"
    )


def build_sjtu_stitched_index(test_info_path: str):
    prepared_eval_root = derive_sjtu_prepared_eval_root(test_info_path)
    stitched_manifest_path = prepared_eval_root / "annotations" / "manifests" / "stitched_frames_manifest.csv"
    index = {}
    with stitched_manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["bag_name"], int(row["ros_msg_seq"]))
            index[key] = row["stitched_image_path"]
    return stitched_manifest_path, index


def resolve_sjtu_stitched_image_path(sample_meta, stitched_index):
    bag_name = sample_meta.get("bag_name", "")
    ros_msg_seq_curr = sample_meta.get("ros_msg_seq_curr")
    if not bag_name or ros_msg_seq_curr in {"", None}:
        return None
    try:
        return stitched_index.get((bag_name, int(ros_msg_seq_curr)))
    except (TypeError, ValueError):
        return None


def get_fullres_proj_pix_curr(pred_data):
    proj_pix_curr_fullres = pred_data.get("proj_pix_curr_fullres")
    if proj_pix_curr_fullres is None:
        raise KeyError(
            "Prediction file does not contain proj_pix_curr_fullres. "
            "Rerun test.py with --save_pred_npy using the updated full-resolution mapping payload."
        )
    proj_pix_curr_fullres = np.asarray(proj_pix_curr_fullres)
    if proj_pix_curr_fullres.ndim != 3 or proj_pix_curr_fullres.shape[2] != 3:
        raise ValueError(
            f"Expected proj_pix_curr_fullres to have shape (H, W, 3), got {proj_pix_curr_fullres.shape}"
        )
    return proj_pix_curr_fullres


def map_rv_point_to_stitched_orig(x_rv, y_rv, pred_data, stitched_img_width):
    proj_pix_curr = get_fullres_proj_pix_curr(pred_data)
    affine_matrix = pred_data.get("affine_matrix")
    if affine_matrix is None:
        return {
            "mapping_valid": 0,
            "cam_idx": "",
            "u_proc": "",
            "v_proc": "",
            "x_stitched_orig": "",
            "y_stitched_orig": "",
        }

    affine_matrix = np.asarray(affine_matrix, dtype=np.float32)

    h_rv, w_rv = proj_pix_curr.shape[:2]
    x_rv_i = int(np.clip(round(float(x_rv)), 0, w_rv - 1))
    y_rv_i = int(np.clip(round(float(y_rv)), 0, h_rv - 1))
    cam_idx, u_proc, v_proc = proj_pix_curr[y_rv_i, x_rv_i].tolist()

    if cam_idx < 0 or u_proc < 0 or v_proc < 0:
        return {
            "mapping_valid": 0,
            "cam_idx": "",
            "u_proc": "",
            "v_proc": "",
            "x_stitched_orig": "",
            "y_stitched_orig": "",
        }

    inv_affine = np.linalg.inv(affine_matrix).astype(np.float32)
    pix_proc = np.array([u_proc, v_proc, 1.0], dtype=np.float32)
    pix_orig = inv_affine @ pix_proc
    pix_orig /= max(float(pix_orig[2]), 1e-6)

    camera_width = stitched_img_width / float(len(CAMERA_CHANNELS))
    x_stitched_orig = float(cam_idx) * camera_width + float(pix_orig[0])
    y_stitched_orig = float(pix_orig[1])

    return {
        "mapping_valid": 1,
        "cam_idx": int(cam_idx),
        "u_proc": float(u_proc),
        "v_proc": float(v_proc),
        "x_stitched_orig": x_stitched_orig,
        "y_stitched_orig": y_stitched_orig,
    }


def map_rv_point_to_stitched_proc(x_rv, y_rv, pred_data, stitched_img_width):
    proj_pix_curr = get_fullres_proj_pix_curr(pred_data)

    h_rv, w_rv = proj_pix_curr.shape[:2]
    x_rv_i = int(np.clip(round(float(x_rv)), 0, w_rv - 1))
    y_rv_i = int(np.clip(round(float(y_rv)), 0, h_rv - 1))
    cam_idx, u_proc, v_proc = proj_pix_curr[y_rv_i, x_rv_i].tolist()

    if cam_idx < 0 or u_proc < 0 or v_proc < 0:
        return {
            "mapping_valid": 0,
            "cam_idx": "",
            "u_proc": "",
            "v_proc": "",
            "x_stitched_proc": "",
            "y_stitched_proc": "",
        }

    camera_width = stitched_img_width / float(len(CAMERA_CHANNELS))
    x_stitched_proc = float(cam_idx) * camera_width + float(u_proc)
    y_stitched_proc = float(v_proc)

    return {
        "mapping_valid": 1,
        "cam_idx": int(cam_idx),
        "u_proc": float(u_proc),
        "v_proc": float(v_proc),
        "x_stitched_proc": x_stitched_proc,
        "y_stitched_proc": y_stitched_proc,
    }


def deduplicate_collision_points(collisions, radius_px=8.0):
    if not collisions:
        return []
    pts_xy = np.array([[pt[0], pt[1]] for pt in collisions], dtype=np.float32)
    clu = DBSCAN(eps=radius_px, min_samples=1).fit(pts_xy)
    merged = []
    for label in sorted(set(clu.labels_)):
        members = [collisions[idx] for idx, lab in enumerate(clu.labels_) if lab == label]
        members.sort(key=lambda pt: (pt[2], -pt[4]))
        merged.append(members[0])
    return merged


def build_collision_csv_rows(
    sample_idx,
    sample_meta,
    stitched_image_path,
    overlay_image_path,
    depth_overlay_image_path,
    collisions,
    pred_data,
    stitched_img_width,
    proposal_source="rule_based",
    collision_details=None,
):
    rows = []
    detection_count = len(collisions)
    collision_details = collision_details or []
    if detection_count == 0:
        rows.append(
            {
                "sample_idx": sample_idx,
                "scene_indice": sample_meta["scene_indice"],
                "bag_name": sample_meta["bag_name"],
                "ros_msg_seq_prev": sample_meta["ros_msg_seq_prev"],
                "ros_msg_seq_curr": sample_meta["ros_msg_seq_curr"],
                "time_diff_cam_us": sample_meta["time_diff_cam_us"],
                "proposal_source": proposal_source,
                "selection_mode": "",
                "has_detection": 0,
                "detection_count": 0,
                "detection_rank": -1,
                "x_rv": "",
                "y_rv": "",
                "ttc_s": "",
                "scale_pred": "",
                "risk_pred": "",
                "cam_idx": "",
                "u_proc": "",
                "v_proc": "",
                "x_stitched_orig": "",
                "y_stitched_orig": "",
                "mapping_valid": "",
                "det_camera_channel": "",
                "det_label": "",
                "det_conf": "",
                "det_x1": "",
                "det_y1": "",
                "det_x2": "",
                "det_y2": "",
                "matched_point_count": "",
                "stitched_image_path": stitched_image_path or "",
                "overlay_image_path": overlay_image_path,
                "depth_overlay_image_path": depth_overlay_image_path or "",
            }
        )
        return rows

    for detection_rank, pt in enumerate(collisions):
        x_rv, y_rv, ttc_s, scale_pred, risk_pred = pt
        detail = collision_details[detection_rank] if detection_rank < len(collision_details) else {}
        if detail.get("mapping_valid", None) is not None:
            mapping = {
                "cam_idx": detail.get("cam_idx", ""),
                "u_proc": detail.get("u_proc", ""),
                "v_proc": detail.get("v_proc", ""),
                "x_stitched_orig": detail.get("x_stitched_orig", ""),
                "y_stitched_orig": detail.get("y_stitched_orig", ""),
                "mapping_valid": detail.get("mapping_valid", ""),
            }
        else:
            mapping = map_rv_point_to_stitched_orig(
                x_rv,
                y_rv,
                pred_data,
                stitched_img_width=stitched_img_width,
            )
        rows.append(
            {
                "sample_idx": sample_idx,
                "scene_indice": sample_meta["scene_indice"],
                "bag_name": sample_meta["bag_name"],
                "ros_msg_seq_prev": sample_meta["ros_msg_seq_prev"],
                "ros_msg_seq_curr": sample_meta["ros_msg_seq_curr"],
                "time_diff_cam_us": sample_meta["time_diff_cam_us"],
                "proposal_source": detail.get("proposal_source", proposal_source),
                "selection_mode": detail.get("selection_mode", ""),
                "has_detection": 1,
                "detection_count": detection_count,
                "detection_rank": detection_rank,
                "x_rv": x_rv,
                "y_rv": y_rv,
                "ttc_s": ttc_s,
                "scale_pred": scale_pred,
                "risk_pred": risk_pred,
                "cam_idx": mapping["cam_idx"],
                "u_proc": mapping["u_proc"],
                "v_proc": mapping["v_proc"],
                "x_stitched_orig": mapping["x_stitched_orig"],
                "y_stitched_orig": mapping["y_stitched_orig"],
                "mapping_valid": mapping["mapping_valid"],
                "det_camera_channel": detail.get("det_camera_channel", ""),
                "det_label": detail.get("det_label", ""),
                "det_conf": detail.get("det_conf", ""),
                "det_x1": detail.get("det_x1", ""),
                "det_y1": detail.get("det_y1", ""),
                "det_x2": detail.get("det_x2", ""),
                "det_y2": detail.get("det_y2", ""),
                "matched_point_count": detail.get("matched_point_count", ""),
                "stitched_image_path": stitched_image_path or "",
                "overlay_image_path": overlay_image_path,
                "depth_overlay_image_path": depth_overlay_image_path or "",
            }
        )
    return rows


def save_collision_predictions_csv(csv_rows, csv_path):
    fieldnames = [
        "sample_idx",
        "scene_indice",
        "bag_name",
        "ros_msg_seq_prev",
        "ros_msg_seq_curr",
        "time_diff_cam_us",
        "proposal_source",
        "selection_mode",
        "has_detection",
        "detection_count",
        "detection_rank",
        "x_rv",
        "y_rv",
        "ttc_s",
        "scale_pred",
        "risk_pred",
        "cam_idx",
        "u_proc",
        "v_proc",
        "x_stitched_orig",
        "y_stitched_orig",
        "mapping_valid",
        "det_camera_channel",
        "det_label",
        "det_conf",
        "det_x1",
        "det_y1",
        "det_x2",
        "det_y2",
        "matched_point_count",
        "stitched_image_path",
        "overlay_image_path",
        "depth_overlay_image_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def save_rgb_image(path, rgb_image):
    rgb_u8 = np.clip(np.asarray(rgb_image, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(rgb_u8).save(path)


def infer_processed_hw_from_proj(proj_pix_curr, fallback_hw):
    proj_pix_curr = np.asarray(proj_pix_curr)
    if proj_pix_curr.ndim != 3 or proj_pix_curr.shape[2] != 3:
        return fallback_hw
    valid = proj_pix_curr[..., 0] >= 0
    if not np.any(valid):
        return fallback_hw
    u_vals = proj_pix_curr[..., 1][valid]
    v_vals = proj_pix_curr[..., 2][valid]
    return int(v_vals.max()) + 1, int(u_vals.max()) + 1


def process_rgb_with_affine_matrix(image_path, affine_matrix, output_hw):
    crop_h, crop_w = output_hw
    affine_matrix = np.asarray(affine_matrix, dtype=np.float32)
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    is_simple_affine = (
        affine_matrix.shape == (3, 3)
        and np.allclose(affine_matrix[2], np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)
        and np.isclose(float(affine_matrix[0, 1]), 0.0, atol=1e-6)
        and np.isclose(float(affine_matrix[1, 0]), 0.0, atol=1e-6)
        and float(affine_matrix[0, 0]) > 0.0
        and np.isclose(float(affine_matrix[0, 0]), float(affine_matrix[1, 1]), atol=1e-6)
    )

    if is_simple_affine:
        scale = float(affine_matrix[0, 0])
        crop_x = int(round(-float(affine_matrix[0, 2])))
        crop_y = int(round(-float(affine_matrix[1, 2])))
        resize_w = int(round(orig_w * scale))
        resize_h = int(round(orig_h * scale))
        resized = pil_img.resize((resize_w, resize_h), Image.BILINEAR)
        cropped = resized.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        return np.asarray(cropped, dtype=np.uint8)

    rgb = np.asarray(pil_img, dtype=np.uint8)
    inv_affine = np.linalg.inv(affine_matrix).astype(np.float32)
    u_grid, v_grid = np.meshgrid(
        np.arange(crop_w, dtype=np.float32),
        np.arange(crop_h, dtype=np.float32),
    )
    ones = np.ones_like(u_grid, dtype=np.float32)
    pix_proc = np.stack([u_grid, v_grid, ones], axis=-1).reshape(-1, 3).T
    pix_orig = inv_affine @ pix_proc
    pix_orig /= np.clip(pix_orig[2:3, :], 1e-6, None)
    map_x = pix_orig[0].reshape(crop_h, crop_w).astype(np.float32)
    map_y = pix_orig[1].reshape(crop_h, crop_w).astype(np.float32)
    remapped = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return remapped.astype(np.uint8)


def load_processed_curr_camera_rgbs(sample_info, pred_data, fallback_hw):
    affine_matrix = pred_data.get("affine_matrix")
    proj_pix_curr = pred_data.get("proj_pix_curr")
    if affine_matrix is None or proj_pix_curr is None:
        raise KeyError("pred_data must contain affine_matrix and proj_pix_curr for reprojection diagnostics")

    output_hw = infer_processed_hw_from_proj(proj_pix_curr, fallback_hw)
    processed = {}
    for ch in CAMERA_CHANNELS:
        image_path = sample_info["curr_camera_data"][ch]["filename"]
        processed[ch] = process_rgb_with_affine_matrix(image_path, affine_matrix, output_hw)
    return processed, output_hw


def load_raw_curr_camera_rgbs(sample_info):
    raw_rgb = {}
    for ch in CAMERA_CHANNELS:
        image_path = sample_info["curr_camera_data"][ch]["filename"]
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        raw_rgb[ch] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return raw_rgb


def get_sam2_cache_path(sam2_cache_dir, sample_idx):
    if not sam2_cache_dir:
        raise ValueError("--sam2_cache_dir is required when --proposal_source sam2_rgb_fusion is used")
    cache_path = Path(sam2_cache_dir) / f"pred_{sample_idx}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"SAM2 cache not found for sample_idx={sample_idx}: {cache_path}")
    return cache_path


def load_sam2_sample_cache(sam2_cache_dir, sample_idx):
    cache_path = get_sam2_cache_path(sam2_cache_dir, sample_idx)
    return np.load(str(cache_path), allow_pickle=True)


def sam2_cache_has_camera(sample_cache, camera_channel):
    prefix = f"{camera_channel}__"
    return f"{prefix}mask_count" in sample_cache.files


def decode_sam2_camera_cache(sample_cache, camera_channel):
    prefix = f"{camera_channel}__"
    if not sam2_cache_has_camera(sample_cache, camera_channel):
        return {
            "mask_count": 0,
            "mask_shape": (0, 0),
            "masks": np.zeros((0, 0, 0), dtype=bool),
            "bbox_xyxy": np.zeros((0, 4), dtype=np.int32),
            "area": np.zeros((0,), dtype=np.int32),
            "predicted_iou": np.zeros((0,), dtype=np.float32),
            "stability_score": np.zeros((0,), dtype=np.float32),
        }
    mask_count = int(np.asarray(sample_cache[f"{prefix}mask_count"]).reshape(-1)[0])
    mask_shape = tuple(int(v) for v in np.asarray(sample_cache[f"{prefix}mask_shape"]).tolist())
    mask_bits = np.asarray(sample_cache[f"{prefix}mask_bits"], dtype=np.uint8)
    bbox_xyxy = np.asarray(sample_cache[f"{prefix}bbox_xyxy"], dtype=np.int32)
    area = np.asarray(sample_cache[f"{prefix}area"], dtype=np.int32)
    predicted_iou = np.asarray(sample_cache[f"{prefix}predicted_iou"], dtype=np.float32)
    stability_score = np.asarray(sample_cache[f"{prefix}stability_score"], dtype=np.float32)

    if mask_count == 0:
        masks = np.zeros((0, mask_shape[0], mask_shape[1]), dtype=bool)
    else:
        flat_bits = np.unpackbits(mask_bits, axis=1)[:, : mask_shape[0] * mask_shape[1]]
        masks = flat_bits.reshape(mask_count, mask_shape[0], mask_shape[1]).astype(bool)

    return {
        "mask_count": mask_count,
        "mask_shape": mask_shape,
        "masks": masks,
        "bbox_xyxy": bbox_xyxy,
        "area": area,
        "predicted_iou": predicted_iou,
        "stability_score": stability_score,
    }


def build_raw_camera_mosaic(raw_curr_rgb, tile_hw=(160, 320)):
    tiles = []
    tile_h, tile_w = tile_hw
    for ch in CAMERA_CHANNELS:
        rgb = np.asarray(raw_curr_rgb[ch], dtype=np.uint8)
        tile = cv2.resize(rgb, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def compute_support_points_by_camera(scale_map, orientation_map, pred_data, delta_t, risk_time_threshold, orientation_threshold):
    ttc_map = compute_ttc_map(scale_map, delta_t)
    support_mask = build_orientation_support_mask(ttc_map, orientation_map, risk_time_threshold, orientation_threshold).astype(bool)
    proj_pix_curr = get_fullres_proj_pix_curr(pred_data)
    affine_matrix = pred_data.get("affine_matrix")
    if affine_matrix is None:
        raise KeyError("pred_data must contain affine_matrix for SAM2 RGB fusion")
    affine_matrix = np.asarray(affine_matrix, dtype=np.float32)
    inv_affine = np.linalg.inv(affine_matrix).astype(np.float32)

    valid_proj = support_mask & (proj_pix_curr[..., 0] >= 0)
    ys, xs = np.where(valid_proj)
    if xs.size == 0:
        return ttc_map, support_mask, {ch: [] for ch in CAMERA_CHANNELS}

    sampled = proj_pix_curr[ys, xs]
    cam_idx = sampled[:, 0].astype(np.int32)
    u_proc = sampled[:, 1].astype(np.float32)
    v_proc = sampled[:, 2].astype(np.float32)
    pix_proc = np.stack([u_proc, v_proc, np.ones_like(u_proc, dtype=np.float32)], axis=1).T
    pix_orig = inv_affine @ pix_proc
    pix_orig /= np.clip(pix_orig[2:3, :], 1e-6, None)
    u_orig = pix_orig[0]
    v_orig = pix_orig[1]

    support_by_camera = {ch: [] for ch in CAMERA_CHANNELS}
    for idx, ch in enumerate(CAMERA_CHANNELS):
        cam_mask = cam_idx == idx
        if not np.any(cam_mask):
            continue
        cam_entries = []
        cam_xs = xs[cam_mask]
        cam_ys = ys[cam_mask]
        cam_u_orig = u_orig[cam_mask]
        cam_v_orig = v_orig[cam_mask]
        cam_ttc = ttc_map[cam_ys, cam_xs]
        cam_scale = scale_map[cam_ys, cam_xs]
        cam_orientation = orientation_map[cam_ys, cam_xs]
        for j in range(cam_xs.shape[0]):
            cam_entries.append(
                {
                    "x_rv": int(cam_xs[j]),
                    "y_rv": int(cam_ys[j]),
                    "u_orig": float(cam_u_orig[j]),
                    "v_orig": float(cam_v_orig[j]),
                    "ttc": float(cam_ttc[j]),
                    "scale": float(cam_scale[j]),
                    "orientation": float(cam_orientation[j]),
                }
            )
        support_by_camera[ch] = cam_entries
    return ttc_map, support_mask.astype(np.uint8), support_by_camera


def build_mask_support_regions(sam2_sample_cache, support_by_camera):
    proposal_regions = []
    for ch in CAMERA_CHANNELS:
        camera_cache = decode_sam2_camera_cache(sam2_sample_cache, ch)
        mask_count = camera_cache["mask_count"]
        if mask_count == 0:
            continue

        masks = camera_cache["masks"]
        areas = camera_cache["area"]
        predicted_iou = camera_cache["predicted_iou"]
        stability_score = camera_cache["stability_score"]
        bbox_xyxy = camera_cache["bbox_xyxy"]
        h, w = camera_cache["mask_shape"]
        assignments = defaultdict(list)

        for entry in support_by_camera.get(ch, []):
            u = int(np.clip(round(entry["u_orig"]), 0, w - 1))
            v = int(np.clip(round(entry["v_orig"]), 0, h - 1))
            hit_ids = np.where(masks[:, v, u])[0]
            if hit_ids.size == 0:
                continue
            if hit_ids.size > 1:
                order = np.lexsort((-predicted_iou[hit_ids], areas[hit_ids]))
                chosen = int(hit_ids[order[0]])
            else:
                chosen = int(hit_ids[0])
            enriched = dict(entry)
            enriched["u_idx"] = u
            enriched["v_idx"] = v
            assignments[chosen].append(enriched)

        for mask_id in range(mask_count):
            assigned = assignments.get(mask_id, [])
            bbox = bbox_xyxy[mask_id].tolist()
            region = {
                "component_id": len(proposal_regions),
                "camera_channel": ch,
                "mask_id": int(mask_id),
                "mask_shape": (h, w),
                "mask_area": int(areas[mask_id]),
                "predicted_iou": float(predicted_iou[mask_id]),
                "stability_score": float(stability_score[mask_id]),
                "bbox_xyxy": [int(v) for v in bbox],
                "mask": masks[mask_id],
                "support_points": assigned,
                "support_count": len(assigned),
                "support_fraction": float(len(assigned)) / float(max(int(areas[mask_id]), 1)),
                "ttc_p10": float("inf"),
                "orientation_p20": float("inf"),
                "support_cc_max_area": 0,
                "support_cc_max_fraction": 0.0,
                "passed_support": False,
                "anchor_boundary_dist": 0.0,
            }
            if assigned:
                ttc_vals = np.array([p["ttc"] for p in assigned], dtype=np.float32)
                ori_vals = np.array([p["orientation"] for p in assigned], dtype=np.float32)
                region["ttc_p10"] = float(np.percentile(ttc_vals, 10))
                region["orientation_p20"] = float(np.percentile(ori_vals, 20))

                local_support = np.zeros((h, w), dtype=np.uint8)
                for p in assigned:
                    u_idx = int(np.clip(p["u_idx"], 0, w - 1))
                    v_idx = int(np.clip(p["v_idx"], 0, h - 1))
                    local_support[v_idx, u_idx] = 1
                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(local_support, connectivity=8)
                if num_labels > 1:
                    region["support_cc_max_area"] = int(stats[1:, cv2.CC_STAT_AREA].max())
                region["support_cc_max_fraction"] = float(region["support_cc_max_area"]) / float(max(region["support_count"], 1))
                region["passed_support"] = (
                    region["support_count"] >= 25
                    and region["support_fraction"] >= 0.01
                    and region["support_cc_max_area"] >= 12
                    and region["support_cc_max_fraction"] >= 0.35
                )
            proposal_regions.append(region)
    return proposal_regions


def binary_mask_iou(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    inter = int(np.count_nonzero(mask_a & mask_b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(mask_a | mask_b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def binary_mask_min_coverage(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    inter = int(np.count_nonzero(mask_a & mask_b))
    if inter == 0:
        return 0.0
    area_a = int(np.count_nonzero(mask_a))
    area_b = int(np.count_nonzero(mask_b))
    denom = min(area_a, area_b)
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


def compute_projected_rv_region_shape_stats(region, rv_width, valid_y_min, valid_y_max):
    bbox = region.get("bbox_valid")
    coords = np.asarray(region.get("coords_valid"), dtype=np.int32)
    if bbox is None or coords.shape[0] == 0:
        region["bbox_width"] = 0
        region["bbox_height"] = 0
        region["bbox_area_valid"] = 0
        region["fill_ratio"] = 0.0
        region["aspect_ratio"] = 0.0
        region["bottom_gap"] = float("inf")
        region["horizontal_span_ratio"] = 0.0
        region["vertical_span_ratio"] = 0.0
        region["centroid_y_valid"] = float("inf")
        return region

    bbox_width, bbox_height, bbox_area = bbox_dims(bbox)
    valid_band_h = max(int(valid_y_max - valid_y_min), 1)
    region["bbox_width"] = bbox_width
    region["bbox_height"] = bbox_height
    region["bbox_area_valid"] = bbox_area
    region["fill_ratio"] = float(region["area_valid"]) / float(max(bbox_area, 1))
    region["aspect_ratio"] = float(bbox_width) / float(max(bbox_height, 1))
    region["bottom_gap"] = float((valid_y_max - 1) - bbox[3])
    region["horizontal_span_ratio"] = float(bbox_width) / float(max(int(rv_width), 1))
    region["vertical_span_ratio"] = float(bbox_height) / float(valid_band_h)
    region["centroid_y_valid"] = float(coords[:, 1].mean())
    return region


def apply_projected_rv_ground_like_filter(region):
    region["quality_passed"] = True
    region["quality_drop_reason"] = ""
    region["ground_like_passed"] = True

    bbox = region.get("bbox_valid")
    if bbox is None:
        return region

    is_ground_like = (
        float(region.get("bottom_gap", float("inf"))) <= 2.0
        and float(region.get("aspect_ratio", 0.0)) >= 3.2
        and float(region.get("horizontal_span_ratio", 0.0)) >= 0.09
        and float(region.get("bbox_width", 0.0)) >= 160.0
        and float(region.get("vertical_span_ratio", float("inf"))) <= 0.80
        and float(region.get("fill_ratio", 0.0)) >= 0.18
    )
    if is_ground_like:
        region["quality_passed"] = False
        region["ground_like_passed"] = False
        region["quality_drop_reason"] = "drop_ground_like"
    return region


def apply_projected_rv_small_box_filter(region, min_bbox_width=30, min_bbox_height=30):
    bbox = region.get("bbox_valid")
    if bbox is None:
        return region
    if (
        int(region.get("bbox_width", 0)) < int(min_bbox_width)
        and int(region.get("bbox_height", 0)) < int(min_bbox_height)
    ):
        region["quality_passed"] = False
        region["ground_like_passed"] = False
        region["quality_drop_reason"] = "drop_small_box"
    return region


def apply_projected_rv_high_orientation_filter(
    region,
    orientation_map,
    orientation_threshold,
    min_high_orientation_count=40,
    min_high_orientation_fraction=0.30,
):
    coords = np.asarray(region.get("coords_valid", []), dtype=np.int32)
    if coords.shape[0] == 0:
        region["high_orientation_count"] = 0
        region["high_orientation_fraction"] = 0.0
        return region

    xs = coords[:, 0]
    ys = coords[:, 1]
    orientation_vals = np.asarray(orientation_map[ys, xs], dtype=np.float32)
    high_mask = orientation_vals > float(orientation_threshold)
    high_count = int(np.count_nonzero(high_mask))
    high_fraction = float(high_count) / float(max(coords.shape[0], 1))
    region["high_orientation_count"] = high_count
    region["high_orientation_fraction"] = high_fraction

    if (
        high_count >= int(min_high_orientation_count)
        and high_fraction >= float(min_high_orientation_fraction)
    ):
        region["quality_passed"] = False
        region["ground_like_passed"] = False
        region["quality_drop_reason"] = "drop_high_orientation"
    return region


def build_projected_rv_mask_regions(
    projection_debug,
    ttc_map,
    orientation_map,
    valid_y_min,
    valid_y_max,
    min_valid_area=40,
    iou_merge_threshold=0.60,
    coverage_merge_threshold=0.85,
):
    camera_channel = projection_debug["camera_channel"]
    camera_cache = projection_debug["camera_cache"]
    mask_rv_stack = np.asarray(projection_debug["mask_rv_stack"], dtype=bool)
    mask_count = int(camera_cache["mask_count"])
    h_rv, w_rv = projection_debug["mask_aggregate_ids"].shape

    raw_regions = []
    for mask_id in range(mask_count):
        rv_mask = mask_rv_stack[mask_id]
        ys, xs = np.where(rv_mask)
        if xs.size == 0:
            continue
        coords = np.stack([xs, ys], axis=1).astype(np.int32)
        region = build_region_from_coords(coords, ttc_map, orientation_map, valid_y_min, valid_y_max)
        if region["area_valid"] < int(min_valid_area):
            continue

        rv_mask_valid = np.zeros((h_rv, w_rv), dtype=bool)
        if region["area_valid"] > 0:
            coords_valid = region["coords_valid"]
            rv_mask_valid[coords_valid[:, 1], coords_valid[:, 0]] = True

        region.update(
            {
                "component_id": len(raw_regions),
                "camera_channel": camera_channel,
                "mask_id": int(mask_id),
                "rv_mask": rv_mask.copy(),
                "rv_mask_valid": rv_mask_valid,
                "rv_mask_area_valid": int(region["area_valid"]),
                "source_mask_ids": [int(mask_id)],
                "source_predicted_iou_max": float(camera_cache["predicted_iou"][mask_id]),
                "source_stability_score_max": float(camera_cache["stability_score"][mask_id]),
                "source_mask_area_min": int(camera_cache["area"][mask_id]),
                "source_mask_count": 1,
            }
        )
        compute_projected_rv_region_shape_stats(region, w_rv, valid_y_min, valid_y_max)
        raw_regions.append(region)

    if not raw_regions:
        return [], []

    parent = list(range(len(raw_regions)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(raw_regions)):
        for j in range(i + 1, len(raw_regions)):
            region_a = raw_regions[i]
            region_b = raw_regions[j]
            iou = binary_mask_iou(region_a["rv_mask_valid"], region_b["rv_mask_valid"])
            coverage = binary_mask_min_coverage(region_a["rv_mask_valid"], region_b["rv_mask_valid"])
            if iou >= float(iou_merge_threshold) or coverage >= float(coverage_merge_threshold):
                union(i, j)

    grouped = {}
    for idx, region in enumerate(raw_regions):
        grouped.setdefault(find(idx), []).append(region)

    merged_regions = []
    for merged_id, members in enumerate(grouped.values()):
        merged_mask = np.zeros((h_rv, w_rv), dtype=bool)
        merged_xy = []
        source_mask_ids = []
        source_predicted_iou_vals = []
        source_stability_vals = []
        source_area_vals = []
        for member in members:
            merged_mask |= member["rv_mask"]
            merged_xy.append(member["coords_all"])
            source_mask_ids.extend(member["source_mask_ids"])
            source_predicted_iou_vals.append(member["source_predicted_iou_max"])
            source_stability_vals.append(member["source_stability_score_max"])
            source_area_vals.append(member["source_mask_area_min"])
        merged_xy = np.unique(np.vstack(merged_xy), axis=0)
        merged_region = build_region_from_coords(merged_xy, ttc_map, orientation_map, valid_y_min, valid_y_max)
        if merged_region["area_valid"] < int(min_valid_area):
            continue

        rv_mask_valid = np.zeros((h_rv, w_rv), dtype=bool)
        coords_valid = merged_region["coords_valid"]
        if coords_valid.shape[0] > 0:
            rv_mask_valid[coords_valid[:, 1], coords_valid[:, 0]] = True

        merged_region.update(
            {
                "component_id": merged_id,
                "camera_channel": camera_channel,
                "mask_id": -1,
                "rv_mask": merged_mask,
                "rv_mask_valid": rv_mask_valid,
                "rv_mask_area_valid": int(merged_region["area_valid"]),
                "source_mask_ids": sorted(set(int(v) for v in source_mask_ids)),
                "source_predicted_iou_max": float(max(source_predicted_iou_vals)) if source_predicted_iou_vals else 0.0,
                "source_stability_score_max": float(max(source_stability_vals)) if source_stability_vals else 0.0,
                "source_mask_area_min": int(min(source_area_vals)) if source_area_vals else 0,
                "source_mask_count": len(set(int(v) for v in source_mask_ids)),
            }
        )
        compute_projected_rv_region_shape_stats(merged_region, w_rv, valid_y_min, valid_y_max)
        merged_regions.append(merged_region)

    return raw_regions, merged_regions


def enrich_projected_rv_region_support(region, support_mask, ttc_map, orientation_map):
    coords = np.asarray(region["coords_valid"], dtype=np.int32)
    if coords.shape[0] == 0:
        region["support_coords"] = np.empty((0, 2), dtype=np.int32)
        region["support_count"] = 0
        region["support_fraction"] = 0.0
        region["ttc_p10"] = float("inf")
        region["orientation_p20"] = float("inf")
        region["support_cc_max_area"] = 0
        region["support_cc_max_fraction"] = 0.0
        region["passed_support"] = False
        region["support_drop_reason"] = "drop_no_valid_mask_area"
        return region

    xs = coords[:, 0]
    ys = coords[:, 1]
    support_keep = np.asarray(support_mask[ys, xs] > 0, dtype=bool)
    support_coords = coords[support_keep]
    region["support_coords"] = support_coords
    region["support_count"] = int(support_coords.shape[0])
    region["support_fraction"] = float(region["support_count"]) / float(max(region["rv_mask_area_valid"], 1))
    region["ttc_p10"] = float("inf")
    region["orientation_p20"] = float("inf")
    region["support_cc_max_area"] = 0
    region["support_cc_max_fraction"] = 0.0
    region["passed_support"] = False
    region["support_drop_reason"] = ""

    if support_coords.shape[0] == 0:
        region["support_drop_reason"] = "drop_no_support"
        return region

    support_xs = support_coords[:, 0]
    support_ys = support_coords[:, 1]
    support_ttc = ttc_map[support_ys, support_xs]
    support_orientation = orientation_map[support_ys, support_xs]
    region["ttc_p10"] = float(np.percentile(support_ttc, 10))
    region["orientation_p20"] = float(np.percentile(support_orientation, 20))

    bbox = region["bbox_valid"]
    if bbox is None:
        region["support_drop_reason"] = "drop_no_valid_bbox"
        return region
    x1, y1, x2, y2 = bbox
    local_support = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    local_support[support_ys - y1, support_xs - x1] = 1
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(local_support, connectivity=8)
    if num_labels > 1:
        region["support_cc_max_area"] = int(stats[1:, cv2.CC_STAT_AREA].max())
    region["support_cc_max_fraction"] = float(region["support_cc_max_area"]) / float(max(region["support_count"], 1))

    region["passed_support"] = (
        region["support_count"] >= 25
        and region["support_fraction"] >= 0.01
        and region["support_cc_max_area"] >= 12
        and region["support_cc_max_fraction"] >= 0.35
    )
    if not region["passed_support"]:
        region["support_drop_reason"] = "drop_sparse_support"
    return region


def select_sam2_region_anchor(region):
    support_points = region["support_points"]
    if not support_points:
        return None, 0.0

    support_ttc = np.array([p["ttc"] for p in support_points], dtype=np.float32)
    support_orientation = np.array([p["orientation"] for p in support_points], dtype=np.float32)
    support_u = np.array([p["u_idx"] for p in support_points], dtype=np.int32)
    support_v = np.array([p["v_idx"] for p in support_points], dtype=np.int32)
    ttc_thr = np.percentile(support_ttc, 25)
    orientation_thr = np.percentile(support_orientation, 35)
    core_mask = (support_ttc <= ttc_thr) & (support_orientation <= orientation_thr)
    core_indices = np.where(core_mask)[0]
    if core_indices.size < 10:
        order = np.lexsort((support_orientation, support_ttc))
        core_indices = order[: min(30, len(support_points))]
    if core_indices.size == 0:
        return None, 0.0

    mask = np.asarray(region["mask"], dtype=np.uint8) * 255
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    best_idx = None
    best_key = None
    best_dist = 0.0
    for idx in core_indices:
        u = int(support_u[idx])
        v = int(support_v[idx])
        dist_to_boundary = float(dist_map[v, u])
        key = (-dist_to_boundary, float(support_ttc[idx]), float(support_orientation[idx]))
        if best_key is None or key < best_key:
            best_key = key
            best_idx = int(idx)
            best_dist = dist_to_boundary

    if best_idx is None:
        return None, 0.0
    p = support_points[best_idx]
    anchor = [int(p["x_rv"]), int(p["y_rv"]), float(p["ttc"]), float(p["scale"]), float(p["orientation"])]
    return anchor, best_dist


def detect_sam2_rgb_collision_regions(sample_idx, sample_info, pred_data, args, scale_map, orientation_map, delta_t):
    sam2_sample_cache = load_sam2_sample_cache(args.sam2_cache_dir, sample_idx)
    ttc_map = compute_ttc_map(scale_map, delta_t)
    support_mask = build_orientation_support_mask(
        ttc_map,
        orientation_map,
        args.risk_time_threshold,
        args.risk_pred_threshold,
    ).astype(np.uint8)
    valid_y_min, valid_y_max = get_vertical_valid_band(scale_map.shape[0])
    projection_debug = project_camera_masks_to_rv(pred_data, sam2_sample_cache, args.sam2_camera_channel)
    proposal_regions, merged_regions = build_projected_rv_mask_regions(
        projection_debug=projection_debug,
        ttc_map=ttc_map,
        orientation_map=orientation_map,
        valid_y_min=valid_y_min,
        valid_y_max=valid_y_max,
        min_valid_area=40,
        iou_merge_threshold=0.60,
        coverage_merge_threshold=0.85,
    )
    ground_like_drop_count = 0
    small_box_drop_count = 0
    high_orientation_drop_count = 0
    support_input_regions = []
    for region in merged_regions:
        apply_projected_rv_small_box_filter(region)
        if region.get("quality_drop_reason") == "drop_small_box":
            small_box_drop_count += 1
            continue
        apply_projected_rv_ground_like_filter(region)
        if not region["ground_like_passed"]:
            ground_like_drop_count += 1
            continue
        apply_projected_rv_high_orientation_filter(
            region,
            orientation_map=orientation_map,
            orientation_threshold=args.risk_pred_threshold,
        )
        if region.get("quality_drop_reason") == "drop_high_orientation":
            high_orientation_drop_count += 1
            continue
        support_input_regions.append(region)

    passed_regions = []
    for region in support_input_regions:
        enrich_projected_rv_region_support(region, support_mask, ttc_map, orientation_map)
        if region["passed_support"]:
            passed_regions.append(region)

    ranked = []
    collision_details = []
    for region in passed_regions:
        support_anchor, support_anchor_boundary_dist = select_scale_region_anchor(scale_map, orientation_map, ttc_map, region)
        if support_anchor is None:
            continue
        display_anchor, display_anchor_boundary_dist = select_region_center_anchor(scale_map, orientation_map, ttc_map, region)
        if display_anchor is None:
            continue
        region["anchor_boundary_dist"] = display_anchor_boundary_dist
        region["support_anchor_boundary_dist"] = support_anchor_boundary_dist
        region["support_anchor"] = support_anchor
        region["display_anchor"] = display_anchor
        ranked.append(
            {
                "region": region,
                "anchor": display_anchor,
                "ttc_p10": region["ttc_p10"],
                "orientation_p20": region["orientation_p20"],
                "support_count": region["support_count"],
                "support_fraction": region["support_fraction"],
            }
        )
    ranked.sort(
        key=lambda item: (
            item["ttc_p10"],
            item["orientation_p20"],
            -item["support_count"],
            -item["support_fraction"],
        )
    )
    collisions = []
    for item in ranked[:1]:
        collisions.append(item["anchor"])
        region = item["region"]
        collision_details.append(
            {
                "proposal_source": "sam2_rgb_fusion",
                "selection_mode": "sam2_projected_rv_mask",
                "det_camera_channel": region["camera_channel"],
                "det_label": "sam2_projected_rv_mask",
                "det_conf": region["source_predicted_iou_max"],
                "matched_point_count": region["support_count"],
                "det_x1": "",
                "det_y1": "",
                "det_x2": "",
                "det_y2": "",
                "region_id": region["component_id"],
            }
        )

    debug = {
        "ttc_map": ttc_map,
        "raw_mask": support_mask,
        "proposal_regions": proposal_regions,
        "merged_regions": merged_regions,
        "passed_regions": passed_regions,
        "selected_region_id": ranked[0]["region"]["component_id"] if ranked else None,
        "valid_y_min": valid_y_min,
        "valid_y_max": valid_y_max,
        "sam2_sample_cache": sam2_sample_cache,
        "projection_debug": projection_debug,
        "ground_like_drop_count": ground_like_drop_count,
        "small_box_drop_count": small_box_drop_count,
        "high_orientation_drop_count": high_orientation_drop_count,
    }
    return collisions, debug, collision_details


def render_mask_overlay(rgb_image, mask, color, alpha=0.45):
    rgb = np.asarray(rgb_image, dtype=np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    rgb[mask] = rgb[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def render_sam2_mosaic_panel(raw_curr_rgb, sam2_sample_cache, highlight_mask_key=None, support_points_by_camera=None, collisions=None, tile_hw=(160, 320)):
    tiles = []
    support_points_by_camera = support_points_by_camera or {}
    collisions = collisions or []
    for ch in CAMERA_CHANNELS:
        tile = np.asarray(raw_curr_rgb[ch], dtype=np.uint8).copy()
        camera_cache = decode_sam2_camera_cache(sam2_sample_cache, ch)
        for mask_id in range(camera_cache["mask_count"]):
            color = color_for_index(mask_id)
            alpha = 0.25
            if highlight_mask_key == (ch, mask_id):
                alpha = 0.45
            tile = render_mask_overlay(tile, camera_cache["masks"][mask_id], color, alpha=alpha)
            x1, y1, x2, y2 = camera_cache["bbox_xyxy"][mask_id].tolist()
            cv2.rectangle(tile, (x1, y1), (x2, y2), color, 3 if highlight_mask_key == (ch, mask_id) else 1)
        for p in support_points_by_camera.get(ch, []):
            cv2.circle(tile, (int(p["u_idx"]), int(p["v_idx"])), 3, (255, 255, 0), -1)
        for p in collisions:
            if p.get("camera_channel") != ch:
                continue
            cv2.circle(tile, (int(p["u_idx"]), int(p["v_idx"])), 9, (0, 0, 0), -1)
        tile = cv2.resize(tile, (tile_hw[1], tile_hw[0]), interpolation=cv2.INTER_LINEAR)
        tile = annotate_rgb_tile(tile, ch)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def project_camera_masks_to_rv(pred_data, sam2_sample_cache, camera_channel):
    proj_pix_curr = get_fullres_proj_pix_curr(pred_data)
    affine_matrix = pred_data.get("affine_matrix")
    if affine_matrix is None:
        raise KeyError("pred_data must contain affine_matrix for SAM2 RV projection debugging")
    affine_matrix = np.asarray(affine_matrix, dtype=np.float32)
    inv_affine = np.linalg.inv(affine_matrix).astype(np.float32)

    camera_cache = decode_sam2_camera_cache(sam2_sample_cache, camera_channel)
    mask_count = int(camera_cache["mask_count"])
    h_rv, w_rv = proj_pix_curr.shape[:2]
    if mask_count == 0:
        return {
            "camera_channel": camera_channel,
            "camera_idx": CAMERA_CHANNELS.index(camera_channel),
            "mask_rv_stack": np.zeros((0, h_rv, w_rv), dtype=bool),
            "mask_aggregate_ids": np.full((h_rv, w_rv), -1, dtype=np.int32),
            "mask_aggregate_valid": np.zeros((h_rv, w_rv), dtype=bool),
            "camera_cache": camera_cache,
        }

    cam_idx = CAMERA_CHANNELS.index(camera_channel)
    rv_cam_mask = proj_pix_curr[..., 0].astype(np.int32) == cam_idx
    ys, xs = np.where(rv_cam_mask)
    if xs.size == 0:
        return {
            "camera_channel": camera_channel,
            "camera_idx": cam_idx,
            "mask_rv_stack": np.zeros((mask_count, h_rv, w_rv), dtype=bool),
            "mask_aggregate_ids": np.full((h_rv, w_rv), -1, dtype=np.int32),
            "mask_aggregate_valid": np.zeros((h_rv, w_rv), dtype=bool),
            "camera_cache": camera_cache,
        }

    sampled = proj_pix_curr[ys, xs]
    u_proc = sampled[:, 1].astype(np.float32)
    v_proc = sampled[:, 2].astype(np.float32)
    pix_proc = np.stack([u_proc, v_proc, np.ones_like(u_proc, dtype=np.float32)], axis=1).T
    pix_orig = inv_affine @ pix_proc
    pix_orig /= np.clip(pix_orig[2:3, :], 1e-6, None)
    u_orig = np.rint(pix_orig[0]).astype(np.int32)
    v_orig = np.rint(pix_orig[1]).astype(np.int32)

    h_img, w_img = camera_cache["mask_shape"]
    in_bounds = (
        (u_orig >= 0)
        & (u_orig < int(w_img))
        & (v_orig >= 0)
        & (v_orig < int(h_img))
    )

    mask_rv_stack = np.zeros((mask_count, h_rv, w_rv), dtype=bool)
    aggregate_ids = np.full((h_rv, w_rv), -1, dtype=np.int32)
    aggregate_valid = np.zeros((h_rv, w_rv), dtype=bool)
    if not np.any(in_bounds):
        return {
            "camera_channel": camera_channel,
            "camera_idx": cam_idx,
            "mask_rv_stack": mask_rv_stack,
            "mask_aggregate_ids": aggregate_ids,
            "mask_aggregate_valid": aggregate_valid,
            "camera_cache": camera_cache,
        }

    ys_valid = ys[in_bounds]
    xs_valid = xs[in_bounds]
    u_valid = u_orig[in_bounds]
    v_valid = v_orig[in_bounds]
    masks = camera_cache["masks"]
    hits = masks[:, v_valid, u_valid].T  # (N_valid, mask_count)

    if hits.size == 0:
        return {
            "camera_channel": camera_channel,
            "camera_idx": cam_idx,
            "mask_rv_stack": mask_rv_stack,
            "mask_aggregate_ids": aggregate_ids,
            "mask_aggregate_valid": aggregate_valid,
            "camera_cache": camera_cache,
        }

    for mask_id in range(mask_count):
        mask_hit = hits[:, mask_id]
        if np.any(mask_hit):
            mask_rv_stack[mask_id, ys_valid[mask_hit], xs_valid[mask_hit]] = True

    hit_any = hits.any(axis=1)
    if np.any(hit_any):
        hit_rows = hits[hit_any]
        hit_ys = ys_valid[hit_any]
        hit_xs = xs_valid[hit_any]
        area = camera_cache["area"]
        predicted_iou = camera_cache["predicted_iou"]
        for row_idx, hit_row in enumerate(hit_rows):
            hit_ids = np.flatnonzero(hit_row)
            if hit_ids.size == 1:
                chosen = int(hit_ids[0])
            else:
                order = np.lexsort((-predicted_iou[hit_ids], area[hit_ids]))
                chosen = int(hit_ids[order[0]])
            aggregate_ids[hit_ys[row_idx], hit_xs[row_idx]] = chosen
            aggregate_valid[hit_ys[row_idx], hit_xs[row_idx]] = True

    return {
        "camera_channel": camera_channel,
        "camera_idx": cam_idx,
        "mask_rv_stack": mask_rv_stack,
        "mask_aggregate_ids": aggregate_ids,
        "mask_aggregate_valid": aggregate_valid,
        "camera_cache": camera_cache,
    }


def render_rv_mask_overlay(base_rgb, projection_debug, alpha=0.35):
    canvas = np.asarray(base_rgb, dtype=np.uint8).copy()
    aggregate_ids = projection_debug["mask_aggregate_ids"]
    aggregate_valid = projection_debug["mask_aggregate_valid"]
    if not np.any(aggregate_valid):
        return canvas
    for mask_id in np.unique(aggregate_ids[aggregate_valid]):
        color = np.asarray(color_for_index(int(mask_id)), dtype=np.uint8)
        mask = aggregate_ids == int(mask_id)
        canvas[mask] = (canvas[mask].astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    return canvas


def render_projected_rv_masks_only(projection_debug):
    aggregate_ids = projection_debug["mask_aggregate_ids"]
    aggregate_valid = projection_debug["mask_aggregate_valid"]
    h_rv, w_rv = aggregate_ids.shape
    canvas = np.full((h_rv, w_rv, 3), 255, dtype=np.uint8)
    if not np.any(aggregate_valid):
        return canvas
    for mask_id in np.unique(aggregate_ids[aggregate_valid]):
        color = np.asarray(color_for_index(int(mask_id)), dtype=np.uint8)
        canvas[aggregate_ids == int(mask_id)] = color
    return canvas


def ensure_uint8_rgb(image):
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.copy()
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size == 0:
            return arr.astype(np.uint8)
        if float(arr.max()) <= 1.0 + 1e-6:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def fit_rgb_to_canvas(rgb_image, target_hw, bg_color=(255, 255, 255)):
    rgb = ensure_uint8_rgb(rgb_image)
    target_h, target_w = target_hw
    canvas = np.full((target_h, target_w, 3), np.asarray(bg_color, dtype=np.uint8), dtype=np.uint8)
    src_h, src_w = rgb.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return canvas

    scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas


def annotate_panel_tile(rgb_image, title):
    canvas = ensure_uint8_rgb(rgb_image)
    cv2.putText(
        canvas,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_front_sam2_rv_overlay_panel(sample_idx, sample_info, scale_map_full, raw_curr_rgb, sam2_sample_cache, projection_debug, out_path):
    camera_channel = projection_debug["camera_channel"]
    scale_panel = ensure_uint8_rgb(make_scale_analysis_rgb(scale_map_full))
    rgb_panel = np.asarray(raw_curr_rgb[camera_channel], dtype=np.uint8).copy()
    camera_cache = decode_sam2_camera_cache(sam2_sample_cache, camera_channel)
    for mask_id in range(camera_cache["mask_count"]):
        color = color_for_index(mask_id)
        rgb_panel = render_mask_overlay(rgb_panel, camera_cache["masks"][mask_id], color, alpha=0.35)
        x1, y1, x2, y2 = camera_cache["bbox_xyxy"][mask_id].tolist()
        cv2.rectangle(rgb_panel, (x1, y1), (x2, y2), color, 2)
        area = int(camera_cache["area"][mask_id])
        iou = float(camera_cache["predicted_iou"][mask_id])
        cv2.putText(
            rgb_panel,
            f"m{mask_id} a={area} iou={iou:.2f}",
            (x1, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    projected_panel = render_projected_rv_masks_only(projection_debug)
    overlay_panel = render_rv_mask_overlay(scale_panel, projection_debug, alpha=0.35)

    rv_h, rv_w = scale_panel.shape[:2]
    rgb_panel = fit_rgb_to_canvas(rgb_panel, (rv_h, rv_w), bg_color=(255, 255, 255))

    scale_panel = annotate_panel_tile(scale_panel, "Scale RV")
    rgb_panel = annotate_panel_tile(rgb_panel, f"{camera_channel} RGB Masks")
    projected_panel = annotate_panel_tile(projected_panel, "Projected RV Masks")
    overlay_panel = annotate_panel_tile(overlay_panel, "Scale + RV Mask Overlay")

    top_row = np.concatenate([scale_panel, rgb_panel], axis=1)
    bottom_row = np.concatenate([projected_panel, overlay_panel], axis=1)
    canvas = np.concatenate([top_row, bottom_row], axis=0)
    title = (
        f"sample={sample_idx} scene={sample_info.get('scene_indice', '')} "
        f"camera={camera_channel} rv_mask_ids={int(np.unique(projection_debug['mask_aggregate_ids'][projection_debug['mask_aggregate_valid']]).size)}"
    )
    cv2.putText(
        canvas,
        title,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def draw_sam2_region_boxes_on_mosaic(mosaic_rgb, raw_curr_rgb, proposal_regions, color_fn, tile_hw=(160, 320)):
    canvas = np.asarray(mosaic_rgb, dtype=np.uint8).copy()
    tile_h, tile_w = tile_hw
    for cam_idx, ch in enumerate(CAMERA_CHANNELS):
        raw_h, raw_w = raw_curr_rgb[ch].shape[:2]
        scale_x = float(tile_w) / float(raw_w)
        scale_y = float(tile_h) / float(raw_h)
        x_offset = cam_idx * tile_w
        for region in proposal_regions:
            if region["camera_channel"] != ch:
                continue
            x1, y1, x2, y2 = region["bbox_xyxy"]
            mx1 = x_offset + int(round(x1 * scale_x))
            my1 = int(round(y1 * scale_y))
            mx2 = x_offset + int(round(x2 * scale_x))
            my2 = int(round(y2 * scale_y))
            color, text, thickness = color_fn(region)
            cv2.rectangle(canvas, (mx1, my1), (mx2, my2), color, thickness)
            cv2.putText(
                canvas,
                text,
                (mx1 + 4, max(16, my1 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                lineType=cv2.LINE_AA,
            )
    return canvas


def build_range_rgb_from_proj(processed_curr_rgb, proj_pix_curr):
    proj_pix_curr = np.asarray(proj_pix_curr, dtype=np.int32)
    h_rv, w_rv = proj_pix_curr.shape[:2]
    range_rgb = np.zeros((h_rv, w_rv, 3), dtype=np.uint8)
    valid_mask = proj_pix_curr[..., 0] >= 0

    for cam_idx, ch in enumerate(CAMERA_CHANNELS):
        cam_mask = valid_mask & (proj_pix_curr[..., 0] == cam_idx)
        if not np.any(cam_mask):
            continue
        u = proj_pix_curr[..., 1][cam_mask]
        v = proj_pix_curr[..., 2][cam_mask]
        range_rgb[cam_mask] = processed_curr_rgb[ch][v, u]

    return range_rgb, valid_mask


def scatter_range_rgb_to_cameras(range_rgb, proj_pix_curr, processed_hw):
    crop_h, crop_w = processed_hw
    proj_pix_curr = np.asarray(proj_pix_curr, dtype=np.int32)
    range_rgb = np.asarray(range_rgb, dtype=np.uint8)

    reconstructed = {}
    valid_masks = {}
    hit_counts = {}

    flat_rgb = range_rgb.reshape(-1, 3)
    flat_proj = proj_pix_curr.reshape(-1, 3)

    for cam_idx, ch in enumerate(CAMERA_CHANNELS):
        cam_entries = flat_proj[:, 0] == cam_idx
        if not np.any(cam_entries):
            reconstructed[ch] = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
            valid_masks[ch] = np.zeros((crop_h, crop_w), dtype=bool)
            hit_counts[ch] = np.zeros((crop_h, crop_w), dtype=np.int32)
            continue

        cam_proj = flat_proj[cam_entries]
        cam_rgb = flat_rgb[cam_entries]
        flat_dest = cam_proj[:, 2].astype(np.int64) * crop_w + cam_proj[:, 1].astype(np.int64)

        last_src_index = np.full((crop_h * crop_w,), -1, dtype=np.int64)
        last_src_index[flat_dest] = np.arange(flat_dest.shape[0], dtype=np.int64)
        hit_count_flat = np.bincount(flat_dest, minlength=crop_h * crop_w).astype(np.int32)
        valid_flat = last_src_index >= 0

        warp_flat = np.zeros((crop_h * crop_w, 3), dtype=np.uint8)
        warp_flat[valid_flat] = cam_rgb[last_src_index[valid_flat]]

        reconstructed[ch] = warp_flat.reshape(crop_h, crop_w, 3)
        valid_masks[ch] = valid_flat.reshape(crop_h, crop_w)
        hit_counts[ch] = hit_count_flat.reshape(crop_h, crop_w)

    return reconstructed, valid_masks, hit_counts


def annotate_rgb_tile(rgb_image, text):
    tile = np.asarray(rgb_image, dtype=np.uint8).copy()
    cv2.putText(
        tile,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def make_abs_diff_heatmap_rgb(gt_rgb, warp_rgb, valid_mask, clip_value=64.0):
    gt_rgb = np.asarray(gt_rgb, dtype=np.float32)
    warp_rgb = np.asarray(warp_rgb, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    diff = np.abs(warp_rgb - gt_rgb).mean(axis=2)
    norm = np.clip(diff / max(float(clip_value), 1e-6), 0.0, 1.0)
    heat_u8 = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    heat_rgb[~valid_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return heat_rgb


def make_hit_count_debug_rgb(hit_count, valid_mask, clip_value=4):
    hit_count = np.asarray(hit_count, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    norm = np.clip(hit_count / max(float(clip_value), 1e-6), 0.0, 1.0)
    heat_u8 = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_VIRIDIS)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    heat_rgb[~valid_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dup_mask = hit_count > 1
    heat_rgb[dup_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return heat_rgb


def save_range_rgb_curr(range_rgb, vis_dir, sample_idx):
    out_path = os.path.join(vis_dir, f"range_rgb_curr_{sample_idx}.png")
    save_rgb_image(out_path, np.asarray(range_rgb, dtype=np.float32) / 255.0)
    return out_path


def save_camera_reprojection_compare(
    processed_curr_rgb,
    reconstructed_rgb,
    valid_masks,
    hit_counts,
    vis_dir,
    sample_idx,
):
    rows = []
    for row_kind in ("gt", "warp", "diff", "debug"):
        tiles = []
        for ch in CAMERA_CHANNELS:
            if row_kind == "gt":
                tile = annotate_rgb_tile(processed_curr_rgb[ch], f"GT {ch}")
            elif row_kind == "warp":
                tile = annotate_rgb_tile(reconstructed_rgb[ch], f"WARP {ch}")
            elif row_kind == "diff":
                tile = annotate_rgb_tile(
                    (make_abs_diff_heatmap_rgb(processed_curr_rgb[ch], reconstructed_rgb[ch], valid_masks[ch]) * 255.0).astype(np.uint8),
                    f"DIFF {ch}",
                )
            else:
                tile = annotate_rgb_tile(
                    (make_hit_count_debug_rgb(hit_counts[ch], valid_masks[ch]) * 255.0).astype(np.uint8),
                    f"HIT {ch}",
                )
            tiles.append(tile.astype(np.uint8))
        rows.append(np.concatenate(tiles, axis=1))

    compare_rgb = np.concatenate(rows, axis=0).astype(np.uint8)
    out_path = os.path.join(vis_dir, f"camera_reproj_compare_{sample_idx}.png")
    Image.fromarray(compare_rgb).save(out_path)
    return out_path


def build_camera_reprojection_metric_rows(
    sample_idx,
    sample_meta,
    processed_curr_rgb,
    reconstructed_rgb,
    valid_masks,
    hit_counts,
):
    rows = []
    for ch in CAMERA_CHANNELS:
        gt_rgb = np.asarray(processed_curr_rgb[ch], dtype=np.float32)
        warp_rgb = np.asarray(reconstructed_rgb[ch], dtype=np.float32)
        valid_mask = np.asarray(valid_masks[ch], dtype=bool)
        hit_count = np.asarray(hit_counts[ch], dtype=np.int32)

        total_pixels = int(valid_mask.size)
        valid_pixel_count = int(valid_mask.sum())
        coverage_ratio = float(valid_pixel_count / max(total_pixels, 1))
        duplicate_pixel_count = int((hit_count > 1).sum())
        duplicate_ratio = float(duplicate_pixel_count / max(valid_pixel_count, 1)) if valid_pixel_count > 0 else 0.0

        if valid_pixel_count > 0:
            diff_values = np.abs(warp_rgb - gt_rgb).mean(axis=2)[valid_mask]
            mean_abs_diff = float(diff_values.mean())
            median_abs_diff = float(np.median(diff_values))
            p95_abs_diff = float(np.percentile(diff_values, 95))
        else:
            mean_abs_diff = 0.0
            median_abs_diff = 0.0
            p95_abs_diff = 0.0

        rows.append(
            {
                "sample_idx": sample_idx,
                "scene_indice": sample_meta["scene_indice"],
                "ros_msg_seq_curr": sample_meta["ros_msg_seq_curr"],
                "camera_channel": ch,
                "valid_pixel_count": valid_pixel_count,
                "coverage_ratio": coverage_ratio,
                "mean_abs_diff": mean_abs_diff,
                "median_abs_diff": median_abs_diff,
                "p95_abs_diff": p95_abs_diff,
                "duplicate_pixel_count": duplicate_pixel_count,
                "duplicate_ratio": duplicate_ratio,
            }
        )
    return rows


def save_camera_reprojection_metrics_csv(metric_rows, csv_path):
    fieldnames = [
        "sample_idx",
        "scene_indice",
        "ros_msg_seq_curr",
        "camera_channel",
        "valid_pixel_count",
        "coverage_ratio",
        "mean_abs_diff",
        "median_abs_diff",
        "p95_abs_diff",
        "duplicate_pixel_count",
        "duplicate_ratio",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)


def resize_prediction_map_to_shape(map_array, target_shape, interpolation=cv2.INTER_AREA):
    map_array = np.asarray(map_array, dtype=np.float32)
    target_h, target_w = target_shape
    if map_array.shape == (target_h, target_w):
        return map_array
    resized = cv2.resize(map_array, (target_w, target_h), interpolation=interpolation)
    return resized.astype(np.float32)


def resolve_collision_prediction_maps(pred_data):
    scale_map_full = np.asarray(pred_data["scale_pred"], dtype=np.float32)
    risk_map_full = np.asarray(pred_data["risk_pred"], dtype=np.float32)
    proj_pix_curr = get_fullres_proj_pix_curr(pred_data).astype(np.int32, copy=False)
    rv_shape = proj_pix_curr.shape[:2]
    scale_map_rv = resize_prediction_map_to_shape(scale_map_full, rv_shape, interpolation=cv2.INTER_AREA)
    risk_map_rv = resize_prediction_map_to_shape(risk_map_full, rv_shape, interpolation=cv2.INTER_AREA)
    return scale_map_full, risk_map_full, scale_map_rv, risk_map_rv, rv_shape


def resolve_input_image_path(args, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    if args.sjtu_test:
        repo_candidate = (Path.cwd() / path).resolve()
        if repo_candidate.exists():
            return repo_candidate
    else:
        data_candidate = (Path(args.dataroot) / path).resolve()
        if data_candidate.exists():
            return data_candidate
    return path


def scale_collision_points_for_visualization(collisions, src_shape, dst_shape):
    if not collisions:
        return []
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    scaled = []
    for x_low, y_low, ttc, scale_pred, risk_pred in collisions:
        x_vis = (float(x_low) + 0.5) * scale_x - 0.5
        y_vis = (float(y_low) + 0.5) * scale_y - 0.5
        scaled.append([x_vis, y_vis, ttc, scale_pred, risk_pred])
    return scaled


def preprocess_scale_map(scale_map, grid_size, approach_threshold):
    """
    裁剪 + 归一化 scale_map（用于特征点检测）
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
    计算有效梯度 mask 和网格级有效 map
    """
    variance = get_ttc_var(scale_map, grid_len=grid_size)
    gy, gx, gy2, gx2 = get_second_grad(scale_map, stride=grad_step)

    mask_grad    = np.abs(gy) < (risk_rate * np.mean(np.abs(gy)))
    mask_thresh1 = scale_map[grad_step:, :] < approach_threshold
    mask_thresh2 = scale_map[1:-1, :]       < approach_threshold

    valid_grad = np.logical_and(mask_grad, mask_thresh1)
    valid_grad = np.logical_and(valid_grad, mask_thresh2)

    grid_map, _ = get_grid_ttc(scale_map, variance, grid_len=grid_size)
    grid_map = grid_map.astype(bool)

    return valid_grad, grid_map


def extract_candidate_points(scale_map, norm_scale, valid_grad_mask, grid_map, grid_size, grad_step, thresh, kp_thresh):
    """
    FAST + 网格候选点
    """
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

    for x,y in invalid:
        if dense_map[y,x]==0:
            continue
        area=7
        y0,y1 = max(y-area//2,0), min(y+area//2+1,proc_h)
        x0,x1 = max(x-area//2,0), min(x+area//2+1,proc_w)
        orig = scale_map[y0:y1, x0:x1]
        if orig.min()<thresh and orig.max()<1.04:
            candidates.append([x,y,np.min(norm_scale[y0:y1, x0:x1])])

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
    两阶段 DBSCAN 聚类并计算碰撞时间
    返回 collision_points: [x, y, ttc, scale, risk]
    """
    collisions = []
    if candidates.size==0:
        return collisions

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

    clu2 = DBSCAN(eps=100, min_samples=3).fit(sec_centers)
    regions = {}
    for idx,label in enumerate(clu2.labels_):
        regions.setdefault(label,[]).append(sec_regions[idx])

    for label, regs in regions.items():
        if label<0:
            for reg in regs:
                best_point = select_best_collision_point(
                    scale_map, risk_map, reg, delta_t, risk_time_threshold
                )
                if best_point is not None:
                    collisions.append(best_point)
        else:
            merged = np.vstack(regs)
            best_point = select_best_collision_point(
                scale_map, risk_map, merged, delta_t, risk_time_threshold
            )
            if best_point is not None:
                collisions.append(best_point)
    return collisions


def filter_by_risk(collisions, risk_pred_threshold):
    """
    根据 risk_pred_map 过滤碰撞点
    """
    return [pt for pt in collisions if pt[-1] < risk_pred_threshold]


def render_range_image_bgr(map_to_vis, collisions, kind='scale'):
    """
    生成与 test.py 一致的分析图，并用 OpenCV 叠加圆点与文本。
    """
    if kind == 'scale':
        rgb = make_scale_analysis_rgb(map_to_vis)
    elif kind in {'risk', 'orientation'}:
        rgb = make_risk_analysis_rgb(map_to_vis)
    else:
        raise ValueError("kind must be 'scale', 'risk', or 'orientation'")

    bgr = cv2.cvtColor((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    for x, y, ttc, _, risk in collisions:
        cv2.circle(bgr, (int(x), int(y)), 5, (0, 0, 0), -1)
        cv2.putText(bgr, f"t={ttc:.2f}", (int(x)+6, int(y)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"r={risk:.2f}", (int(x)+6, int(y)+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return bgr


def overlay_surround_views_sjtu(collisions, pred_data, sample_meta, stitched_image_path, args, idx, collision_details=None):
    """
    在原始尺寸 stitched curr 图上叠加碰撞点并保存。
    即使 collisions 为空，也会输出一张未标注的 stitched 图，用于后续人工核查。
    """
    overlay_path = os.path.join(args.vis_dir, f"surround_sjtu_curr_{idx}.jpg")
    if not stitched_image_path:
        raise FileNotFoundError(
            f"Failed to resolve stitched image for scene={sample_meta['scene_indice']} "
            f"ros_msg_seq_curr={sample_meta['ros_msg_seq_curr']}"
        )

    cv_img = cv2.imread(stitched_image_path, cv2.IMREAD_COLOR)
    if cv_img is None:
        raise FileNotFoundError(f"Failed to load stitched image: {stitched_image_path}")

    stitched_h, stitched_w = cv_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    line_h = int(36 * font_scale)
    collision_details = collision_details or []

    for det_idx, (x_rv, y_rv, ttc, _, risk) in enumerate(collisions):
        detail = collision_details[det_idx] if det_idx < len(collision_details) else {}
        if detail.get("mapping_valid", None) is not None:
            mapping = {
                "mapping_valid": detail.get("mapping_valid", 0),
                "x_stitched_orig": detail.get("x_stitched_orig", ""),
                "y_stitched_orig": detail.get("y_stitched_orig", ""),
            }
        else:
            mapping = map_rv_point_to_stitched_orig(
                x_rv,
                y_rv,
                pred_data,
                stitched_img_width=stitched_w,
            )
        if not mapping["mapping_valid"]:
            continue
        xp = int(np.clip(round(mapping["x_stitched_orig"]), 0, stitched_w - 1))
        yp = int(np.clip(round(mapping["y_stitched_orig"]), 0, stitched_h - 1))

        if detail.get("det_x1", "") != "":
            x1 = int(np.clip(round(detail["det_x1"]), 0, stitched_w - 1))
            y1 = int(np.clip(round(detail["det_y1"]), 0, stitched_h - 1))
            x2 = int(np.clip(round(detail["det_x2"]), 0, stitched_w - 1))
            y2 = int(np.clip(round(detail["det_y2"]), 0, stitched_h - 1))
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 255), 4)
            det_text = detail.get("det_label", "det")
            det_conf = detail.get("det_conf", "")
            if det_conf != "":
                det_text = f"{det_text} {float(det_conf):.2f}"
            cv2.putText(
                cv_img,
                det_text,
                (x1 + 8, max(32, y1 + 32)),
                font,
                0.9,
                (0, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        cv2.circle(cv_img, (xp, yp), 12, (255, 0, 0), -1)
        cv2.putText(
            cv_img,
            f"t={ttc:.2f}",
            (xp + 14, max(24, yp - 10)),
            font,
            font_scale,
            (255, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            cv_img,
            f"r={risk:.2f}",
            (xp + 14, max(24 + line_h, yp - 10 + line_h)),
            font,
            font_scale,
            (255, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    cv2.imwrite(overlay_path, cv_img)
    return overlay_path, stitched_w


def overlay_stitched_depth_sjtu(collisions, pred_data, depth_stitched_path, args, idx, collision_details=None):
    """
    在 stitched curr depth 可视化图上叠加碰撞点并保存。
    """
    if not depth_stitched_path:
        return None

    depth_stitched_path = str(depth_stitched_path)
    cv_img = cv2.imread(depth_stitched_path, cv2.IMREAD_COLOR)
    if cv_img is None:
        return None

    overlay_path = os.path.join(args.vis_dir, f"surround_sjtu_curr_depth_{idx}.jpg")
    stitched_h, stitched_w = cv_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    line_h = int(36 * font_scale)
    collision_details = collision_details or []

    for det_idx, (x_rv, y_rv, ttc, _, risk) in enumerate(collisions):
        mapping = map_rv_point_to_stitched_proc(
            x_rv,
            y_rv,
            pred_data,
            stitched_img_width=stitched_w,
        )
        if not mapping["mapping_valid"]:
            continue
        xp = int(np.clip(round(mapping["x_stitched_proc"]), 0, stitched_w - 1))
        yp = int(np.clip(round(mapping["y_stitched_proc"]), 0, stitched_h - 1))

        cv2.circle(cv_img, (xp, yp), 12, (255, 0, 0), -1)
        cv2.putText(
            cv_img,
            f"t={ttc:.2f}",
            (xp + 14, max(24, yp - 10)),
            font,
            font_scale,
            (255, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            cv_img,
            f"r={risk:.2f}",
            (xp + 14, max(24 + line_h, yp - 10 + line_h)),
            font,
            font_scale,
            (255, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    cv2.imwrite(overlay_path, cv_img)
    return overlay_path


def overlay_surround_views(collisions, test_info, args, idx):
    """
    在环视图上叠加碰撞点、裁剪、拼接并保存
    """
    gt = np.load(
        os.path.join(test_info['gt_map_path'], 'range_image_prev.npy'),
        allow_pickle=True
    ).item()
    depth_map = gt['depth']
    metas     = test_info['sensor_metas_prev']
    prev_cam  = test_info['prev_camera_data']

    empty_pts = np.zeros((0, 3))
    proj_all = project_lidar_to_surround_view_img(
        empty_pts, prev_cam, metas, min_dist=1.0
    )
    surround_imgs = {}
    for ch, data in proj_all.items():
        img_np = np.array(data['original_img'])
        surround_imgs[ch] = Image.fromarray(img_np)

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

                pil_img = surround_imgs[ch]
                cv_img  = np.array(pil_img)

                xp, yp = pts2d[:, 0]
                cv2.circle(cv_img, (xp, yp), 30, (255, 0, 0), -1)
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                thickness  = 5
                line_h     = int(30 * font_scale)

                cv2.putText(cv_img, f"t={ttc:.2f}", (xp + 30, yp - 30),
                            font, font_scale, (255, 0, 0), thickness, lineType=cv2.LINE_AA)
                cv2.putText(cv_img, f"r={risk:.2f}", (xp + 30, yp - 30 + line_h),
                            font, font_scale, (255, 0, 0), thickness, lineType=cv2.LINE_AA)

                surround_imgs[ch] = Image.fromarray(cv_img)

    out_path = os.path.join(args.vis_dir, f"surround_{idx}.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resize_param = [args.image_size[0], args.image_size[1]*6]
    combine_surround_views(surround_imgs, resize_param, out_path)


def combine_surround_views(surround_imgs, resize_param, out_path):
    """
    surround_imgs: dict[str, PIL.Image]，6 路原始环视图
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

    resized_list = []
    for ch in camera_channels:
        img = surround_imgs[ch]
        arr = np.array(img)
        resized_arr = cv2.resize(arr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        resized_img = Image.fromarray(resized_arr)
        resized_list.append(resized_img)

    widths, heights = zip(*(im.size for im in resized_list))
    total_w = sum(widths)
    max_h = max(heights)

    canvas = Image.new('RGB', (total_w, max_h))
    x_offset = 0
    for im in resized_list:
        canvas.paste(im, (x_offset, 0))
        x_offset += im.width

    os.makedirs(os.path.dirname(out_path), exist_ok=True
               ) if os.path.dirname(out_path) else None
    canvas.save(out_path)


def main():
    args = parse_args()
    if args.save_front_sam2_rv_debug:
        if not args.sjtu_test:
            raise NotImplementedError("--save_front_sam2_rv_debug is currently implemented for --sjtu_test only")
        if not args.sam2_cache_dir:
            raise ValueError("--sam2_cache_dir is required when --save_front_sam2_rv_debug is used")
        os.makedirs(args.vis_dir, exist_ok=True)
        test_infos = load_test_infos(args.test_info_path)
        sample_idx = int(args.front_sam2_rv_sample_idx)
        sample_info = get_sample_info(test_infos, sample_idx)
        pred_path = os.path.join(args.pred_npy_dir, f"pred_{sample_idx}.npy")
        pred_data = load_prediction(pred_path)
        scale_map_full, _, _, _, _ = resolve_collision_prediction_maps(pred_data)
        raw_curr_rgb = load_raw_curr_camera_rgbs(sample_info)
        sam2_sample_cache = load_sam2_sample_cache(args.sam2_cache_dir, sample_idx)
        projection_debug = project_camera_masks_to_rv(pred_data, sam2_sample_cache, args.sam2_camera_channel)
        out_path = os.path.join(args.vis_dir, f"sam2_{args.sam2_camera_channel.lower()}_rv_overlay_{sample_idx}.png")
        save_front_sam2_rv_overlay_panel(
            sample_idx=sample_idx,
            sample_info=sample_info,
            scale_map_full=scale_map_full,
            raw_curr_rgb=raw_curr_rgb,
            sam2_sample_cache=sam2_sample_cache,
            projection_debug=projection_debug,
            out_path=out_path,
        )
        print(f"[SAM2->RV] saved {args.sam2_camera_channel} RV projection panel to {out_path}")
        return

    if args.proposal_source == "sam2_rgb_fusion":
        if not args.sjtu_test:
            raise NotImplementedError("sam2_rgb_fusion is currently implemented for --sjtu_test only")
        if not args.sam2_cache_dir:
            raise ValueError("--sam2_cache_dir is required when --proposal_source sam2_rgb_fusion is used")

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.vis_dir = os.path.join(args.vis_dir, f'collision_{ts}_ttc{args.risk_time_threshold}_risk{args.risk_pred_threshold}')
    os.makedirs(args.vis_dir, exist_ok=True)
    test_infos = load_test_infos(args.test_info_path)
    files = get_pred_npy_files(args.pred_npy_dir)

    idx_offset = int(files[0].split('_')[-1].split('.')[0]) if files else 0
    csv_rows = []
    reprojection_metric_rows = []
    stitched_manifest_path = None
    stitched_index = None
    if args.sjtu_test:
        stitched_manifest_path, stitched_index = build_sjtu_stitched_index(args.test_info_path)
        print(f"[SJTU] stitched manifest: {stitched_manifest_path}")

    win_name = "range_image_vis"
    if args.keyboard or not args.save:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    for local_idx, fn in enumerate(tqdm(files)):
        sample_idx = local_idx + idx_offset

        # if idx < 101 or idx > 449:
        #     continue

        pred_path = os.path.join(args.pred_npy_dir, fn)
        pred_data = load_prediction(pred_path)
        risk_map = pred_data.get('risk_pred')
        if risk_map is None:
            raise KeyError(
                f"Prediction file {pred_path} does not contain risk_pred. "
                "Rerun test.py with --scale_only false."
            )
        if pred_data.get("proj_pix_curr_fullres") is None:
            raise KeyError(
                f"Prediction file {pred_path} does not contain proj_pix_curr_fullres. "
                "Rerun test.py with --save_pred_npy using the updated full-resolution mapping payload."
            )
        scale_map_full, risk_map_full, scale_map_rv, risk_map_rv, rv_shape = resolve_collision_prediction_maps(pred_data)

        sample_info = get_sample_info(test_infos, sample_idx)
        sample_meta = get_effective_sample_meta(sample_info, pred_data)
        delta_t = resolve_delta_t_seconds(args, sample_meta)
        collision_details = []

        if args.sjtu_test and args.save and args.save_reprojection_debug:
            processed_curr_rgb, processed_hw = load_processed_curr_camera_rgbs(
                sample_info,
                pred_data,
                fallback_hw=(args.image_size[0], args.image_size[1]),
            )
            range_rgb_curr, _ = build_range_rgb_from_proj(processed_curr_rgb, pred_data["proj_pix_curr"])
            reconstructed_rgb, reproj_valid_masks, reproj_hit_counts = scatter_range_rgb_to_cameras(
                range_rgb_curr,
                pred_data["proj_pix_curr"],
                processed_hw,
            )
            save_range_rgb_curr(range_rgb_curr, args.vis_dir, sample_idx)
            save_camera_reprojection_compare(
                processed_curr_rgb,
                reconstructed_rgb,
                reproj_valid_masks,
                reproj_hit_counts,
                args.vis_dir,
                sample_idx,
            )
            reprojection_metric_rows.extend(
                build_camera_reprojection_metric_rows(
                    sample_idx,
                    sample_meta,
                    processed_curr_rgb,
                    reconstructed_rgb,
                    reproj_valid_masks,
                    reproj_hit_counts,
                )
            )

        if args.proposal_source == "sam2_rgb_fusion":
            collisions, cluster_debug, collision_details = detect_sam2_rgb_collision_regions(
                sample_idx=sample_idx,
                sample_info=sample_info,
                pred_data=pred_data,
                args=args,
                scale_map=scale_map_rv,
                orientation_map=risk_map_rv,
                delta_t=delta_t,
            )
        else:
            collisions, cluster_debug = detect_dense_collision_regions(
                scale_map=scale_map_rv,
                orientation_map=risk_map_rv,
                delta_t=delta_t,
                risk_time_threshold=args.risk_time_threshold,
                orientation_threshold=args.risk_pred_threshold,
            )
        vis_collisions = scale_collision_points_for_visualization(collisions, rv_shape, scale_map_full.shape)
        if args.save and args.sjtu_test:
            if args.proposal_source == "sam2_rgb_fusion":
                save_sam2_cluster_region_panel(
                    args=args,
                    sample_idx=sample_idx,
                    sample_info=sample_info,
                    scale_map_full=scale_map_full,
                    sam2_sample_cache=cluster_debug["sam2_sample_cache"],
                    projection_debug=cluster_debug["projection_debug"],
                    proposal_regions=cluster_debug["proposal_regions"],
                    merged_regions=cluster_debug["merged_regions"],
                    passed_regions=cluster_debug["passed_regions"],
                    collisions=collisions,
                    collision_details=collision_details,
                    valid_y_min=cluster_debug["valid_y_min"],
                    valid_y_max=cluster_debug["valid_y_max"],
                )
            else:
                save_cluster_region_panel(
                    args=args,
                    sample_idx=sample_idx,
                    scale_map_full=scale_map_full,
                    grad_mag=cluster_debug["grad_mag"],
                    edge_barrier=cluster_debug["edge_barrier"],
                    closed_contour_mask=cluster_debug["closed_contour_mask"],
                    filled_proposal_mask=cluster_debug["filled_proposal_mask"],
                    proposal_regions=cluster_debug["proposal_regions"],
                    merged_regions=cluster_debug["merged_regions"],
                    collisions=collisions,
                    valid_y_min=cluster_debug["valid_y_min"],
                    valid_y_max=cluster_debug["valid_y_max"],
                    selected_region_id=cluster_debug["selected_region_id"],
                )

        # ----- 生成并保存/显示：pred scale -----
        bgr = render_range_image_bgr(scale_map_full, vis_collisions, kind='scale')
        if args.save:
            out_colli_scale_pred = os.path.join(args.vis_dir, f'colli_scale_pred_{sample_idx}.png')
            cv2.imwrite(out_colli_scale_pred, bgr)
        if args.keyboard:
            cv2.imshow(win_name, bgr)
            key = cv2.waitKey(0)
            if key == 27:  # ESC 退出
                break
        elif not args.save:
            cv2.imshow(win_name, bgr)
            cv2.waitKey(1)

        if not args.sjtu_test:
            # ----- 环视图叠加 -----
            overlay_surround_views(collisions, sample_info, args, sample_idx)

            # ----- gt scale -----
            gt = np.load(
                os.path.join(sample_info['gt_map_path'], 'range_image_prev.npy'),
                allow_pickle=True
            ).item()
            gt_scale_map = gt['scale']
            gt_vis_collisions = scale_collision_points_for_visualization(collisions, rv_shape, gt_scale_map.shape)
            bgr = render_range_image_bgr(gt_scale_map, gt_vis_collisions, kind='scale')
            if args.save:
                out_colli_scale_gt = os.path.join(args.vis_dir, f'colli_scale_gt_{sample_idx}.png')
                cv2.imwrite(out_colli_scale_gt, bgr)

            # ----- gt risk -----
            gt = np.load(
                os.path.join(sample_info['gt_map_path'], 'range_image_prev.npy'),
                allow_pickle=True
            ).item()
            gt_risk_map = gt['risk_score']
            bgr = render_range_image_bgr(gt_risk_map, gt_vis_collisions, kind='risk')
            if args.save:
                out_colli_risk_gt = os.path.join(args.vis_dir, f'colli_risk_gt_{sample_idx}.png')
                cv2.imwrite(out_colli_risk_gt, bgr)

            # ----- pred risk -----
            bgr = render_range_image_bgr(risk_map_full, vis_collisions, kind='risk')
            if args.save:
                out_colli_orientation_pred = os.path.join(args.vis_dir, f'colli_orientation_pred_{sample_idx}.png')
                out_colli_risk_pred = os.path.join(args.vis_dir, f'colli_risk_pred_{sample_idx}.png')
                cv2.imwrite(out_colli_orientation_pred, bgr)
                cv2.imwrite(out_colli_risk_pred, bgr)
        else:
            if args.save:
                bgr = render_range_image_bgr(risk_map_full, vis_collisions, kind='orientation')
                out_colli_orientation_pred = os.path.join(args.vis_dir, f'colli_orientation_pred_{sample_idx}.png')
                cv2.imwrite(out_colli_orientation_pred, bgr)
            stitched_image_path = resolve_sjtu_stitched_image_path(sample_meta, stitched_index)
            overlay_image_path, stitched_img_width = overlay_surround_views_sjtu(
                collisions=collisions,
                pred_data=pred_data,
                sample_meta=sample_meta,
                stitched_image_path=stitched_image_path,
                args=args,
                idx=sample_idx,
                collision_details=collision_details,
            )
            depth_overlay_image_path = overlay_stitched_depth_sjtu(
                collisions=collisions,
                pred_data=pred_data,
                depth_stitched_path=pred_data.get("curr_depth_stitched_path"),
                args=args,
                idx=sample_idx,
                collision_details=collision_details,
            )
            csv_rows.extend(
                build_collision_csv_rows(
                    sample_idx=sample_idx,
                    sample_meta=sample_meta,
                    stitched_image_path=stitched_image_path,
                    overlay_image_path=overlay_image_path,
                    depth_overlay_image_path=depth_overlay_image_path,
                    collisions=collisions,
                    pred_data=pred_data,
                    stitched_img_width=stitched_img_width,
                    proposal_source=args.proposal_source,
                    collision_details=collision_details,
                )
            )

    if args.sjtu_test:
        csv_path = os.path.join(args.vis_dir, "collision_predictions.csv")
        save_collision_predictions_csv(csv_rows, csv_path)
        print(f"[SJTU] saved structured collision results to {csv_path}")
        if args.save_reprojection_debug:
            reproj_csv_path = os.path.join(args.vis_dir, "camera_reprojection_metrics.csv")
            save_camera_reprojection_metrics_csv(reprojection_metric_rows, reproj_csv_path)
            print(f"[SJTU] saved camera reprojection metrics to {reproj_csv_path}")

    if args.keyboard or not args.save:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
