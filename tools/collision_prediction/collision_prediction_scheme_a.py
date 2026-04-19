#!/usr/bin/env python3
import argparse
import csv
import os
import pickle
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.draw import make_risk_analysis_rgb, make_scale_analysis_rgb

matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=FutureWarning)


CAMERA_CHANNELS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scheme-A collision area extraction based on scale/orientation maps"
    )
    parser.add_argument("--pred_npy_dir", type=str, required=True, help="Directory of pred_*.npy files")
    parser.add_argument("--test_info_path", type=str, required=True, help="Path to scene overlap pkl")
    parser.add_argument("--vis_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--panel_prefix", type=str, default="schemeA", help="Prefix for saved panel filenames")
    parser.add_argument("--candidate_scale_threshold", type=float, default=0.97)
    parser.add_argument("--candidate_res_threshold", type=float, default=0.015)
    parser.add_argument("--seed_scale_threshold", type=float, default=0.94)
    parser.add_argument("--seed_res_threshold", type=float, default=0.03)
    parser.add_argument("--candidate_min_area", type=int, default=80)
    parser.add_argument("--seed_min_area", type=int, default=30)
    parser.add_argument("--approaching_min_area", type=int, default=300)
    parser.add_argument("--approaching_min_height", type=int, default=18)
    parser.add_argument("--approaching_med_scale_threshold", type=float, default=0.96)
    parser.add_argument("--approaching_med_res_threshold", type=float, default=0.018)
    parser.add_argument("--touch_top_margin", type=int, default=4)
    parser.add_argument("--touch_top_height_limit", type=int, default=20)
    parser.add_argument("--risk_scale_threshold", type=float, default=0.9)
    parser.add_argument("--orientation_threshold", type=float, default=0.1)
    parser.add_argument("--risk_min_area", type=int, default=60)
    parser.add_argument("--ignore_top_ratio", type=float, default=0.10)
    parser.add_argument("--ignore_bottom_ratio", type=float, default=0.10)
    return parser.parse_args()


def load_test_infos(test_info_path):
    with open(test_info_path, "rb") as f:
        return pickle.load(f)


def get_pred_npy_files(pred_dir):
    files = [f for f in os.listdir(pred_dir) if f.endswith(".npy")]
    files.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
    return files


def load_prediction(pred_path):
    data = np.load(pred_path, allow_pickle=True).item()
    return data["scale_pred"].astype(np.float32), data["risk_pred"].astype(np.float32)


def filter_small(mask, min_area):
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(lbl.ravel())
    keep = counts >= min_area
    keep[0] = False
    return keep[lbl]


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
    resized = cv2.resize(
        rgb,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
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
        0.72,
        (255, 255, 255),
        2,
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
    mask_u8 = (np.asarray(mask, dtype=np.uint8) * 255)
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)


def make_heatmap_rgb(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(values):
        values = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX)
    values_u8 = values.astype(np.uint8)
    heat_bgr = cv2.applyColorMap(values_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def build_current_stitched_rgb(sample_info):
    frames = []
    for channel in CAMERA_CHANNELS:
        image_path = sample_info["curr_camera_data"][channel]["filename"]
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            blank = np.full((1080, 1920, 3), 255, dtype=np.uint8)
            cv2.putText(
                blank,
                f"Missing: {channel}",
                (48, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            rgb = blank
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cv2.putText(
            rgb,
            channel,
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(rgb)
    return np.concatenate(frames, axis=1)


def extract_scheme_a_masks(args, scale_map, orientation_map):
    a = -np.log(scale_map + 1e-6)
    a_smooth = cv2.bilateralFilter(a.astype(np.float32), d=9, sigmaColor=0.03, sigmaSpace=9)
    a_bg = cv2.GaussianBlur(a_smooth, (0, 0), sigmaX=120, sigmaY=18)
    a_res = a_smooth - a_bg

    candidate = (scale_map < args.candidate_scale_threshold) & (a_res > args.candidate_res_threshold)
    seed = (scale_map < args.seed_scale_threshold) & (a_res > args.seed_res_threshold)

    candidate = ndi.binary_opening(candidate, structure=np.ones((3, 3), dtype=bool))
    candidate = ndi.binary_closing(candidate, structure=np.ones((5, 5), dtype=bool))
    candidate = filter_small(candidate, args.candidate_min_area)

    seed = ndi.binary_opening(seed, structure=np.ones((3, 3), dtype=bool))
    seed = filter_small(seed, args.seed_min_area)

    markers = label(seed)
    dist = ndi.distance_transform_edt(candidate)
    watershed_labels = watershed(-dist, markers, mask=candidate)

    approaching_mask = np.zeros_like(candidate, dtype=bool)
    kept_regions = 0
    total_regions = 0
    for region in regionprops(watershed_labels):
        total_regions += 1
        region_mask = watershed_labels == region.label
        y0, x0, y1, x1 = region.bbox
        h = y1 - y0
        med_scale = float(np.median(scale_map[region_mask]))
        med_res = float(np.median(a_res[region_mask]))
        touch_top = y0 <= args.touch_top_margin
        keep = bool(
            region.area >= args.approaching_min_area
            and h >= args.approaching_min_height
            and med_scale < args.approaching_med_scale_threshold
            and med_res > args.approaching_med_res_threshold
            and not (touch_top and h < args.touch_top_height_limit)
        )
        if keep:
            approaching_mask |= region_mask
            kept_regions += 1

    orientation_threshold_mask = orientation_map < args.orientation_threshold
    raw_risk_mask = approaching_mask & (scale_map < args.risk_scale_threshold) & orientation_threshold_mask
    raw_risk_mask = ndi.binary_opening(raw_risk_mask, structure=np.ones((3, 3), dtype=bool))
    raw_risk_mask = ndi.binary_closing(raw_risk_mask, structure=np.ones((5, 5), dtype=bool))

    risk_mask, risk_filter_stats = filter_final_risk_components(
        raw_risk_mask,
        min_area=args.risk_min_area,
        top_ratio=args.ignore_top_ratio,
        bottom_ratio=args.ignore_bottom_ratio,
    )

    stats = {
        "candidate_area": int(candidate.sum()),
        "seed_area": int(seed.sum()),
        "approaching_area": int(approaching_mask.sum()),
        "raw_risk_area": int(raw_risk_mask.sum()),
        "risk_area": int(risk_mask.sum()),
        "watershed_regions": int(total_regions),
        "kept_regions": int(kept_regions),
    }
    stats.update(risk_filter_stats)

    return {
        "a_res": a_res,
        "candidate_mask": candidate,
        "seed_mask": seed,
        "watershed_labels": watershed_labels.astype(np.int32),
        "approaching_mask": approaching_mask,
        "orientation_threshold_mask": orientation_threshold_mask,
        "raw_risk_mask": raw_risk_mask,
        "risk_mask": risk_mask,
        "stats": stats,
    }


def filter_final_risk_components(risk_mask, min_area, top_ratio, bottom_ratio):
    risk_mask = np.asarray(risk_mask, dtype=bool)
    h, _ = risk_mask.shape
    top_limit = int(round(h * top_ratio))
    bottom_start = int(round(h * (1.0 - bottom_ratio)))

    labels, num = ndi.label(risk_mask)
    final_mask = np.zeros_like(risk_mask, dtype=bool)
    kept = 0
    drop_top_bottom = 0
    drop_small = 0

    for label_id in range(1, num + 1):
        component = labels == label_id
        ys, xs = np.where(component)
        if ys.size == 0:
            continue
        area = int(ys.size)
        centroid_y = float(np.mean(ys))
        if centroid_y < top_limit or centroid_y >= bottom_start:
            drop_top_bottom += 1
            continue
        if area < min_area:
            drop_small += 1
            continue
        final_mask |= component
        kept += 1

    return final_mask, {
        "risk_components_raw": int(num),
        "risk_components_kept": int(kept),
        "risk_drop_top_bottom": int(drop_top_bottom),
        "risk_drop_small": int(drop_small),
    }


def save_scheme_a_panel(args, sample_idx, sample_info, scale_map, orientation_map, results, out_path):
    panel_h, panel_w = scale_map.shape
    stitched_rgb = build_current_stitched_rgb(sample_info)
    stitched_panel = fit_rgb_to_canvas(stitched_rgb, (panel_h, panel_w), bg_color=(255, 255, 255))

    scale_panel = ensure_uint8_rgb(make_scale_analysis_rgb(scale_map))
    a_res_panel = make_heatmap_rgb(results["a_res"])
    candidate_panel = make_binary_mask_rgb(results["candidate_mask"])
    seed_panel = make_binary_mask_rgb(results["seed_mask"])
    approaching_panel = make_binary_mask_rgb(results["approaching_mask"])

    orientation_panel = ensure_uint8_rgb(make_risk_analysis_rgb(orientation_map))
    orientation_panel = render_mask_overlay(
        orientation_panel,
        results["orientation_threshold_mask"],
        color=(0, 255, 0),
        alpha=0.32,
    )

    risk_panel = ensure_uint8_rgb(make_scale_analysis_rgb(scale_map))
    risk_panel = render_mask_overlay(risk_panel, results["raw_risk_mask"], color=(255, 255, 0), alpha=0.18)
    risk_panel = render_mask_overlay(risk_panel, results["risk_mask"], color=(255, 0, 0), alpha=0.42)
    top_limit = int(round(scale_map.shape[0] * args.ignore_top_ratio))
    bottom_start = int(round(scale_map.shape[0] * (1.0 - args.ignore_bottom_ratio)))
    cv2.line(risk_panel, (0, top_limit), (panel_w - 1, top_limit), (0, 0, 0), 1)
    cv2.line(risk_panel, (0, bottom_start), (panel_w - 1, bottom_start), (0, 0, 0), 1)

    stats = results["stats"]
    panel_tiles = [
        annotate_panel_tile(stitched_panel, "Current Stitched RGB"),
        annotate_panel_tile(scale_panel, "Scale"),
        annotate_panel_tile(a_res_panel, "a_res"),
        annotate_panel_tile(candidate_panel, f"Candidate area={stats['candidate_area']}"),
        annotate_panel_tile(seed_panel, f"Seed area={stats['seed_area']}"),
        annotate_panel_tile(approaching_panel, f"Approaching kept={stats['kept_regions']}/{stats['watershed_regions']}"),
        annotate_panel_tile(
            orientation_panel,
            f"Orientation + mask(r<{args.orientation_threshold:.2f})",
        ),
        annotate_panel_tile(
            risk_panel,
            (
                f"Risk raw={stats['raw_risk_area']} final={stats['risk_area']} "
                f"dropTB={stats['risk_drop_top_bottom']} dropSmall={stats['risk_drop_small']}"
            ),
        ),
    ]

    top_row = np.concatenate(panel_tiles[:4], axis=1)
    bottom_row = np.concatenate(panel_tiles[4:], axis=1)
    canvas = np.concatenate([top_row, bottom_row], axis=0)
    title = (
        f"sample={sample_idx} scene={sample_info.get('scene_indice', '')} "
        f"ros_seq={sample_info.get('ros_msg_seq_curr', '')}"
    )
    cv2.putText(
        canvas,
        title,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    os.makedirs(args.vis_dir, exist_ok=True)
    npz_dir = os.path.join(args.vis_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)

    test_infos = load_test_infos(args.test_info_path)
    pred_files = get_pred_npy_files(args.pred_npy_dir)

    summary_rows = []
    for fn in pred_files:
        sample_idx = int(os.path.splitext(fn)[0].split("_")[-1])
        pred_path = os.path.join(args.pred_npy_dir, fn)
        scale_map, orientation_map = load_prediction(pred_path)
        sample_info = test_infos[sample_idx]

        results = extract_scheme_a_masks(args, scale_map, orientation_map)
        stats = results["stats"]

        np.savez_compressed(
            os.path.join(npz_dir, f"{args.panel_prefix}_{sample_idx}.npz"),
            scale=scale_map.astype(np.float32),
            orientation=orientation_map.astype(np.float32),
            a_res=results["a_res"].astype(np.float32),
            candidate_mask=results["candidate_mask"].astype(np.uint8),
            seed_mask=results["seed_mask"].astype(np.uint8),
            watershed_labels=results["watershed_labels"].astype(np.int32),
            approaching_mask=results["approaching_mask"].astype(np.uint8),
            orientation_threshold_mask=results["orientation_threshold_mask"].astype(np.uint8),
            raw_risk_mask=results["raw_risk_mask"].astype(np.uint8),
            risk_mask=results["risk_mask"].astype(np.uint8),
        )

        panel_path = os.path.join(args.vis_dir, f"{args.panel_prefix}_{sample_idx}.png")
        save_scheme_a_panel(args, sample_idx, sample_info, scale_map, orientation_map, results, panel_path)

        summary_rows.append(
            {
                "sample_idx": sample_idx,
                "candidate_area": stats["candidate_area"],
                "seed_area": stats["seed_area"],
                "approaching_area": stats["approaching_area"],
                "raw_risk_area": stats["raw_risk_area"],
                "risk_area": stats["risk_area"],
                "watershed_regions": stats["watershed_regions"],
                "kept_regions": stats["kept_regions"],
                "risk_components_raw": stats["risk_components_raw"],
                "risk_components_kept": stats["risk_components_kept"],
                "risk_drop_top_bottom": stats["risk_drop_top_bottom"],
                "risk_drop_small": stats["risk_drop_small"],
                "panel_path": panel_path,
            }
        )

    summary_path = os.path.join(args.vis_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "candidate_area",
                "seed_area",
                "approaching_area",
                "raw_risk_area",
                "risk_area",
                "watershed_regions",
                "kept_regions",
                "risk_components_raw",
                "risk_components_kept",
                "risk_drop_top_bottom",
                "risk_drop_small",
                "panel_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    nonzero = sum(int(row["risk_area"]) > 0 for row in summary_rows)
    print(f"[schemeA] processed={len(summary_rows)} nonzero_risk={nonzero} vis_dir={args.vis_dir}")


if __name__ == "__main__":
    main()
