"""
Visualize Waymo GT projection quality.

For each selected pair:
  * project GT range-image xyz back onto the 5 camera images
  * save current scale and risk GT range-view visualizations
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.waymo_paths import WAYMO_CAMERAS, resolve_waymo_path
from dataloader.waymo_dataset import build_waymo_lidar_to_camera_projection


REPRESENTATIVE_SEGMENT_ID = "10017090168044687777_6380_000_6400_000"


def infer_output_root_from_info_pkl(info_pkl: str | Path) -> Path:
    info_pkl = Path(info_pkl)
    search = [info_pkl.parent] + list(info_pkl.parents)
    for candidate in search:
        if candidate.name == "2_trainval_test_infos":
            return candidate.parent
    return info_pkl.parent


def project_vehicle_points_to_waymo_image(points_xyz: np.ndarray, cam_calib: dict):
    """
    Forward project vehicle-frame points into the raw Waymo image, including
    radial+tangential distortion.
    """
    if points_xyz.size == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    _, K, dist, R_l2c, t_l2c = build_waymo_lidar_to_camera_projection(cam_calib)
    pts_cam = (R_l2c @ points_xyz.T) + t_l2c.reshape(3, 1)
    z = pts_cam[2]
    valid = z > 1e-6
    valid_idx = np.nonzero(valid)[0]
    if valid_idx.size == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    x = (pts_cam[0, valid] / z[valid]).astype(np.float64)
    y = (pts_cam[1, valid] / z[valid]).astype(np.float64)
    k1, k2, p1, p2, k3 = dist.astype(np.float64)
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x_d + cx
    v = fy * y_d + cy
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return uv, z[valid].astype(np.float32), valid_idx.astype(np.int64)


def render_camera_overlay(
    frame_key: str,
    info: dict,
    range_image: dict,
    dataset_root: Path,
    output_path: Path,
    max_points_per_camera: int = 4000,
):
    cameras_key = f"{frame_key}_cameras"
    frame_idx_key = f"{frame_key}_frame_idx"
    cameras = info[cameras_key]
    frame_idx = info[frame_idx_key]
    segment_id = info["segment_id"]

    xyz = np.asarray(range_image["xyz"], dtype=np.float32)
    scale = np.asarray(range_image["scale"], dtype=np.float32)
    mask = np.asarray(range_image["mask"], dtype=bool)

    pts = xyz[mask]
    scale_vals = scale[mask]
    scale_vals = np.clip(scale_vals, 0.5, 1.5)

    fig, axes = plt.subplots(1, len(WAYMO_CAMERAS), figsize=(28, 6), dpi=150)
    for ax, cam_name in zip(axes, WAYMO_CAMERAS):
        img_path = dataset_root / "camera_images" / cam_name / f"segment_{segment_id}_frame_{frame_idx:04d}.jpg"
        img = np.asarray(Image.open(img_path).convert("RGB"))
        ax.imshow(img)

        cam_calib = cameras[cam_name]
        uv, cam_depth, cam_point_idx = project_vehicle_points_to_waymo_image(pts, cam_calib)
        if uv.shape[0] > 0:
            w = int(cam_calib["width"])
            h = int(cam_calib["height"])
            valid = (
                (uv[:, 0] >= 0) & (uv[:, 0] < w) &
                (uv[:, 1] >= 0) & (uv[:, 1] < h)
            )
            uv = uv[valid]
            cam_depth = cam_depth[valid]
            cam_scale = scale_vals[cam_point_idx][valid]

            # Prefer nearer points when there are too many.
            if uv.shape[0] > max_points_per_camera:
                order = np.argsort(cam_depth)
                keep = order[:max_points_per_camera]
                uv = uv[keep]
                cam_scale = cam_scale[keep]

            ax.scatter(
                uv[:, 0],
                uv[:, 1],
                c=cam_scale,
                s=4,
                cmap="turbo",
                vmin=0.5,
                vmax=1.5,
                alpha=0.75,
                linewidths=0,
            )

        ax.set_title(f"{cam_name}\n{frame_key} frame {frame_idx:04d}")
        ax.axis("off")

    fig.suptitle(
        f"Waymo GT Projection Overlay | segment={segment_id} | {frame_key}",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_range_view_map(
    data: np.ndarray,
    mask: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    vis = np.array(data, dtype=np.float32)
    vis = np.ma.array(vis, mask=~mask)

    fig, ax = plt.subplots(figsize=(18, 3), dpi=160)
    im = ax.imshow(vis, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(args):
    info_pkl = Path(args.info_pkl)
    gt_root = Path(args.gt_root) if args.gt_root else infer_output_root_from_info_pkl(info_pkl)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    with open(info_pkl, "rb") as f:
        infos = pickle.load(f)

    selected = [
        info for info in infos
        if info["segment_id"] == args.segment_id
    ]
    selected.sort(key=lambda x: (x["curr_frame_idx"], x["prev_frame_idx"]))
    selected = selected[args.start_index: args.start_index + args.max_pairs]

    if not selected:
        raise ValueError(
            f"No Waymo infos found for segment_id={args.segment_id} in {info_pkl}"
        )

    for info in selected:
        pair_name = info["sf_folder"]
        gt_map_dir = resolve_waymo_path(info["gt_map_path"], gt_root)
        range_prev = np.load(gt_map_dir / "range_image_prev.npy", allow_pickle=True).item()
        range_curr = np.load(gt_map_dir / "range_image_curr.npy", allow_pickle=True).item()

        pair_dir = output_dir / pair_name
        render_camera_overlay("prev", info, range_prev, dataset_root, pair_dir / "prev_overlay.png")
        render_camera_overlay("curr", info, range_curr, dataset_root, pair_dir / "curr_overlay.png")

        curr_mask = np.asarray(range_curr["mask"], dtype=bool)
        render_range_view_map(
            range_curr["scale"],
            curr_mask,
            f"{pair_name} | scale",
            pair_dir / "scale_curr.png",
            cmap="RdBu_r",
            vmin=0.5,
            vmax=1.5,
        )
        render_range_view_map(
            range_curr["risk_score"],
            curr_mask,
            f"{pair_name} | risk/orientation",
            pair_dir / "risk_curr.png",
            cmap="twilight",
            vmin=0.0,
            vmax=np.pi,
        )

    print(f"Saved {len(selected)} visualized pairs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Waymo GT projection quality")
    parser.add_argument(
        "--info_pkl",
        type=str,
        required=True,
        help="Waymo info PKL produced by range_projection_waymo.py",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/data2/waymo_processed",
        help="Waymo processed root that contains camera_images / camera_calibrations / depth",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default=None,
        help="Root for gt_map/info outputs. If omitted, infer from info_pkl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the visualization PNGs",
    )
    parser.add_argument(
        "--segment_id",
        type=str,
        default=REPRESENTATIVE_SEGMENT_ID,
        help="Representative Waymo segment id to visualize",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=3)
    args = parser.parse_args()
    main(args)
