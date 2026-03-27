"""
Generate DepthAnythingV2 metric depth maps for Waymo camera images.

Reads camera images from ./Datasets/waymo/camera_images/{cam_name}/
and saves depth predictions to ./Datasets/waymo/4_depth_map/{cam_name}/

Usage:
    python tools/create_depth_map/create_waymo_depth_map.py \
        --dataset_root ./Datasets/waymo \
        --encoder vitl \
        --max_depth 80 \
        --skip_existing
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from utils.waymo_paths import WAYMO_CAMERAS


def resolve_checkpoint_path(encoder: str, metric_dataset: str, checkpoint_path: str | None = None) -> str:
    """Locate the DepthAnythingV2 metric depth checkpoint."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path

    ckpt_name = f"depth_anything_v2_metric_{metric_dataset}_{encoder}.pth"
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "pretrained" / "new" / ckpt_name,
        repo_root / "pretrained" / ckpt_name,
        repo_root / "depthanything" / "checkpoints" / ckpt_name,
        repo_root / "depthanything" / "metric_depth" / "checkpoints" / ckpt_name,
        repo_root.parent / "FP-TTC" / "pretrained" / "new" / ckpt_name,
        repo_root.parent / "FP-TTC" / "depthanything" / "metric_depth" / "checkpoints" / ckpt_name,
        repo_root.parent / "Depth-Anything-V2" / "checkpoints" / ckpt_name,
        repo_root.parent / "Depth-Anything-V2" / "metric_depth" / "checkpoints" / ckpt_name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Cannot find checkpoint {ckpt_name}. Tried:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )


def build_model(encoder: str, metric_dataset: str, max_depth: float, ckpt_path: str | None = None):
    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**{**configs[encoder], "max_depth": max_depth})
    path = resolve_checkpoint_path(encoder, metric_dataset, ckpt_path)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to("cuda").eval()
    print(f"Loaded {encoder} model from {path}")
    return model


def main(args):
    camera_images_dir = os.path.join(args.dataset_root, "camera_images")
    depth_output_dir = os.path.join(args.dataset_root, "4_depth_map")

    # Collect all images to process
    image_jobs = []
    for cam_name in WAYMO_CAMERAS:
        cam_dir = os.path.join(camera_images_dir, cam_name)
        if not os.path.isdir(cam_dir):
            print(f"Skipping {cam_name}: directory not found")
            continue
        for img_path in sorted(glob(os.path.join(cam_dir, "*.jpg"))):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            depth_path = os.path.join(depth_output_dir, cam_name, f"{basename}.npy")
            if args.skip_existing and os.path.exists(depth_path):
                continue
            image_jobs.append((cam_name, img_path, depth_path))

    if not image_jobs:
        print("No images to process (all depth maps already exist or no images found).")
        return

    print(f"Processing {len(image_jobs)} images across {len(WAYMO_CAMERAS)} cameras")

    model = build_model(args.encoder, args.metric_dataset, args.max_depth, args.checkpoint_path)

    for cam_name, img_path, depth_path in tqdm(image_jobs, desc="Generating depth maps"):
        # Read image (DepthAnythingV2 expects BGR)
        raw_bgr = cv2.imread(img_path) 
        if raw_bgr is None:
            print(f"Warning: could not read {img_path}")
            continue

        # Infer depth
        with torch.no_grad():
            depth = model.infer_image(raw_bgr, input_size=args.depth_input_size)
        # depth: (H, W) float32, metric depth in meters

        # Save
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        np.save(depth_path, depth.astype(np.float32))

    print(f"Done. Depth maps saved to {depth_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate depth maps for Waymo camera images")
    parser.add_argument("--dataset_root", type=str, default="./Datasets/waymo")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--metric_dataset", type=str, default="vkitti")
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--depth_input_size", type=int, default=518,
                        help="Input size for DepthAnything inference")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()
    main(args)
