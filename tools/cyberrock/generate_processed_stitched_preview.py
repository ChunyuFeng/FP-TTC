#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from tools.cyberrock.sjtu_pipeline_utils import CAMERA_CHANNELS, REF_CHANNEL, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a single processed SJTU stitched surround-view preview in model input coordinates."
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default="Datasets/cyberrock_keyframe_2hz/prepared_eval_data",
        help="Prepared evaluation root containing manifests/ and sequences/.",
    )
    parser.add_argument(
        "--sequence_dir_name",
        type=str,
        default="s1_front_static__2026-04-13-17-21-09",
        help="Sequence directory name under prepared_eval_data/sequences/.",
    )
    parser.add_argument(
        "--group_id",
        type=int,
        default=51,
        help="Aligned group_id to preview.",
    )
    parser.add_argument(
        "--expected_ros_msg_seq",
        type=int,
        default=44432,
        help="Optional seq sanity check for the chosen preview frame.",
    )
    return parser.parse_args()


def load_processed_row(processed_bags_csv: Path, sequence_dir_name: str) -> dict[str, str]:
    with open(processed_bags_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("sequence_dir_name") == sequence_dir_name:
                return row
    raise KeyError(f"Sequence {sequence_dir_name} was not found in {processed_bags_csv}")


def load_aligned_row(aligned_manifest_csv: Path, group_id: int) -> dict[str, str]:
    with open(aligned_manifest_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["group_id"]) == group_id:
                return row
    raise KeyError(f"group_id={group_id} was not found in {aligned_manifest_csv}")


def load_curr_rectified_images(aligned_row: dict[str, str]) -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    for channel in CAMERA_CHANNELS:
        image_path = Path(aligned_row[f"{channel}_rectified_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing rectified image for {channel}: {image_path}")
        images[channel] = Image.open(image_path).convert("RGB")
    return images


def sample_preview_params(img_size: tuple[int, int], crop_size: tuple[int, int] = (160, 320)) -> dict[str, object]:
    # Mirrors NuscRangeImageAugmentor.sample_params() for do_flip=False, rotate=False.
    w, h = img_size
    crop_h, crop_w = crop_size
    scale_h = crop_h / h
    scale_w = crop_w / w
    scale = max(scale_h, scale_w)
    resize_w, resize_h = int(w * scale), int(h * scale)
    crop_x = (resize_w - crop_w) // 2
    crop_y = resize_h - crop_h
    return {
        "scale": scale,
        "resize": (resize_w, resize_h),
        "crop": (crop_x, crop_y),
        "flip_h": False,
        "flip_v": False,
        "rotate": False,
        "angle": 0,
    }


def apply_affine_rgb(img: Image.Image, params: dict[str, object], crop_size: tuple[int, int] = (160, 320)) -> np.ndarray:
    crop_h, crop_w = crop_size
    img = img.resize(params["resize"], Image.BILINEAR)
    crop_x, crop_y = params["crop"]
    img = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    return np.array(img)


def stitch_processed_images(images_by_channel: dict[str, np.ndarray]) -> Image.Image:
    stitched = np.concatenate([images_by_channel[channel] for channel in CAMERA_CHANNELS], axis=1)
    return Image.fromarray(stitched.astype(np.uint8), mode="RGB")


def main():
    args = parse_args()
    processed_root = Path(args.processed_root).resolve()
    processed_bags_csv = processed_root / "manifests" / "processed_bags.csv"
    processed_row = load_processed_row(processed_bags_csv, args.sequence_dir_name)

    aligned_manifest_csv = Path(processed_row["aligned_manifest_csv"]).resolve()
    aligned_row = load_aligned_row(aligned_manifest_csv, args.group_id)

    ros_msg_seq = int(aligned_row[f"{REF_CHANNEL}_seq"])
    if args.expected_ros_msg_seq >= 0 and ros_msg_seq != args.expected_ros_msg_seq:
        raise ValueError(
            f"Expected ros_msg_seq={args.expected_ros_msg_seq}, but got {ros_msg_seq} "
            f"for group_id={args.group_id} in {aligned_manifest_csv}"
        )

    curr_images = load_curr_rectified_images(aligned_row)
    orig_size = next(iter(curr_images.values())).size

    affine_params = sample_preview_params(orig_size, crop_size=(160, 320))
    processed_images = {
        channel: apply_affine_rgb(curr_images[channel], affine_params, crop_size=(160, 320))
        for channel in CAMERA_CHANNELS
    }

    for channel in CAMERA_CHANNELS:
        arr = processed_images[channel]
        if arr.shape[:2] != (160, 320):
            raise ValueError(f"Processed image for {channel} has shape {arr.shape}, expected (160, 320, 3)")

    stitched_image = stitch_processed_images(processed_images)
    if stitched_image.size != (1920, 160):
        raise ValueError(f"Stitched image has size {stitched_image.size}, expected (1920, 160)")

    preview_root = ensure_dir(
        processed_root / "annotations" / "preview_processed_stitched" / args.sequence_dir_name
    )
    stitched_filename = (
        f"curr__group_{args.group_id:06d}__seq_{ros_msg_seq}__stitched_160x1920.png"
    )
    stitched_path = preview_root / stitched_filename
    stitched_image.save(stitched_path)

    manifest_path = preview_root / "preview_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_dir_name",
                "group_id",
                "ros_msg_seq",
                "frame_key",
                "stitched_image_path",
                "source_aligned_manifest_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sequence_dir_name": args.sequence_dir_name,
                "group_id": str(args.group_id),
                "ros_msg_seq": str(ros_msg_seq),
                "frame_key": "curr",
                "stitched_image_path": str(stitched_path.resolve()),
                "source_aligned_manifest_path": str(aligned_manifest_csv),
            }
        )

    print(f"[preview] wrote stitched preview to {stitched_path}")
    print(f"[preview] wrote manifest to {manifest_path}")
    print(f"[preview] affine params: {affine_params}")


if __name__ == "__main__":
    main()
