"""
Extract camera images from Waymo TFRecord files and save as JPEGs.

This is a preprocessing step: later scripts (range projection, depth map
generation) expect camera images to be available on disk.

Output layout (under <save_path>/camera_images/):
    FRONT/segment_<id>_frame_<idx>.jpg
    FRONT_LEFT/segment_<id>_frame_<idx>.jpg
    ...
"""
from __future__ import annotations

import os
import sys
import glob
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tools.generate_scene_flow_waymo.waymo_reader import (
    iter_frames,
    extract_camera_images,
    extract_camera_calibrations,
    extract_ego_pose,
    extract_timestamp_us,
    extract_segment_id,
)
from utils.waymo_paths import WAYMO_CAMERA_ID_TO_NAME

import pickle
import numpy as np


def extract_segment_cameras(
    tfrecord_path: str,
    save_root: str,
    save_calibrations: bool = True,
):
    """
    Extract all camera images from one TFRecord and save as JPEG files.
    Optionally save per-segment calibration metadata as a pickle.
    """
    segment_id = (
        os.path.basename(tfrecord_path)
        .replace("segment-", "")
        .replace("_with_camera_labels.tfrecord", "")
    )

    calib_records = []  # per-frame calibration info

    for frame_idx, frame in enumerate(tqdm(
        iter_frames(tfrecord_path),
        desc=f"Extracting cameras: {segment_id[:30]}...",
    )):
        # Extract JPEG bytes for each camera
        images = extract_camera_images(frame)

        for cam_id, jpeg_bytes in images.items():
            cam_name = WAYMO_CAMERA_ID_TO_NAME.get(cam_id)
            if cam_name is None:
                continue

            out_dir = os.path.join(save_root, "camera_images", cam_name)
            os.makedirs(out_dir, exist_ok=True)

            filename = f"segment_{segment_id}_frame_{frame_idx:04d}.jpg"
            out_path = os.path.join(out_dir, filename)

            with open(out_path, "wb") as f:
                f.write(jpeg_bytes)

        # Collect calibration info
        if save_calibrations:
            calibs = extract_camera_calibrations(frame)
            pose = extract_ego_pose(frame)
            timestamp = extract_timestamp_us(frame)

            calib_record = {
                "frame_idx": frame_idx,
                "timestamp_us": timestamp,
                "segment_id": segment_id,
                "ego_pose": pose,  # (4, 4)
                "cameras": {},
            }
            for cam_id, cal in calibs.items():
                cam_name = WAYMO_CAMERA_ID_TO_NAME.get(cam_id)
                if cam_name is None:
                    continue
                calib_record["cameras"][cam_name] = {
                    "extrinsic": cal["extrinsic"],   # (4, 4) camera → vehicle
                    "intrinsic": cal["intrinsic"],   # (9,)
                    "width": cal["width"],
                    "height": cal["height"],
                    "image_path": f"camera_images/{cam_name}/segment_{segment_id}_frame_{frame_idx:04d}.jpg",
                }
            calib_records.append(calib_record)

    # Save calibration pkl
    if save_calibrations and calib_records:
        calib_dir = os.path.join(save_root, "camera_calibrations")
        os.makedirs(calib_dir, exist_ok=True)
        pkl_path = os.path.join(calib_dir, f"segment_{segment_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(calib_records, f)
        print(f"  Saved calibration PKL: {pkl_path}")

    return len(calib_records)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Extract camera images from Waymo TFRecords")
    parser.add_argument("--dataroot", type=str, default="/mnt/data2/waymo")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--save_path", type=str, default="./Datasets/waymo",
                        help="Root output directory")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--no_calibrations", action="store_true",
                        help="Skip saving calibration PKLs")
    args = parser.parse_args()

    pattern = os.path.join(args.dataroot, args.split, "*.tfrecord")
    tfrecord_files = sorted(glob.glob(pattern))
    if not tfrecord_files:
        print(f"No TFRecords at {pattern}")
        sys.exit(1)

    selected = tfrecord_files[args.start : args.end]
    print(f"Extracting cameras from {len(selected)} segments [{args.start}:{args.end}]")

    for tf_path in selected:
        n = extract_segment_cameras(
            tf_path,
            save_root=args.save_path,
            save_calibrations=not args.no_calibrations,
        )
        print(f"  Extracted {n} frames")
