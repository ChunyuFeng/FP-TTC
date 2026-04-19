#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import cv2

from tools.cyberrock.sjtu_pipeline_utils import CAMERA_CHANNELS, REF_CHANNEL, ensure_dir


MANIFEST_FIELDS = [
    "sequence_dir_name",
    "bag_name",
    "scene",
    "pos_neg",
    "conditions",
    "tag",
    "group_id",
    "ros_msg_seq",
    "timestamp_us",
    "stitched_image_path",
    "aligned_manifest_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate stitched surround-view images and a frame manifest from SJTU aligned manifests."
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        required=True,
        help="Prepared SJTU root, e.g. Datasets/cyberrock_keyframe_2hz/prepared_eval_data",
    )
    parser.add_argument(
        "--stitched_root",
        type=str,
        default=None,
        help="Optional override for stitched image root. Defaults to <processed_root>/annotations/stitched_frames",
    )
    parser.add_argument(
        "--manifest_out",
        type=str,
        default=None,
        help="Optional override for stitched manifest CSV path. Defaults to <processed_root>/annotations/manifests/stitched_frames_manifest.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stitched images if they already exist.",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format for stitched views.",
    )
    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=95,
        help="JPEG quality for stitched views when --image_format=jpg.",
    )
    return parser.parse_args()


def load_processed_rows(processed_manifest_csv: Path) -> List[Dict[str, str]]:
    with open(processed_manifest_csv, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_aligned_rows(aligned_manifest_csv: Path) -> List[Dict[str, str]]:
    with open(aligned_manifest_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["group_id"]))
    return rows


def build_stitched_image(aligned_row: Dict[str, str]):
    images = []
    for channel in CAMERA_CHANNELS:
        image_path = Path(aligned_row[f"{channel}_rectified_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read rectified image: {image_path}")
        images.append(image)

    heights = {img.shape[0] for img in images}
    if len(heights) != 1:
        raise ValueError(f"Camera heights do not match for group_id={aligned_row['group_id']}: {sorted(heights)}")

    return cv2.hconcat(images)


def write_stitched_image(stitched_path: Path, stitched_image, image_format: str, jpg_quality: int) -> None:
    if image_format == "jpg":
        ok = cv2.imwrite(str(stitched_path), stitched_image, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])
    elif image_format == "png":
        ok = cv2.imwrite(str(stitched_path), stitched_image)
    else:
        raise ValueError(f"Unsupported image format: {image_format}")
    if not ok:
        raise IOError(f"Failed to write stitched image: {stitched_path}")


def ensure_annotation_dirs(processed_root: Path) -> Dict[str, Path]:
    annotations_root = ensure_dir(processed_root / "annotations")
    return {
        "annotations_root": annotations_root,
        "stitched_root": ensure_dir(annotations_root / "stitched_frames"),
        "manifests_root": ensure_dir(annotations_root / "manifests"),
        "cvat_exports_root": ensure_dir(annotations_root / "cvat_exports"),
        "gt_root": ensure_dir(annotations_root / "gt"),
    }


def main():
    args = parse_args()
    processed_root = Path(args.processed_root).resolve()
    dirs = ensure_annotation_dirs(processed_root)

    stitched_root = Path(args.stitched_root).resolve() if args.stitched_root else dirs["stitched_root"]
    manifest_out = (
        Path(args.manifest_out).resolve()
        if args.manifest_out
        else dirs["manifests_root"] / "stitched_frames_manifest.csv"
    )

    processed_manifest_csv = processed_root / "manifests" / "processed_bags.csv"
    if not processed_manifest_csv.exists():
        raise FileNotFoundError(f"Missing processed bag manifest: {processed_manifest_csv}")

    processed_rows = load_processed_rows(processed_manifest_csv)
    manifest_rows = []
    stitched_count = 0
    sequence_count = 0

    for seq_idx, bag_row in enumerate(processed_rows, start=1):
        sequence_dir_name = bag_row.get("sequence_dir_name") or Path(bag_row["sequence_root"]).name
        bag_name = bag_row["bag_name"]
        aligned_manifest_csv = Path(bag_row["aligned_manifest_csv"])
        if not aligned_manifest_csv.exists():
            raise FileNotFoundError(f"Missing aligned manifest: {aligned_manifest_csv}")

        sequence_stitched_root = ensure_dir(stitched_root / sequence_dir_name)
        aligned_rows = load_aligned_rows(aligned_manifest_csv)
        sequence_written = 0
        print(
            f"[stitched][{seq_idx}/{len(processed_rows)}] {sequence_dir_name}: "
            f"{len(aligned_rows)} groups -> {sequence_stitched_root}"
        )

        for aligned_row in aligned_rows:
            group_id = int(aligned_row["group_id"])
            ros_msg_seq = int(aligned_row.get(f"{REF_CHANNEL}_seq") or aligned_row["group_id"])
            timestamp_us = int(aligned_row[f"{REF_CHANNEL}_header_time_us"])
            stitched_name = f"{group_id:06d}__seq_{ros_msg_seq}.{args.image_format}"
            stitched_path = sequence_stitched_root / stitched_name

            if args.overwrite or not stitched_path.exists():
                stitched_image = build_stitched_image(aligned_row)
                write_stitched_image(stitched_path, stitched_image, args.image_format, args.jpg_quality)
                stitched_count += 1
                sequence_written += 1

            manifest_rows.append(
                {
                    "sequence_dir_name": sequence_dir_name,
                    "bag_name": bag_name,
                    "scene": bag_row.get("scene", ""),
                    "pos_neg": bag_row.get("pos_neg", ""),
                    "conditions": bag_row.get("conditions", ""),
                    "tag": bag_row.get("tag", ""),
                    "group_id": str(group_id),
                    "ros_msg_seq": str(ros_msg_seq),
                    "timestamp_us": str(timestamp_us),
                    "stitched_image_path": str(stitched_path.resolve()),
                    "aligned_manifest_path": str(aligned_manifest_csv.resolve()),
                }
            )
        sequence_count += 1
        print(
            f"[stitched][{seq_idx}/{len(processed_rows)}] completed {sequence_dir_name}: "
            f"wrote {sequence_written} files, manifest rows +{len(aligned_rows)}"
        )

    ensure_dir(manifest_out.parent)
    with open(manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[stitched] completed {sequence_count} sequences")
    print(f"[stitched] wrote {stitched_count} stitched images under {stitched_root}")
    print(f"[stitched] wrote manifest with {len(manifest_rows)} rows to {manifest_out}")


if __name__ == "__main__":
    main()
