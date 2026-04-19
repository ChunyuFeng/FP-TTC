#!/usr/bin/env python3
"""Initialize canonical GT CSV templates for SJTU real-vehicle annotation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tools.cyberrock.real_vehicle_annotation_common import (
    COLLISION_BOX_HEADERS,
    FRAME_LABEL_HEADERS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize frame_labels.csv and collision_boxes.csv from stitched frame manifest."
    )
    parser.add_argument(
        "--stitched-manifest",
        required=True,
        help="Path to stitched_frames_manifest.csv",
    )
    parser.add_argument(
        "--gt-root",
        required=True,
        help="Output directory for canonical GT csv files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV files if they already exist",
    )
    return parser.parse_args()


def ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. Use --force to overwrite the existing file."
        )


def load_stitched_rows(stitched_manifest: Path) -> list[dict[str, str]]:
    with stitched_manifest.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in stitched manifest: {stitched_manifest}")
    return rows


def write_frame_labels(rows: list[dict[str, str]], output_path: Path) -> int:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_LABEL_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence_dir_name": row["sequence_dir_name"],
                    "bag_name": row["bag_name"],
                    "group_id": row["group_id"],
                    "ros_msg_seq": row["ros_msg_seq"],
                    "timestamp_us": row["timestamp_us"],
                    "stitched_image_path": row["stitched_image_path"],
                    "frame_label": "",
                    "event_id": "",
                    "annotator": "",
                    "review_status": "draft",
                    "notes": "",
                }
            )
    return len(rows)


def write_collision_boxes(output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLLISION_BOX_HEADERS)
        writer.writeheader()


def main() -> None:
    args = parse_args()
    stitched_manifest = Path(args.stitched_manifest).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve()
    gt_root.mkdir(parents=True, exist_ok=True)

    frame_labels_path = gt_root / "frame_labels.csv"
    collision_boxes_path = gt_root / "collision_boxes.csv"

    ensure_can_write(frame_labels_path, args.force)
    ensure_can_write(collision_boxes_path, args.force)

    rows = load_stitched_rows(stitched_manifest)
    frame_count = write_frame_labels(rows, frame_labels_path)
    write_collision_boxes(collision_boxes_path)

    print(f"stitched_manifest: {stitched_manifest}")
    print(f"gt_root: {gt_root}")
    print(f"frame_labels: {frame_labels_path} ({frame_count} rows)")
    print(f"collision_boxes: {collision_boxes_path} (header only)")


if __name__ == "__main__":
    main()
