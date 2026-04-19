#!/usr/bin/env python3
"""Shared utilities for the SJTU real-vehicle annotation workflow."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


FRAME_LABEL_HEADERS = [
    "sequence_dir_name",
    "bag_name",
    "group_id",
    "ros_msg_seq",
    "timestamp_us",
    "stitched_image_path",
    "frame_label",
    "event_id",
    "annotator",
    "review_status",
    "notes",
]

COLLISION_BOX_HEADERS = [
    "sequence_dir_name",
    "group_id",
    "event_id",
    "target_id",
    "box_id",
    "x1",
    "y1",
    "x2",
    "y2",
]

FRAME_LABEL_VALUES = ("pos", "neg", "ignore")
POSITIVE_SCENES = {"S1", "S2", "S3", "S4"}
NEGATIVE_SCENES = {"S5", "S6", "S7"}
DEFAULT_SCENE_ORDER = ["S1", "S2", "S5", "S7", "S3", "S4", "S6"]
ROUND_BY_SCENE = {
    "S1": "Round 1",
    "S2": "Round 1",
    "S3": "Round 2",
    "S4": "Round 2",
    "S5": "Round 1",
    "S6": "Round 2",
    "S7": "Round 1",
}

CVAT_LABEL_SPECS = [
    {"name": "pos", "type": "tag", "color": "#5086EB"},
    {"name": "neg", "type": "tag", "color": "#2699D8"},
    {"name": "ignore", "type": "tag", "color": "#C040E0"},
    {"name": "collision_area", "type": "rectangle", "color": "#B3ADB3"},
]


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv_rows(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path | str, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def infer_annotations_root(stitched_manifest: Path | str) -> Path:
    stitched_manifest = Path(stitched_manifest).resolve()
    if (
        stitched_manifest.name == "stitched_frames_manifest.csv"
        and stitched_manifest.parent.name == "manifests"
        and stitched_manifest.parent.parent.name == "annotations"
    ):
        return stitched_manifest.parent.parent
    return stitched_manifest.parent


def load_stitched_groups(stitched_manifest: Path | str) -> dict[str, list[dict[str, str]]]:
    rows = load_csv_rows(stitched_manifest)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_dir_name"]].append(row)
    for seq_rows in grouped.values():
        seq_rows.sort(key=lambda row: int(row["group_id"]))
    return dict(grouped)


def load_processed_index(processed_manifest: Path | str | None) -> dict[str, dict[str, str]]:
    if not processed_manifest:
        return {}
    rows = load_csv_rows(processed_manifest)
    return {row["sequence_dir_name"]: row for row in rows if row.get("sequence_dir_name")}


def scene_sort_key(scene: str, scene_order: list[str] | None = None) -> tuple[int, str]:
    scene_order = scene_order or DEFAULT_SCENE_ORDER
    try:
        return scene_order.index(scene), scene
    except ValueError:
        return len(scene_order), scene


def summarize_sequence_rows(
    sequence_dir_name: str,
    seq_rows: list[dict[str, str]],
    processed_row: dict[str, str] | None = None,
) -> dict[str, str]:
    first = seq_rows[0]
    processed_row = processed_row or {}
    stitched_dir = str(Path(first["stitched_image_path"]).resolve().parent)
    valid_start = processed_row.get("valid_start_bag_sec", "")
    valid_end = processed_row.get("valid_end_bag_sec", "")
    valid_range = f"{valid_start} - {valid_end}" if valid_start and valid_end else ""
    return {
        "scene": first.get("scene", ""),
        "round": ROUND_BY_SCENE.get(first.get("scene", ""), ""),
        "pos_neg": first.get("pos_neg", ""),
        "sequence_dir_name": sequence_dir_name,
        "bag_name": first.get("bag_name", ""),
        "keep_t2": processed_row.get("keep_t2", ""),
        "valid_range": valid_range,
        "conditions": first.get("conditions", ""),
        "tag": first.get("tag", ""),
        "frame_count": str(len(seq_rows)),
        "first_group_id": seq_rows[0]["group_id"],
        "last_group_id": seq_rows[-1]["group_id"],
        "stitched_dir": stitched_dir,
    }


def build_empty_frame_rows(stitched_groups: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sequence_dir_name in sorted(stitched_groups.keys()):
        for row in stitched_groups[sequence_dir_name]:
            rows.append(
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
    return rows
