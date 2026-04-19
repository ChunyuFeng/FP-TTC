#!/usr/bin/env python3
"""End-to-end helpers for the SJTU real-vehicle annotation workflow.

This tool covers the parts of the workflow that can be automated:

1. Build a per-sequence registry from the stitched manifest.
2. Create one CVAT task per sequence with the frozen labels.
3. Pull CVAT annotations back into canonical GT CSV files.
4. Evaluate collision prediction CSV outputs against the GT and emit
   ready-to-fill Table 2 snippets.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cyberrock.real_vehicle_annotation_common import (
    COLLISION_BOX_HEADERS,
    CVAT_LABEL_SPECS,
    DEFAULT_SCENE_ORDER,
    FRAME_LABEL_HEADERS,
    FRAME_LABEL_VALUES,
    NEGATIVE_SCENES,
    build_empty_frame_rows,
    ensure_dir,
    infer_annotations_root,
    load_csv_rows,
    load_processed_index,
    load_stitched_groups,
    scene_sort_key,
    summarize_sequence_rows,
    write_csv_rows,
)


REGISTRY_HEADERS = [
    "scene",
    "round",
    "pos_neg",
    "sequence_dir_name",
    "bag_name",
    "keep_t2",
    "valid_range",
    "frame_count",
    "first_group_id",
    "last_group_id",
    "stitched_dir",
    "conditions",
    "tag",
    "suggested_task_name",
    "task_name",
    "task_id",
    "task_url",
    "task_status",
    "sync_status",
    "label_counts",
    "event_count",
    "box_count",
    "last_sync_time",
    "notes",
]

SEQUENCE_SYNC_HEADERS = [
    "scene",
    "sequence_dir_name",
    "bag_name",
    "task_id",
    "task_name",
    "frame_count",
    "pos_frames",
    "neg_frames",
    "ignore_frames",
    "pos_events",
    "box_count",
    "status",
    "error_count",
    "warning_count",
    "notes",
]

FRAME_EVAL_HEADERS = [
    "scene",
    "sequence_dir_name",
    "bag_name",
    "group_id",
    "ros_msg_seq",
    "timestamp_us",
    "frame_label",
    "event_id",
    "prediction_status",
    "matched_detection_count",
    "valid_detection_count",
    "first_matched_ttc_s",
    "stitched_image_path",
]

SEQUENCE_METRIC_HEADERS = [
    "scene",
    "sequence_dir_name",
    "bag_name",
    "seq_count",
    "valid_frames",
    "pos_frames",
    "neg_frames",
    "ignore_frames",
    "pos_events",
    "tp",
    "fn",
    "fa",
    "recall",
    "far",
    "event_recall",
    "avg_first_tp_offset_frames",
    "avg_first_tp_offset_s",
    "avg_first_tp_ttc_s",
    "missing_prediction_frames",
    "invalid_prediction_rows",
]

SCENE_METRIC_HEADERS = [
    "scene",
    "seq_count",
    "valid_frames",
    "pos_frames",
    "neg_frames",
    "ignore_frames",
    "pos_events",
    "tp",
    "fn",
    "fa",
    "recall",
    "far",
    "event_recall",
    "avg_first_tp_offset_frames",
    "avg_first_tp_offset_s",
    "avg_first_tp_ttc_s",
    "missing_prediction_frames",
    "invalid_prediction_rows",
]


class CvatApiError(RuntimeError):
    """Raised when a CVAT API request fails."""


@dataclass
class SequenceBundle:
    sequence_dir_name: str
    meta: dict[str, str]
    rows: list[dict[str, str]]


class CvatClient:
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        timeout_s: float = 60.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.auth = (username, password)
        self.timeout_s = timeout_s

    def _full_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        if not path_or_url.startswith("/"):
            path_or_url = f"/{path_or_url}"
        return f"{self.host}{path_or_url}"

    def request(self, method: str, path_or_url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout_s)
        response = requests.request(
            method=method,
            url=self._full_url(path_or_url),
            auth=self.auth,
            **kwargs,
        )
        if response.status_code >= 400:
            body = response.text.strip()
            raise CvatApiError(
                f"CVAT API {method} {path_or_url} failed with {response.status_code}: {body}"
            )
        return response

    def list_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        url = "/api/tasks?page_size=500"
        while url:
            response = self.request("GET", url)
            payload = response.json()
            tasks.extend(payload.get("results", []))
            url = payload.get("next")
        return tasks

    def get_task(self, task_id: int) -> dict[str, Any]:
        return self.request("GET", f"/api/tasks/{task_id}").json()

    def delete_task(self, task_id: int) -> None:
        self.request("DELETE", f"/api/tasks/{task_id}")

    def create_task(self, name: str, labels: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "name": name,
            "labels": labels,
        }
        return self.request("POST", "/api/tasks", json=payload).json()

    def get_task_labels(self, task_id: int) -> list[dict[str, Any]]:
        response = self.request("GET", f"/api/labels?task_id={task_id}&page_size=100")
        return response.json().get("results", [])

    def get_task_annotations(self, task_id: int) -> dict[str, Any]:
        return self.request("GET", f"/api/tasks/{task_id}/annotations/").json()

    def get_task_data_meta(self, task_id: int) -> dict[str, Any]:
        return self.request("GET", f"/api/tasks/{task_id}/data/meta").json()

    def wait_for_request(
        self,
        request_id: str,
        timeout_s: float = 900.0,
        poll_interval_s: float = 1.0,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        last_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            payload = self.request("GET", f"/api/requests/{request_id}").json()
            last_payload = payload
            status = payload.get("status")
            if status == "finished":
                return payload
            if status == "failed":
                message = payload.get("message", "")
                raise CvatApiError(f"Background request {request_id} failed: {message}")
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Timed out while waiting for CVAT request {request_id}. "
            f"Last payload: {last_payload}"
        )

    def upload_task_images(
        self,
        task_id: int,
        image_paths: list[Path],
        image_quality: int = 95,
        bulk_max_mb: float = 128.0,
        bulk_max_files: int = 200,
    ) -> dict[str, Any]:
        if not image_paths:
            raise ValueError("No image paths were provided for CVAT upload.")

        bulk_groups = split_upload_groups(
            image_paths,
            max_bytes=int(max(1, bulk_max_mb) * 1024 * 1024),
            max_files=max(1, bulk_max_files),
        )
        upload_order = [path.name for path in image_paths]
        url = f"/api/tasks/{task_id}/data/"

        if len(bulk_groups) > 1:
            self.request("POST", url, json={}, headers={"Upload-Start": ""})

        request_id = None
        for group_index, group in enumerate(bulk_groups):
            headers: dict[str, str]
            form_data: dict[str, Any] = {
                "image_quality": str(image_quality),
                "sorting_method": "predefined",
            }
            if group_index == len(bulk_groups) - 1:
                headers = {"Upload-Finish": ""}
                form_data["upload_file_order"] = upload_order
            else:
                headers = {"Upload-Multiple": ""}

            with ExitStack() as stack:
                files = {}
                for item_index, image_path in enumerate(group):
                    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                    files[f"client_files[{item_index}]"] = (
                        image_path.name,
                        stack.enter_context(image_path.open("rb")),
                        mime_type,
                    )
                response = self.request(
                    "POST",
                    url,
                    data=form_data,
                    files=files,
                    headers=headers,
                    timeout=max(self.timeout_s, 600.0),
                )
            if group_index == len(bulk_groups) - 1:
                payload = response.json()
                request_id = payload["rq_id"]

        if request_id is None:
            raise RuntimeError("CVAT upload finished without returning a request id.")
        return self.wait_for_request(request_id)


def split_upload_groups(
    image_paths: list[Path],
    max_bytes: int,
    max_files: int,
) -> list[list[Path]]:
    groups: list[list[Path]] = []
    current_group: list[Path] = []
    current_bytes = 0
    for image_path in image_paths:
        file_size = image_path.stat().st_size
        should_split = (
            current_group
            and (len(current_group) >= max_files or current_bytes + file_size > max_bytes)
        )
        if should_split:
            groups.append(current_group)
            current_group = []
            current_bytes = 0
        current_group.append(image_path)
        current_bytes += file_size
    if current_group:
        groups.append(current_group)
    return groups


def coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return default
    return int(float(text))


def coerce_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def format_optional_float(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    return f"{value:.{digits}f}"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_sequence_bundles(
    stitched_manifest: Path,
    processed_manifest: Path | None = None,
    scene_order: list[str] | None = None,
) -> list[SequenceBundle]:
    stitched_groups = load_stitched_groups(stitched_manifest)
    processed_index = load_processed_index(processed_manifest)
    bundles = [
        SequenceBundle(
            sequence_dir_name=sequence_dir_name,
            meta=summarize_sequence_rows(
                sequence_dir_name=sequence_dir_name,
                seq_rows=rows,
                processed_row=processed_index.get(sequence_dir_name),
            ),
            rows=rows,
        )
        for sequence_dir_name, rows in stitched_groups.items()
    ]
    bundles.sort(
        key=lambda bundle: (
            scene_sort_key(bundle.meta.get("scene", ""), scene_order or DEFAULT_SCENE_ORDER),
            bundle.sequence_dir_name,
        )
    )
    return bundles


def include_bundle(
    bundle: SequenceBundle,
    scenes: set[str] | None,
    sequences: set[str] | None,
    keep_t2_only: bool,
) -> bool:
    if scenes and bundle.meta.get("scene", "") not in scenes:
        return False
    if sequences and bundle.sequence_dir_name not in sequences:
        return False
    if keep_t2_only:
        keep_t2 = bundle.meta.get("keep_t2", "").strip().upper()
        if keep_t2 not in {"", "?", "Y"}:
            return False
    return True


def registry_default_path(stitched_manifest: Path) -> Path:
    annotations_root = infer_annotations_root(stitched_manifest)
    return annotations_root / "cvat_exports" / "cvat_task_registry.csv"


def gt_default_root(stitched_manifest: Path) -> Path:
    annotations_root = infer_annotations_root(stitched_manifest)
    return annotations_root / "gt"


def raw_export_default_root(stitched_manifest: Path) -> Path:
    annotations_root = infer_annotations_root(stitched_manifest)
    return annotations_root / "cvat_exports" / "raw_api"


def load_registry_index(registry_path: Path | None) -> dict[str, dict[str, str]]:
    if not registry_path or not registry_path.exists():
        return {}
    rows = load_csv_rows(registry_path)
    return {row["sequence_dir_name"]: row for row in rows if row.get("sequence_dir_name")}


def save_registry_rows(registry_path: Path, rows: list[dict[str, str]]) -> None:
    write_csv_rows(registry_path, REGISTRY_HEADERS, rows)


def build_registry_rows(
    bundles: list[SequenceBundle],
    registry_index: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    registry_index = registry_index or {}
    rows: list[dict[str, str]] = []
    for bundle in bundles:
        existing = registry_index.get(bundle.sequence_dir_name, {})
        suggested_task_name = f"{bundle.meta.get('scene', '')}_{bundle.sequence_dir_name}"
        row = {
            "scene": bundle.meta.get("scene", ""),
            "round": bundle.meta.get("round", ""),
            "pos_neg": bundle.meta.get("pos_neg", ""),
            "sequence_dir_name": bundle.sequence_dir_name,
            "bag_name": bundle.meta.get("bag_name", ""),
            "keep_t2": bundle.meta.get("keep_t2", ""),
            "valid_range": bundle.meta.get("valid_range", ""),
            "frame_count": bundle.meta.get("frame_count", ""),
            "first_group_id": bundle.meta.get("first_group_id", ""),
            "last_group_id": bundle.meta.get("last_group_id", ""),
            "stitched_dir": bundle.meta.get("stitched_dir", ""),
            "conditions": bundle.meta.get("conditions", ""),
            "tag": bundle.meta.get("tag", ""),
            "suggested_task_name": suggested_task_name,
            "task_name": existing.get("task_name", "") or suggested_task_name,
            "task_id": existing.get("task_id", ""),
            "task_url": existing.get("task_url", ""),
            "task_status": existing.get("task_status", ""),
            "sync_status": existing.get("sync_status", ""),
            "label_counts": existing.get("label_counts", ""),
            "event_count": existing.get("event_count", ""),
            "box_count": existing.get("box_count", ""),
            "last_sync_time": existing.get("last_sync_time", ""),
            "notes": existing.get("notes", ""),
        }
        rows.append(row)
    return rows


def command_build_registry(args: argparse.Namespace) -> None:
    stitched_manifest = Path(args.stitched_manifest).resolve()
    processed_manifest = Path(args.processed_manifest).resolve() if args.processed_manifest else None
    registry_path = Path(args.registry_out).resolve() if args.registry_out else registry_default_path(stitched_manifest)

    scene_filter = set(args.scene or []) or None
    sequence_filter = set(args.sequence_dir or []) or None

    bundles = [
        bundle
        for bundle in load_sequence_bundles(
            stitched_manifest=stitched_manifest,
            processed_manifest=processed_manifest,
            scene_order=args.scene_order or DEFAULT_SCENE_ORDER,
        )
        if include_bundle(
            bundle=bundle,
            scenes=scene_filter,
            sequences=sequence_filter,
            keep_t2_only=args.keep_t2_only,
        )
    ]

    existing_index = load_registry_index(registry_path)
    registry_rows = build_registry_rows(bundles, existing_index)
    save_registry_rows(registry_path, registry_rows)

    print(f"stitched_manifest: {stitched_manifest}")
    if processed_manifest:
        print(f"processed_manifest: {processed_manifest}")
    print(f"registry_out: {registry_path}")
    print(f"sequence_count: {len(registry_rows)}")


def command_create_cvat_tasks(args: argparse.Namespace) -> None:
    stitched_manifest = Path(args.stitched_manifest).resolve()
    processed_manifest = Path(args.processed_manifest).resolve() if args.processed_manifest else None
    registry_path = Path(args.registry).resolve() if args.registry else registry_default_path(stitched_manifest)

    bundles = load_sequence_bundles(
        stitched_manifest=stitched_manifest,
        processed_manifest=processed_manifest,
        scene_order=args.scene_order or DEFAULT_SCENE_ORDER,
    )
    scene_filter = set(args.scene or []) or None
    sequence_filter = set(args.sequence_dir or []) or None
    bundles = [
        bundle
        for bundle in bundles
        if include_bundle(bundle, scene_filter, sequence_filter, args.keep_t2_only)
    ]
    registry_rows = build_registry_rows(bundles, load_registry_index(registry_path))
    registry_by_sequence = {row["sequence_dir_name"]: row for row in registry_rows}

    client = CvatClient(
        host=args.cvat_host,
        username=args.username,
        password=args.password,
        timeout_s=args.timeout_s,
    )
    existing_tasks = {task["name"]: task for task in client.list_tasks()}

    created_count = 0
    reused_count = 0
    for bundle in bundles:
        registry_row = registry_by_sequence[bundle.sequence_dir_name]
        task_name = registry_row["task_name"] or registry_row["suggested_task_name"]
        image_paths = [Path(row["stitched_image_path"]).resolve() for row in bundle.rows]
        if not all(path.exists() for path in image_paths):
            missing = [str(path) for path in image_paths if not path.exists()]
            raise FileNotFoundError(
                f"Missing stitched images for {bundle.sequence_dir_name}: {missing[:5]}"
            )

        existing_task = existing_tasks.get(task_name)
        if existing_task and not args.overwrite_existing:
            registry_row["task_id"] = str(existing_task["id"])
            registry_row["task_url"] = existing_task["url"]
            registry_row["task_status"] = existing_task.get("status", "")
            reused_count += 1
            continue

        if args.dry_run:
            print(
                f"[dry-run] would create task {task_name} for {bundle.sequence_dir_name} "
                f"with {len(image_paths)} frames"
            )
            continue

        if existing_task and args.overwrite_existing:
            client.delete_task(int(existing_task["id"]))

        created_task = client.create_task(task_name, CVAT_LABEL_SPECS)
        task_id = int(created_task["id"])
        try:
            client.upload_task_images(
                task_id=task_id,
                image_paths=image_paths,
                image_quality=args.image_quality,
                bulk_max_mb=args.bulk_max_mb,
                bulk_max_files=args.bulk_max_files,
            )
            data_meta = client.get_task_data_meta(task_id)
            if int(data_meta.get("size", 0)) != len(image_paths):
                raise RuntimeError(
                    f"CVAT uploaded frame count mismatch for {task_name}: "
                    f"{data_meta.get('size')} vs expected {len(image_paths)}"
                )
        except Exception:
            client.delete_task(task_id)
            raise

        registry_row["task_name"] = task_name
        registry_row["task_id"] = str(task_id)
        registry_row["task_url"] = created_task["url"]
        registry_row["task_status"] = created_task.get("status", "annotation")
        created_count += 1

    save_registry_rows(registry_path, registry_rows)

    print(f"registry_out: {registry_path}")
    print(f"created_tasks: {created_count}")
    print(f"reused_tasks: {reused_count}")
    print(f"selected_sequences: {len(bundles)}")


def label_id_to_name_map(labels: list[dict[str, Any]]) -> dict[int, str]:
    return {int(label["id"]): label["name"] for label in labels}


def normalize_box(points: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(points) < 4:
        raise ValueError(f"Rectangle points must contain at least 4 numbers, got {points}")
    x_vals = [float(points[0]), float(points[2])]
    y_vals = [float(points[1]), float(points[3])]
    x1, x2 = min(x_vals), max(x_vals)
    y1, y2 = min(y_vals), max(y_vals)
    x1_i = int(round(max(0.0, min(x1, width - 1)))) if width > 0 else int(round(x1))
    x2_i = int(round(max(0.0, min(x2, width - 1)))) if width > 0 else int(round(x2))
    y1_i = int(round(max(0.0, min(y1, height - 1)))) if height > 0 else int(round(y1))
    y2_i = int(round(max(0.0, min(y2, height - 1)))) if height > 0 else int(round(y2))
    return x1_i, y1_i, x2_i, y2_i


def convert_task_annotations_to_gt(
    *,
    bundle: SequenceBundle,
    task: dict[str, Any],
    labels: list[dict[str, Any]],
    data_meta: dict[str, Any],
    annotations: dict[str, Any],
    annotator: str,
    review_status: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str], list[str], dict[str, str]]:
    id_to_name = label_id_to_name_map(labels)
    manifest_by_name = {
        Path(row["stitched_image_path"]).name: row for row in bundle.rows
    }
    frame_names = [frame_meta["name"] for frame_meta in data_meta.get("frames", [])]
    errors: list[str] = []
    warnings: list[str] = []
    if len(frame_names) != len(bundle.rows):
        errors.append(
            f"[{bundle.sequence_dir_name}] CVAT frame count {len(frame_names)} does not match "
            f"stitched manifest rows {len(bundle.rows)}"
        )

    frame_rows: list[dict[str, Any]] = []
    for frame_index, frame_meta in enumerate(data_meta.get("frames", [])):
        frame_name = frame_meta["name"]
        manifest_row = manifest_by_name.get(frame_name)
        if manifest_row is None:
            errors.append(
                f"[{bundle.sequence_dir_name}] CVAT frame '{frame_name}' is not present in stitched manifest"
            )
            continue
        frame_rows.append(
            {
                "frame_index": frame_index,
                "frame_name": frame_name,
                "width": int(frame_meta.get("width", 0) or 0),
                "height": int(frame_meta.get("height", 0) or 0),
                "manifest_row": manifest_row,
                "label": "",
                "boxes": [],
                "event_id": "",
            }
        )

    tags_by_frame: dict[int, set[str]] = defaultdict(set)
    for tag in annotations.get("tags", []):
        frame_index = int(tag["frame"])
        label_name = id_to_name.get(int(tag["label_id"]))
        if label_name in FRAME_LABEL_VALUES:
            tags_by_frame[frame_index].add(label_name)
        else:
            warnings.append(
                f"[{bundle.sequence_dir_name}] ignored unknown tag label_id={tag['label_id']}"
            )

    boxes_by_frame: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for shape in annotations.get("shapes", []):
        frame_index = int(shape["frame"])
        label_name = id_to_name.get(int(shape["label_id"]))
        if label_name != "collision_area":
            warnings.append(
                f"[{bundle.sequence_dir_name}] ignored non-collision shape label_id={shape['label_id']}"
            )
            continue
        if shape.get("type") != "rectangle":
            errors.append(
                f"[{bundle.sequence_dir_name}] only rectangle collision_area is supported, got {shape.get('type')}"
            )
            continue
        if frame_index >= len(frame_rows):
            errors.append(
                f"[{bundle.sequence_dir_name}] shape references missing frame index {frame_index}"
            )
            continue
        width = frame_rows[frame_index]["width"]
        height = frame_rows[frame_index]["height"]
        boxes_by_frame[frame_index].append(normalize_box(shape["points"], width, height))

    if annotations.get("tracks"):
        errors.append(
            f"[{bundle.sequence_dir_name}] tracks are not allowed for this workflow; only tags and shapes are supported"
        )

    for frame_row in frame_rows:
        frame_index = frame_row["frame_index"]
        labels_for_frame = sorted(tags_by_frame.get(frame_index, set()))
        boxes = sorted(boxes_by_frame.get(frame_index, []))
        if len(labels_for_frame) != 1:
            errors.append(
                f"[{bundle.sequence_dir_name}] frame {frame_row['frame_name']} has invalid tags: {labels_for_frame}"
            )
            continue
        frame_row["label"] = labels_for_frame[0]
        frame_row["boxes"] = boxes

    current_event_id = ""
    event_counter = 0
    for frame_row in frame_rows:
        label = frame_row["label"]
        if label == "pos":
            if not current_event_id:
                event_counter += 1
                current_event_id = f"{bundle.sequence_dir_name}_evt{event_counter:02d}"
            frame_row["event_id"] = current_event_id
        else:
            current_event_id = ""
            frame_row["event_id"] = ""

        box_count = len(frame_row["boxes"])
        if label == "pos" and box_count < 1:
            errors.append(
                f"[{bundle.sequence_dir_name}] positive frame {frame_row['frame_name']} does not contain any collision_area box"
            )
        if label != "pos" and box_count > 0:
            errors.append(
                f"[{bundle.sequence_dir_name}] non-positive frame {frame_row['frame_name']} contains collision_area boxes"
            )
        if label == "pos" and box_count > 2:
            errors.append(
                f"[{bundle.sequence_dir_name}] positive frame {frame_row['frame_name']} has {box_count} boxes; v1 expects 1 or 2"
            )

    gt_frame_rows: list[dict[str, str]] = []
    gt_box_rows: list[dict[str, str]] = []
    for frame_row in frame_rows:
        manifest_row = frame_row["manifest_row"]
        gt_frame_rows.append(
            {
                "sequence_dir_name": manifest_row["sequence_dir_name"],
                "bag_name": manifest_row["bag_name"],
                "group_id": manifest_row["group_id"],
                "ros_msg_seq": manifest_row["ros_msg_seq"],
                "timestamp_us": manifest_row["timestamp_us"],
                "stitched_image_path": manifest_row["stitched_image_path"],
                "frame_label": frame_row["label"],
                "event_id": frame_row["event_id"],
                "annotator": annotator,
                "review_status": review_status,
                "notes": "",
            }
        )
        if frame_row["label"] == "pos":
            for box_index, (x1, y1, x2, y2) in enumerate(sorted(frame_row["boxes"]), start=1):
                gt_box_rows.append(
                    {
                        "sequence_dir_name": manifest_row["sequence_dir_name"],
                        "group_id": manifest_row["group_id"],
                        "event_id": frame_row["event_id"],
                        "target_id": "1",
                        "box_id": str(box_index),
                        "x1": str(x1),
                        "y1": str(y1),
                        "x2": str(x2),
                        "y2": str(y2),
                    }
                )

    label_counter = defaultdict(int)
    for row in gt_frame_rows:
        label_counter[row["frame_label"]] += 1
    summary = {
        "scene": bundle.meta.get("scene", ""),
        "sequence_dir_name": bundle.sequence_dir_name,
        "bag_name": bundle.meta.get("bag_name", ""),
        "task_id": str(task["id"]),
        "task_name": task["name"],
        "frame_count": str(len(gt_frame_rows)),
        "pos_frames": str(label_counter.get("pos", 0)),
        "neg_frames": str(label_counter.get("neg", 0)),
        "ignore_frames": str(label_counter.get("ignore", 0)),
        "pos_events": str(event_counter),
        "box_count": str(len(gt_box_rows)),
    }
    return gt_frame_rows, gt_box_rows, errors, warnings, summary


def load_existing_gt(
    gt_root: Path,
    stitched_groups: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    frame_labels_path = gt_root / "frame_labels.csv"
    collision_boxes_path = gt_root / "collision_boxes.csv"
    if frame_labels_path.exists():
        frame_rows = load_csv_rows(frame_labels_path)
    else:
        frame_rows = build_empty_frame_rows(stitched_groups)
    if collision_boxes_path.exists():
        box_rows = load_csv_rows(collision_boxes_path)
    else:
        box_rows = []
    return frame_rows, box_rows


def manifest_order_index(stitched_groups: dict[str, list[dict[str, str]]]) -> dict[tuple[str, str], int]:
    order: dict[tuple[str, str], int] = {}
    cursor = 0
    for sequence_dir_name in sorted(stitched_groups.keys()):
        for row in stitched_groups[sequence_dir_name]:
            order[(row["sequence_dir_name"], row["group_id"])] = cursor
            cursor += 1
    return order


def save_raw_task_dump(
    export_root: Path,
    bundle: SequenceBundle,
    task: dict[str, Any],
    data_meta: dict[str, Any],
    labels: list[dict[str, Any]],
    annotations: dict[str, Any],
) -> None:
    task_root = ensure_dir(export_root / bundle.sequence_dir_name)
    (task_root / "task.json").write_text(
        json.dumps(task, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (task_root / "data_meta.json").write_text(
        json.dumps(data_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (task_root / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (task_root / "annotations.json").write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_task_mapping(
    *,
    client: CvatClient,
    bundles: list[SequenceBundle],
    registry_path: Path | None,
) -> dict[str, dict[str, str]]:
    registry_index = load_registry_index(registry_path)
    tasks_by_name = {task["name"]: task for task in client.list_tasks()}
    mapping: dict[str, dict[str, str]] = {}
    for bundle in bundles:
        registry_row = registry_index.get(bundle.sequence_dir_name, {})
        task_id = registry_row.get("task_id", "").strip()
        task_name = registry_row.get("task_name", "").strip()
        if task_id:
            mapping[bundle.sequence_dir_name] = {
                "task_id": task_id,
                "task_name": task_name or f"{bundle.meta.get('scene', '')}_{bundle.sequence_dir_name}",
            }
            continue
        fallback_name = task_name or f"{bundle.meta.get('scene', '')}_{bundle.sequence_dir_name}"
        if fallback_name in tasks_by_name:
            mapping[bundle.sequence_dir_name] = {
                "task_id": str(tasks_by_name[fallback_name]["id"]),
                "task_name": fallback_name,
            }
    return mapping


def command_sync_cvat_to_gt(args: argparse.Namespace) -> None:
    stitched_manifest = Path(args.stitched_manifest).resolve()
    processed_manifest = Path(args.processed_manifest).resolve() if args.processed_manifest else None
    registry_path = Path(args.registry).resolve() if args.registry else registry_default_path(stitched_manifest)
    export_root = Path(args.export_root).resolve() if args.export_root else raw_export_default_root(stitched_manifest)
    gt_root = Path(args.gt_root).resolve() if args.gt_root else gt_default_root(stitched_manifest)

    bundles = load_sequence_bundles(
        stitched_manifest=stitched_manifest,
        processed_manifest=processed_manifest,
        scene_order=args.scene_order or DEFAULT_SCENE_ORDER,
    )
    scene_filter = set(args.scene or []) or None
    sequence_filter = set(args.sequence_dir or []) or None
    bundles = [
        bundle
        for bundle in bundles
        if include_bundle(bundle, scene_filter, sequence_filter, args.keep_t2_only)
    ]

    stitched_groups = load_stitched_groups(stitched_manifest)
    existing_frame_rows, existing_box_rows = load_existing_gt(gt_root, stitched_groups)
    existing_frame_map = {
        (row["sequence_dir_name"], row["group_id"]): row for row in existing_frame_rows
    }
    manifest_index = manifest_order_index(stitched_groups)

    client = CvatClient(
        host=args.cvat_host,
        username=args.username,
        password=args.password,
        timeout_s=args.timeout_s,
    )
    task_mapping = resolve_task_mapping(client=client, bundles=bundles, registry_path=registry_path)
    registry_rows = build_registry_rows(bundles, load_registry_index(registry_path))
    registry_by_sequence = {row["sequence_dir_name"]: row for row in registry_rows}

    new_sequence_frames: dict[str, list[dict[str, str]]] = {}
    new_sequence_boxes: dict[str, list[dict[str, str]]] = {}
    sequence_summaries: list[dict[str, str]] = []
    report_payload = {
        "generated_at": now_iso(),
        "stitched_manifest": str(stitched_manifest),
        "gt_root": str(gt_root),
        "export_root": str(export_root),
        "sequences": [],
    }
    invalid_sequences: list[str] = []

    for bundle in bundles:
        registry_row = registry_by_sequence[bundle.sequence_dir_name]
        task_info = task_mapping.get(bundle.sequence_dir_name)
        if not task_info:
            notes = "CVAT task not found"
            registry_row["sync_status"] = "未找到 task"
            registry_row["notes"] = notes
            sequence_summaries.append(
                {
                    "scene": bundle.meta.get("scene", ""),
                    "sequence_dir_name": bundle.sequence_dir_name,
                    "bag_name": bundle.meta.get("bag_name", ""),
                    "task_id": "",
                    "task_name": registry_row.get("task_name", ""),
                    "frame_count": bundle.meta.get("frame_count", ""),
                    "pos_frames": "",
                    "neg_frames": "",
                    "ignore_frames": "",
                    "pos_events": "",
                    "box_count": "",
                    "status": "task_missing",
                    "error_count": "1",
                    "warning_count": "0",
                    "notes": notes,
                }
            )
            invalid_sequences.append(bundle.sequence_dir_name)
            continue

        task_id = int(task_info["task_id"])
        task = client.get_task(task_id)
        labels = client.get_task_labels(task_id)
        data_meta = client.get_task_data_meta(task_id)
        annotations = client.get_task_annotations(task_id)
        save_raw_task_dump(export_root, bundle, task, data_meta, labels, annotations)

        frame_rows, box_rows, errors, warnings, summary = convert_task_annotations_to_gt(
            bundle=bundle,
            task=task,
            labels=labels,
            data_meta=data_meta,
            annotations=annotations,
            annotator=args.annotator,
            review_status=args.review_status,
        )
        report_payload["sequences"].append(
            {
                "sequence_dir_name": bundle.sequence_dir_name,
                "task_id": task_id,
                "task_name": task["name"],
                "errors": errors,
                "warnings": warnings,
                "summary": summary,
            }
        )

        notes = "; ".join(errors[:3] + warnings[:2])
        sequence_summaries.append(
            {
                **summary,
                "status": "ok" if not errors else "invalid",
                "error_count": str(len(errors)),
                "warning_count": str(len(warnings)),
                "notes": notes,
            }
        )

        if errors:
            registry_row["sync_status"] = "同步失败"
            registry_row["notes"] = notes
            invalid_sequences.append(bundle.sequence_dir_name)
            if args.strict:
                continue

        new_sequence_frames[bundle.sequence_dir_name] = frame_rows
        new_sequence_boxes[bundle.sequence_dir_name] = box_rows
        registry_row["task_id"] = str(task_id)
        registry_row["task_name"] = task["name"]
        registry_row["task_url"] = task["url"]
        registry_row["task_status"] = task.get("status", "")
        registry_row["sync_status"] = "已同步" if not errors else "部分同步"
        registry_row["label_counts"] = (
            f"{summary['pos_frames']}/{summary['neg_frames']}/{summary['ignore_frames']}"
        )
        registry_row["event_count"] = summary["pos_events"]
        registry_row["box_count"] = summary["box_count"]
        registry_row["last_sync_time"] = now_iso()
        registry_row["notes"] = notes

    merged_frame_map = dict(existing_frame_map)
    for sequence_dir_name, rows in new_sequence_frames.items():
        for row in rows:
            merged_frame_map[(row["sequence_dir_name"], row["group_id"])] = row

    merged_box_rows = [
        row for row in existing_box_rows if row["sequence_dir_name"] not in new_sequence_boxes
    ]
    for sequence_dir_name in sorted(new_sequence_boxes.keys()):
        merged_box_rows.extend(new_sequence_boxes[sequence_dir_name])

    ordered_frame_keys = sorted(
        merged_frame_map.keys(),
        key=lambda key: manifest_index.get(key, 10**9),
    )
    ordered_frame_rows = [merged_frame_map[key] for key in ordered_frame_keys]
    ordered_box_rows = sorted(
        merged_box_rows,
        key=lambda row: (
            manifest_index.get((row["sequence_dir_name"], row["group_id"]), 10**9),
            coerce_int(row.get("box_id"), 0) or 0,
        ),
    )

    write_csv_rows(gt_root / "frame_labels.csv", FRAME_LABEL_HEADERS, ordered_frame_rows)
    write_csv_rows(gt_root / "collision_boxes.csv", COLLISION_BOX_HEADERS, ordered_box_rows)
    write_csv_rows(export_root / "gt_sync_summary.csv", SEQUENCE_SYNC_HEADERS, sequence_summaries)
    (export_root / "gt_sync_report.json").write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_registry_rows(registry_path, registry_rows)

    print(f"gt_root: {gt_root}")
    print(f"export_root: {export_root}")
    print(f"synced_sequences: {len(new_sequence_frames)}")
    print(f"invalid_sequences: {len(invalid_sequences)}")
    if args.strict and invalid_sequences:
        raise SystemExit(
            f"Strict sync detected invalid sequences: {', '.join(sorted(invalid_sequences))}"
        )


def load_frame_boxes(box_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[tuple[int, int, int, int]]]:
    mapping: dict[tuple[str, str], list[tuple[int, int, int, int]]] = defaultdict(list)
    for row in box_rows:
        mapping[(row["sequence_dir_name"], row["group_id"])].append(
            (
                int(row["x1"]),
                int(row["y1"]),
                int(row["x2"]),
                int(row["y2"]),
            )
        )
    for key in mapping:
        mapping[key].sort()
    return mapping


def point_in_boxes(x: float, y: float, boxes: list[tuple[int, int, int, int]]) -> bool:
    for x1, y1, x2, y2 in boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def scene_map_from_manifest(stitched_manifest: Path) -> dict[tuple[str, str], str]:
    scene_map: dict[tuple[str, str], str] = {}
    for rows in load_stitched_groups(stitched_manifest).values():
        for row in rows:
            scene_map[(row["sequence_dir_name"], row["group_id"])] = row.get("scene", "")
    return scene_map


def build_gt_frame_eval_records(
    frame_rows: list[dict[str, str]],
    box_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    scene_lookup: dict[tuple[str, str], str],
    selected_scenes: set[str] | None,
    selected_sequences: set[str] | None,
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    list[dict[str, str]],
]:
    boxes_by_frame = load_frame_boxes(box_rows)

    gt_key_by_bag_and_seq: dict[tuple[str, str], tuple[str, str]] = {}
    gt_rows_by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        key = (row["bag_name"], row["ros_msg_seq"])
        gt_key_by_bag_and_seq[key] = (row["sequence_dir_name"], row["group_id"])
        gt_rows_by_sequence[row["sequence_dir_name"]].append(row)

    prediction_by_frame: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows": [],
            "valid_points": [],
            "invalid_rows": 0,
        }
    )
    for pred_row in prediction_rows:
        bag_name = pred_row.get("bag_name", "")
        ros_msg_seq = pred_row.get("ros_msg_seq_curr", "")
        gt_key = gt_key_by_bag_and_seq.get((bag_name, ros_msg_seq))
        if gt_key is None:
            continue
        entry = prediction_by_frame[gt_key]
        entry["rows"].append(pred_row)
        if not is_truthy(pred_row.get("has_detection")):
            continue
        if not is_truthy(pred_row.get("mapping_valid")):
            entry["invalid_rows"] += 1
            continue
        x_value = coerce_float(pred_row.get("x_stitched_orig"))
        y_value = coerce_float(pred_row.get("y_stitched_orig"))
        if x_value is None or y_value is None:
            entry["invalid_rows"] += 1
            continue
        entry["valid_points"].append(
            {
                "x": x_value,
                "y": y_value,
                "ttc_s": coerce_float(pred_row.get("ttc_s")),
            }
        )

    frame_eval_rows: list[dict[str, str]] = []
    sequence_stats: dict[str, dict[str, Any]] = {}
    event_rows: list[dict[str, str]] = []

    for sequence_dir_name, rows in gt_rows_by_sequence.items():
        rows.sort(key=lambda row: int(row["group_id"]))
        if selected_sequences and sequence_dir_name not in selected_sequences:
            continue
        scene_name = scene_lookup.get((sequence_dir_name, rows[0]["group_id"]), "")
        if selected_scenes and scene_name not in selected_scenes:
            continue

        stats = {
            "scene": scene_name,
            "sequence_dir_name": sequence_dir_name,
            "bag_name": rows[0]["bag_name"],
            "seq_count": 1,
            "valid_frames": 0,
            "pos_frames": 0,
            "neg_frames": 0,
            "ignore_frames": 0,
            "pos_events": 0,
            "tp": 0,
            "fn": 0,
            "fa": 0,
            "missing_prediction_frames": 0,
            "invalid_prediction_rows": 0,
        }
        sequence_stats[sequence_dir_name] = stats

        current_event: dict[str, Any] | None = None

        def finalize_current_event() -> None:
            nonlocal current_event
            if current_event is None:
                return
            event_rows.append(
                {
                    "scene": current_event["scene"],
                    "sequence_dir_name": sequence_dir_name,
                    "event_id": current_event["event_id"],
                    "event_detected": "1" if current_event["detected"] else "0",
                    "first_tp_offset_frames": ""
                    if current_event["first_tp_offset_frames"] is None
                    else str(current_event["first_tp_offset_frames"]),
                    "first_tp_offset_s": ""
                    if current_event["first_tp_offset_s"] is None
                    else f"{current_event['first_tp_offset_s']:.6f}",
                    "first_tp_ttc_s": ""
                    if current_event["first_tp_ttc_s"] is None
                    else f"{current_event['first_tp_ttc_s']:.6f}",
                    "start_group_id": current_event["start_group_id"],
                }
            )
            current_event = None

        for row in rows:
            key = (sequence_dir_name, row["group_id"])
            scene_name = scene_lookup.get(key, scene_name)
            label = row["frame_label"]
            pred_entry = prediction_by_frame.get(key, {"valid_points": [], "invalid_rows": 0, "rows": []})
            valid_points = pred_entry["valid_points"]
            invalid_rows = pred_entry["invalid_rows"]
            if not pred_entry["rows"]:
                stats["missing_prediction_frames"] += 1
            stats["invalid_prediction_rows"] += invalid_rows

            matched_points = []
            boxes = boxes_by_frame.get(key, [])
            for point in valid_points:
                if point_in_boxes(point["x"], point["y"], boxes):
                    matched_points.append(point)
            matched_ttc_values = [
                point["ttc_s"] for point in matched_points if point["ttc_s"] is not None
            ]
            first_matched_ttc_s = max(matched_ttc_values) if matched_ttc_values else None

            prediction_status = "IGNORED"
            if label == "ignore":
                finalize_current_event()
                stats["ignore_frames"] += 1
            elif label == "pos":
                stats["valid_frames"] += 1
                stats["pos_frames"] += 1
                event_id = row["event_id"]
                if current_event is None or event_id != current_event["event_id"]:
                    finalize_current_event()
                    stats["pos_events"] += 1
                    current_event = {
                        "scene": scene_name,
                        "event_id": event_id,
                        "start_group_id": row["group_id"],
                        "start_timestamp_us": int(row["timestamp_us"]),
                        "detected": False,
                        "first_tp_offset_frames": None,
                        "first_tp_offset_s": None,
                        "first_tp_ttc_s": None,
                        "pos_frame_offset": 0,
                    }

                if matched_points:
                    prediction_status = "TP"
                    stats["tp"] += 1
                    if current_event is not None and not current_event["detected"]:
                        current_event["detected"] = True
                        current_event["first_tp_offset_frames"] = current_event["pos_frame_offset"]
                        current_event["first_tp_offset_s"] = (
                            int(row["timestamp_us"]) - current_event["start_timestamp_us"]
                        ) / 1_000_000.0
                        current_event["first_tp_ttc_s"] = first_matched_ttc_s
                else:
                    prediction_status = "FN"
                    stats["fn"] += 1
                if current_event is not None:
                    current_event["pos_frame_offset"] += 1
            elif label == "neg":
                finalize_current_event()
                stats["valid_frames"] += 1
                stats["neg_frames"] += 1
                if valid_points:
                    prediction_status = "FA"
                    stats["fa"] += 1
                else:
                    prediction_status = "TN"
            else:
                finalize_current_event()
                prediction_status = "UNLABELED"

            frame_eval_rows.append(
                {
                    "scene": scene_name,
                    "sequence_dir_name": sequence_dir_name,
                    "bag_name": row["bag_name"],
                    "group_id": row["group_id"],
                    "ros_msg_seq": row["ros_msg_seq"],
                    "timestamp_us": row["timestamp_us"],
                    "frame_label": label,
                    "event_id": row["event_id"],
                    "prediction_status": prediction_status,
                    "matched_detection_count": str(len(matched_points)),
                    "valid_detection_count": str(len(valid_points)),
                    "first_matched_ttc_s": ""
                    if first_matched_ttc_s is None
                    else format_optional_float(first_matched_ttc_s),
                    "stitched_image_path": row["stitched_image_path"],
                }
            )

        finalize_current_event()

    return frame_eval_rows, sequence_stats, event_rows


def append_metric_fields(stats: dict[str, Any], event_rows: list[dict[str, str]]) -> dict[str, Any]:
    pos_frames = int(stats["pos_frames"])
    neg_frames = int(stats["neg_frames"])
    pos_events = int(stats["pos_events"])
    tp = int(stats["tp"])
    fn = int(stats["fn"])
    fa = int(stats["fa"])

    detected_event_rows = [row for row in event_rows if row["event_detected"] == "1"]
    avg_offset_frames = None
    avg_offset_s = None
    avg_ttc_s = None
    if detected_event_rows:
        offset_frames_values = [
            coerce_float(row["first_tp_offset_frames"])
            for row in detected_event_rows
            if coerce_float(row["first_tp_offset_frames"]) is not None
        ]
        offset_s_values = [
            coerce_float(row["first_tp_offset_s"])
            for row in detected_event_rows
            if coerce_float(row["first_tp_offset_s"]) is not None
        ]
        ttc_values = [
            coerce_float(row["first_tp_ttc_s"])
            for row in detected_event_rows
            if coerce_float(row["first_tp_ttc_s"]) is not None
        ]
        avg_offset_frames = (
            sum(offset_frames_values) / len(offset_frames_values) if offset_frames_values else None
        )
        avg_offset_s = sum(offset_s_values) / len(offset_s_values) if offset_s_values else None
        avg_ttc_s = sum(ttc_values) / len(ttc_values) if ttc_values else None

    return {
        **stats,
        "recall": None if pos_frames <= 0 else tp / pos_frames,
        "far": None if neg_frames <= 0 else fa / neg_frames,
        "event_recall": None if pos_events <= 0 else len(detected_event_rows) / pos_events,
        "avg_first_tp_offset_frames": avg_offset_frames,
        "avg_first_tp_offset_s": avg_offset_s,
        "avg_first_tp_ttc_s": avg_ttc_s,
    }


def aggregate_scene_metrics(
    sequence_stats: dict[str, dict[str, Any]],
    event_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_rows_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in event_rows:
        event_rows_by_scene[row["scene"]].append(row)

    scene_metrics: list[dict[str, Any]] = []
    scene_names = sorted(
        {stats["scene"] for stats in sequence_stats.values()},
        key=lambda scene: scene_sort_key(scene, DEFAULT_SCENE_ORDER),
    )
    for scene_name in scene_names:
        stats_list = [stats for stats in sequence_stats.values() if stats["scene"] == scene_name]
        aggregated = {
            "scene": scene_name,
            "seq_count": sum(int(stats["seq_count"]) for stats in stats_list),
            "valid_frames": sum(int(stats["valid_frames"]) for stats in stats_list),
            "pos_frames": sum(int(stats["pos_frames"]) for stats in stats_list),
            "neg_frames": sum(int(stats["neg_frames"]) for stats in stats_list),
            "ignore_frames": sum(int(stats["ignore_frames"]) for stats in stats_list),
            "pos_events": sum(int(stats["pos_events"]) for stats in stats_list),
            "tp": sum(int(stats["tp"]) for stats in stats_list),
            "fn": sum(int(stats["fn"]) for stats in stats_list),
            "fa": sum(int(stats["fa"]) for stats in stats_list),
            "missing_prediction_frames": sum(int(stats["missing_prediction_frames"]) for stats in stats_list),
            "invalid_prediction_rows": sum(int(stats["invalid_prediction_rows"]) for stats in stats_list),
        }
        scene_metrics.append(append_metric_fields(aggregated, event_rows_by_scene.get(scene_name, [])))

    overall_events = event_rows
    overall_base = {
        "scene": "Overall S1-S7",
        "seq_count": sum(int(stats["seq_count"]) for stats in sequence_stats.values()),
        "valid_frames": sum(int(stats["valid_frames"]) for stats in sequence_stats.values()),
        "pos_frames": sum(int(stats["pos_frames"]) for stats in sequence_stats.values()),
        "neg_frames": sum(int(stats["neg_frames"]) for stats in sequence_stats.values()),
        "ignore_frames": sum(int(stats["ignore_frames"]) for stats in sequence_stats.values()),
        "pos_events": sum(int(stats["pos_events"]) for stats in sequence_stats.values()),
        "tp": sum(int(stats["tp"]) for stats in sequence_stats.values()),
        "fn": sum(int(stats["fn"]) for stats in sequence_stats.values()),
        "fa": sum(int(stats["fa"]) for stats in sequence_stats.values()),
        "missing_prediction_frames": sum(
            int(stats["missing_prediction_frames"]) for stats in sequence_stats.values()
        ),
        "invalid_prediction_rows": sum(
            int(stats["invalid_prediction_rows"]) for stats in sequence_stats.values()
        ),
    }
    overall_metric = append_metric_fields(overall_base, overall_events)
    return scene_metrics, overall_metric


def metric_rows_to_csv_ready(rows: list[dict[str, Any]], headers: list[str]) -> list[dict[str, str]]:
    csv_rows: list[dict[str, str]] = []
    for row in rows:
        csv_row: dict[str, str] = {}
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                csv_row[header] = f"{value:.6f}"
            elif value is None:
                csv_row[header] = ""
            else:
                csv_row[header] = str(value)
        csv_rows.append(csv_row)
    return csv_rows


def render_scene_metrics_markdown(scene_metrics: list[dict[str, Any]], overall_metric: dict[str, Any]) -> str:
    lines = [
        "| Scene | #Seqs | #Valid Frames | #Pos Frames | #Neg Frames | #Ignore Frames | #Pos Events | TP | FN | FA | Recall | FAR | Event Recall | Avg. Lead Time | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in scene_metrics + [overall_metric]:
        pos_events = int(row["pos_events"])
        avg_lead = format_optional_float(row.get("avg_first_tp_ttc_s"))
        notes = ""
        if row["scene"] in NEGATIVE_SCENES and pos_events == 0:
            notes = "负样本场景重点看 FAR"
        lines.append(
            "| {scene} | {seq_count} | {valid_frames} | {pos_frames} | {neg_frames} | {ignore_frames} | "
            "{pos_events} | {tp_display} | {fn_display} | {fa} | {recall} | {far} | {event_recall} | {avg_lead} | {notes} |".format(
                scene=f"`{row['scene']}`",
                seq_count=row["seq_count"],
                valid_frames=row["valid_frames"],
                pos_frames=row["pos_frames"],
                neg_frames=row["neg_frames"],
                ignore_frames=row["ignore_frames"],
                pos_events=row["pos_events"],
                tp_display="N/A" if pos_events == 0 else row["tp"],
                fn_display="N/A" if pos_events == 0 else row["fn"],
                fa=row["fa"],
                recall=format_optional_float(row.get("recall")),
                far=format_optional_float(row.get("far")),
                event_recall=format_optional_float(row.get("event_recall")),
                avg_lead=avg_lead,
                notes=notes,
            )
        )
    return "\n".join(lines) + "\n"


def render_paper_fill_line(method_name: str, overall_metric: dict[str, Any], scenes: list[str]) -> str:
    scenario_text = "/".join(scenes) if scenes else "S1-S7"
    return (
        f"| {method_name} | {format_optional_float(overall_metric.get('recall'))} | "
        f"{format_optional_float(overall_metric.get('far'))} | "
        f"{format_optional_float(overall_metric.get('event_recall'))} | "
        f"`{overall_metric['valid_frames']} ({overall_metric['pos_frames']}/{overall_metric['neg_frames']})` | "
        f"`{overall_metric['seq_count']}` | `{scenario_text}` |\n"
    )


def command_evaluate_predictions(args: argparse.Namespace) -> None:
    stitched_manifest = Path(args.stitched_manifest).resolve()
    gt_root = Path(args.gt_root).resolve() if args.gt_root else gt_default_root(stitched_manifest)
    prediction_csv = Path(args.prediction_csv).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else ensure_dir(prediction_csv.parent / f"{prediction_csv.stem}_table2_eval")
    )
    ensure_dir(out_dir)

    frame_rows = load_csv_rows(gt_root / "frame_labels.csv")
    box_rows = load_csv_rows(gt_root / "collision_boxes.csv")
    prediction_rows = load_csv_rows(prediction_csv)
    scene_lookup = scene_map_from_manifest(stitched_manifest)

    selected_scenes = set(args.scene or []) or None
    selected_sequences = set(args.sequence_dir or []) or None

    frame_eval_rows, sequence_stats_base, event_rows = build_gt_frame_eval_records(
        frame_rows=frame_rows,
        box_rows=box_rows,
        prediction_rows=prediction_rows,
        scene_lookup=scene_lookup,
        selected_scenes=selected_scenes,
        selected_sequences=selected_sequences,
    )

    event_rows_by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in event_rows:
        event_rows_by_sequence[row["sequence_dir_name"]].append(row)

    sequence_metrics = [
        append_metric_fields(stats, event_rows_by_sequence.get(sequence_dir_name, []))
        for sequence_dir_name, stats in sorted(sequence_stats_base.items())
    ]
    scene_metrics, overall_metric = aggregate_scene_metrics(sequence_stats_base, event_rows)

    write_csv_rows(out_dir / "frame_evaluation.csv", FRAME_EVAL_HEADERS, frame_eval_rows)
    write_csv_rows(
        out_dir / "sequence_metrics.csv",
        SEQUENCE_METRIC_HEADERS,
        metric_rows_to_csv_ready(sequence_metrics, SEQUENCE_METRIC_HEADERS),
    )
    write_csv_rows(
        out_dir / "scene_metrics.csv",
        SCENE_METRIC_HEADERS,
        metric_rows_to_csv_ready(scene_metrics + [overall_metric], SCENE_METRIC_HEADERS),
    )
    write_csv_rows(
        out_dir / "event_metrics.csv",
        [
            "scene",
            "sequence_dir_name",
            "event_id",
            "event_detected",
            "first_tp_offset_frames",
            "first_tp_offset_s",
            "first_tp_ttc_s",
            "start_group_id",
        ],
        event_rows,
    )

    method_name = args.method_name or prediction_csv.stem
    scene_metric_markdown = render_scene_metrics_markdown(scene_metrics, overall_metric)
    paper_fill_line = render_paper_fill_line(
        method_name=method_name,
        overall_metric=overall_metric,
        scenes=[row["scene"] for row in scene_metrics] or ["S1-S7"],
    )
    (out_dir / "tracking_table_fill.md").write_text(scene_metric_markdown, encoding="utf-8")
    (out_dir / "paper_table2_fill.md").write_text(paper_fill_line, encoding="utf-8")
    (out_dir / "overall_metrics.json").write_text(
        json.dumps(
            {
                **overall_metric,
                "method_name": method_name,
                "prediction_csv": str(prediction_csv),
                "gt_root": str(gt_root),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"prediction_csv: {prediction_csv}")
    print(f"gt_root: {gt_root}")
    print(f"out_dir: {out_dir}")
    print(f"Recall: {format_optional_float(overall_metric.get('recall'))}")
    print(f"FAR: {format_optional_float(overall_metric.get('far'))}")
    print(f"Event Recall: {format_optional_float(overall_metric.get('event_recall'))}")
    print(f"#Frames(Pos/Neg): {overall_metric['valid_frames']} ({overall_metric['pos_frames']}/{overall_metric['neg_frames']})")
    print(f"#Seqs: {overall_metric['seq_count']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automation helpers for the SJTU real-vehicle annotation and Table 2 workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry_parser = subparsers.add_parser(
        "build-registry",
        help="Build or refresh the per-sequence annotation registry from the stitched manifest.",
    )
    registry_parser.add_argument("--stitched-manifest", required=True)
    registry_parser.add_argument("--processed-manifest", default="")
    registry_parser.add_argument("--registry-out", default="")
    registry_parser.add_argument("--scene", nargs="*", default=[])
    registry_parser.add_argument("--sequence-dir", nargs="*", default=[])
    registry_parser.add_argument("--keep-t2-only", action="store_true")
    registry_parser.add_argument("--scene-order", nargs="*", default=DEFAULT_SCENE_ORDER)
    registry_parser.set_defaults(func=command_build_registry)

    create_parser = subparsers.add_parser(
        "create-cvat-tasks",
        help="Create one CVAT task per sequence and upload stitched frames.",
    )
    create_parser.add_argument("--stitched-manifest", required=True)
    create_parser.add_argument("--processed-manifest", default="")
    create_parser.add_argument("--registry", default="")
    create_parser.add_argument("--cvat-host", default="http://localhost:8080")
    create_parser.add_argument("--username", required=True)
    create_parser.add_argument("--password", required=True)
    create_parser.add_argument("--scene", nargs="*", default=[])
    create_parser.add_argument("--sequence-dir", nargs="*", default=[])
    create_parser.add_argument("--keep-t2-only", action="store_true")
    create_parser.add_argument("--scene-order", nargs="*", default=DEFAULT_SCENE_ORDER)
    create_parser.add_argument("--image-quality", type=int, default=95)
    create_parser.add_argument("--bulk-max-mb", type=float, default=128.0)
    create_parser.add_argument("--bulk-max-files", type=int, default=200)
    create_parser.add_argument("--overwrite-existing", action="store_true")
    create_parser.add_argument("--dry-run", action="store_true")
    create_parser.add_argument("--timeout-s", type=float, default=60.0)
    create_parser.set_defaults(func=command_create_cvat_tasks)

    sync_parser = subparsers.add_parser(
        "sync-cvat-to-gt",
        help="Fetch CVAT task annotations and convert them into canonical GT CSV files.",
    )
    sync_parser.add_argument("--stitched-manifest", required=True)
    sync_parser.add_argument("--processed-manifest", default="")
    sync_parser.add_argument("--registry", default="")
    sync_parser.add_argument("--gt-root", default="")
    sync_parser.add_argument("--export-root", default="")
    sync_parser.add_argument("--cvat-host", default="http://localhost:8080")
    sync_parser.add_argument("--username", required=True)
    sync_parser.add_argument("--password", required=True)
    sync_parser.add_argument("--scene", nargs="*", default=[])
    sync_parser.add_argument("--sequence-dir", nargs="*", default=[])
    sync_parser.add_argument("--keep-t2-only", action="store_true")
    sync_parser.add_argument("--scene-order", nargs="*", default=DEFAULT_SCENE_ORDER)
    sync_parser.add_argument("--annotator", default="cvat")
    sync_parser.add_argument("--review-status", default="annotated")
    sync_parser.add_argument("--timeout-s", type=float, default=60.0)
    sync_parser.add_argument(
        "--strict",
        action="store_true",
        help="Skip invalid sequences and exit non-zero if any task violates the protocol.",
    )
    sync_parser.set_defaults(func=command_sync_cvat_to_gt)

    eval_parser = subparsers.add_parser(
        "evaluate-predictions",
        help="Evaluate collision_predictions.csv against canonical GT CSVs.",
    )
    eval_parser.add_argument("--stitched-manifest", required=True)
    eval_parser.add_argument("--gt-root", default="")
    eval_parser.add_argument("--prediction-csv", required=True)
    eval_parser.add_argument("--out-dir", default="")
    eval_parser.add_argument("--method-name", default="")
    eval_parser.add_argument("--scene", nargs="*", default=[])
    eval_parser.add_argument("--sequence-dir", nargs="*", default=[])
    eval_parser.set_defaults(func=command_evaluate_predictions)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
