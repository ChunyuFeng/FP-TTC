#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional

import cv2
import rosbag

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from tools.cyberrock.sjtu_pipeline_utils import (
    CAMERA_CHANNELS,
    REF_CHANNEL,
    TOPIC_TO_CHANNEL,
    build_sequence_dir_name,
    compute_rectify_map_and_kud,
    ensure_dir,
    load_camera_projections,
    strip_markdown_code,
)


MANIFEST_FIELDS = [
    "bag_name",
    "sequence_dir_name",
    "bag_path",
    "scene",
    "pos_neg",
    "valid_start_bag_sec",
    "valid_end_bag_sec",
    "keep_t2",
    "conditions",
    "tag",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SJTU bag manifest and process bags into raw/rectified/aligned assets."
    )
    parser.add_argument("--output_root", required=True, help="Pipeline output root directory.")
    parser.add_argument("--yaml_path", type=str, default=None, help="Camera calibration YAML path.")
    parser.add_argument(
        "--scene_info_md",
        type=str,
        default=None,
        help="scene_info.md path. When provided, the script can export a structured CSV manifest.",
    )
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=None,
        help="Structured manifest CSV path. Used as batch input after export.",
    )
    parser.add_argument(
        "--bags_root",
        type=str,
        default=None,
        help="Root directory for bags referenced by scene_info.md or manifest CSV.",
    )
    parser.add_argument(
        "--export_manifest_only",
        action="store_true",
        help="Only export manifest CSV from scene_info.md, without processing bags.",
    )
    parser.add_argument(
        "--keep_t2_only",
        action="store_true",
        help="Process only manifest rows with Keep T2 == Y.",
    )
    parser.add_argument(
        "--ref_channel",
        type=str,
        default=REF_CHANNEL,
        choices=CAMERA_CHANNELS,
        help="Reference channel used for multi-camera synchronization.",
    )
    parser.add_argument(
        "--residual_tolerance_ms",
        type=float,
        default=5.0,
        help="Maximum corrected timestamp residual allowed during synchronization.",
    )
    parser.add_argument(
        "--bag_path",
        type=str,
        default=None,
        help="Optional single-bag processing mode.",
    )
    parser.add_argument("--valid_start_bag_sec", type=float, default=None)
    parser.add_argument("--valid_end_bag_sec", type=float, default=None)
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--pos_neg", type=str, default="")
    parser.add_argument("--keep_t2", type=str, default="?")
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    return parser.parse_args()


def parse_valid_range(text: str) -> Optional[tuple[float, float]]:
    normalized = strip_markdown_code(text)
    if not normalized or normalized in {"待填", "nan"}:
        return None
    match = re.search(r"([0-9]+\.[0-9]+)\s*~\s*([0-9]+\.[0-9]+)", normalized)
    if not match:
        return None
    start = float(match.group(1))
    end = float(match.group(2))
    if end <= start:
        return None
    return start, end


def extract_tag(notes: str) -> str:
    match = re.search(r"tag=([A-Za-z0-9_\-]+)", notes)
    return match.group(1) if match else ""


def iter_bag_table_rows(scene_info_md: Path) -> Iterable[Dict[str, str]]:
    lines = scene_info_md.read_text(encoding="utf-8").splitlines()
    in_section = False
    header = None
    separator_skipped = False
    for line in lines:
        if line.startswith("## Bag 主清单"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if not separator_skipped:
            separator_skipped = True
            continue
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        bag_name = strip_markdown_code(row.get("Bag", ""))
        if not bag_name or bag_name == "Bag":
            continue
        yield row


def export_manifest_from_scene_info(scene_info_md: Path, bags_root: Path, manifest_csv: Path) -> List[Dict[str, str]]:
    rows = []
    for row in iter_bag_table_rows(scene_info_md):
        bag_name = strip_markdown_code(row.get("Bag", ""))
        valid_range = parse_valid_range(row.get("Valid Range", ""))
        if not bag_name or valid_range is None:
            continue

        notes = strip_markdown_code(row.get("Notes", ""))
        scene = strip_markdown_code(row.get("S", ""))
        tag = extract_tag(notes)
        manifest_row = {
            "bag_name": bag_name,
            "sequence_dir_name": build_sequence_dir_name(bag_name, scene, tag),
            "bag_path": str((bags_root / bag_name).resolve()),
            "scene": scene,
            "pos_neg": strip_markdown_code(row.get("Pos/Neg", "")),
            "valid_start_bag_sec": f"{valid_range[0]:.9f}",
            "valid_end_bag_sec": f"{valid_range[1]:.9f}",
            "keep_t2": strip_markdown_code(row.get("Keep T2", "")),
            "conditions": strip_markdown_code(row.get("Conditions", "")),
            "tag": tag,
            "notes": notes,
        }
        rows.append(manifest_row)

    ensure_dir(manifest_csv.parent)
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_manifest_csv(manifest_csv: Path) -> List[Dict[str, str]]:
    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if not row.get("bag_name"):
                continue
            rows.append(row)
    return rows


def compute_timestamp_us(stamp) -> int:
    return int(stamp.secs) * 1_000_000 + int(stamp.nsecs) // 1_000


def bag_output_dirs(output_root: Path, sequence_dir_name: str) -> Dict[str, Path]:
    sequence_root = output_root / "sequences" / sequence_dir_name
    return {
        "sequence_root": sequence_root,
        "raw_root": ensure_dir(sequence_root / "raw"),
        "rectified_root": ensure_dir(sequence_root / "rectified"),
        "manifests_root": ensure_dir(sequence_root / "manifests"),
        "reports_root": ensure_dir(sequence_root / "reports"),
    }


def infer_bag_time_range(bag_path: Path) -> tuple[float, float]:
    with rosbag.Bag(str(bag_path), "r") as bag:
        return float(bag.get_start_time()), float(bag.get_end_time())


def extract_raw_frames(
    bag_path: Path,
    valid_start_bag_sec: float,
    valid_end_bag_sec: float,
    raw_root: Path,
) -> Dict[str, List[dict]]:
    frames_by_channel = {ch: [] for ch in CAMERA_CHANNELS}
    with rosbag.Bag(str(bag_path), "r") as bag:
        total = sum(
            bag.get_type_and_topic_info()[1].get(topic, type("T", (), {"message_count": 0})()).message_count
            for topic in TOPIC_TO_CHANNEL
        )
        iterator = bag.read_messages(topics=list(TOPIC_TO_CHANNEL.keys()))
        for topic, msg, bag_time in tqdm(iterator, total=total, desc=f"Extract {bag_path.name}", unit="msg"):
            bag_sec = bag_time.to_sec()
            if bag_sec < valid_start_bag_sec or bag_sec > valid_end_bag_sec:
                continue

            data = getattr(msg, "data", None)
            header = getattr(msg, "header", None)
            if data is None or header is None:
                continue

            channel = TOPIC_TO_CHANNEL[topic]
            channel_dir = ensure_dir(raw_root / channel)
            header_time_us = compute_timestamp_us(header.stamp)
            bag_time_us = int(round(bag_sec * 1_000_000))
            seq = int(header.seq)
            filename = f"{channel}__seq_{seq}__{header_time_us}.jpg"
            raw_path = channel_dir / filename
            with open(raw_path, "wb") as f:
                f.write(data)

            frames_by_channel[channel].append(
                {
                    "channel": channel,
                    "seq": seq,
                    "bag_time_us": bag_time_us,
                    "header_time_us": header_time_us,
                    "raw_path": str(raw_path.resolve()),
                }
            )

    for channel in CAMERA_CHANNELS:
        frames_by_channel[channel].sort(key=lambda item: item["header_time_us"])
    return frames_by_channel


def estimate_period_us(frames_by_channel: Dict[str, List[dict]]) -> int:
    diffs = []
    for frames in frames_by_channel.values():
        diffs.extend(
            frames[i]["header_time_us"] - frames[i - 1]["header_time_us"]
            for i in range(1, len(frames))
        )
    if not diffs:
        raise RuntimeError("Failed to estimate sampling period: not enough frames.")
    return int(round(median(diffs)))


def wrap_phase_diff(diff_us: int, period_us: int) -> int:
    half = period_us / 2.0
    wrapped = ((diff_us + half) % period_us) - half
    return int(round(wrapped))


def nearest_header_diff(target_us: int, header_times_us: List[int], period_us: int) -> int:
    idx = bisect.bisect_left(header_times_us, target_us)
    candidates = []
    if idx < len(header_times_us):
        candidates.append(header_times_us[idx])
    if idx > 0:
        candidates.append(header_times_us[idx - 1])
    if not candidates:
        raise RuntimeError("No candidates available for phase offset estimation.")
    best = min(candidates, key=lambda ts: abs(ts - target_us))
    return wrap_phase_diff(best - target_us, period_us)


def estimate_phase_offsets(
    frames_by_channel: Dict[str, List[dict]],
    ref_channel: str,
    period_us: int,
) -> Dict[str, int]:
    offsets = {ref_channel: 0}
    ref_times = [frame["header_time_us"] for frame in frames_by_channel[ref_channel]]
    if not ref_times:
        raise RuntimeError(f"Reference channel {ref_channel} has no frames.")

    for channel in CAMERA_CHANNELS:
        if channel == ref_channel:
            continue
        channel_times = [frame["header_time_us"] for frame in frames_by_channel[channel]]
        if not channel_times:
            raise RuntimeError(f"Channel {channel} has no frames within valid range.")
        diffs = [nearest_header_diff(target, channel_times, period_us) for target in ref_times]
        offsets[channel] = int(round(median(diffs)))
    return offsets


def find_monotonic_match(
    corrected_times_us: List[int],
    target_us: int,
    start_idx: int,
    tolerance_us: int,
) -> tuple[Optional[int], Optional[int]]:
    idx = bisect.bisect_left(corrected_times_us, target_us, lo=start_idx)
    candidates = []
    if idx < len(corrected_times_us):
        candidates.append(idx)
    if idx > start_idx:
        candidates.append(idx - 1)
    if not candidates:
        return None, None
    best_idx = min(candidates, key=lambda cand_idx: abs(corrected_times_us[cand_idx] - target_us))
    residual_us = corrected_times_us[best_idx] - target_us
    if abs(residual_us) > tolerance_us:
        return None, None
    return best_idx, residual_us


def align_frames(
    frames_by_channel: Dict[str, List[dict]],
    ref_channel: str,
    period_us: int,
    phase_offsets_us: Dict[str, int],
    residual_tolerance_us: int,
) -> tuple[List[dict], Dict[str, dict]]:
    ref_frames = frames_by_channel[ref_channel]
    corrected_by_channel = {
        channel: [frame["header_time_us"] - phase_offsets_us[channel] for frame in frames]
        for channel, frames in frames_by_channel.items()
    }
    next_idx = {channel: 0 for channel in CAMERA_CHANNELS if channel != ref_channel}

    aligned_groups = []
    residual_stats = {channel: [] for channel in CAMERA_CHANNELS}
    for ref_frame in ref_frames:
        target_us = ref_frame["header_time_us"]
        group_frames = {
            ref_channel: {
                **ref_frame,
                "corrected_header_time_us": target_us,
                "residual_us": 0,
            }
        }
        proposed_next_idx = {}
        success = True

        for channel in CAMERA_CHANNELS:
            if channel == ref_channel:
                continue
            best_idx, residual_us = find_monotonic_match(
                corrected_by_channel[channel],
                target_us,
                next_idx[channel],
                residual_tolerance_us,
            )
            if best_idx is None:
                success = False
                break
            frame = frames_by_channel[channel][best_idx]
            group_frames[channel] = {
                **frame,
                "corrected_header_time_us": corrected_by_channel[channel][best_idx],
                "residual_us": residual_us,
            }
            proposed_next_idx[channel] = best_idx + 1

        if not success:
            continue

        next_idx.update(proposed_next_idx)
        for channel, frame in group_frames.items():
            residual_stats[channel].append(frame["residual_us"])
        aligned_groups.append(
            {
                "group_id": len(aligned_groups),
                "ref_channel": ref_channel,
                "period_us": period_us,
                "frames": group_frames,
            }
        )

    residual_summary = {}
    for channel, residuals in residual_stats.items():
        if not residuals:
            residual_summary[channel] = {}
            continue
        abs_residuals = [abs(value) for value in residuals]
        residual_summary[channel] = {
            "min_us": int(min(residuals)),
            "max_us": int(max(residuals)),
            "median_abs_us": float(median(abs_residuals)),
        }
    return aligned_groups, residual_summary


def build_rectifiers(calib: Dict[str, dict], aligned_groups: List[dict]) -> Dict[str, dict]:
    rectifiers = {}
    if not aligned_groups:
        return rectifiers
    for channel in CAMERA_CHANNELS:
        raw_path = Path(aligned_groups[0]["frames"][channel]["raw_path"])
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load raw image for rectification: {raw_path}")
        height, width = image.shape[:2]
        map1, map2, K_ud = compute_rectify_map_and_kud(
            calib[channel]["K_src"],
            calib[channel]["dist_src"],
            width,
            height,
        )
        rectifiers[channel] = {
            "map1": map1,
            "map2": map2,
            "K_ud": K_ud,
            "image_size_hw": (height, width),
        }
    return rectifiers


def rectify_aligned_groups(
    aligned_groups: List[dict],
    rectified_root: Path,
    calib: Dict[str, dict],
) -> Dict[str, dict]:
    rectifiers = build_rectifiers(calib, aligned_groups)
    written = {}
    for group in tqdm(aligned_groups, desc=f"Rectify {rectified_root.parent.name}", unit="frame"):
        for channel in CAMERA_CHANNELS:
            raw_path = Path(group["frames"][channel]["raw_path"])
            dst_dir = ensure_dir(rectified_root / channel)
            dst_path = dst_dir / raw_path.name
            if dst_path not in written:
                image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Failed to load image for rectification: {raw_path}")
                map1 = rectifiers[channel]["map1"]
                map2 = rectifiers[channel]["map2"]
                if map1 is None or map2 is None:
                    rectified = image
                else:
                    rectified = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
                ok = cv2.imwrite(str(dst_path), rectified)
                if not ok:
                    raise IOError(f"Failed to write rectified image: {dst_path}")
                written[dst_path] = True
            group["frames"][channel]["rectified_path"] = str(dst_path.resolve())
    return rectifiers


def write_aligned_manifest(
    aligned_groups: List[dict],
    bag_meta: Dict[str, str],
    aligned_csv_path: Path,
    aligned_json_path: Path,
):
    fieldnames = [
        "bag_name",
        "scene",
        "pos_neg",
        "keep_t2",
        "conditions",
        "tag",
        "group_id",
        "ref_channel",
        "period_us",
    ]
    for channel in CAMERA_CHANNELS:
        fieldnames.extend(
            [
                f"{channel}_seq",
                f"{channel}_bag_time_us",
                f"{channel}_header_time_us",
                f"{channel}_corrected_header_time_us",
                f"{channel}_residual_us",
                f"{channel}_raw_path",
                f"{channel}_rectified_path",
            ]
        )

    csv_rows = []
    json_rows = []
    for group in aligned_groups:
        row = {
            "bag_name": bag_meta["bag_name"],
            "scene": bag_meta.get("scene", ""),
            "pos_neg": bag_meta.get("pos_neg", ""),
            "keep_t2": bag_meta.get("keep_t2", ""),
            "conditions": bag_meta.get("conditions", ""),
            "tag": bag_meta.get("tag", ""),
            "group_id": group["group_id"],
            "ref_channel": group["ref_channel"],
            "period_us": group["period_us"],
        }
        json_group = dict(row)
        json_group["frames"] = {}
        for channel in CAMERA_CHANNELS:
            frame = group["frames"][channel]
            row[f"{channel}_seq"] = frame["seq"]
            row[f"{channel}_bag_time_us"] = frame["bag_time_us"]
            row[f"{channel}_header_time_us"] = frame["header_time_us"]
            row[f"{channel}_corrected_header_time_us"] = frame["corrected_header_time_us"]
            row[f"{channel}_residual_us"] = frame["residual_us"]
            row[f"{channel}_raw_path"] = frame["raw_path"]
            row[f"{channel}_rectified_path"] = frame["rectified_path"]
            json_group["frames"][channel] = {
                "seq": frame["seq"],
                "bag_time_us": frame["bag_time_us"],
                "header_time_us": frame["header_time_us"],
                "corrected_header_time_us": frame["corrected_header_time_us"],
                "residual_us": frame["residual_us"],
                "raw_path": frame["raw_path"],
                "rectified_path": frame["rectified_path"],
            }
        csv_rows.append(row)
        json_rows.append(json_group)

    ensure_dir(aligned_csv_path.parent)
    with open(aligned_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    with open(aligned_json_path, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2, ensure_ascii=False)


def write_report(
    bag_meta: Dict[str, str],
    report_path: Path,
    raw_counts: Dict[str, int],
    ref_channel: str,
    period_us: int,
    phase_offsets_us: Dict[str, int],
    residual_summary: Dict[str, dict],
    aligned_groups: List[dict],
):
    report = {
        "bag_name": bag_meta["bag_name"],
        "sequence_dir_name": bag_meta.get("sequence_dir_name", ""),
        "bag_path": bag_meta["bag_path"],
        "scene": bag_meta.get("scene", ""),
        "pos_neg": bag_meta.get("pos_neg", ""),
        "keep_t2": bag_meta.get("keep_t2", ""),
        "conditions": bag_meta.get("conditions", ""),
        "tag": bag_meta.get("tag", ""),
        "valid_start_bag_sec": float(bag_meta["valid_start_bag_sec"]),
        "valid_end_bag_sec": float(bag_meta["valid_end_bag_sec"]),
        "raw_counts": raw_counts,
        "period_us": period_us,
        "ref_channel": ref_channel,
        "phase_offsets_us": phase_offsets_us,
        "aligned_group_count": len(aligned_groups),
        "dropped_ref_frames": raw_counts.get(ref_channel, 0) - len(aligned_groups),
        "residual_summary": residual_summary,
        "rectified_output_count": len(aligned_groups),
    }
    ensure_dir(report_path.parent)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def process_one_bag(
    bag_meta: Dict[str, str],
    output_root: Path,
    calib: Dict[str, dict],
    ref_channel: str,
    residual_tolerance_us: int,
) -> Dict[str, str]:
    bag_path = Path(bag_meta["bag_path"])
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    sequence_dir_name = bag_meta.get("sequence_dir_name") or build_sequence_dir_name(
        bag_meta["bag_name"],
        bag_meta.get("scene", ""),
        bag_meta.get("tag", ""),
    )
    dirs = bag_output_dirs(output_root, sequence_dir_name)
    bag_stem = Path(bag_meta["bag_name"]).stem
    aligned_csv_path = dirs["manifests_root"] / f"{bag_stem}.aligned.csv"
    aligned_json_path = dirs["manifests_root"] / f"{bag_stem}.aligned.json"
    report_path = dirs["reports_root"] / f"{bag_stem}.json"

    frames_by_channel = extract_raw_frames(
        bag_path=bag_path,
        valid_start_bag_sec=float(bag_meta["valid_start_bag_sec"]),
        valid_end_bag_sec=float(bag_meta["valid_end_bag_sec"]),
        raw_root=dirs["raw_root"],
    )
    raw_counts = {channel: len(frames) for channel, frames in frames_by_channel.items()}
    period_us = estimate_period_us(frames_by_channel)
    phase_offsets_us = estimate_phase_offsets(frames_by_channel, ref_channel, period_us)
    aligned_groups, residual_summary = align_frames(
        frames_by_channel=frames_by_channel,
        ref_channel=ref_channel,
        period_us=period_us,
        phase_offsets_us=phase_offsets_us,
        residual_tolerance_us=residual_tolerance_us,
    )
    rectify_aligned_groups(aligned_groups, dirs["rectified_root"], calib)
    write_aligned_manifest(aligned_groups, bag_meta, aligned_csv_path, aligned_json_path)
    write_report(
        bag_meta=bag_meta,
        report_path=report_path,
        raw_counts=raw_counts,
        ref_channel=ref_channel,
        period_us=period_us,
        phase_offsets_us=phase_offsets_us,
        residual_summary=residual_summary,
        aligned_groups=aligned_groups,
    )
    return {
        "bag_name": bag_meta["bag_name"],
        "sequence_dir_name": sequence_dir_name,
        "aligned_manifest_csv": str(aligned_csv_path.resolve()),
        "aligned_manifest_json": str(aligned_json_path.resolve()),
        "report_json": str(report_path.resolve()),
        "sequence_root": str(dirs["sequence_root"].resolve()),
    }


def write_processed_manifest(output_root: Path, processed_rows: List[Dict[str, str]]):
    if not processed_rows:
        return
    manifest_path = output_root / "manifests" / "processed_bags.csv"
    ensure_dir(manifest_path.parent)
    fieldnames = list(processed_rows[0].keys())
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)


def normalize_manifest_row(row: Dict[str, str], bags_root: Optional[Path]) -> Dict[str, str]:
    normalized = dict(row)
    bag_path = Path(strip_markdown_code(normalized["bag_path"]))
    if not bag_path.is_absolute() and bags_root is not None:
        bag_path = bags_root / bag_path
    normalized["bag_path"] = str(bag_path.resolve())
    normalized["keep_t2"] = strip_markdown_code(normalized.get("keep_t2", ""))
    normalized["conditions"] = strip_markdown_code(normalized.get("conditions", ""))
    normalized["tag"] = strip_markdown_code(normalized.get("tag", ""))
    normalized["scene"] = strip_markdown_code(normalized.get("scene", ""))
    normalized["pos_neg"] = strip_markdown_code(normalized.get("pos_neg", ""))
    normalized["sequence_dir_name"] = strip_markdown_code(
        normalized.get("sequence_dir_name", "")
    ) or build_sequence_dir_name(
        normalized.get("bag_name", ""),
        normalized.get("scene", ""),
        normalized.get("tag", ""),
    )
    return normalized


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    manifest_csv_path = Path(args.manifest_csv).resolve() if args.manifest_csv else None
    bags_root = Path(args.bags_root).resolve() if args.bags_root else None

    if args.scene_info_md:
        scene_info_md = Path(args.scene_info_md).resolve()
        if bags_root is None:
            bags_root = scene_info_md.parent
        if manifest_csv_path is None:
            manifest_csv_path = output_root / "manifests" / "scene_info_manifest.csv"
        exported_rows = export_manifest_from_scene_info(scene_info_md, bags_root, manifest_csv_path)
        print(f"[scene_info] exported {len(exported_rows)} rows to {manifest_csv_path}")
        if args.export_manifest_only:
            return

    if args.yaml_path is None:
        raise ValueError("--yaml_path is required unless --export_manifest_only is used.")
    calib = load_camera_projections(args.yaml_path)
    residual_tolerance_us = int(round(args.residual_tolerance_ms * 1_000.0))

    manifest_rows = []
    if args.bag_path:
        bag_path = Path(args.bag_path).resolve()
        if args.valid_start_bag_sec is None or args.valid_end_bag_sec is None:
            inferred_start, inferred_end = infer_bag_time_range(bag_path)
            valid_start = args.valid_start_bag_sec if args.valid_start_bag_sec is not None else inferred_start
            valid_end = args.valid_end_bag_sec if args.valid_end_bag_sec is not None else inferred_end
        else:
            valid_start = args.valid_start_bag_sec
            valid_end = args.valid_end_bag_sec
        manifest_rows = [
            {
                "bag_name": bag_path.name,
                "sequence_dir_name": build_sequence_dir_name(bag_path.name, args.scene, args.tag),
                "bag_path": str(bag_path),
                "scene": args.scene,
                "pos_neg": args.pos_neg,
                "valid_start_bag_sec": f"{valid_start:.9f}",
                "valid_end_bag_sec": f"{valid_end:.9f}",
                "keep_t2": args.keep_t2,
                "conditions": args.conditions,
                "tag": args.tag,
                "notes": args.notes,
            }
        ]
    elif manifest_csv_path is not None:
        manifest_rows = [normalize_manifest_row(row, bags_root) for row in load_manifest_csv(manifest_csv_path)]
    else:
        raise ValueError("Provide either --bag_path or --scene_info_md/--manifest_csv.")

    if args.keep_t2_only:
        manifest_rows = [row for row in manifest_rows if row.get("keep_t2", "").upper() == "Y"]

    processed_rows = []
    for row in manifest_rows:
        valid_start = float(row["valid_start_bag_sec"])
        valid_end = float(row["valid_end_bag_sec"])
        result = process_one_bag(
            bag_meta=row,
            output_root=output_root,
            calib=calib,
            ref_channel=args.ref_channel,
            residual_tolerance_us=residual_tolerance_us,
        )
        processed_rows.append({**row, **result})

    write_processed_manifest(output_root, processed_rows)
    print(f"[done] processed {len(processed_rows)} bags under {output_root}")


if __name__ == "__main__":
    main()
