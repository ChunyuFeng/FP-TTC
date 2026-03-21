import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes


FRAME_REGEX = re.compile(
    r"^scene_(?P<scene_idx>\d+)_(?P<kind>scale_map|risk_score_map)_(?P<frame_id>\d+)\.png$"
)


@dataclass(frozen=True)
class FrameRecord:
    split: str
    category: str
    scene_idx: int
    frame_id: int
    file_name: str
    old_path: Path
    new_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build old-vs-new comparison videos from existing range projection PNGs."
    )
    parser.add_argument(
        "--old_vis_root",
        type=Path,
        default=Path("/mnt/data/fpttc_ground_truth/3_visualization_150_keyframes"),
    )
    parser.add_argument(
        "--new_vis_root",
        type=Path,
        default=Path("/mnt/data/fpttc_ground_truth/150_keyframes_classaware_mixedctx_v1/3_visualization"),
    )
    parser.add_argument(
        "--local30_scene_list",
        type=Path,
        default=Path("/mnt/data/fpttc_ground_truth/150_keyframes_classaware_mixedctx_v1/scene_lists/scene_150_local30_subset.txt"),
    )
    parser.add_argument(
        "--all_scene_list",
        type=Path,
        default=Path("tools/generate_scene_flow_nuscenes/scene_lists/scene_150_all.txt"),
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("./Datasets/nuscenes"),
    )
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/mnt/data/fpttc_ground_truth/150_keyframes_compare_videos_v1"),
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--scene_pause_frames", type=int, default=12)
    return parser.parse_args()


def load_scene_names(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_scene_name_to_index(version: str, dataroot: Path) -> Dict[str, int]:
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    return {scene["name"]: idx for idx, scene in enumerate(nusc.scene)}


def resolve_local30_indices(
    local30_scene_list: Path, all_scene_list: Path, version: str, dataroot: Path
) -> Tuple[set[int], set[int]]:
    name_to_index = build_scene_name_to_index(version, dataroot)
    local_names = load_scene_names(local30_scene_list)
    all_names = load_scene_names(all_scene_list)
    local_indices = {name_to_index[name] for name in local_names}
    all_indices = {name_to_index[name] for name in all_names}
    return local_indices, all_indices - local_indices


def get_vis_subdir(category: str) -> Path:
    if category == "scale":
        return Path("scale_map/key_frames_160_1920_fov_8_15")
    if category == "orientation":
        return Path("risk_score_map/key_frames_160_1920_fov_8_15_theta")
    raise ValueError(f"Unsupported category {category}")


def expected_kind(category: str) -> str:
    return "scale_map" if category == "scale" else "risk_score_map"


def collect_records(
    old_root: Path, new_root: Path, category: str, local30_indices: set[int]
) -> Dict[str, List[FrameRecord]]:
    groups: Dict[str, List[FrameRecord]] = {"local30": [], "complement": []}
    expected = expected_kind(category)
    subdir = get_vis_subdir(category)
    for split in ("train", "val", "test"):
        old_dir = old_root / split / subdir
        new_dir = new_root / split / subdir
        old_names = {path.name for path in old_dir.glob("*.png")}
        new_names = {path.name for path in new_dir.glob("*.png")}
        shared_names = sorted(old_names & new_names)
        for name in shared_names:
            match = FRAME_REGEX.match(name)
            if not match:
                continue
            if match.group("kind") != expected:
                continue
            scene_idx = int(match.group("scene_idx"))
            frame_id = int(match.group("frame_id"))
            group = "local30" if scene_idx in local30_indices else "complement"
            groups[group].append(
                FrameRecord(
                    split=split,
                    category=category,
                    scene_idx=scene_idx,
                    frame_id=frame_id,
                    file_name=name,
                    old_path=old_dir / name,
                    new_path=new_dir / name,
                )
            )
    for group in groups:
        groups[group].sort(key=lambda item: (item.scene_idx, item.frame_id))
    return groups


def draw_label_block(
    image: "cv2.typing.MatLike", lines: Sequence[str], origin: Tuple[int, int]
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    line_height = 32
    width = 0
    for line in lines:
        (line_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        width = max(width, line_width)
    block_h = 18 + len(lines) * line_height
    cv2.rectangle(image, (x, y), (x + width + 24, y + block_h), (0, 0, 0), -1)
    cv2.rectangle(image, (x, y), (x + width + 24, y + block_h), (255, 255, 255), 2)
    for idx, line in enumerate(lines):
        baseline_y = y + 30 + idx * line_height
        cv2.putText(
            image,
            line,
            (x + 12, baseline_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def render_comparison_frame(record: FrameRecord, title: str) -> "cv2.typing.MatLike":
    old_img = cv2.imread(str(record.old_path), cv2.IMREAD_COLOR)
    new_img = cv2.imread(str(record.new_path), cv2.IMREAD_COLOR)
    if old_img is None or new_img is None:
        raise RuntimeError(f"Failed to read image pair for {record.file_name}")
    if old_img.shape != new_img.shape:
        raise RuntimeError(
            f"Image size mismatch for {record.file_name}: {old_img.shape} vs {new_img.shape}"
        )

    image_h, image_w = old_img.shape[:2]
    title_h = 74
    canvas = 255 * np.ones((title_h + image_h * 2, image_w, 3), dtype=old_img.dtype)
    canvas[title_h : title_h + image_h] = old_img
    canvas[title_h + image_h :] = new_img

    cv2.rectangle(canvas, (0, 0), (image_w, title_h), (24, 24, 24), -1)
    cv2.putText(
        canvas,
        title,
        (24, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    base_lines = [
        f"split {record.split}",
        f"scene{record.scene_idx}",
        f"frame {record.frame_id}",
        f"type {record.category}",
    ]
    draw_label_block(canvas, ["OLD"] + base_lines, (24, title_h + 18))
    draw_label_block(canvas, ["NEW"] + base_lines, (24, title_h + image_h + 18))
    return canvas


def ensure_writer(
    writer: Optional[cv2.VideoWriter],
    frame: "cv2.typing.MatLike",
    output_path: Path,
    fps: int,
) -> cv2.VideoWriter:
    if writer is not None:
        return writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    return writer


def write_video(
    records: Iterable[FrameRecord],
    output_path: Path,
    title: str,
    fps: int,
    scene_pause_frames: int,
) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[cv2.VideoWriter] = None
    frame_count = 0
    scene_count = 0
    last_scene_idx = None
    for record in records:
        frame = render_comparison_frame(record, title)
        writer = ensure_writer(writer, frame, output_path, fps)
        repeat = scene_pause_frames if record.scene_idx != last_scene_idx else 1
        if record.scene_idx != last_scene_idx:
            scene_count += 1
        for _ in range(repeat):
            writer.write(frame)
            frame_count += 1
        last_scene_idx = record.scene_idx
    if writer is None:
        raise RuntimeError(f"No frames collected for {output_path}")
    writer.release()
    return {"frame_count": frame_count, "scene_count": scene_count}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local30_indices, complement_indices = resolve_local30_indices(
        args.local30_scene_list, args.all_scene_list, args.version, args.dataroot
    )

    manifest = {
        "fps": args.fps,
        "scene_pause_frames": args.scene_pause_frames,
        "local30_scene_count": len(local30_indices),
        "complement_scene_count": len(complement_indices),
        "videos": {},
    }

    for category in ("scale", "orientation"):
        grouped = collect_records(args.old_vis_root, args.new_vis_root, category, local30_indices)
        for group_name, records in grouped.items():
            title = f"{group_name} {category}"
            output_name = f"{group_name}_{category}_old_vs_new.mp4"
            output_path = args.output_dir / output_name
            stats = write_video(
                records=records,
                output_path=output_path,
                title=title,
                fps=args.fps,
                scene_pause_frames=args.scene_pause_frames,
            )
            manifest["videos"][output_name] = {
                "category": category,
                "group": group_name,
                "output_path": str(output_path),
                "paired_sample_count": len(records),
                **stats,
            }
            print(
                f"Saved {output_name}: paired_samples={len(records)}, "
                f"frames={stats['frame_count']}, scenes={stats['scene_count']}"
            )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
