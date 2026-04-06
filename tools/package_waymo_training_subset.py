#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import json
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path


WAYMO_CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
DEFAULT_SRC_ROOT = Path("/mnt/data2/waymo_projection_fix_v2_scene150")
DEFAULT_DST_ROOT = Path("/mnt/data2/waymo_projection_fix_v2_scene150_trainval_minpack_v1")
DEFAULT_CACHE_NAME = (
    "waymo_scene150_keyframes_160x320_proj40x400_"
    "fov2_17_toplidar_nusc_vehrot_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a minimal Waymo train/val subset for server-side training."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT,
        help="Source Waymo dataset root.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DEFAULT_DST_ROOT,
        help="Destination subset root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
        help="Dataset splits to include.",
    )
    parser.add_argument(
        "--cache-name",
        type=str,
        default=DEFAULT_CACHE_NAME,
        help="Projection-cache directory name under 5_proj_cache.",
    )
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="How to stage files into the subset root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing destination root before packaging.",
    )
    return parser.parse_args()


def load_build_manifest(src_root: Path) -> dict:
    manifest_path = src_root / "build_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing build manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def load_infos(info_pkl: Path) -> list[dict]:
    with open(info_pkl, "rb") as f:
        return pickle.load(f)


def ensure_clean_destination(dst_root: Path, overwrite: bool):
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_root}. "
                "Pass --overwrite to remove it first."
            )
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)


def add_file(
    files_by_relpath: dict[Path, Path],
    relpath: Path,
    src_path: Path,
):
    src_resolved = src_path.resolve(strict=True)
    existing = files_by_relpath.get(relpath)
    if existing is not None and existing != src_resolved:
        raise ValueError(
            f"Conflicting source paths for {relpath}: {existing} vs {src_resolved}"
        )
    files_by_relpath[relpath] = src_resolved


def build_file_map(src_root: Path, splits: list[str], cache_name: str):
    build_manifest = load_build_manifest(src_root)
    files_by_relpath: dict[Path, Path] = {}
    stats = defaultdict(lambda: {"files": 0, "bytes": 0})

    def remember(category: str, relpath: Path, src_path: Path):
        already_present = relpath in files_by_relpath
        add_file(files_by_relpath, relpath, src_path)
        if not already_present:
            stats[category]["files"] += 1
            stats[category]["bytes"] += src_path.resolve(strict=True).stat().st_size

    for split in splits:
        split_meta = build_manifest.get("splits", {}).get(split)
        if split_meta is None:
            raise KeyError(f"Split {split} not found in build_manifest.json")

        info_pkl = Path(split_meta["info_pkl"])
        if not info_pkl.is_absolute():
            info_pkl = src_root / info_pkl
        info_rel = Path("2_trainval_test_infos") / split / info_pkl.name
        remember("info_pkls", info_rel, info_pkl)

        infos = load_infos(info_pkl)
        for info in infos:
            for frame_key in ("prev", "curr"):
                frame_cameras = info[f"{frame_key}_cameras"]
                frame_idx = int(info[f"{frame_key}_frame_idx"])
                segment_name = f"segment_{info['segment_id']}"

                for cam_name in WAYMO_CAMERAS:
                    image_rel = Path(frame_cameras[cam_name]["image_path"])
                    remember("camera_images", image_rel, src_root / image_rel)

                    depth_rel = (
                        Path("4_depth_map")
                        / cam_name
                        / f"{segment_name}_frame_{frame_idx:04d}.npy"
                    )
                    remember("depth_maps", depth_rel, src_root / depth_rel)

            gt_rel = Path(info["gt_map_path"]) / "range_image_curr.npy"
            remember("gt_maps", gt_rel, src_root / gt_rel)

    cache_root = src_root / "5_proj_cache" / cache_name
    if not cache_root.exists():
        raise FileNotFoundError(f"Projection cache root does not exist: {cache_root}")

    remember(
        "proj_cache",
        Path("5_proj_cache") / cache_name / "cache_meta.json",
        cache_root / "cache_meta.json",
    )

    for split in splits:
        split_root = cache_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Projection cache split does not exist: {split_root}")

        for name in ("manifest.json", "manifest.pkl"):
            remember(
                "proj_cache",
                Path("5_proj_cache") / cache_name / split / name,
                split_root / name,
            )

        for npz_path in sorted(split_root.glob("*.npz")):
            remember(
                "proj_cache",
                Path("5_proj_cache") / cache_name / split / npz_path.name,
                npz_path,
            )

    return files_by_relpath, stats


def stage_file(src_path: Path, dst_path: Path, mode: str):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return

    if mode == "hardlink":
        try:
            os.link(src_path, dst_path)
            return
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK):
                raise

    shutil.copy2(src_path, dst_path)


def write_subset_manifest(
    dst_root: Path,
    src_root: Path,
    splits: list[str],
    cache_name: str,
    mode: str,
    files_by_relpath: dict[Path, Path],
    stats: dict[str, dict[str, int]],
):
    manifest = {
        "source_root": str(src_root),
        "subset_root": str(dst_root),
        "splits": splits,
        "cache_name": cache_name,
        "stage_mode": mode,
        "total_files": len(files_by_relpath),
        "total_bytes": sum(v["bytes"] for v in stats.values()),
        "categories": stats,
    }
    (dst_root / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )


def main():
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    splits = list(dict.fromkeys(args.splits))

    ensure_clean_destination(dst_root, args.overwrite)
    files_by_relpath, stats = build_file_map(src_root, splits, args.cache_name)

    for relpath, src_path in files_by_relpath.items():
        stage_file(src_path, dst_root / relpath, args.mode)

    write_subset_manifest(
        dst_root=dst_root,
        src_root=src_root,
        splits=splits,
        cache_name=args.cache_name,
        mode=args.mode,
        files_by_relpath=files_by_relpath,
        stats=stats,
    )

    print(f"subset_root={dst_root}")
    print(f"total_files={len(files_by_relpath)}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
