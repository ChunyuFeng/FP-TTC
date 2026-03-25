from __future__ import annotations

from pathlib import Path


DEFAULT_NUSC_DATASET_ROOT = Path("./Datasets/nuscenes")
NUSC_PATH_ANCHORS = (
    "0_scene_flow",
    "1_gt_map",
    "2_trainval_test_infos",
    "3_visualization",
    "4_depth_map",
    "5_proj_cache",
    "samples",
    "sweeps",
    "maps",
    "lidarseg",
    "v1.0-trainval",
    "v1.0-test",
    "v1.0-mini",
)


def resolve_nusc_path(path_value: str | Path | None, dataset_root: str | Path = DEFAULT_NUSC_DATASET_ROOT) -> Path | None:
    if path_value in (None, ""):
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    return Path(dataset_root) / path


def resolve_nusc_depth_pred_path(camera_info: dict | None, dataset_root: str | Path = DEFAULT_NUSC_DATASET_ROOT) -> Path | None:
    if not camera_info:
        return None

    depth_pred = camera_info.get("depth_pred")
    if depth_pred not in (None, ""):
        return resolve_nusc_path(depth_pred, dataset_root)

    filename = camera_info.get("filename")
    if filename in (None, ""):
        return None

    depth_rel = Path("4_depth_map") / Path(filename).with_suffix(".npy")
    return resolve_nusc_path(depth_rel, dataset_root)


def make_nusc_relative_path(path_value: str | Path | None) -> str | None:
    if path_value in (None, ""):
        return path_value

    path = Path(path_value)
    parts = path.parts
    for index, part in enumerate(parts):
        if part in NUSC_PATH_ANCHORS:
            return Path(*parts[index:]).as_posix()

    return path.as_posix()


def infer_nusc_dataset_root(path_value: str | Path) -> Path:
    path = Path(path_value)
    search_chain = [path] + list(path.parents)
    for candidate in search_chain:
        if candidate.name == "2_trainval_test_infos":
            return candidate.parent
    return path.parent
