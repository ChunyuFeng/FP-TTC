import argparse
import os
import pickle
from pathlib import Path

from utils.nusc_paths import make_nusc_relative_path


def migrate_entry(info: dict) -> tuple[dict, int]:
    changes = 0

    for key in ("gt_map_path", "scene_flow_path"):
        old_value = info.get(key)
        new_value = make_nusc_relative_path(old_value)
        if new_value != old_value:
            info[key] = new_value
            changes += 1

    for camera_key in ("prev_camera_data", "curr_camera_data"):
        camera_data = info.get(camera_key, {})
        for camera_info in camera_data.values():
            old_value = camera_info.get("depth_pred")
            new_value = make_nusc_relative_path(old_value)
            if new_value != old_value:
                camera_info["depth_pred"] = new_value
                changes += 1

    return info, changes


def migrate_pkl_file(pkl_path: Path) -> None:
    with pkl_path.open("rb") as f:
        infos = pickle.load(f)

    total_changes = 0
    for idx, info in enumerate(infos):
        infos[idx], changes = migrate_entry(info)
        total_changes += changes

    tmp_path = pkl_path.with_suffix(pkl_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(infos, f)
    os.replace(tmp_path, pkl_path)

    print(f"Migrated {pkl_path} with {total_changes} field updates across {len(infos)} items.")


def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite nuScenes info PKLs to use relative derived-data paths.")
    parser.add_argument("--pkl_paths", nargs="+", required=True, help="PKL files to rewrite in place.")
    return parser.parse_args()


def main():
    args = parse_args()
    for pkl_path_str in args.pkl_paths:
        migrate_pkl_file(Path(pkl_path_str))


if __name__ == "__main__":
    main()
