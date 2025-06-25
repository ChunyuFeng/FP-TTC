import os
import pickle
import argparse
import shutil
from argparse import ArgumentParser
from nuscenes.nuscenes import NuScenes

def copy_nusc(rel_path: str, data_root: str, save_root: str):
    """
    拷贝一个 sample_data 文件（.jpg/.png/.bin/...）
    """
    src = os.path.join(data_root, rel_path)
    dst = os.path.join(save_root, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def main(args):
    with open(args.pkl_file, 'rb') as f:
        pkl_data = pickle.load(f)
    
    count = 0

    save_root = f'/mnt/data/custom_nusc_scene{args.scene_idx}'
    os.makedirs(save_root, exist_ok=True)
    for info in pkl_data:
        if int(info['scene_indice']) != args.scene_idx:
            continue
        # copy CAM images
        for channel, cam_data in info['prev_camera_data'].items():
            copy_nusc(
                rel_path=cam_data['filename'],
                data_root='/mnt/data/nuScenes',
                save_root=save_root,
            )
        for channel, cam_data in info['curr_camera_data'].items():
            copy_nusc(
                rel_path=cam_data['filename'],
                data_root='/mnt/data/nuScenes',
                save_root=save_root,
            )
        # cpy LIDAR data
        copy_nusc(
                rel_path=info['prev_lidar_data']['filename'],
                data_root='/mnt/data/nuScenes',
                save_root=save_root,
            )
        copy_nusc(
                rel_path=info['curr_lidar_data']['filename'],
                data_root='/mnt/data/nuScenes',
                save_root=save_root,
            )
        
        count += 1

    print(f"Copied {count} items from nuscenes scene-{args.scene_idx}.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Copy Nuscenes scene data to a specified directory."
    )
    parser.add_argument(
        '--pkl_file', type=str, required=True,
        help="Path to the .pkl file"
    )
    parser.add_argument(
        '--scene_idx', type=int, default=0,
        help="Scene index to load from the .pkl file"
    )
    args = parser.parse_args()

    main(args)
