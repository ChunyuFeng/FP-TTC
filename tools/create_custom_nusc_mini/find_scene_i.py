#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from argparse import ArgumentParser
from nuscenes.nuscenes import NuScenes

def copy_one_sd(nusc: NuScenes, sd_token: str, dataroot: str, save_root: str):
    """
    拷贝一个 sample_data 文件（.jpg/.png/.bin/...）
    """
    sd = nusc.get('sample_data', sd_token)
    rel_path = sd['filename']  # 相对 dataroot 的路径，比如 "samples/CAM_FRONT/xxx.jpg" 或 "sweeps/CAM_FRONT/yyy.jpg"
    src = os.path.join(dataroot, rel_path)
    dst = os.path.join(save_root,  rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def copy_scene_data(nusc: NuScenes, scene_idx: int, save_root: str):
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']

    # 用来去重，避免同一个 sweep 被拷贝多次
    visited = set()

    while sample_token:
        sample = nusc.get('sample', sample_token)

        for sensor_name, sd_token in sample['data'].items():
            # 只处理相机和 LiDAR
            if not (sensor_name.startswith('CAM_') or sensor_name.startswith('LIDAR_')):
                continue

            # 广度优先遍历：关键帧 + prev/next 所在的所有 sweeps
            queue = [sd_token]
            while queue:
                tok = queue.pop(0)
                if tok in visited:
                    continue
                visited.add(tok)

                # 拷贝当前这个 sample_data 对应的文件
                copy_one_sd(nusc, tok, nusc.dataroot, save_root)

                # 如果有 prev/next，就加入队列，继续拷贝
                sd = nusc.get('sample_data', tok)
                if sd['prev']:
                    queue.append(sd['prev'])
                if sd['next']:
                    queue.append(sd['next'])

        sample_token = sample['next'] or None


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Copy NuScenes scene data (key frames + sweeps)，并保持目录结构"
    )
    parser.add_argument(
        '--dataroot',  type=str, required=True,
        help="NuScenes 数据集根目录（指向 v1.0-trainval，包含 samples/、sweeps/、*.json）"
    )
    parser.add_argument(
        '--scene_idx', type=int, default=3,
        help="要复制的场景索引（0-based），对应 nusc.scene 列表"
    )
    parser.add_argument(
        '--save_root', type=str, required=True,
        help="目标保存目录"
    )
    args = parser.parse_args()

    nusc = NuScenes(
        version='v1.0-trainval',
        dataroot=args.dataroot,
        verbose=False
    )

    print(f"开始复制 scene #{args.scene_idx} …")
    copy_scene_data(nusc, args.scene_idx, args.save_root)
    print("全部完成！")
