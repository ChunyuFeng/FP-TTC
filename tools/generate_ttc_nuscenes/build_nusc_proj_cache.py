import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataloader.dataset import (
    CAMERA_CHANNELS,
    DEFAULT_NUSC_PROJ_CACHE_NAME,
    build_frame_mapping,
)
from dataloader.utils.augmentor import NuscRangeImageAugmentor
from utils.nusc_paths import make_nusc_relative_path, resolve_nusc_depth_pred_path


DEFAULT_SPLIT_FILES = {
    'train': 'nusc_train_infos_key_frames_160_1920_fov_8_15.pkl',
    'val': 'nusc_val_infos_key_frames_160_1920_fov_8_15.pkl',
    'test': 'nusc_test_infos_key_frames_160_1920_fov_8_15.pkl',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_info_root',
        default='./Datasets/nuscenes/2_trainval_test_infos',
        type=str,
        help='Directory containing train/val/test pkl folders',
    )
    parser.add_argument(
        '--dataset_root',
        default='./Datasets/nuscenes',
        type=str,
        help='Logical nuScenes dataset root',
    )
    parser.add_argument(
        '--cache_root',
        default=f'./Datasets/nuscenes/5_proj_cache/{DEFAULT_NUSC_PROJ_CACHE_NAME}',
        type=str,
        help='Root directory for projection cache',
    )
    parser.add_argument(
        '--splits',
        default=['train', 'val', 'test'],
        nargs='+',
        choices=['train', 'val', 'test'],
        help='Dataset splits to cache',
    )
    parser.add_argument(
        '--image_size',
        default=[160, 320],
        nargs=2,
        type=int,
        help='Training crop size (H W)',
    )
    parser.add_argument(
        '--proj_hw',
        default=[40, 480],
        nargs=2,
        type=int,
        help='Projection cache size (H W)',
    )
    parser.add_argument(
        '--fov',
        default=[8.0, 15.0],
        nargs=2,
        type=float,
        help='Range projection FOV (up down_abs)',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Rewrite existing cache files',
    )
    parser.add_argument(
        '--max_samples',
        default=None,
        type=int,
        help='Optional debug limit per split',
    )
    return parser.parse_args()


def has_complete_depth(info, dataset_root: Path) -> bool:
    for frame_key in ['prev_camera_data', 'curr_camera_data']:
        for channel in CAMERA_CHANNELS:
            depth_path = resolve_nusc_depth_pred_path(info[frame_key][channel], dataset_root)
            if depth_path is None or not depth_path.exists():
                return False
    return True


def build_affine(image_size):
    augmentor = NuscRangeImageAugmentor(
        crop_size=image_size,
        do_flip=False,
        rotate=False,
        rotate_prob=0.1,
        rotate_angle=90,
    )
    affine_params = augmentor.sample_params((1600, 900))
    affine_matrix = augmentor.get_affine_matrix(affine_params)
    return affine_params, affine_matrix


def build_manifest_entry(info, split: str, original_index: int, cache_relpath: Path):
    return {
        'split': split,
        'original_index': int(original_index),
        'scene_indice': info.get('scene_indice'),
        'sample_token': info.get('sample_token'),
        'curr_sample_data_token': info.get('curr_sample_data_token'),
        'gt_map_path': make_nusc_relative_path(info.get('gt_map_path')),
        'prev_camera_files': {
            channel: info['prev_camera_data'][channel]['filename']
            for channel in CAMERA_CHANNELS
        },
        'curr_camera_files': {
            channel: info['curr_camera_data'][channel]['filename']
            for channel in CAMERA_CHANNELS
        },
        'prev_depth_files': {
            channel: make_nusc_relative_path(info['prev_camera_data'][channel].get('depth_pred'))
            for channel in CAMERA_CHANNELS
        },
        'curr_depth_files': {
            channel: make_nusc_relative_path(info['curr_camera_data'][channel].get('depth_pred'))
            for channel in CAMERA_CHANNELS
        },
        'complete_depth': True,
        'cache_relpath': cache_relpath.as_posix(),
    }


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    train_info_root = Path(args.train_info_root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    affine_params, affine_matrix = build_affine(args.image_size)
    summary = {}

    for split in args.splits:
        pkl_path = train_info_root / split / DEFAULT_SPLIT_FILES[split]
        if not pkl_path.exists():
            raise FileNotFoundError(f'Missing pkl file: {pkl_path}')

        with open(pkl_path, 'rb') as f:
            infos = pickle.load(f)

        split_root = cache_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        entries = []

        iterator = enumerate(infos)
        if args.max_samples is not None:
            iterator = list(iterator)[:args.max_samples]

        total = len(infos) if args.max_samples is None else min(len(infos), args.max_samples)
        for original_index, info in tqdm(iterator, total=total, desc=f'Caching {split}'):
            if not has_complete_depth(info, dataset_root):
                raise FileNotFoundError(
                    f'Sample {split}[{original_index}] does not have complete depth maps. '
                    'Regenerate 4_depth_map before building the projection cache.'
                )

            token = info.get('curr_sample_data_token') or info.get('sample_token') or f'{split}_{original_index:06d}'
            token_safe = str(token).replace('/', '_')
            cache_relpath = Path(split) / f'{original_index:06d}_{token_safe}.npz'
            cache_path = cache_root / cache_relpath

            if args.overwrite or not cache_path.exists():
                _, proj_pix_prev = build_frame_mapping(
                    info, 'nusc', 'prev', None, affine_matrix, idx=None,
                    dataset_root=dataset_root, H_r=args.proj_hw[0], W_r=args.proj_hw[1]
                )
                _, proj_pix_curr = build_frame_mapping(
                    info, 'nusc', 'curr', None, affine_matrix, idx=None,
                    dataset_root=dataset_root, H_r=args.proj_hw[0], W_r=args.proj_hw[1]
                )
                np.savez_compressed(
                    cache_path,
                    proj_pix_prev=proj_pix_prev.astype(np.int16),
                    proj_pix_curr=proj_pix_curr.astype(np.int16),
                )

            entries.append(build_manifest_entry(info, split, original_index, cache_relpath))

        manifest_obj = {'split': split, 'count': len(entries), 'entries': entries}
        with open(split_root / 'manifest.pkl', 'wb') as f:
            pickle.dump(manifest_obj, f)
        (split_root / 'manifest.json').write_text(json.dumps(manifest_obj, indent=2, ensure_ascii=False))
        summary[split] = len(entries)

    cache_meta = {
        'cache_name': cache_root.name,
        'dataset_root': str(dataset_root),
        'train_info_root': str(train_info_root),
        'image_size': [int(args.image_size[0]), int(args.image_size[1])],
        'camera_order': CAMERA_CHANNELS,
        'H_r': int(args.proj_hw[0]),
        'W_r': int(args.proj_hw[1]),
        'fov_up': float(args.fov[0]),
        'fov_down': -float(args.fov[1]),
        'augmentor': {
            'do_flip': False,
            'rotate': False,
            'rotate_prob': 0.1,
            'rotate_angle': 90,
            'affine_params': affine_params,
        },
        'split_counts': summary,
    }
    (cache_root / 'cache_meta.json').write_text(json.dumps(cache_meta, indent=2, ensure_ascii=False))

    print(f'cache_root={cache_root}')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
