#!/usr/bin/env python3
import os
import sys
import bisect
import pickle
import argparse
import yaml
import numpy as np
import cv2
import shutil
from scipy.spatial.transform import Rotation


def load_camera_projections(yaml_path):
    """
    读取包含多相机内外参以及畸变参数的 YAML 文件，返回每个相机的标定。
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    calib = {}
    cameras = config.get('cameras', {})

    for cam_name, cam_cfg in cameras.items():
        # 内参
        intr = cam_cfg['intrinsics']
        fx, fy = intr['fx'], intr['fy']
        cx, cy = intr['cx'], intr['cy']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

        # 畸变（按原逻辑：若无则空数组）
        if 'distortion' in intr and intr['distortion']:
            dist = np.array(intr['distortion'], dtype=float)
            has_dist = True
        else:
            dist = np.array([], dtype=float)
            has_dist = False
            print(f"[Warning] Camera '{cam_name}' missing distortion params in YAML.", file=sys.stderr)

        # 外参（原定义）
        ext = cam_cfg['extrinsics']
        t = np.array(ext['translation'], dtype=float)
        q = ext['rotation']  # 四元数 [x, y, z, w]
        R = Rotation.from_quat(q).as_matrix()

        calib[cam_name] = {
            'K_src':           K,     # 保留原始K
            'dist_src':        dist,  # 保留原始畸变
            'has_distortion':  has_dist,
            'R':               R,     # 原 R/t（假设与原推理相同语义）
            't':               t,
        }

    return calib


def build_image_index(base_dir):
    """
    遍历 base_dir 下的每个子文件夹（channel），收集符合
    {channel}__seq_{seq}__{ts_usec}.jpg 的图像，按 seq 升序。
    """
    if not os.path.isdir(base_dir):
        print(f"ERROR: 目录不存在: {base_dir}", file=sys.stderr)
        sys.exit(1)

    img_dict = {}
    for channel in os.listdir(base_dir):
        channel_dir = os.path.join(base_dir, channel)
        if not os.path.isdir(channel_dir):
            continue

        items = []
        for fn in os.listdir(channel_dir):
            if not fn.lower().endswith('.jpg'):
                continue
            parts = fn.split('__')
            if len(parts) < 3 or not parts[1].startswith('seq_'):
                continue
            try:
                seq = int(parts[1].split('_', 1)[1])
                ts  = int(parts[2].split('.')[0])
            except ValueError:
                continue
            items.append((seq, ts, os.path.join(channel_dir, fn)))

        if not items:
            continue

        items.sort(key=lambda x: x[0])

        img_dict[channel] = {
            'filename':  [path for seq, ts, path in items],
            'timestamp': [ts   for seq, ts, path in items],
            'sequence':  [seq  for seq, ts, path in items],
        }

    return img_dict


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_rectify_map_and_Kud(K_src, dist_src, w, h):
    """
    按原代码风格（pinhole）计算一次 map1/map2 与去畸变后的 K_ud。
    若 dist 为空，则返回 (None, None, K_src)。
    """
    if dist_src is None or dist_src.size == 0:
        return None, None, K_src

    # 与原推理保持一致：alpha=0，不裁剪，输出分辨率与输入一致
    K_ud, _ = cv2.getOptimalNewCameraMatrix(K_src, dist_src, (w, h), 0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K_src, dist_src, None, K_ud, (w, h), cv2.CV_32FC1)
    return map1, map2, K_ud


def rectify_and_save_channel(channel, items, rectified_root, K_src, dist_src):
    """
    对单个通道批量去畸变并落盘到 rectified_root/channel 下，文件名不变。
    返回：
      - rectified_fns: 与 items 对齐的去畸变后文件路径列表
      - K_ud: 去畸变后的内参（若无畸变则等于 K_src）
      - size_hw: (h, w)
    """
    if len(items) == 0:
        return [], K_src, None

    # 用第一帧确定尺寸
    h, w = cv2.imread(items[0], cv2.IMREAD_COLOR).shape[:2]
    map1, map2, K_ud = compute_rectify_map_and_Kud(K_src, dist_src, w, h)

    out_dir = os.path.join(rectified_root, channel)
    ensure_dir(out_dir)

    rectified_fns = []
    for src_path in items:
        fn = os.path.basename(src_path)
        dst_path = os.path.join(out_dir, fn)
        if map1 is None:
            # 无畸变：直接拷贝即可（更快）
            # 若你更希望完全一致的编码，也可以读/写一遍
            shutil.copy2(src_path, dst_path)
        else:
            img = cv2.imread(src_path, cv2.IMREAD_COLOR)
            img_ud = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(dst_path, img_ud)
        rectified_fns.append(dst_path)

    return rectified_fns, K_ud, (h, w)


def main():
    parser = argparse.ArgumentParser(description='Generate SJTU test infos (pre-rectified images)')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='原始根目录，包含各通道子文件夹')
    parser.add_argument('--rectified_root', type=str, required=True,
                        help='去畸变图像输出根目录（与 base_dir 平行）')
    parser.add_argument('--pkl_save_path', type=str, required=True,
                        help='保存输出 pkl 文件的目录')
    parser.add_argument('--image_size', type=int, nargs=2, required=True,
                        metavar=('HEIGHT','WIDTH'),
                        help='单通道图像的目标高和宽（仅用于命名，不改变原图尺寸）')
    parser.add_argument('--scene_idx', type=int, default=0,
                        help='场景索引')
    parser.add_argument('--yaml_path', type=str, required=True,
                        help='包含相机内外参的 YAML 文件路径')
    args = parser.parse_args()

    # 1) 读取标定
    calib_src = load_camera_projections(args.yaml_path)

    # 2) 扫描原始目录
    img_dict = build_image_index(args.base_dir)
    channels = sorted(img_dict.keys())
    if not channels:
        print("ERROR: 未发现任何通道图像。", file=sys.stderr)
        sys.exit(1)

    # 3) 每通道批量去畸变 -> 写入 rectified_root，并记录去畸变后的 K、尺寸
    ensure_dir(args.rectified_root)
    rectified_img_dict = {}
    per_channel_meta = {}

    for ch in channels:
        src_fns = img_dict[ch]['filename']
        rect_fns, K_ud, size_hw = rectify_and_save_channel(
            channel=ch,
            items=src_fns,
            rectified_root=args.rectified_root,
            K_src=calib_src[ch]['K_src'],
            dist_src=calib_src[ch]['dist_src']
        )
        rectified_img_dict[ch] = {
            'filename':  rect_fns,
            'timestamp': img_dict[ch]['timestamp'],
            'sequence':  img_dict[ch]['sequence'],
        }

        # 预写好后续要用到的 meta（去畸变后的内参与尺寸；以及 R_l2c/t_l2c）
        R = calib_src[ch]['R']
        t = calib_src[ch]['t']
        R_l2c = R.T               # 与你原推理里的一致
        t_l2c = -R_l2c @ t

        per_channel_meta[ch] = {
            # 源（备查）
            'K_src': calib_src[ch]['K_src'],
            'dist_src': calib_src[ch]['dist_src'],
            # 生效（去畸变后）
            'K': K_ud,
            'dist': np.array([], dtype=float),   # 表明图像已去畸变
            'is_rectified': True,
            'image_size_rectified': (size_hw[0], size_hw[1]) if size_hw else None,
            # 外参（统一提供 lidar->camera）
            'R_l2c': R_l2c,
            't_l2c': t_l2c,
            # 同时保留原始 R/t（与之前保持兼容）
            'R': R,
            't': t,
        }

    # 4) 生成测试 info（文件路径改为去畸变后的路径）
    sjtu_test_infos = []
    frame_count = len(rectified_img_dict[channels[0]]['timestamp'])

    for i in range(1, frame_count):
        seq = rectified_img_dict[channels[0]]['sequence'][i]

        # if seq < 24399 or seq > 24707:
        #     continue

        prev_data, curr_data = {}, {}
        for ch in channels:
            prev_data[ch] = {
                'filename':  rectified_img_dict[ch]['filename'][i-1],
                'timestamp': rectified_img_dict[ch]['timestamp'][i-1]
            }
            curr_data[ch] = {
                'filename':  rectified_img_dict[ch]['filename'][i],
                'timestamp': rectified_img_dict[ch]['timestamp'][i]
            }

        # 同一份 meta（不随时间变化）可直接复用
        sensor_metas_prev = {ch: per_channel_meta[ch] for ch in channels}
        sensor_metas_curr = {ch: per_channel_meta[ch] for ch in channels}

        info = {
            'prev_camera_data':  prev_data,
            'curr_camera_data':  curr_data,
            'prev_lidar_data':   None,
            'curr_lidar_data':   None,
            'sensor_metas_prev': sensor_metas_prev,
            'sensor_metas_curr': sensor_metas_curr,
            'gt_map_path':       None,
            'scene_flow_path':   None,
            'scene_indice':      None,
            'ros_msg_seq':       seq
        }
        sjtu_test_infos.append(info)

    # 5) 存 pkl
    os.makedirs(args.pkl_save_path, exist_ok=True)
    H, W = args.image_size
    pkl_name = f"scene_{args.scene_idx}_sjtu_test_infos_{H}_{W*len(channels)}_rectified.pkl"
    pkl_path = os.path.join(args.pkl_save_path, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(sjtu_test_infos, f)

    print(f"Saved {len(sjtu_test_infos)} info entries to {pkl_path}")
    print(f"Rectified images written under: {args.rectified_root}")
    for ch in channels:
        print(f"  - {ch}: {len(rectified_img_dict[ch]['filename'])} frames")
    print("Done.")
    

if __name__ == "__main__":
    main()
