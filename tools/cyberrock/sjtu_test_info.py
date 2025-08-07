#!/usr/bin/env python3
import os
import sys
import bisect
import pickle
import argparse
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def load_camera_projections(yaml_path):
    """
    读取包含多相机内外参以及畸变参数的 YAML 文件，返回每个相机的投影矩阵 P（3x4）及畸变系数。

    Args:
        yaml_path (str): YAML 文件路径。

    Returns:
        dict: 键为相机名称，值为包含以下键的字典：
              - 'K': 3x3 相机内参矩阵
              - 'dist': 畸变系数数组（如果未定义则为空数组）
              - 'has_distortion': 布尔，是否定义了畸变系数
              - 'R': 3x3 旋转矩阵
              - 't': 3x1 平移向量
              - 'P': 3x4 投影矩阵
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

        # 判断是否包含畸变参数
        if 'distortion' in intr and intr['distortion']:
            dist = np.array(intr['distortion'], dtype=float)
            has_dist = True
        else:
            dist = np.array([], dtype=float)
            has_dist = False
            print(f"[Warning] Camera '{cam_name}' missing distortion params in YAML.", file=sys.stderr)

        # 外参
        ext = cam_cfg['extrinsics']
        t = np.array(ext['translation'], dtype=float)
        q = ext['rotation']  # 四元数 [x, y, z, w]
        R = Rotation.from_quat(q).as_matrix()

        # 投影矩阵 P = K * [R | t]
        RT = np.hstack((R, t.reshape(3, 1)))  # 3x4 矩阵
        P = K.dot(RT)

        calib[cam_name] = {
            'K':              K,
            'dist':           dist,
            'has_distortion': has_dist,
            'R':              R,
            't':              t,
            'P':              P,
        }

    return calib

def undistort_image(image, K, dist):
    """
    使用相机内参和畸变系数对图像进行畸变矫正。

    Args:
        image (ndarray): 输入畸变图像
        K (ndarray): 3x3 相机内参矩阵
        dist (ndarray): 畸变系数

    Returns:
        undistorted (ndarray): 矫正后的图像
        new_K (ndarray): 优化后的相机内参
        roi (tuple): 纠正后图像感兴趣区域
    """
    h, w = image.shape[:2]
    if dist.size == 0:
        return image, K, (0, 0, w, h)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    undistorted = cv2.undistort(image, K, dist, None, new_K)
    return undistorted, new_K, roi

def build_image_index(base_dir):
    """
    遍历 base_dir 下的每个子文件夹（channel），
    收集其中所有符合 `{channel}__seq_{seq}__{ts_usec}.jpg` 格式的文件，
    按 seq（sequence）从小到大排序，
    返回结构：
      {
        channel_name: {
          'filename': [...],
          'timestamp': [...],
          'sequence': [...]
        },
        ...
      }
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
            # 期望格式：{channel}__seq_{seq}__{ts_usec}.jpg
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

        # 只按 seq 排序
        items.sort(key=lambda x: x[0])

        img_dict[channel] = {
            'filename':  [path for seq, ts, path in items],
            'timestamp': [ts   for seq, ts, path in items],
            'sequence':  [seq  for seq, ts, path in items],
        }

    return img_dict

def sync_image_dict(img_dict, hz=10):
    """
    同步所有 channel 到相同 10Hz 列表，生成近似同一时刻采集的帧组
    """
    channels = sorted(img_dict.keys())
    # 公共时间区间
    start_us = max(img_dict[ch]['timestamp'][0] for ch in channels)
    end_us   = min(img_dict[ch]['timestamp'][-1] for ch in channels)
    interval = int(1e6 / hz)
    target_ts = list(range(start_us, end_us+1, interval))

    synced = {ch: {'filename': [], 'timestamp': []} for ch in channels}

    for t in target_ts:
        for ch in channels:
            ts_list = img_dict[ch]['timestamp']
            fn_list = img_dict[ch]['filename']
            idx = bisect.bisect_left(ts_list, t)
            if idx == 0:
                pick = 0
            elif idx >= len(ts_list):
                pick = len(ts_list) - 1
            else:
                before, after = ts_list[idx-1], ts_list[idx]
                pick = idx-1 if abs(before-t) <= abs(after-t) else idx

            synced[ch]['timestamp'].append(ts_list[pick])
            synced[ch]['filename'].append(fn_list[pick])

    return synced

def compute_time_diffs(synced):
    """
    （可选）计算各帧组内 6 通道时间戳差值的最大/最小，用于验证
    """
    channels = sorted(synced.keys())
    n = len(synced[channels[0]]['timestamp'])
    max_diffs, min_diffs = [], []
    for i in range(n):
        ts = [synced[ch]['timestamp'][i]/1e6 for ch in channels]
        diffs = [abs(a-b) for idx,a in enumerate(ts) for b in ts[idx+1:]]
        max_diffs.append(max(diffs) if diffs else 0)
        pos = [d for d in diffs if d>0]
        min_diffs.append(min(pos) if pos else 0)
    return max_diffs, min_diffs

def compute_max_timestamp_diff(synced):
    """
    计算同步后每组数据中，各通道时间戳的最大差值。

    Args:
        synced (dict): sync_image_dict 输出。

    Returns:
        list: 每个同步时刻的最大时间戳差值（单位：微秒）。
    """
    channels = sorted(synced.keys())
    n = len(synced[channels[0]]['timestamp'])
    max_diffs = []
    for i in range(n):
        ts_group = [synced[ch]['timestamp'][i] for ch in channels]
        max_diffs.append(max(ts_group) - min(ts_group))
    return max_diffs

def main():
    parser = argparse.ArgumentParser(description='Generate SJTU test infos from synced camera frames')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='根目录，包含各通道 sweeps 子文件夹')
    parser.add_argument('--pkl_save_path', type=str, required=True,
                        help='保存输出 pkl 文件的目录')
    parser.add_argument('--image_size', type=int, nargs=2, required=True,
                        metavar=('HEIGHT','WIDTH'),
                        help='单通道图像的高和宽')
    parser.add_argument('--scene_idx', type=int, default=0,
                        help='场景索引')
    parser.add_argument('--yaml_path', type=str, required=True,
                        help='包含相机内外参的 YAML 文件路径')
    args = parser.parse_args()

    # 构建和同步帧字典
    img_dict = build_image_index(args.base_dir)
    calib    = load_camera_projections(args.yaml_path)

    # 生成测试信息列表
    sjtu_test_infos = []
    channels = sorted(img_dict.keys())
    frame_count = len(img_dict[channels[0]]['timestamp'])

    for i in range(1, frame_count):
        seq = img_dict[channels[0]]['sequence'][i]
        if seq < 1640 or seq > 10060:
            continue
        prev_data, curr_data = {}, {}
        for ch in channels:
            prev_data[ch] = {
                'filename':  img_dict[ch]['filename'][i-1],
                'timestamp': img_dict[ch]['timestamp'][i-1]
            }
            curr_data[ch] = {
                'filename':  img_dict[ch]['filename'][i],
                'timestamp': img_dict[ch]['timestamp'][i]
            }
        info = {
            'prev_camera_data':  prev_data,
            'curr_camera_data':  curr_data,
            'prev_lidar_data':   None,
            'curr_lidar_data':   None,
            'sensor_metas_prev': calib,
            'sensor_metas_curr': calib,
            'gt_map_path':       None,
            'scene_flow_path':   None,
            'scene_indice':      None,
            'ros_msg_seq':       seq
        }
        sjtu_test_infos.append(info)

    # 确保保存目录存在
    os.makedirs(args.pkl_save_path, exist_ok=True)
    pkl_name = f"scene_{args.scene_idx}_sjtu_test_infos_{args.image_size[0]}_{args.image_size[1]*len(channels)}.pkl"
    pkl_path = os.path.join(args.pkl_save_path, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(sjtu_test_infos, f)

    print(f"Saved {len(sjtu_test_infos)} info entries to {pkl_path}")

if __name__ == "__main__":
    main()