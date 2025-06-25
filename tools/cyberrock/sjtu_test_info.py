#!/usr/bin/env python3
import os
import sys
import bisect
import pickle
import pprint
import argparse

def build_image_index(base_dir):
    """
    遍历 base_dir 下的每个子文件夹（channel），
    收集其中所有 .jpg 文件，按文件名中的时间戳排序，
    返回结构：
      { channel_name: { 'filename': [...], 'timestamp': [...] }, ... }
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
            try:
                ts = int(fn.rsplit('__', 1)[1].split('.')[0])
            except (IndexError, ValueError):
                continue
            items.append((ts, os.path.join(channel_dir, fn)))

        if not items:
            continue

        items.sort(key=lambda x: x[0])
        img_dict[channel] = {
            'filename':  [path for ts, path in items],
            'timestamp': [ts   for ts, path in items]
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SJTU test infos from synced camera frames')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='根目录，包含各通道 sweeps 子文件夹')
    parser.add_argument('--pkl_save_path', type=str, required=True,
                        help='保存输出 pkl 文件的目录')
    parser.add_argument('--image_size', type=int, nargs=2, required=True,
                        metavar=('HEIGHT','WIDTH'),
                        help='单通道图像的高和宽')
    parser.add_argument('--hz', type=int, default=10,
                        help='同步采样的目标帧率 (Hz)')
    args = parser.parse_args()

    # 构建和同步帧字典
    img_dict = build_image_index(args.base_dir)
    synced   = sync_image_dict(img_dict, hz=args.hz)

    # 生成测试信息列表
    sjtu_test_infos = []
    channels = sorted(synced.keys())
    frame_count = len(synced[channels[0]]['timestamp'])

    for i in range(1, frame_count):
        prev_data, curr_data = {}, {}
        for ch in channels:
            prev_data[ch] = {
                'filename':  synced[ch]['filename'][i-1],
                'timestamp': synced[ch]['timestamp'][i-1]
            }
            curr_data[ch] = {
                'filename':  synced[ch]['filename'][i],
                'timestamp': synced[ch]['timestamp'][i]
            }

        info = {
            'prev_camera_data': prev_data,
            'curr_camera_data': curr_data,
            'prev_lidar_data':  None,
            'curr_lidar_data':  None,
            'sensor_metas':     None,
            'gt_map_path':      None,
            'scene_flow_path':  None,
            'scene_indice':     None,
        }
        sjtu_test_infos.append(info)

    # 确保保存目录存在
    os.makedirs(args.pkl_save_path, exist_ok=True)
    pkl_name = f"sjtu_test_infos_{args.image_size[0]}_{args.image_size[1]*len(channels)}.pkl"
    pkl_path = os.path.join(args.pkl_save_path, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(sjtu_test_infos, f)

    print(f"Saved {len(sjtu_test_infos)} info entries to {pkl_path}")
