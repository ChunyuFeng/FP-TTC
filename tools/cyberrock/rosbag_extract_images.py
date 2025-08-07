#!/usr/bin/env python3
import os
import sys
import argparse
import rosbag
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="从 ROS bag 中提取指定相机话题的压缩图像并保存为 JPEG"
    )
    parser.add_argument(
        '--bag_path', '-b',
        required=True,
        help='输入的 ROS bag 文件路径'
    )
    parser.add_argument(
        '--output_root', '-o',
        required=True,
        help='输出图片保存的根目录'
    )
    args = parser.parse_args()

    bag_path = args.bag_path
    output_root = args.output_root

    # 只保留这 6 条话题
    topic_map = {
        '/multi/camera/wide_len1/compressed': 'CAM_FRONT',
        '/multi/camera/wide_len2/compressed': 'CAM_FRONT_RIGHT',
        '/multi/camera/wide_len3/compressed': 'CAM_BACK_RIGHT',
        '/multi/camera/wide_len4/compressed': 'CAM_BACK',
        '/multi/camera/wide_len5/compressed': 'CAM_BACK_LEFT',
        '/multi/camera/wide_len6/compressed': 'CAM_FRONT_LEFT',
    }

    # 检查 bag 文件是否存在
    if not os.path.exists(bag_path):
        print(f"ERROR: 找不到 rosbag: {bag_path}")
        sys.exit(1)

    # 创建输出目录
    for folder in topic_map.values():
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    count = 0
    with rosbag.Bag(bag_path, 'r') as bag:
        messages = bag.read_messages(topics=list(topic_map.keys()))
        for topic, msg, _ in tqdm(messages, desc="Extracting images", unit="msg"):
            try:
                data = msg.data
                # fmt = msg.format  # 可用于检查图像格式
            except AttributeError:
                continue

            folder = topic_map[topic]
            ts_usec = msg.header.stamp.secs * 1_000_000 + (msg.header.stamp.nsecs // 1_000)
            seq = msg.header.seq
            # 构造文件名：CAMERA__seq_{idx}__timestamp.jpg
            filename = f"{folder}__seq_{seq}__{ts_usec}.jpg"
            filepath = os.path.join(output_root, folder, filename)

            with open(filepath, 'wb') as f:
                f.write(data)
            count += 1

    print(f"Done. 总共保存了 {count} 张图片到 {output_root}。")

if __name__ == '__main__':
    main()
