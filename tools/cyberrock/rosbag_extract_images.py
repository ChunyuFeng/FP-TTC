#!/usr/bin/env python3
import os, sys
import rosbag

# 只保留这 6 条话题
topic_map = {
    '/multi/camera/wide_len1/compressed': 'CAM_FRONT',
    '/multi/camera/wide_len2/compressed': 'CAM_FRONT_RIGHT',
    '/multi/camera/wide_len3/compressed': 'CAM_BACK_RIGHT',
    '/multi/camera/wide_len4/compressed': 'CAM_BACK',
    '/multi/camera/wide_len5/compressed': 'CAM_BACK_LEFT',
    '/multi/camera/wide_len6/compressed': 'CAM_FRONT_LEFT',
}

bag_path     = './Datasets/cyberrock/rosbag/wide_len1-6_6-50.bag'
output_root  = './Datasets/cyberrock/sweeps'

# 检查 bag 文件是否存在
if not os.path.exists(bag_path):
    print(f"ERROR: 找不到 rosbag: {bag_path}")
    sys.exit(1)

# 创建输出目录
for folder in topic_map.values():
    os.makedirs(os.path.join(output_root, folder), exist_ok=True)

count = 0
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=list(topic_map.keys())):
        try:
            data = msg.data
            fmt  = msg.format  # 'bgr8; jpeg compressed bgr8'
        except AttributeError:
            continue

        folder = topic_map[topic]
        ts_usec = msg.header.stamp.secs * 1_000_000 + (msg.header.stamp.nsecs // 1_000)
        filename = f"{folder}__{ts_usec}.jpg"
        filepath = os.path.join(output_root, folder, filename)

        with open(filepath, 'wb') as f:
            f.write(data)
        count += 1

print(f"Done. 总共保存了 {count} 张图片到 {output_root}。")
