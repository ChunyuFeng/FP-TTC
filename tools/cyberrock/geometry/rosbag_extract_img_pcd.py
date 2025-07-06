#!/usr/bin/env python3
import os
import sys
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2

# 六路相机 + 一路激光
topic_map = {
    # 相机话题 → 文件夹名
    '/multi/camera/wide_len1/compressed': 'CAM_FRONT',
    '/multi/camera/wide_len2/compressed': 'CAM_FRONT_RIGHT',
    '/multi/camera/wide_len3/compressed': 'CAM_BACK_RIGHT',
    '/multi/camera/wide_len4/compressed': 'CAM_BACK',
    '/multi/camera/wide_len5/compressed': 'CAM_BACK_LEFT',
    '/multi/camera/wide_len6/compressed': 'CAM_FRONT_LEFT',
    # 激光雷达话题 → 文件夹名
    '/driver/hesai/pandar':            'LIDAR_TOP',
}

bag_path    = './Datasets/cyberrock/rosbag/proj.bag'
output_root = './Datasets/cyberrock/proj_test/sweeps'

# 检查输入
if not os.path.exists(bag_path):
    print(f"ERROR: 找不到 rosbag: {bag_path}")
    sys.exit(1)

# 创建输出子目录
for folder in set(topic_map.values()):
    os.makedirs(os.path.join(output_root, folder), exist_ok=True)

count_img = 0
count_pc  = 0

with rosbag.Bag(bag_path, 'r') as bag:
    # 一次性订阅所有七个话题
    for topic, msg, _ in bag.read_messages(topics=list(topic_map.keys())):
        folder   = topic_map[topic]
        # 微秒级时间戳，用于文件名
        ts_usec = msg.header.stamp.secs * 1_000_000 + (msg.header.stamp.nsecs // 1_000)

        out_dir  = os.path.join(output_root, folder)

        if folder == 'LIDAR_TOP':
            # 处理 PointCloud2：只提取 x,y,z，保存为 .npy
            points = []
            for x, y, z in pc2.read_points(msg,
                                           field_names=('x','y','z'),
                                           skip_nans=True):
                points.append((x, y, z))
            points = np.asarray(points, dtype=np.float32)

            filename = f"{folder}__{ts_usec}.npy"
            filepath = os.path.join(out_dir, filename)
            np.save(filepath, points)
            count_pc += 1

        else:
            # 处理压缩图像
            try:
                data = msg.data
            except AttributeError:
                # 万一消息格式不符，跳过
                continue

            filename = f"{folder}__{ts_usec}.jpg"
            filepath = os.path.join(out_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(data)
            count_img += 1

print(f"Done. 保存了 {count_img} 张图片，{count_pc} 个点云文件 到 {output_root}。")
