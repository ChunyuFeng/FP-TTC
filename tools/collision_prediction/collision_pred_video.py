import os
import cv2
import numpy as np

# 配置参数
image_folder = '/mnt/fpttc_data/output_vis/collision_detect/collision_points_on_rgb'  # 存放图像的文件夹
output_video_file = '/mnt/fpttc_data/output_vis/collision_detect/collision_on_rgb.mp4'  # 输出视频文件路径（mp4格式）
fps = 5  # 视频帧率

# 实际所用的6个通道顺序
channel_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

# 按index分组图像，字典结构: {index: {channel: file_path}}
images_by_index = {}

for filename in os.listdir(image_folder):
    if not filename.endswith('.jpg'):
        continue
    # 遍历每个通道，确保匹配时后面紧跟数字，避免误匹配
    for channel in channel_order:
        if filename.startswith(channel) and len(filename) > len(channel) and filename[len(channel)].isdigit():
            # 提取index部分，例如 "CAM_FRONT_RIGHT227.jpg" 中 index 为 "227"
            index = filename[len(channel):-4]
            if index not in images_by_index:
                images_by_index[index] = {}
            images_by_index[index][channel] = os.path.join(image_folder, filename)
            break

# 按照index（转换为整数）顺序排序
sorted_indices = sorted(images_by_index.keys(), key=lambda x: int(x))

stitched_images = []  # 存放拼接后的图像

for index in sorted_indices:
    channels_dict = images_by_index[index]
    # 检查6个通道是否齐全
    if all(channel in channels_dict for channel in channel_order):
        row1_imgs = []
        row2_imgs = []
        # 第一行：CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT
        for channel in channel_order[:3]:
            img = cv2.imread(channels_dict[channel])
            if img is None:
                print(f"读取图像失败: {channels_dict[channel]}")
                continue
            row1_imgs.append(img)
        # 第二行：CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
        for channel in channel_order[3:]:
            img = cv2.imread(channels_dict[channel])
            if img is None:
                print(f"读取图像失败: {channels_dict[channel]}")
                continue
            row2_imgs.append(img)
        
        if len(row1_imgs) == 3 and len(row2_imgs) == 3:
            # 假设各图像尺寸一致，进行水平拼接和垂直拼接
            row1 = np.hstack(row1_imgs)
            row2 = np.hstack(row2_imgs)
            stitched = np.vstack([row1, row2])
            stitched_images.append(stitched)
        else:
            print(f"Index {index} 部分图像读取失败，无法拼接。")
    else:
        print(f"Index {index} 缺少部分通道，跳过拼接。")

# 若存在拼接后的图像，则生成视频
if stitched_images:
    height, width, layers = stitched_images[0].shape
    # 使用mp4v编码生成mp4视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))
    
    for img in stitched_images:
        video_writer.write(img)
    
    video_writer.release()
    print("视频已保存至:", output_video_file)
else:
    print("没有生成拼接图像，无法生成视频。")
