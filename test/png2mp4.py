import cv2
import os
from glob import glob


def create_video_from_images(image_folder, output_video, frame_rate=30):
    """
    将指定文件夹中的 pred{i}.jpg 图像按数字顺序拼接为 MP4 视频。

    Args:
        image_folder (str): 图像所在的文件夹路径。
        output_video (str): 输出的视频文件路径，例如 'output.mp4'。
        frame_rate (int): 视频帧率（每秒显示的帧数），默认值为 30。
    """

    # 获取所有 pred{i}.jpg 文件并按文件名中的数字排序
    image_files = sorted(
        glob(os.path.join(image_folder, 'gt*.jpg')),
        key=lambda x: int(os.path.basename(x).split('gt')[-1].split('.jpg')[0])
    )

    if not image_files:
        print("No 'pred*.jpg' files found in the folder.")
        return

    # 读取第一张图像以获取宽和高
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # 逐帧写入视频
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img.shape[:2] != (height, width):
            print(f"Skipping {image_file}, size does not match.")
            continue
        video_writer.write(img)

    # 释放资源
    video_writer.release()
    print(f"Video saved to {output_video}")


# 示例用法
if __name__ == "__main__":
    image_folder = "./test/25_02_21-13_40_36_selfcon_ttc"  # 图像所在的文件夹路径
    output_video = "./test/range_image_gt.mp4"  # 输出视频文件路径
    frame_rate = 10  # 帧率

    create_video_from_images(image_folder, output_video, frame_rate)
