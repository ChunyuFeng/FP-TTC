import cv2
import os
from glob import glob

def create_video_from_images(image_folder, output_video, frame_rate=30):
    """
    将指定文件夹中的 PNG 图像（形如 scale_i.png）拼接为 MP4 视频。

    Args:
        image_folder (str): 图像所在的文件夹路径。
        output_video (str): 输出的视频文件路径，例如 'output.mp4'。
        frame_rate (int): 视频帧率（每秒显示的帧数），默认值为 30。
    """
    # 获取所有符合 scale_*.png 的文件
    image_files = glob(os.path.join(image_folder, 'seismic_clip_range_*.png'))
    if not image_files:
        print("No PNG files found in the folder.")
        return

    # 按文件名中的数字进行排序
    # file_name 形如: scale_10.png -> basename: scale_10.png
    # 我们从 "scale_" 后面提取数值 "10" 再转成 int
    image_files = sorted(
        image_files,
        key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
    )

    # 读取第一张图像以获取宽和高
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print("Failed to read the first image. Exiting.")
        return

    height, width, _ = first_image.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # 逐帧写入视频
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: failed to read {image_file}, skipping.")
            continue

        # 检查尺寸是否匹配
        if img.shape[0] != height or img.shape[1] != width:
            print(f"Skipping {image_file}, size does not match.")
            continue

        video_writer.write(img)

    # 释放资源
    video_writer.release()
    print(f"Video saved to {output_video}")


if __name__ == "__main__":
    # 输入图像所在目录
    image_folder = "/mnt/fpttc_data/output_vis/31_range_img"
    # 输出视频文件路径
    output_video = "/mnt/fpttc_data/output_vis/31_range_img/seismic_clip_range_video.mp4"
    # 帧率可自行调整
    frame_rate = 30

    create_video_from_images(image_folder, output_video, frame_rate)
