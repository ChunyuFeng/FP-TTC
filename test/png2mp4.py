# import cv2
# import os
# from glob import glob


# def create_video_from_images(image_folder, output_video, frame_rate=30):
#     """
#     将指定文件夹中的 pred{i}.jpg 图像按数字顺序拼接为 MP4 视频。

#     Args:
#         image_folder (str): 图像所在的文件夹路径。
#         output_video (str): 输出的视频文件路径，例如 'output.mp4'。
#         frame_rate (int): 视频帧率（每秒显示的帧数），默认值为 30。
#     """

#     # 获取所有 pred{i}.jpg 文件并按文件名中的数字排序
#     image_files = sorted(
#         glob(os.path.join(image_folder, 'pred*.jpg')),
#         key=lambda x: int(os.path.basename(x).split('pred')[-1].split('.jpg')[0])
#     )

#     if not image_files:
#         print("No 'pred*.jpg' files found in the folder.")
#         return

#     # 读取第一张图像以获取宽和高
#     first_image = cv2.imread(image_files[0])
#     height, width, layers = first_image.shape

#     # 初始化视频写入器
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式
#     video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

#     # 逐帧写入视频
#     for image_file in image_files:
#         img = cv2.imread(image_file)
#         if img.shape[:2] != (height, width):
#             print(f"Skipping {image_file}, size does not match.")
#             continue
#         video_writer.write(img)

#     # 释放资源
#     video_writer.release()
#     print(f"Video saved to {output_video}")


# # 示例用法
# if __name__ == "__main__":
#     image_folder = "./test/25_02_21-13_40_36_selfcon_ttc"  # 图像所在的文件夹路径
#     output_video = "./test/range_image_pred.mp4"  # 输出视频文件路径
#     frame_rate = 10  # 帧率

#     create_video_from_images(image_folder, output_video, frame_rate)

# import cv2
# import os
# import numpy as np
# from glob import glob

# def create_video_from_images(gt_folder, pred_folder, output_video, frame_rate=30):
#     """
#     将指定文件夹中的 gt{i}.jpg 和 pred{i}.jpg 图像按数字顺序拼接为 MP4 视频，
#     上半部分为 gt，下半部分为 pred。

#     Args:
#         gt_folder (str): gt 图像所在的文件夹路径。
#         pred_folder (str): pred 图像所在的文件夹路径。
#         output_video (str): 输出的视频文件路径，例如 'output.mp4'。
#         frame_rate (int): 视频帧率（每秒显示的帧数），默认值为 30。
#     """

#     # 获取所有 gt{i}.jpg 和 pred{i}.jpg 文件并按文件名中的数字排序
#     gt_files = sorted(
#         glob(os.path.join(gt_folder, 'gt*.jpg')),
#         key=lambda x: int(os.path.basename(x).split('gt')[-1].split('.jpg')[0])
#     )
#     pred_files = sorted(
#         glob(os.path.join(pred_folder, 'pred*.jpg')),
#         key=lambda x: int(os.path.basename(x).split('pred')[-1].split('.jpg')[0])
#     )

#     if not gt_files or not pred_files:
#         print("No 'gt*.jpg' or 'pred*.jpg' files found in the folders.")
#         return

#     # 确保两者图像数量一致
#     if len(gt_files) != len(pred_files):
#         print("The number of 'gt*.jpg' and 'pred*.jpg' files do not match.")
#         return

#     # 读取第一张图像以获取宽和高
#     first_gt_image = cv2.imread(gt_files[0])
#     height, width, layers = first_gt_image.shape

#     # 初始化视频写入器
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式
#     video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height * 2))

#     # 逐帧写入视频
#     for gt_file, pred_file in zip(gt_files, pred_files):
#         gt_img = cv2.imread(gt_file)
#         pred_img = cv2.imread(pred_file)

#         # 确保两张图像尺寸一致
#         if gt_img.shape[:2] != (height, width) or pred_img.shape[:2] != (height, width):
#             print(f"Skipping {gt_file} or {pred_file}, size does not match.")
#             continue

#         # 将两张图像上下拼接
#         combined_img = np.vstack((gt_img, pred_img))

#         # 写入视频
#         video_writer.write(combined_img)

#     # 释放资源
#     video_writer.release()
#     print(f"Video saved to {output_video}")


# if __name__ == "__main__":
#     # gt_folder = "./test/25_02_21-13_40_36_selfcon_ttc"  # gt 图像所在的文件夹路径
#     # pred_folder = "./test/25_02_21-13_40_36_selfcon_ttc"  # pred 图像所在的文件夹路径
#     gt_folder = "/home/chunyu/WorkSpace/BugStudio/FP-TTC/test/25_03_21-11_50_03_selfcon_ttc"
#     pred_folder = "/home/chunyu/WorkSpace/BugStudio/FP-TTC/test/25_03_21-11_50_03_selfcon_ttc"

#     output_video = "./test/gt_pred_160_320_full_frame.mp4"  # 输出视频文件路径
#     frame_rate = 10  # 帧率

#     create_video_from_images(gt_folder, pred_folder, output_video, frame_rate)

# import cv2
# import os
# from glob import glob

# def create_video_from_images(image_folder, output_video, frame_rate=30):
#     """
#     将指定文件夹中的 risk_score_map_{i}.png 图像按数字顺序拼接为 MP4 视频。

#     参数:
#         image_folder (str): 图像所在的文件夹路径。
#         output_video (str): 输出的视频文件路径，例如 'output.mp4'。
#         frame_rate (int): 视频帧率（每秒显示的帧数），默认值为 30。
#     """
#     # 获取所有 proj_risk_mask{i}.png 文件，并按文件名中的数字排序
#     image_files = sorted(
#         glob(os.path.join(image_folder, 'risk_score_map_*.png')),
#         key=lambda x: int(os.path.basename(x).split('risk_score_map_')[-1].split('.png')[0])
#     )

#     if not image_files:
#         print("No 'risk_score_map_*.png' files found in the folder.")
#         return

#     # 读取第一张图像以获取宽和高
#     first_image = cv2.imread(image_files[0])
#     if first_image is None:
#         print(f"Failed to read {image_files[0]}")
#         return
#     height, width, layers = first_image.shape

#     # 初始化视频写入器
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式
#     video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

#     # 逐帧写入视频
#     for image_file in image_files:
#         img = cv2.imread(image_file)
#         if img is None:
#             print(f"Failed to read {image_file}, skipping.")
#             continue
#         if img.shape[:2] != (height, width):
#             print(f"Skipping {image_file}, size does not match.")
#             continue
#         video_writer.write(img)

#     # 释放资源
#     video_writer.release()
#     print(f"Video saved to {output_video}")


# # 示例用法
# if __name__ == "__main__":
#     image_folder = "/mnt/data/fpttc_ground_truth/3_visualization/risk_score_map/160_1920"  # 图像所在的文件夹路径
#     output_video = "/mnt/data/fpttc_ground_truth/3_visualization/risk_score_map/160_1920/risk_score_map.mp4"  # 输出视频文件路径
#     frame_rate = 10  # 帧率

#     create_video_from_images(image_folder, output_video, frame_rate)
# '/home/chunyu/WorkSpace/BugStudio/FP-TTC/test/25_04_20-22_46_45_selfcon_ttc'

import cv2
import os
import re
import argparse

def make_video(input_folder, gt_prefix, pred_prefix, output_path, fps=10):
    """
    Read GT and Pred PNGs from input_folder, vertically stack them, and write to an MP4 at output_path.
    """
    # List and sort files matching the prefixes
    files = os.listdir(input_folder)
    pat = re.compile(rf'{re.escape(gt_prefix)}(\d+)\.png')
    def sorted_files(prefix):
        matched = [f for f in files if f.startswith(prefix) and f.endswith('.png')]
        return sorted(matched, key=lambda x: int(re.findall(r'(\d+)', x)[0]))

    gt_files = sorted_files(gt_prefix)
    pred_files = sorted_files(pred_prefix)
    if len(gt_files) != len(pred_files):
        raise ValueError(f"Number of GT files ({len(gt_files)}) does not match Pred files ({len(pred_files)}).")

    # Read first frame to get dimensions
    first_gt = cv2.imread(os.path.join(input_folder, gt_files[0]))
    h, w = first_gt.shape[:2]

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, 2 * h))

    # Process each pair
    for gt_name, pred_name in zip(gt_files, pred_files):
        gt_img = cv2.imread(os.path.join(input_folder, gt_name))
        pred_img = cv2.imread(os.path.join(input_folder, pred_name))
        stacked = cv2.vconcat([gt_img, pred_img])
        writer.write(stacked)

    writer.release()
    print(f"✔ Video saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate GT vs Pred videos (risk & scale)")
    parser.add_argument('--input_folder', required=True, help="Folder with GT and Pred PNGs")
    parser.add_argument('--output_dir', required=True, help="Directory to save the MP4 videos")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second for the output videos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate risk video
    risk_out = os.path.join(args.output_dir, 'risk.mp4')
    make_video(
        input_folder=args.input_folder,
        gt_prefix='gt_risk_',
        pred_prefix='pred_risk_',
        output_path=risk_out,
        fps=args.fps
    )

    # Generate scale video
    scale_out = os.path.join(args.output_dir, 'scale.mp4')
    make_video(
        input_folder=args.input_folder,
        gt_prefix='gt_scale_',
        pred_prefix='pred_scale_',
        output_path=scale_out,
        fps=args.fps
    )
