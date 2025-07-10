# import cv2
# import os
# import re
# import argparse

# def make_video(input_folder, gt_prefix, pred_prefix, output_path, fps=10):
#     """
#     Read GT and Pred PNGs from input_folder, vertically stack them, and write to an MP4 at output_path.
#     """
#     # List and sort files matching the prefixes
#     files = os.listdir(input_folder)
#     pat = re.compile(rf'{re.escape(gt_prefix)}(\d+)\.png')
#     def sorted_files(prefix):
#         matched = [f for f in files if f.startswith(prefix) and f.endswith('.png')]
#         return sorted(matched, key=lambda x: int(re.findall(r'(\d+)', x)[0]))

#     gt_files = sorted_files(gt_prefix)
#     pred_files = sorted_files(pred_prefix)
#     if len(gt_files) != len(pred_files):
#         raise ValueError(f"Number of GT files ({len(gt_files)}) does not match Pred files ({len(pred_files)}).")

#     # Read first frame to get dimensions
#     first_gt = cv2.imread(os.path.join(input_folder, gt_files[0]))
#     h, w = first_gt.shape[:2]

#     # Prepare VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (w, 2 * h))

#     # Process each pair
#     for gt_name, pred_name in zip(gt_files, pred_files):
#         gt_img = cv2.imread(os.path.join(input_folder, gt_name))
#         pred_img = cv2.imread(os.path.join(input_folder, pred_name))
#         stacked = cv2.vconcat([gt_img, pred_img])
#         writer.write(stacked)

#     writer.release()
#     print(f"✔ Video saved to: {output_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Generate GT vs Pred videos (risk & scale)")
#     parser.add_argument('--input_folder', required=True, help="Folder with GT and Pred PNGs")
#     parser.add_argument('--output_dir', required=True, help="Directory to save the MP4 videos")
#     parser.add_argument('--fps', type=int, default=10, help="Frames per second for the output videos")
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)

#     # Generate risk video
#     risk_out = os.path.join(args.output_dir, 'risk.mp4')
#     make_video(
#         input_folder=args.input_folder,
#         gt_prefix='gt_risk_',
#         pred_prefix='pred_risk_',
#         output_path=risk_out,
#         fps=args.fps
#     )

#     # Generate scale video
#     scale_out = os.path.join(args.output_dir, 'scale.mp4')
#     make_video(
#         input_folder=args.input_folder,
#         gt_prefix='gt_scale_',
#         pred_prefix='pred_scale_',
#         output_path=scale_out,
#         fps=args.fps
#     )

# import cv2
# import os
# import re
# import argparse


# def make_video(input_folder, output_path, fps=10):
#     """
#     从 input_folder 读取 concat_prev_{idx}.png, pred_scale_{idx}.png, pred_risk_{idx}.png
#     按 idx 排序，每帧按顺序垂直拼接三张图像，写入 output_path 指定的 MP4 文件。
#     """
#     # 列出文件并提取所有可用的 idx
#     files = os.listdir(input_folder)
#     pat = re.compile(r'concat_prev_(\d+)\.png')
#     idxs = sorted({int(m.group(1)) for f in files if (m := pat.match(f))})

#     if not idxs:
#         raise ValueError(f"在文件夹 {input_folder} 中未找到任何 concat_prev_{{idx}}.png 文件。")

#     # 读取第一帧来获取尺寸
#     first_idx = idxs[0]
#     first_prev = cv2.imread(os.path.join(input_folder, f"concat_prev_{first_idx}.png"))
#     if first_prev is None:
#         raise ValueError(f"无法读取图片: concat_prev_{first_idx}.png")
#     h, w = first_prev.shape[:2]

#     # 视频写入器，帧大小为 (w, 3*h)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (w, 3 * h))

#     for idx in idxs:
#         prev_path  = os.path.join(input_folder, f"concat_prev_{idx}.png")
#         scale_path = os.path.join(input_folder, f"pred_scale_{idx}.png")
#         risk_path  = os.path.join(input_folder, f"pred_risk_{idx}.png")

#         # 逐张读取并检查
#         prev_img  = cv2.imread(prev_path)
#         scale_img = cv2.imread(scale_path)
#         risk_img  = cv2.imread(risk_path)
#         if prev_img is None:
#             raise ValueError(f"无法读取图片: {prev_path}")
#         if scale_img is None:
#             raise ValueError(f"无法读取图片: {scale_path}")
#         if risk_img is None:
#             raise ValueError(f"无法读取图片: {risk_path}")

#         # 拼接并写入
#         stacked = cv2.vconcat([prev_img, scale_img, risk_img])
#         writer.write(stacked)

#     writer.release()
#     print(f"✔ 视频已保存到: {output_path}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="生成 concat_prev vs pred_scale vs pred_risk 视频")
#     parser.add_argument('--input_folder', required=True, help="包含 PNG 文件的目录")
#     parser.add_argument('--output_path',  required=True, help="输出 MP4 文件路径，如 ./output.mp4")
#     parser.add_argument('--fps', type=int, default=10, help="生成视频的帧率 (默认为 10)")
#     args = parser.parse_args()

#     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
#     make_video(
#         input_folder=args.input_folder,
#         output_path=args.output_path,
#         fps=args.fps
#     )


import cv2
import os
import re
import argparse


def make_video(input_folder, output_path, fps=10):
    """
    从 input_folder 读取 concat_prev_{idx}.png, gt_scale_{idx}.png, pred_scale_{idx}.png,
    gt_risk_{idx}.png, pred_risk_{idx}.png，按 idx 排序，每帧按顺序垂直拼接五张图像，
    写入 output_path 指定的 MP4 文件。
    """
    # 列出文件并提取所有可用的 idx
    files = os.listdir(input_folder)
    pat = re.compile(r'concat_prev_(\d+)\.png')
    idxs = sorted({int(m.group(1)) for f in files if (m := pat.match(f))})

    if not idxs:
        raise ValueError(f"在文件夹 {input_folder} 中未找到任何 concat_prev_{{idx}}.png 文件。")

    # 读取第一帧 concat_prev 来获取尺寸（假定五张图尺寸相同）
    first_idx = idxs[0]
    first_prev = cv2.imread(os.path.join(input_folder, f"concat_prev_{first_idx}.png"))
    if first_prev is None:
        raise ValueError(f"无法读取图片: concat_prev_{first_idx}.png")
    h, w = first_prev.shape[:2]

    # 视频写入器，帧大小为 (w, 5*h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, 5 * h))

    for idx in idxs:
        prev_path      = os.path.join(input_folder, f"concat_prev_{idx}.png")
        gt_scale_path  = os.path.join(input_folder, f"gt_scale_{idx}.png")
        pred_scale_path= os.path.join(input_folder, f"pred_scale_{idx}.png")
        gt_risk_path   = os.path.join(input_folder, f"gt_risk_{idx}.png")
        pred_risk_path = os.path.join(input_folder, f"pred_risk_{idx}.png")

        # 逐张读取并检查
        prev_img       = cv2.imread(prev_path)
        gt_scale_img   = cv2.imread(gt_scale_path)
        pred_scale_img = cv2.imread(pred_scale_path)
        gt_risk_img    = cv2.imread(gt_risk_path)
        pred_risk_img  = cv2.imread(pred_risk_path)

        for path, img in [
            (prev_path, prev_img),
            (gt_scale_path, gt_scale_img),
            (pred_scale_path, pred_scale_img),
            (gt_risk_path, gt_risk_img),
            (pred_risk_path, pred_risk_img),
        ]:
            if img is None:
                raise ValueError(f"无法读取图片: {path}")

        # 垂直拼接五张图并写入视频
        stacked = cv2.vconcat([
            prev_img,
            gt_scale_img,
            pred_scale_img,
            gt_risk_img,
            pred_risk_img
        ])
        writer.write(stacked)

    writer.release()
    print(f"✔ 视频已保存到: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成 concat_prev vs gt_scale vs pred_scale vs gt_risk vs pred_risk 视频")
    parser.add_argument('--input_folder', required=True, help="包含 PNG 文件的目录")
    parser.add_argument('--output_path',  required=True, help="输出 MP4 文件路径，如 ./output.mp4")
    parser.add_argument('--fps', type=int, default=10, help="生成视频的帧率 (默认为 10)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    make_video(
        input_folder=args.input_folder,
        output_path=args.output_path,
        fps=args.fps
    )
