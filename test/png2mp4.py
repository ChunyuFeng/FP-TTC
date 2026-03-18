import cv2
import os
import re
import argparse

def make_video(input_folder, output_path, fps=10, start_idx=None, end_idx=None):
    """
    1. 读取 concat_prev_{idx}, new_pred_scale_{idx}, new_pred_orien_{idx}
    2. 以 new_pred_scale_{idx} 的尺寸为基准，将其他两张图缩放到相同大小
    3. 纵向拼接并生成视频
    """
    # 1. 扫描文件夹，提取所有存在的 idx
    files = os.listdir(input_folder)
    pat = re.compile(r'concat_prev_(\d+)\.png')
    
    all_idxs = sorted({int(m.group(1)) for f in files if (m := pat.match(f))})

    if not all_idxs:
        raise ValueError(f"在文件夹 {input_folder} 中未找到任何 concat_prev_{{idx}}.png 文件。")

    # 2. 根据指定的范围过滤 idx
    idxs = all_idxs
    if start_idx is not None:
        idxs = [i for i in idxs if i >= start_idx]
    if end_idx is not None:
        idxs = [i for i in idxs if i <= end_idx]

    if not idxs:
        raise ValueError(f"在指定的范围 ({start_idx} ~ {end_idx}) 内没有找到有效的文件索引。")

    print(f"即将处理索引范围: {idxs[0]} 到 {idxs[-1]} (共 {len(idxs)} 帧)")

    writer = None
    video_w = 0
    video_h_unit = 0  # 单张图的高度
    
    # 3. 循环处理
    for idx in idxs:
        # 定义当前帧的文件路径
        prev_path   = os.path.join(input_folder, f"concat_prev_{idx}.png")
        scale_path  = os.path.join(input_folder, f"new_pred_scale_{idx}.png")
        orien_path  = os.path.join(input_folder, f"new_pred_orien_{idx}.png")

        # 读取图片
        prev_img   = cv2.imread(prev_path)
        scale_img  = cv2.imread(scale_path)
        orien_img  = cv2.imread(orien_path)

        # 检查文件是否存在
        if prev_img is None:
            print(f"警告: 缺失文件 {prev_path}，跳过此帧。")
            continue
        if scale_img is None:
            print(f"警告: 缺失文件 {scale_path}，跳过此帧。")
            continue
        if orien_img is None:
            print(f"警告: 缺失文件 {orien_path}，跳过此帧。")
            continue

        # 获取当前 scale 图的尺寸
        curr_h, curr_w = scale_img.shape[:2]

        # --- 初始化 VideoWriter (仅在第一帧) ---
        if writer is None:
            if curr_w <= 0 or curr_h <= 0:
                raise ValueError(f"图片尺寸异常: 宽={curr_w}, 高={curr_h}，无法创建视频。")

            video_w = curr_w
            video_h_unit = curr_h
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 视频总高度 = 3 * 单张图高度
            writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, 3 * video_h_unit))
            
            if not writer.isOpened():
                raise RuntimeError("VideoWriter 初始化失败，请检查输出路径或编码格式。")
                
            print(f"视频尺寸初始化为: 宽={video_w}, 总高={3*video_h_unit} (单图高={video_h_unit})")

        # --- 统一缩放逻辑 ---
        # 目标尺寸 (width, height) -> 注意 resize 接受的是 (width, height)
        target_size = (video_w, video_h_unit)

        # 1. 检查 scale_img 是否需要缩放 (防止后续帧尺寸突变)
        if (scale_img.shape[1] != video_w) or (scale_img.shape[0] != video_h_unit):
            scale_img = cv2.resize(scale_img, target_size)

        # 2. 强制缩放 prev_img 到目标尺寸
        if (prev_img.shape[1] != video_w) or (prev_img.shape[0] != video_h_unit):
            prev_img = cv2.resize(prev_img, target_size)

        # 3. 强制缩放 orien_img 到目标尺寸
        if (orien_img.shape[1] != video_w) or (orien_img.shape[0] != video_h_unit):
            orien_img = cv2.resize(orien_img, target_size)

        # 纵向拼接: prev -> scale -> orien
        stacked = cv2.vconcat([prev_img, scale_img, orien_img])
        writer.write(stacked)

    if writer is not None:
        writer.release()
        print(f"✔ 视频已保存到: {output_path}")
    else:
        print("❌ 未生成任何视频帧（可能未找到有效图片）。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成视频 (自动缩放到 new_pred_scale 尺寸)")
    parser.add_argument('--input_folder', required=True, help="包含 PNG 文件的目录")
    parser.add_argument('--output_path',  required=True, help="输出 MP4 文件路径")
    parser.add_argument('--fps', type=int, default=10, help="帧率")
    parser.add_argument('--start_idx', type=int, default=None, help="起始索引")
    parser.add_argument('--end_idx',   type=int, default=None, help="结束索引")

    args = parser.parse_args()
    
    # 自动创建父目录
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    make_video(
        input_folder=args.input_folder,
        output_path=args.output_path,
        fps=args.fps,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
# import cv2
# import os
# import re
# import argparse


# def make_video(input_folder, output_path, fps=10):
#     """
#     从 input_folder 读取 concat_prev_{idx}.png, gt_scale_{idx}.png, pred_scale_{idx}.png,
#     gt_risk_{idx}.png, pred_risk_{idx}.png，按 idx 排序，每帧按顺序垂直拼接五张图像，
#     写入 output_path 指定的 MP4 文件。
#     """
#     # 列出文件并提取所有可用的 idx
#     files = os.listdir(input_folder)
#     pat = re.compile(r'concat_prev_(\d+)\.png')
#     idxs = sorted({int(m.group(1)) for f in files if (m := pat.match(f))})

#     if not idxs:
#         raise ValueError(f"在文件夹 {input_folder} 中未找到任何 concat_prev_{{idx}}.png 文件。")

#     # 读取第一帧 concat_prev 来获取尺寸（假定五张图尺寸相同）
#     first_idx = idxs[0]
#     first_prev = cv2.imread(os.path.join(input_folder, f"concat_prev_{first_idx}.png"))
#     if first_prev is None:
#         raise ValueError(f"无法读取图片: concat_prev_{first_idx}.png")
#     h, w = first_prev.shape[:2]

#     # 视频写入器，帧大小为 (w, 5*h)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (w, 5 * h))

#     for idx in idxs:
#         prev_path      = os.path.join(input_folder, f"concat_prev_{idx}.png")
#         gt_scale_path  = os.path.join(input_folder, f"gt_scale_{idx}.png")
#         pred_scale_path= os.path.join(input_folder, f"pred_scale_{idx}.png")
#         gt_risk_path   = os.path.join(input_folder, f"gt_risk_{idx}.png")
#         pred_risk_path = os.path.join(input_folder, f"pred_risk_{idx}.png")

#         # 逐张读取并检查
#         prev_img       = cv2.imread(prev_path)
#         gt_scale_img   = cv2.imread(gt_scale_path)
#         pred_scale_img = cv2.imread(pred_scale_path)
#         gt_risk_img    = cv2.imread(gt_risk_path)
#         pred_risk_img  = cv2.imread(pred_risk_path)

#         for path, img in [
#             (prev_path, prev_img),
#             (gt_scale_path, gt_scale_img),
#             (pred_scale_path, pred_scale_img),
#             (gt_risk_path, gt_risk_img),
#             (pred_risk_path, pred_risk_img),
#         ]:
#             if img is None:
#                 raise ValueError(f"无法读取图片: {path}")

#         # 垂直拼接五张图并写入视频
#         stacked = cv2.vconcat([
#             prev_img,
#             gt_scale_img,
#             pred_scale_img,
#             gt_risk_img,
#             pred_risk_img
#         ])
#         writer.write(stacked)

#     writer.release()
#     print(f"✔ 视频已保存到: {output_path}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="生成 concat_prev vs gt_scale vs pred_scale vs gt_risk vs pred_risk 视频")
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
