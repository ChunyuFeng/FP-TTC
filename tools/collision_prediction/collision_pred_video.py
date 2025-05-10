#!/usr/bin/env python3
import os
import cv2
import re
import argparse

def make_video_multiple(input_folder, prefixes, exts, labels, output_path, fps=10):
    """
    从 input_folder 中按 prefixes 顺序读取多路图像序列，
    垂直拼接成一帧，在每个子图左上角添加 labels，
    并在最上面一张图像的右上角添加 ttc/risk 信息，写成视频。
    """
    # ——— 从文件夹名里提取 ttc 和 risk ———
    base = os.path.basename(os.path.normpath(input_folder))
    m = re.search(r'ttc([\d\.]+)_risk([\d\.]+)', base)
    if m:
        ttc_val, risk_val = m.group(1), m.group(2)
        info_text = f"TTC {ttc_val}   Risk {risk_val}"
    else:
        ttc_val = risk_val = None
        info_text = None

    files = os.listdir(input_folder)
    def sorted_files(prefix):
        matched = [f for f in files
                   if any(f.startswith(prefix) and f.endswith(ext) for ext in exts)]
        def idx(f):
            m2 = re.search(r'(\d+)', f)
            return int(m2.group(1)) if m2 else -1
        return sorted(matched, key=idx)

    lists = [sorted_files(p) for p in prefixes]
    lengths = [len(l) for l in lists]
    if len(set(lengths)) != 1:
        raise ValueError(f"Frame count mismatch: {dict(zip(prefixes, lengths))}")
    if labels and len(labels) != len(prefixes):
        raise ValueError(f"Labels count {len(labels)} must match prefixes count {len(prefixes)}")

    # 读第一帧确定每个子图大小
    first_imgs = [cv2.imread(os.path.join(input_folder, lists[i][0])) for i in range(len(prefixes))]
    hs, ws = zip(*(img.shape[:2] for img in first_imgs))
    if len(set(ws)) != 1:
        raise ValueError(f"Widths differ: {dict(zip(prefixes, ws))}")
    sub_w, sub_h = ws[0], hs[0]
    frame_w, frame_h = sub_w, sum(hs)

    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, f'output_ttc{ttc_val}_risk{risk_val}.mp4')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness  = 2
    label_color= (0, 255, 0)  # 绿色

    for frames in zip(*lists):
        imgs = [cv2.imread(os.path.join(input_folder, fn)) for fn in frames]

        # 在每张子图左上角绘制 labels
        for i, img in enumerate(imgs):
            text = labels[i] if labels else prefixes[i]
            cv2.putText(img, text, (10, 30), font, font_scale, label_color, thickness)

        # 在最上面那张子图右上角绘制 ttc/risk
        if info_text is not None:
            top_img = imgs[0]
            (text_w, text_h), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
            # 右上角坐标：x = 宽度 - 文本宽度 - 10，y = 30
            x = sub_w - text_w - 10
            y = 30
            cv2.putText(top_img, info_text, (x, y), font, font_scale, label_color, thickness)

        stacked = cv2.vconcat(imgs)
        writer.write(stacked)

    writer.release()
    print(f"✔ Video saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate video by stacking image sequences with labels and ttc/risk info"
    )
    parser.add_argument('--input_folder', required=True,
                        help="Folder containing image sequences, e.g. ..._ttc3.0_risk1.0")
    parser.add_argument('--prefixes', nargs='+', required=True,
                        help="Filename prefixes in stack order")
    parser.add_argument('--labels', nargs='+', default=None,
                        help="Labels for each sequence")
    parser.add_argument('--exts', nargs='+', default=['.jpg','.png'],
                        help="Allowed file extensions")
    parser.add_argument('--output_path', required=True,
                        help="Path or directory to save the output MP4")
    parser.add_argument('--fps', type=int, default=10,
                        help="Frames per second for output video")
    args = parser.parse_args()

    make_video_multiple(
        input_folder=args.input_folder,
        prefixes=args.prefixes,
        exts=args.exts,
        labels=args.labels,
        output_path=args.output_path,
        fps=args.fps
    )
