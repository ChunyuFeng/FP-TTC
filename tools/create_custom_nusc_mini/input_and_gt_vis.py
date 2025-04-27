import os
import re
from PIL import Image
import numpy as np
import cv2

def get_sorted_pngs(folder):
    """
    读取 folder 下所有 .png 文件，根据文件名末尾的 _{index}.png 提取 index 并升序返回文件名列表
    """
    pngs = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    pattern = re.compile(r'_(\d+)\.png$')
    def idx(fn):
        m = pattern.search(fn)
        if not m:
            raise ValueError(f"文件名 '{fn}' 不符合 '_{{数字}}.png' 规则")
        return int(m.group(1))
    return sorted(pngs, key=idx)

def vconcat(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    垂直拼接两张 PIL.Image，宽度取较大者，高度相加
    """
    w1, h1 = img1.size
    w2, h2 = img2.size
    new_w = max(w1, w2)
    new_h = h1 + h2
    new_img = Image.new('RGB', (new_w, new_h), (255, 255, 255))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, h1))
    return new_img

def concat_to_video(folder_a, folder_b, video_path, fps=10):
    """
    将 folder_a 与 folder_b 中按序号对应的 PNG 图上下拼接，并写入一个视频文件。
    :param folder_a: A 文件夹路径
    :param folder_b: B 文件夹路径
    :param video_path: 输出视频路径（如 .mp4）
    :param fps: 帧率
    """
    files_a = get_sorted_pngs(folder_a)
    files_b = get_sorted_pngs(folder_b)
    if len(files_a) != len(files_b):
        raise RuntimeError("A、B 两个文件夹中的 png 数量不一致！")

    # 准备 VideoWriter：先拼接第一帧获取尺寸
    img0_a = Image.open(os.path.join(folder_a, files_a[0]))
    img0_b = Image.open(os.path.join(folder_b, files_b[0]))
    first = vconcat(img0_a, img0_b)
    w, h = first.size  # PIL 的 size 是 (width, height)
    
    # OpenCV VideoWriter 要求尺寸为 (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者 'XVID'
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for fn_a, fn_b in zip(files_a, files_b):
        # 读取并拼接
        img_a = Image.open(os.path.join(folder_a, fn_a))
        img_b = Image.open(os.path.join(folder_b, fn_b))
        combined = vconcat(img_a, img_b)

        # PIL -> numpy (RGB->BGR)
        frame = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    print(f"视频已保存到: {video_path}")

if __name__ == "__main__":
    folder_A = "/mnt/data/fpttc_ground_truth/3_visualization/raw_surround_view_imgs/160_1920"
    folder_B = "/mnt/data/fpttc_ground_truth/3_visualization/scale_map/160_1920"
    output_video = "/mnt/data/fpttc_ground_truth/3_visualization/combined_160_1920.mp4"
    # 可以根据需要调整 fps
    concat_to_video(folder_A, folder_B, output_video, fps=5)
