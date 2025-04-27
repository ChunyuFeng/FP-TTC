import pickle
import argparse
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def resize_and_crop(img, width, height):
    """将图像缩放并裁剪到指定大小"""
    w, h = img.size
    crop_h, crop_w = height, width
    resize = crop_w / w

    resize_h, resize_w = int(h * resize), int(w * resize)
    crop_h_start = 0
    crop_w_start = (resize_w - crop_w) // 2
    crop = (crop_w_start, crop_h_start, crop_w_start + crop_w, crop_h_start + crop_h)

    img = img.resize((resize_w, resize_h), Image.BILINEAR)

    img = img.crop(crop)

    return img

def load_pkl_file(filepath):
    """
    从指定路径读取 pkl 文件，并返回内容.
    """
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def main(args):
    """
    主函数，解析命令行参数并调用加载函数.
    """
    data = load_pkl_file(args.pkl_file_path)

    if data is not None:
        data_length = len(data)
        for i in tqdm(range(data_length), desc="Processing data"):
            prev_camera_data = data[i]['prev_camera_data']
            curr_camera_data = data[i]['curr_camera_data']

            if args.surr_view_imgs_vis:
                # 按预定义顺序读取6个相机通道的图像
                ordered_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
                img_list = []
                for channel in ordered_channels:
                    camera_sample_data = prev_camera_data[channel]
                    file_path = os.path.join(args.nusc_root, camera_sample_data['filename'])
                    img = Image.open(file_path)
                    img = resize_and_crop(img, 320, 160)  # 调整图像大小
                    img_list.append(img)

                # 拼接为一行图像
                combined_img = np.hstack(img_list)

                # 保存拼接后的图像
                combined_img_path = os.path.join(args.vis_dir, f"combined_prev_surround_view_{i}.png")
                plt.imsave(combined_img_path, combined_img)

        print(f"成功处理 {data_length} 条数据.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pkl file and print its content.")
    parser.add_argument('--vis_dir', type=str, default='/mnt/data/fpttc_ground_truth/3_visualization/raw_surround_view_imgs/160_1920',
                        help="Directory to save visualizations.")
    args = parser.parse_args()
    main(args=args)