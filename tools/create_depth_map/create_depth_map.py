import pickle
import argparse
import os
import shutil
from tqdm import tqdm
import cv2
import torch
import numpy as np
from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from dataloader.utils.augmentor import NuscRangeImageAugmentor
from PIL import Image

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

    # 加载图像增强器
    # Instantiate augmentor using --image_size as [height, width]
    resize_height, resize_width = args.image_size  # height, width from CLI
    # crop_size expects (crop_h, crop_w)
    augmentor = NuscRangeImageAugmentor(crop_size=(resize_height, resize_width),
                                        do_flip=False,
                                        rotate=False)

    # 加载 pkl 数据
    data = load_pkl_file(args.pkl_file_path)

    data_new = []

    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'pretrained/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model.to('cuda').eval()

    # raw_img = cv2.imread('your/image/path')
    # depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
        

    print(f"pkl 文件加载成功, 包含 {len(data)} 条数据.")

    channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    for idx, info in enumerate(tqdm(data, desc="Processing data")):
        prev_images = {}
        curr_images = {}
        for channel in channels:
            prev_path = os.path.join(args.nusc_root, info['prev_camera_data'][channel]['filename'])
            curr_path = os.path.join(args.nusc_root, info['curr_camera_data'][channel]['filename'])
            prev_images[channel] = Image.open(prev_path)
            curr_images[channel] = Image.open(curr_path)
        
        # 对 prev 和 curr 图像进行 crop 和 resize
        orig_size = next(iter(prev_images.values())).size  # 获取原始图像大小
        affine_params = augmentor.sample_params(orig_size)
        augmented_prev, _ = augmentor(prev_images, affine_params)
        augmented_curr, _ = augmentor(curr_images, affine_params)


        # 遍历每条 prev 数据
        for channel in channels:
            raw_image = augmented_prev[channel]
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR) 
            depth = model.infer_image(raw_image)

            image_filename = info['prev_camera_data'][channel]['filename']
            depth_filename = os.path.splitext(image_filename)[0] + ".npy"
            depth_save_path = os.path.join(args.output_dir, depth_filename)

            save_dir = os.path.dirname(depth_save_path)
            os.makedirs(save_dir, exist_ok=True)
            np.save(depth_save_path, depth)

            # 将 depth 路径信息添加到 info 中
            info['prev_camera_data'][channel]['depth_pred'] = depth_save_path

            # # 归一化深度图到 [0,255]
            # depth_vis = np.clip(depth, 0, max_depth) / max_depth * 255
            # depth_vis = depth_vis.astype(np.uint8)

            # # 应用伪彩色
            # depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # # 将原图和伪彩色深度图横向拼接
            # combined = np.hstack((raw_image, depth_color))

            # # 显示
            # cv2.imshow(f"{channel} - Image | Depth", combined)
            # cv2.waitKey(1)

        # 遍历每条 curr 数据
        for channel in channels:
            raw_image = augmented_curr[channel]
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR) 
            depth = model.infer_image(raw_image)

            image_filename = info['curr_camera_data'][channel]['filename']
            depth_filename = os.path.splitext(image_filename)[0] + ".npy"
            depth_save_path = os.path.join(args.output_dir, depth_filename)

            save_dir = os.path.dirname(depth_save_path)
            os.makedirs(save_dir, exist_ok=True)
            np.save(depth_save_path, depth)
            
            # 将 depth 路径信息添加到 info 中
            info['curr_camera_data'][channel]['depth_pred'] = depth_save_path
              
        data_new.append(info)
    
    # 保存处理后的数据到新的 pkl 文件
    output_pkl_path = os.path.join('./Datasets/nuscenes/2_trainval_test_infos', 'nusc_trainval_infos_160_1920_fov_8_15_dpt.pkl')
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(data_new, f)
    print(f"处理完成，新的 pkl 文件已保存到: {output_pkl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pkl file and print its content.")
    parser.add_argument('--image_size', default=[160, 320], type=int, nargs='+',
                    help='[height, width] for resize operations')
    parser.add_argument('--pkl_file_path', type=str, required=True, help="Path to the pkl file.")
    parser.add_argument('--nusc_root', type=str, default='/mnt/data/nuScenes', help="Root directory of the nuScenes dataset.")
    parser.add_argument('--output_dir', type=str, default='/mnt/data/fpttc_ground_truth/4_depth_map', help="Output directory for depth map npy file.")
    args = parser.parse_args()
    main(args=args)