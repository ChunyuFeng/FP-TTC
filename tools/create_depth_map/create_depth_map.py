import pickle
import argparse
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import cv2
import torch
import numpy as np
from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from dataloader.utils.augmentor import NuscRangeImageAugmentor
from PIL import Image
from utils.nusc_paths import make_nusc_relative_path

# 读取旧外参
# front -> pandar:
front_to_lidar = np.array([[-0.9994, 0.0268, 0.0243, -0.0992],
    [-0.0242, 0.0019, -0.9997, -0.1998],
    [-0.0269, -0.9996, -0.0012, -0.1768],
    [0, 0, 0, 1.0000]])

# front right -> pandar:
front_right_to_lidar = np.array([[-0.5601, -0.0367, -0.8277, -0.5154],
    [0.8285, -0.0176, -0.5598, -0.1677],
    [0.0060, -0.9991, 0.0402, -0.1925],
    [0, 0, 0, 1.0000]])

# back right-> pandar:
back_right_to_lidar = np.array([[ 0.4354, -0.0227, -0.9000, -0.5385],
    [0.9000, -0.0128, 0.4358, 0.0845],
    [-0.0214, -0.9996, 0.0148, -0.1781],
    [0, 0, 0, 1.0000]])

# back -> pandar:
back_to_lidar = np.array([[1.0000, 0.0029, 0.0009, -0.0471],
    [-0.0009, -0.0053, 1.0000, 1.1084],
    [0.0029, -1.0000, -0.0052, -0.2528],
    [0, 0, 0, 1.0000]])

#back left -> pandar:
back_left_to_lidar = np.array([[0.5358, -0.0117, 0.8443, 0.5212],
    [-0.8444, -0.0029, 0.5358, 0.1203],
    [-0.0039, -0.9999, -0.0114, -0.1698],
    [0, 0, 0, 1.0000]])

# front left -> pandar:
front_left_to_lidar = np.array([[-0.4716, -0.0228, 0.8816, 0.5734],
    [-0.8810, -0.0301, -0.4721, -0.1123],
    [0.0372, -0.9993, -0.0059, -0.1826],
    [0, 0, 0, 1.0000]])

# 旧外参字典
old_extrinsics = {
    'CAM_FRONT':         front_to_lidar,
    'CAM_FRONT_RIGHT':   front_right_to_lidar,
    'CAM_BACK_RIGHT':    back_right_to_lidar,
    'CAM_BACK':          back_to_lidar,
    'CAM_BACK_LEFT':     back_left_to_lidar,
    'CAM_FRONT_LEFT':    front_left_to_lidar,
}
old_extrinsics_ = {}
for ch, extr in old_extrinsics.items():
    R = extr[:3, :3]
    t = extr[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    extr_ = np.hstack([R_inv, t_inv.reshape(3, 1)])  # 3x4
    old_extrinsics_[ch] = extr_

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


def save_pkl_file(filepath, data):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)


def resolve_checkpoint_path(encoder: str, metric_dataset: str, checkpoint_path: Optional[str] = None) -> str:
    if checkpoint_path not in (None, ""):
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        raise FileNotFoundError(f"Depth checkpoint not found: {checkpoint_path}")

    ckpt_name = f'depth_anything_v2_metric_{metric_dataset}_{encoder}.pth'
    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root / 'pretrained' / 'new' / ckpt_name,
        repo_root / 'pretrained' / ckpt_name,
        repo_root / 'depthanything' / 'checkpoints' / ckpt_name,
        repo_root / 'depthanything' / 'metric_depth' / 'checkpoints' / ckpt_name,
        repo_root.parent / 'FP-TTC' / 'pretrained' / 'new' / ckpt_name,
        repo_root.parent / 'FP-TTC' / 'depthanything' / 'metric_depth' / 'checkpoints' / ckpt_name,
    ]
    for path in candidate_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Unable to locate a DepthAnything checkpoint. Tried:\n" +
        "\n".join(str(path) for path in candidate_paths)
    )


def build_depth_model(encoder: str, metric_dataset: str, max_depth: float, checkpoint_path: Optional[str] = None):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    ckpt_path = resolve_checkpoint_path(
        encoder=encoder,
        metric_dataset=metric_dataset,
        checkpoint_path=checkpoint_path,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to('cuda').eval()
    return model


def infer_and_save_depth(
    *,
    model,
    raw_image_rgb: np.ndarray,
    image_filename: str,
    output_dir: str,
    depth_input_size: int,
    skip_existing_depth: bool,
    cache: dict,
):
    cached_rel_path = cache.get(image_filename)
    if cached_rel_path is not None:
        return cached_rel_path

    depth_filename = os.path.splitext(image_filename)[0] + ".npy"
    depth_save_path = os.path.join(output_dir, depth_filename)
    depth_rel_path = make_nusc_relative_path(depth_save_path)

    if not (skip_existing_depth and os.path.exists(depth_save_path)):
        raw_image_bgr = cv2.cvtColor(raw_image_rgb, cv2.COLOR_RGB2BGR)
        depth = model.infer_image(raw_image_bgr, input_size=depth_input_size)
        save_dir = os.path.dirname(depth_save_path)
        os.makedirs(save_dir, exist_ok=True)
        np.save(depth_save_path, depth)

    cache[image_filename] = depth_rel_path
    return depth_rel_path

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
    if data is None:
        raise RuntimeError(f"无法读取 pkl 文件: {args.pkl_file_path}")
    print(f"pkl 文件加载成功, 包含 {len(data)} 条数据.")

    model = build_depth_model(
        encoder=args.encoder,
        metric_dataset=args.metric_dataset,
        max_depth=args.max_depth,
        checkpoint_path=args.checkpoint_path,
    )

    print(f"模型 {args.encoder} 加载成功, 最大深度: {args.max_depth}.")
        
    channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    processed_depth_paths = {}

    if args.sjtu:
        from tools.cyberrock.sjtu_test_info import undistort_image
        for idx, info in enumerate(tqdm(data, desc="Processing SJTU data")):
            # Load images
            prev_images_undistorted = {}
            curr_images_undistorted = {}
            for ch in channels:
                prev_path = info['prev_camera_data'][ch]['filename']
                prev_images_undistorted[ch] = cv2.imread(prev_path, cv2.IMREAD_COLOR)
                prev_images_undistorted[ch], K_prev, roi_prev = undistort_image(prev_images_undistorted[ch], 
                                                                                info['sensor_metas_prev'][ch]['K'],
                                                                                info['sensor_metas_prev'][ch]['dist'])
                prev_images_undistorted[ch] = Image.fromarray(cv2.cvtColor(prev_images_undistorted[ch], cv2.COLOR_BGR2RGB))
                info['sensor_metas_prev'][ch]['K_undist'] = K_prev
                info['sensor_metas_prev'][ch]['old_ext'] = old_extrinsics_[ch]

                curr_path = data[idx]['curr_camera_data'][ch]['filename']
                curr_images_undistorted[ch] = cv2.imread(curr_path, cv2.IMREAD_COLOR)
                curr_images_undistorted[ch], K_curr, roi_curr = undistort_image(curr_images_undistorted[ch],
                                                                                info['sensor_metas_curr'][ch]['K'],
                                                                                info['sensor_metas_curr'][ch]['dist'])
                curr_images_undistorted[ch] = Image.fromarray(cv2.cvtColor(curr_images_undistorted[ch], cv2.COLOR_BGR2RGB))
                info['sensor_metas_curr'][ch]['K_undist'] = K_curr
                info['sensor_metas_curr'][ch]['old_ext'] = old_extrinsics_[ch]
            
            # 对 prev 和 curr 图像进行 crop 和 resize
            orig_size = next(iter(prev_images_undistorted.values())).size
            affine_params = augmentor.sample_params(orig_size)
            augmented_prev, _ = augmentor(prev_images_undistorted, affine_params)
            augmented_curr, _ = augmentor(curr_images_undistorted, affine_params)
            
            # 遍历每条 prev 数据
            for channel in channels:
                image_filename = info['prev_camera_data'][channel]['filename']
                info['prev_camera_data'][channel]['depth_pred'] = infer_and_save_depth(
                    model=model,
                    raw_image_rgb=augmented_prev[channel],
                    image_filename=image_filename,
                    output_dir=args.output_dir,
                    depth_input_size=args.depth_input_size,
                    skip_existing_depth=args.skip_existing_depth,
                    cache=processed_depth_paths,
                )

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
                image_filename = info['curr_camera_data'][channel]['filename']
                info['curr_camera_data'][channel]['depth_pred'] = infer_and_save_depth(
                    model=model,
                    raw_image_rgb=augmented_curr[channel],
                    image_filename=image_filename,
                    output_dir=args.output_dir,
                    depth_input_size=args.depth_input_size,
                    skip_existing_depth=args.skip_existing_depth,
                    cache=processed_depth_paths,
                )
                
        output_pkl_path = args.pkl_file_path if args.rewrite_pkl_in_place else args.pkl_file_path.replace('.pkl', '_dpt.pkl')
        save_pkl_file(output_pkl_path, data)
        print(f"处理完成，新的 pkl 文件已保存到: {output_pkl_path}")

        
    else:
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
                image_filename = info['prev_camera_data'][channel]['filename']
                info['prev_camera_data'][channel]['depth_pred'] = infer_and_save_depth(
                    model=model,
                    raw_image_rgb=augmented_prev[channel],
                    image_filename=image_filename,
                    output_dir=args.output_dir,
                    depth_input_size=args.depth_input_size,
                    skip_existing_depth=args.skip_existing_depth,
                    cache=processed_depth_paths,
                )

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
                image_filename = info['curr_camera_data'][channel]['filename']
                info['curr_camera_data'][channel]['depth_pred'] = infer_and_save_depth(
                    model=model,
                    raw_image_rgb=augmented_curr[channel],
                    image_filename=image_filename,
                    output_dir=args.output_dir,
                    depth_input_size=args.depth_input_size,
                    skip_existing_depth=args.skip_existing_depth,
                    cache=processed_depth_paths,
                )
                
        output_pkl_path = args.pkl_file_path if args.rewrite_pkl_in_place else args.pkl_file_path.replace('.pkl', '_dpt.pkl')
        save_pkl_file(output_pkl_path, data)
        print(f"处理完成，新的 pkl 文件已保存到: {output_pkl_path}")
        print(f"本次运行累计处理/复用的唯一图像数: {len(processed_depth_paths)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pkl file and print its content.")
    parser.add_argument('--image_size', default=[160, 320], type=int, nargs='+',
                    help='[height, width] for resize operations')
    parser.add_argument('--sjtu', action='store_true', help="Use SJTU dataset.")
    parser.add_argument('--pkl_file_path', type=str, required=True, help="Path to the pkl file.")
    parser.add_argument('--nusc_root', type=str, default='./Datasets/nuscenes', help="Root directory of the nuScenes dataset.")
    parser.add_argument('--output_dir', type=str, default='./Datasets/nuscenes/4_depth_map', help="Output directory for depth map npy file.")
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help="DepthAnythingV2 encoder.")
    parser.add_argument('--metric_dataset', type=str, default='vkitti', help="Metric depth checkpoint dataset tag.")
    parser.add_argument('--max_depth', type=float, default=80, help="Maximum depth for metric model.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Optional explicit DepthAnything checkpoint path.")
    parser.add_argument('--depth_input_size', type=int, default=320, help="DepthAnything inference input size.")
    parser.add_argument('--rewrite_pkl_in_place', action='store_true', help="Rewrite the input pkl in place instead of saving a *_dpt.pkl copy.")
    parser.add_argument('--skip_existing_depth', action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse existing depth npy files when present.")
    args = parser.parse_args()
    main(args=args)
