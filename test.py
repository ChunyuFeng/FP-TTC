from PIL import Image
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import datetime
from glob import glob
from fpttc.fp_ttc import FpTTC
from utils.trainer import TTCTrainer
from utils.draw import (
    visual_scale_map_range_image,
    visual_risk_score_map_range_image,
    scale2rgb,
    orientation2rgb
)
from dataloader.utils.augmentor import NuscRangeImageAugmentor
from dataloader.dataset import build_frame_mapping, pack_geocalib_tensors_per_cam_to_lidar, lidar2cam_to_cam2lidar
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.cyberrock.sjtu_test_info import undistort_image
from PIL import ImageDraw
from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import time
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
parser = argparse.ArgumentParser()

# Dataset & evaluation parameters
parser.add_argument('--checkpoint_dir', default='tmp', type=str)
parser.add_argument('--stage', default='chairs', type=str)
parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+')
parser.add_argument('--max_flow', default=400, type=int)
parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                    help='[height, width] for resize operations')
parser.add_argument('--padding_factor', default=16, type=int)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--save_eval_to_file', action='store_true')
parser.add_argument('--evaluate_matched_unmatched', action='store_true')
parser.add_argument('--val_things_clean_only', action='store_true')
parser.add_argument('--with_speed_metric', action='store_true')

# Training hyperparameters (not used in test)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--strict_resume', action='store_true')
parser.add_argument('--no_resume_optimizer', action='store_true')
parser.add_argument('--num_scales', default=1, type=int)
parser.add_argument('--feature_channels', default=128, type=int)
parser.add_argument('--upsample_factor', default=8, type=int)
parser.add_argument('--num_head', default=1, type=int)
parser.add_argument('--ffn_dim_expansion', default=4, type=int)
parser.add_argument('--num_transformer_layers', default=6, type=int)
parser.add_argument('--reg_refine', action='store_true')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--load_opt', action='store_true')
parser.add_argument('--attn_type', default='swin', type=str)
parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+')
parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+')
parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+')
parser.add_argument('--num_reg_refine', default=1, type=int)
parser.add_argument('--gamma', default=0.9, type=float)

# Submission & inference
parser.add_argument('--kittidataset', default='/mnt/pool2/lcl/data/data_scene_flow/training/', type=str)
parser.add_argument('--drivingdataset', default='/mnt/pool2/lcl/data/Driving/', type=str)
parser.add_argument('--submission', action='store_true')
parser.add_argument('--output_path', default='output', type=str)
parser.add_argument('--save_vis_flow', action='store_true')
parser.add_argument('--no_save_flo', action='store_true')
parser.add_argument('--inference_dir', default=None, type=str)
parser.add_argument('--inference_video', default=None, type=str)
parser.add_argument('--inference_size', default=None, type=int, nargs='+')
parser.add_argument('--save_flo_flow', action='store_true')
parser.add_argument('--pred_bidir_flow', action='store_true')
parser.add_argument('--pred_bwd_flow', action='store_true')
parser.add_argument('--fwd_bwd_check', action='store_true')
parser.add_argument('--save_video', action='store_true')
parser.add_argument('--concat_flow_img', action='store_true')
parser.add_argument('--radial_sampling_num', default=8, type=int,
                    help='number of radial sampling points for spherical coordinates')

# Distributed & misc
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')
parser.add_argument('--count_time', action='store_true')
parser.add_argument('--debug', action='store_true')

# sjtu dataset test
parser.add_argument('--sjtu_test', action='store_true',
                    help='Run test on SJTU dataset with specific parameters')

# path
parser.add_argument('--save_pred_npy', action='store_true',
                    help='Save prediction as .npy files for collision map generation')
parser.add_argument('--pred_npy_dir', default='./Datasets/nuscenes/3_visualization/collision_pred',
                    type=str, help='Directory to save prediction .npy files')
parser.add_argument('--test_info_path', default='./Datasets/nuscenes/2_trainval_test_infos/nusc_trainval_infos_160_1920.pkl',
                    type=str, help='Path to test info file (e.g., nusc_trainval_infos_160_1920.pkl)')

# depth-anything & consistency options
parser.add_argument('--use_da_head', action='store_true', help='use DepthAnythingV2 head online and finetune it')
parser.add_argument('--da_encoder', default='vitl', type=str, choices=['vits','vitb','vitl','vitg'])
parser.add_argument('--da_max_depth', default=80.0, type=float)
parser.add_argument('--online_proj', action='store_true', help='rebuild proj_pix from current depth each iteration')
parser.add_argument('--detach_depth_for_proj', action='store_true', help='use depth.detach() when building proj_pix')

# DDCL & MVRCL weights
parser.add_argument('--ddcl_w', default=0.10, type=float)
parser.add_argument('--mvrcl_w', default=0.20, type=float)
parser.add_argument('--smooth_scale_w', default=0.01, type=float)
parser.add_argument('--smooth_risk_w',  default=0.01, type=float)

# FOV for online spherical projection (range-view)
parser.add_argument('--rv_fov_up', default=8.0, type=float)
parser.add_argument('--rv_fov_down', default=-15.0, type=float)
parser.add_argument('--rv_size', default=[40, 480], type=int, nargs='+')  # [H_r, W_r]

args = parser.parse_args()

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate augmentor using --image_size as [height, width]
resize_height, resize_width = args.image_size  # height, width from CLI
# crop_size expects (crop_h, crop_w)
augmentor = NuscRangeImageAugmentor(crop_size=(resize_height, resize_width),
                                   do_flip=False,
                                   rotate=False)
def main():
    # Load model
    model = FpTTC(num_scales             = args.num_scales,
                  feature_channels       = args.feature_channels,
                  upsample_factor        = args.upsample_factor,
                  num_head               = args.num_head,
                  ffn_dim_expansion      = args.ffn_dim_expansion,
                  num_transformer_layers = args.num_transformer_layers,
                  reg_refine             = args.reg_refine,
                  # NEW: consistency & DA
                  use_da_head            = args.use_da_head,
                  da_encoder             = args.da_encoder,
                  da_max_depth           = args.da_max_depth,
                  online_proj            = args.online_proj,
                  detach_depth_for_proj  = args.detach_depth_for_proj,
                  ddcl_w                 = args.ddcl_w,
                  mvrcl_w                = args.mvrcl_w,
                  smooth_scale_w         = args.smooth_scale_w,
                  smooth_risk_w          = args.smooth_risk_w,
                  rv_size                = tuple(args.rv_size),
                  rv_fov_up              = args.rv_fov_up,
                  rv_fov_down            = args.rv_fov_down,).cuda()

    # Optionally resume checkpoint
    if args.resume:
        map_loc = f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=map_loc)
        # Extract raw state dict
        if 'net' in checkpoint:
            raw_state_dict = checkpoint['net']
        elif 'state_dict' in checkpoint:
            raw_state_dict = checkpoint['state_dict']
        else:
            raw_state_dict = checkpoint

        processed_state_dict = {}
        for key, value in raw_state_dict.items():
            new_key = key[len('module.'):] if key.startswith('module.') else key
            processed_state_dict[new_key] = value
        # Load into model
        model.load_state_dict(processed_state_dict)

    model.eval()

    # Prepare output directory
    time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    output_dir = f"./test/{time_stamp}_surround_ttc"
    os.makedirs(output_dir, exist_ok=True)

    # Load test info
    info_path = args.test_info_path
    with open(info_path, 'rb') as f:
        test_entries = pickle.load(f)

    camera_channels = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
        ]
    
    # Load DepthAnythingV2 model for depth prediction
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80 # 20 for indoor model, 80 for outdoor model

    depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(f'pretrained/new/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    depth_model.to('cuda').eval()
    
    if args.sjtu_test:

        undistort_maps = {}
        for ch in camera_channels:
            # 任意一帧的 K/dist 都是固定的：
            K = test_entries[0]['sensor_metas_prev'][ch]['K']
            dist = test_entries[0]['sensor_metas_prev'][ch]['dist']
            w, h = (1920, 1080)  # e.g. (1600, 900)
            # 1) 计算去畸变映射表
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_32FC1)
            undistort_maps[ch] = (map1, map2, newK)
        
        def load_remap(ch, frame_type):
            # frame_type = 'prev' 或 'curr'
            path = test_entries[idx][f'{frame_type}_camera_data'][ch]['filename']
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            map1, map2, K_ud = undistort_maps[ch]
            img_ud = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            # 更新内参
            test_entries[idx][f'sensor_metas_{frame_type}'][ch]['K_undist'] = K_ud
            # 计算 R_l2c, t_l2c
            R = test_entries[idx][f'sensor_metas_{frame_type}'][ch]['R']
            t = test_entries[idx][f'sensor_metas_{frame_type}'][ch]['t']
            R_inv, t_inv = R.T, -R.T @ t
            test_entries[idx][f'sensor_metas_{frame_type}'][ch].update({
                'R_l2c': R_inv, 't_l2c': t_inv
            })
            return ch, img_ud
            
        # for idx in tqdm(range(len(test_entries)), desc='Processing surround view images'):
        #      # if idx <= 180:
        #     #     continue
        #     # 1) Load input images
        #     prev_images_undistorted = {}
        #     curr_images_undistorted = {}

        #     for ch in camera_channels:
        #         # 对 prev 帧的图像去畸变，并且获取去畸变后的内参 K_undist
        #         prev_path = test_entries[idx]['prev_camera_data'][ch]['filename']
        #         prev_images_undistorted[ch] = cv2.imread(prev_path, cv2.IMREAD_COLOR)
        #         prev_images_undistorted[ch], K_prev, _ = undistort_image(prev_images_undistorted[ch], 
        #                                                                  test_entries[idx]['sensor_metas_prev'][ch]['K'],
        #                                                                  test_entries[idx]['sensor_metas_prev'][ch]['dist'])
        #         prev_images_undistorted[ch] = Image.fromarray(cv2.cvtColor(prev_images_undistorted[ch], cv2.COLOR_BGR2RGB))
        #         test_entries[idx]['sensor_metas_prev'][ch]['K_undist'] = K_prev
        #         # 将 相机 -> LiDAR 的 RT 转换为 LiDAR -> 相机 的 RT
        #         R = test_entries[idx]['sensor_metas_prev'][ch]['R']
        #         t = test_entries[idx]['sensor_metas_prev'][ch]['t']
        #         R_inv = R.T
        #         t_inv = -R_inv @ t
        #         test_entries[idx]['sensor_metas_prev'][ch]['R_l2c'] = R_inv
        #         test_entries[idx]['sensor_metas_prev'][ch]['t_l2c'] = t_inv

        #         # 对 curr 帧的图像去畸变，并且获取去畸变后的内参 K_undist
        #         curr_path = test_entries[idx]['curr_camera_data'][ch]['filename']
        #         curr_images_undistorted[ch] = cv2.imread(curr_path, cv2.IMREAD_COLOR)
        #         curr_images_undistorted[ch], K_curr, _ = undistort_image(curr_images_undistorted[ch],
        #                                                                  test_entries[idx]['sensor_metas_curr'][ch]['K'],
        #                                                                  test_entries[idx]['sensor_metas_curr'][ch]['dist'])
        #         curr_images_undistorted[ch] = Image.fromarray(cv2.cvtColor(curr_images_undistorted[ch], cv2.COLOR_BGR2RGB))
        #         test_entries[idx]['sensor_metas_curr'][ch]['K_undist'] = K_curr
        #         # 将 相机 -> LiDAR 的 RT 转换为 LiDAR -> 相机 的 RT
        #         R = test_entries[idx]['sensor_metas_curr'][ch]['R']
        #         t = test_entries[idx]['sensor_metas_curr'][ch]['t']
        #         R_inv = R.T
        #         t_inv = -R_inv @ t
        #         test_entries[idx]['sensor_metas_curr'][ch]['R_l2c'] = R_inv
        #         test_entries[idx]['sensor_metas_curr'][ch]['t_l2c'] = t_inv       

        #     # augment images
        #     orig_size = next(iter(prev_images_undistorted.values())).size  # (W, H)
        #     affine_params = augmentor.sample_params(orig_size)
        #     augmented_prev, _ = augmentor(prev_images_undistorted.copy(), affine_params)
        #     augmented_curr, _ = augmentor(curr_images_undistorted.copy(), affine_params)
        #     affine_matrix = augmentor.get_affine_matrix(affine_params)
        for idx in tqdm(range(len(test_entries)), desc='Processing surround view'):

            # if idx > 100:
            #     continue
            # start = time.perf_counter()
            # 1) 并行加载 + 去畸变

            # 用线程池并行处理所有通道的 prev/curr
            prev_raw, curr_raw = {}, {}
            with ThreadPoolExecutor(max_workers=len(camera_channels)*2) as exe:
                # prev futures
                prev_futs = [exe.submit(load_remap, ch, 'prev') for ch in camera_channels]
                curr_futs = [exe.submit(load_remap, ch, 'curr') for ch in camera_channels]
                for fut in prev_futs:
                    ch, img_ud = fut.result(); prev_raw[ch] = img_ud
                for fut in curr_futs:
                    ch, img_ud = fut.result(); curr_raw[ch] = img_ud

            # 2) 批量仿射增强 —— 用 Albumentations 一次性处理所有视角
            orig_h, orig_w = next(iter(prev_raw.values())).shape[:2]
            params = augmentor.sample_params_sjtu((orig_w, orig_h))
            resize_w, resize_h = params['resize']      # (width, height)
            crop_x, crop_y = params['crop']            # (x offset, y offset)
            crop_h, crop_w = augmentor.crop_size       # from your class

            # 2) 构造 Transform 列表
            tfms = [
                # 2.1 Resize 到 (resize_h, resize_w)
                A.Resize(height=resize_h, width=resize_w, 
                        interpolation=cv2.INTER_LINEAR),
                # 2.2 Crop 出 (crop_h, crop_w) 大小的窗口
                A.Crop(x_min=crop_x, y_min=crop_y,
                    x_max=crop_x+crop_w, y_max=crop_y+crop_h),
            ]
            # 2.3 水平 / 垂直 翻转
            if params['flip_h']:
                tfms.append(A.HorizontalFlip(p=1.0))
            if params['flip_v']:
                tfms.append(A.VerticalFlip(p=1.0))
            # 2.4 旋转
            if params['rotate']:
                # 固定 angle；border_mode 可根据你想要的背景填充方式调整
                tfms.append(A.Rotate(limit=(params['angle'], params['angle']),
                                    p=1.0,
                                    border_mode=cv2.BORDER_CONSTANT))

            # 3) 最终 Compose
            alb_tf = A.Compose(tfms, p=1.0)

            # 4) 对每张图像分别应用（用 dict comprehension 也行）
            augmented_prev = {
                ch: alb_tf(image=prev_raw[ch])['image']
                for ch in camera_channels
            }
            augmented_curr = {
                ch: alb_tf(image=curr_raw[ch])['image']
                for ch in camera_channels
            }
            affine_matrix = augmentor.get_affine_matrix(params)

            # end1 = time.perf_counter()
            # print(f"Image loading and augmentation took {(end1 - start)*1000:.2f} ms")

            # Convert to tensors and stack
            prev_tensors = [torch.from_numpy(augmented_prev[ch]).permute(2,0,1).float() for ch in camera_channels]
            curr_tensors = [torch.from_numpy(augmented_curr[ch]).permute(2,0,1).float() for ch in camera_channels]

            prev_batch = torch.stack(prev_tensors, dim=0).unsqueeze(0).to(device)
            curr_batch = torch.stack(curr_tensors, dim=0).unsqueeze(0).to(device)

            # 2) Load Depth Pred Map (DepthAnythingV2)
            prev_depth_pred_map = {}
            curr_depth_pred_map = {}
            prev_depth_pred_map_tensor = {}
            curr_depth_pred_map_tensor = {}

            ######################### 从预先生成的 Depth Map 中加载 #########################
            # for channel in camera_channels:
            #     depth_pred_prev_path = test_entries[idx]['prev_camera_data'][channel]['depth_pred']
            #     depth_pred_curr_path = test_entries[idx]['curr_camera_data'][channel]['depth_pred']

            #     prev_depth_pred_map[channel] = np.load(depth_pred_prev_path)
            #     curr_depth_pred_map[channel] = np.load(depth_pred_curr_path)

            #     prev_depth_pred_map[channel] = torch.from_numpy(prev_depth_pred_map[channel])
            #     curr_depth_pred_map[channel] = torch.from_numpy(curr_depth_pred_map[channel])

            ######################### 用 DepthAnythingV2 推理生成 Depth Map #########################
            # 测算 DepthAnythingV2 的推理时间
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event   = torch.cuda.Event(enable_timing=True)
            # start_event.record()

            for channel in camera_channels:
                raw_image_prev = cv2.cvtColor(augmented_prev[channel], cv2.COLOR_RGB2BGR)  # Convert to BGR for DepthAnythingV2
                prev_depth_pred_map[channel] = depth_model.infer_image(raw_image_prev)

                raw_image_curr = cv2.cvtColor(augmented_curr[channel], cv2.COLOR_RGB2BGR)  # Convert to BGR for DepthAnythingV2
                curr_depth_pred_map[channel] = depth_model.infer_image(raw_image_curr)

                prev_depth_pred_map_tensor[channel] = torch.from_numpy(prev_depth_pred_map[channel])
                curr_depth_pred_map_tensor[channel] = torch.from_numpy(curr_depth_pred_map[channel])

            prev_depths_pred_tensor_stacked = torch.stack([prev_depth_pred_map_tensor[channel] for channel in camera_channels], dim=0).unsqueeze(1)
            curr_depths_pred_tensor_stacked = torch.stack([curr_depth_pred_map_tensor[channel] for channel in camera_channels], dim=0).unsqueeze(1)

            prev_depths_pred_batch = prev_depths_pred_tensor_stacked.unsqueeze(0).to(device)
            curr_depths_pred_batch = curr_depths_pred_tensor_stacked.unsqueeze(0).to(device)

            # end_event.record()
            # torch.cuda.synchronize()
            # elapsed_ms = start_event.elapsed_time(end_event)
            # print(f"test：{elapsed_ms:.3f} ms")

            # 3) load 环视图像 uv 坐标与 range view uv 坐标之间的对应关系 (DepthAnythingV2)
            # proj_range_prev, proj_pix_prev = build_frame_mapping(test_entries, 'sjtu', 'prev', prev_depth_pred_map,
            #                                                      affine_matrix, idx, H_r=40, W_r=480, visualize=False)
            # proj_range_curr, proj_pix_curr = build_frame_mapping(test_entries, 'sjtu', 'curr', curr_depth_pred_map,
            #                                                      affine_matrix, idx, H_r=40, W_r=480, visualize=False)
            proj_pix_prev = np.zeros((1,3), dtype=np.int64)   # (M, 3)
            proj_pix_curr = np.zeros((1,3), dtype=np.int64)   # (M, 3)
            # 转换为 tensor
            proj_pix_prev_tensor = torch.from_numpy(proj_pix_prev.astype(np.int64))   # (M, 3)
            proj_pix_curr_tensor = torch.from_numpy(proj_pix_curr.astype(np.int64))   # (M, 3)
            # 打包 batch
            proj_pix_prev_batch = proj_pix_prev_tensor.unsqueeze(0).to(device)
            proj_pix_curr_batch = proj_pix_curr_tensor.unsqueeze(0).to(device)

            affine_matrix = torch.from_numpy(affine_matrix).unsqueeze(0).to(device)

            K_ = []
            T_Cam2Lidar = []
            flip_xy_matrix = np.array([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]], dtype=np.float32)
            for channel in camera_channels:
                k = test_entries[idx]['sensor_metas_prev'][channel]['K_undist']
                k = k.astype(np.float32)
                K_.append(k)
                t_cam2lidar = lidar2cam_to_cam2lidar(
                    test_entries[idx]['sensor_metas_prev'][channel]['R_l2c'],
                    test_entries[idx]['sensor_metas_prev'][channel]['t_l2c']
                )
                t_cam2lidar = t_cam2lidar.astype(np.float32)
                # 乘以 flip_xy_matrix 来修正坐标系
                t_cam2lidar[:3, :3] = flip_xy_matrix @ t_cam2lidar[:3, :3]
                T_Cam2Lidar.append(t_cam2lidar)
            K_curr = torch.stack([torch.from_numpy(k) for k in K_], dim=0).unsqueeze(0).to(device)  # (1, 6, 3, 3)
            T_Camera2Lidar = torch.stack([torch.from_numpy(t) for t in T_Cam2Lidar], dim=0).unsqueeze(0).to(device)  # (1, 6, 4, 4)

            # Inference
            with torch.no_grad():
                scale_pred, risk_pred    = model.forward(
                        img_prev         = prev_batch,
                        img_curr         = curr_batch,
                        depth_prev       = prev_depths_pred_batch,
                        depth_curr       = curr_depths_pred_batch,
                        proj_pix_prev    = proj_pix_prev_batch,
                        proj_pix_curr    = proj_pix_curr_batch,
                        attn_type        = args.attn_type,
                        attn_splits_list = args.attn_splits_list,
                        corr_radius_list = args.corr_radius_list,
                        prop_radius_list = args.prop_radius_list,
                        num_reg_refine   = args.num_reg_refine,
                        scale_only       = False,
                        affine_matrix    = affine_matrix,
                        K_curr           = K_curr,
                        T_E_from_C_curr  = T_Camera2Lidar
                    )
            # end2 = time.perf_counter()
            # print(f"Inference took {(end2 - end1)*1000:.2f} ms")

            # print(f"The whole process took {(end2 - start)*1000:.2f} ms")

            # Visualization
            # visualize RGB images
            concat_prev = np.concatenate([augmented_prev[ch] for ch in camera_channels], axis=1)
            concat_prev = concat_prev.astype(np.uint8)
            concat_prev = cv2.cvtColor(concat_prev, cv2.COLOR_BGR2RGB)
            concat_prev_img = Image.fromarray(concat_prev)
            concat_prev_img.save(os.path.join(output_dir, f"concat_prev_{idx}.png"))

            scale_prediction_array = scale_pred[0].squeeze(0).cpu().numpy()
            # scale_prediction_mask = (scale_prediction_array > 0.3) & (scale_prediction_array < 3.0)
            # normalized_pred_scale_image = visual_scale_map_range_image(scale_prediction_array, scale_prediction_mask)

            risk_prediction_array = risk_pred[0].squeeze(0).cpu().numpy()
            # normalized_pred_risk_image = visual_risk_score_map_range_image(risk_prediction_array, None)

            # new vis method
            scale_vis = np.clip(scale_prediction_array, 0.0, 2.0)
            vis = scale2rgb(scale_vis)
            vis = vis*255.0
            cv2.imwrite(os.path.join(output_dir, f"new_pred_scale_{idx}.png"), vis)

            orien = np.clip(risk_prediction_array, 0.0, np.pi)
            orien_vis = orientation2rgb(orien)
            orien_vis = orien_vis * 255.0
            cv2.imwrite(os.path.join(output_dir, f"new_pred_orien_{idx}.png"), orien_vis)

            # save prediction as .npy files for collision map generation
            if args.save_pred_npy:
                # 添加以时间戳命名的子目录
                pred_npy_subdir = os.path.join(args.pred_npy_dir, f"pred_npy_{time_stamp}")
                os.makedirs(pred_npy_subdir, exist_ok=True)
                pred_data = {
                    "scale_pred": scale_prediction_array,
                    "risk_pred": risk_prediction_array
                }
                np.save(os.path.join(pred_npy_subdir, f"pred_{idx}.npy"), pred_data)

            # Save visuals
            # plt.imsave(os.path.join(output_dir, f"pred_scale_{idx}.png"),
            #         -normalized_pred_scale_image, cmap='seismic', vmin=-1, vmax=1)
            # plt.imsave(os.path.join(output_dir, f"pred_risk_{idx}.png"),
            #         -normalized_pred_risk_image, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)
            
    else:
        for idx in tqdm(range(len(test_entries)), desc='Processing surround view images'):
            # if test_entries[idx]['scene_indice'] != '7':
            #     continue
            scene_indice = test_entries[idx]['scene_indice']
            # 1) load 相邻两帧的输入图像
            # Load images
            prev_images = {}
            curr_images = {}
            for ch in camera_channels:
                prev_path = os.path.join('./Datasets/nuscenes', test_entries[idx]['prev_camera_data'][ch]['filename'])
                curr_path = os.path.join('./Datasets/nuscenes', test_entries[idx]['curr_camera_data'][ch]['filename'])
                prev_images[ch] = Image.open(prev_path)
                curr_images[ch] = Image.open(curr_path)

            # augment images
            orig_size = next(iter(prev_images.values())).size  # (W, H)
            affine_params = augmentor.sample_params(orig_size)
            augmented_prev, _ = augmentor(prev_images, affine_params)
            augmented_curr, _ = augmentor(curr_images, affine_params)
            affine_matrix = augmentor.get_affine_matrix(affine_params)

            # Convert to tensors and stack
            prev_tensors = [torch.from_numpy(augmented_prev[ch]).permute(2,0,1).float() for ch in camera_channels]
            curr_tensors = [torch.from_numpy(augmented_curr[ch]).permute(2,0,1).float() for ch in camera_channels]

            prev_batch = torch.stack(prev_tensors, dim=0).unsqueeze(0).to(device)
            curr_batch = torch.stack(curr_tensors, dim=0).unsqueeze(0).to(device)

            # 2) load Depth Pred Map (DepthAnythingV2)
            prev_depth_pred_map = {}
            curr_depth_pred_map = {}
            for channel in camera_channels:
                depth_pred_prev = depth_model.infer_image(augmented_prev[channel], input_size=320)
                depth_pred_curr = depth_model.infer_image(augmented_curr[channel], input_size=320)

                prev_depth_pred_map[channel] = torch.from_numpy(depth_pred_prev)
                curr_depth_pred_map[channel] = torch.from_numpy(depth_pred_curr)
            # for channel in camera_channels:
            #     depth_pred_prev_path = test_entries[idx]['prev_camera_data'][channel]['depth_pred']
            #     depth_pred_curr_path = test_entries[idx]['curr_camera_data'][channel]['depth_pred']

            #     prev_depth_pred_map[channel] = np.load(depth_pred_prev_path)
            #     curr_depth_pred_map[channel] = np.load(depth_pred_curr_path)

            #     prev_depth_pred_map[channel] = torch.from_numpy(prev_depth_pred_map[channel])
            #     curr_depth_pred_map[channel] = torch.from_numpy(curr_depth_pred_map[channel])

            prev_depths_pred_tensor = torch.stack([prev_depth_pred_map[channel] for channel in camera_channels], dim=0).unsqueeze(1)
            curr_depths_pred_tensor = torch.stack([curr_depth_pred_map[channel] for channel in camera_channels], dim=0).unsqueeze(1)

            prev_depths_pred_batch = prev_depths_pred_tensor.unsqueeze(0).to(device)
            curr_depths_pred_batch = curr_depths_pred_tensor.unsqueeze(0).to(device)

            # 3) load 真值，在 test 流程中仅用于可视化
            if test_entries[idx]['gt_map_path'] is not None:
                # Load ground-truth maps
                gt_item = np.load(os.path.join(test_entries[idx]['gt_map_path'], 'range_image_curr.npy'), allow_pickle=True).item()
                gt_scale_map = torch.from_numpy(gt_item['scale']).float()
                gt_risk_map = torch.from_numpy(gt_item['risk_score']).float()

                # Build masked GT tensors
                valid_mask = (gt_scale_map > 0.3) & (gt_scale_map < 3.0)
                gt_scale_tensor = torch.cat([
                    gt_scale_map.unsqueeze(0),
                    valid_mask.unsqueeze(0).float()
                ], dim=0).unsqueeze(0).to(device)
                gt_risk_tensor = torch.cat([
                    gt_risk_map.unsqueeze(0),
                    valid_mask.unsqueeze(0).float()
                ], dim=0).unsqueeze(0).to(device)
            else:
                gt_scale_tensor = torch.zeros((1, 2, prev_batch.shape[2], prev_batch.shape[3])).to(device)
                gt_risk_tensor = torch.zeros((1, 2, prev_batch.shape[2], prev_batch.shape[3])).to(device)

            # 4) load 环视图像 uv 坐标与 range view uv 坐标之间的对应关系 (DepthAnythingV2)
            # proj_range_prev, proj_pix_prev = build_frame_mapping(test_entries, 'nusc', 'prev', depth_pred_prev, 
            #                                                      affine_matrix, idx, H_r=40, W_r=480)
            # proj_range_curr, proj_pix_curr = build_frame_mapping(test_entries, 'nusc', 'curr', depth_pred_curr,
            #                                                      affine_matrix, idx, H_r=40, W_r=480)
            proj_pix_prev = np.zeros((1,3), dtype=np.int64)   # (M, 3)
            proj_pix_curr = np.zeros((1,3), dtype=np.int64)   # (M, 3)
            # 转换为 tensor
            proj_pix_prev_tensor = torch.from_numpy(proj_pix_prev.astype(np.int64))   # (M, 3)
            proj_pix_curr_tensor = torch.from_numpy(proj_pix_curr.astype(np.int64))   # (M, 3)
            # 打包 batch
            proj_pix_prev_batch = proj_pix_prev_tensor.unsqueeze(0).to(device)
            proj_pix_curr_batch = proj_pix_curr_tensor.unsqueeze(0).to(device)

            affine_matrix = torch.from_numpy(affine_matrix).unsqueeze(0).to(device)

            sensor_metas_prev = test_entries[idx]['sensor_metas_prev']
            sensor_metas_curr = test_entries[idx]['sensor_metas_curr']
            K_curr, T_E_from_C_curr, _ = pack_geocalib_tensors_per_cam_to_lidar(
                sensor_metas_prev = sensor_metas_prev, 
                sensor_metas_curr = sensor_metas_curr, 
                camera_channels   = camera_channels)
            K_curr = K_curr.unsqueeze(0).to(device)
            T_E_from_C_curr = T_E_from_C_curr.unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                scale_pred, risk_pred = model.forward(
                        img_prev         = prev_batch,
                        img_curr         = curr_batch,
                        depth_prev       = prev_depths_pred_batch,
                        depth_curr       = curr_depths_pred_batch,
                        proj_pix_prev    = proj_pix_prev_batch,
                        proj_pix_curr    = proj_pix_curr_batch,
                        attn_type        = args.attn_type,
                        attn_splits_list = args.attn_splits_list,
                        corr_radius_list = args.corr_radius_list,
                        prop_radius_list = args.prop_radius_list,
                        num_reg_refine   = args.num_reg_refine,
                        scale_only       = False,
                        affine_matrix    = affine_matrix,
                        K_curr           = K_curr,
                        T_E_from_C_curr  = T_E_from_C_curr  
                    )

            # # Visualization
            # gt_scale_array = gt_scale_tensor[0,0].cpu().numpy()
            # gt_scale_mask_array = gt_scale_tensor[0,1].cpu().bool().numpy()
            # normalized_gt_scale_image = visual_scale_map_range_image(gt_scale_array, gt_scale_mask_array)

            # scale_prediction_array = scale_pred[0].squeeze(0).cpu().numpy()
            # scale_prediction_mask = (scale_prediction_array > 0.3) & (scale_prediction_array < 3.0)
            # normalized_pred_scale_image = visual_scale_map_range_image(scale_prediction_array, scale_prediction_mask)

            # gt_risk_array = gt_risk_tensor[0,0].cpu().numpy()
            # gt_risk_mask_array = gt_risk_tensor[0,1].cpu().bool().numpy()
            # normalized_gt_risk_image = visual_risk_score_map_range_image(gt_risk_array, gt_risk_mask_array)

            # risk_prediction_array = risk_pred[0].squeeze(0).cpu().numpy()
            # normalized_pred_risk_image = visual_risk_score_map_range_image(risk_prediction_array, None)

            gt_scale_array = gt_scale_tensor[0,0].cpu().numpy()
            gt_scale_vis = np.clip(gt_scale_array, 0.0, 2.0)
            gt_scale_vis = scale2rgb(gt_scale_vis)
            gt_scale_vis = gt_scale_vis * 255.0
            cv2.imwrite(os.path.join(output_dir, f"gt_scale_{idx}.png"), gt_scale_vis)
            # gt_scale_mask_array = gt_scale_tensor[0,1].cpu().bool().numpy()
            # normalized_gt_scale_image = visual_scale_map_range_image(gt_scale_array, gt_scale_mask_array)

            scale_prediction_array = scale_pred[0].squeeze(0).cpu().numpy()
            scale_vis = np.clip(scale_prediction_array, 0.0, 2.0)
            vis = scale2rgb(scale_vis)
            vis = vis * 255.0
            cv2.imwrite(os.path.join(output_dir, f"pred_scale_{idx}.png"), vis)
            # scale_prediction_mask = (scale_prediction_array > 0.3) & (scale_prediction_array < 3.0)
            # normalized_pred_scale_image = visual_scale_map_range_image(scale_prediction_array, scale_prediction_mask)

            gt_risk_array = gt_risk_tensor[0,0].cpu().numpy()
            gt_risk_vis = np.clip(gt_risk_array, 0.0, np.pi)
            gt_risk_vis = orientation2rgb(gt_risk_vis)
            gt_risk_vis = gt_risk_vis * 255.0
            cv2.imwrite(os.path.join(output_dir, f"gt_risk_{idx}.png"), gt_risk_vis)
            # gt_risk_mask_array = gt_risk_tensor[0,1].cpu().bool().numpy()
            # normalized_gt_risk_image = visual_risk_score_map_range_image(gt_risk_array, gt_risk_mask_array)

            risk_prediction_array = risk_pred[0].squeeze(0).cpu().numpy()
            risk_vis = np.clip(risk_prediction_array, 0.0, np.pi)
            risk_vis = orientation2rgb(risk_vis)
            risk_vis = risk_vis * 255.0
            cv2.imwrite(os.path.join(output_dir, f"pred_risk_{idx}.png"), risk_vis)
            # normalized_pred_risk_image = visual_risk_score_map_range_image(risk_prediction_array, None)

            # save prediction as .npy files
            # for collision map generation
            if args.save_pred_npy:
                # 添加以时间戳命名的子目录
                pred_npy_subdir = os.path.join(args.pred_npy_dir, f"pred_npy_{time_stamp}")
                os.makedirs(pred_npy_subdir, exist_ok=True)
                pred_data = {
                    "scale_pred": scale_prediction_array,
                    "risk_pred": risk_prediction_array
                }
                np.save(os.path.join(pred_npy_subdir, f"scene_{scene_indice}_pred_{idx}.npy"), pred_data)

            # Save visuals
            # 将 augmented_prev 中的图像按照channel顺序拼接，并保存为 concat_prev_{idx}.png
            concat_prev = np.concatenate([augmented_prev[ch] for ch in camera_channels], axis=1)
            concat_prev = concat_prev.astype(np.uint8)
            concat_prev_img = Image.fromarray(concat_prev)
            concat_prev_img.save(os.path.join(output_dir, f"concat_prev_{idx}.png"))

            # plt.imsave(os.path.join(output_dir, f"pred_scale_{idx}.png"),
            #         -normalized_pred_scale_image, cmap='seismic', vmin=-1, vmax=1)
            # plt.imsave(os.path.join(output_dir, f"gt_scale_{idx}.png"),
            #         -normalized_gt_scale_image, cmap='seismic', vmin=-1, vmax=1)
            # plt.imsave(os.path.join(output_dir, f"pred_risk_{idx}.png"),
            #         -normalized_pred_risk_image, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)
            # plt.imsave(os.path.join(output_dir, f"gt_risk_{idx}.png"),
            #         -normalized_gt_risk_image, cmap='seismic', vmin=-np.pi/2, vmax=np.pi/2)

            # Cleanup
            del prev_batch, curr_batch, scale_pred, risk_pred
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()