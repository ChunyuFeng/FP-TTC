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
    visual_risk_score_map_range_image
)
from dataloader.utils.augmentor import NuscRangeImageAugmentor
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from fpttc.scale_net.utils.spherical import build_spherical_voxels, project_voxel_to_camera
from tools.cyberrock.sjtu_test_info import undistort_image, intersect_rois
from PIL import ImageDraw
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

args = parser.parse_args()

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate augmentor using --image_size as [height, width]
resize_height, resize_width = args.image_size  # height, width from CLI
# crop_size expects (crop_h, crop_w)
augmentor = NuscRangeImageAugmentor(crop_size=(resize_height, resize_width),
                                   do_flip=False,
                                   rotate=False)
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

def main():
    # Load model
    model = FpTTC(num_scales             = args.num_scales,
                  feature_channels       = args.feature_channels,
                  upsample_factor        = args.upsample_factor,
                  num_head               = args.num_head,
                  ffn_dim_expansion      = args.ffn_dim_expansion,
                  num_transformer_layers = args.num_transformer_layers,
                  reg_refine             = args.reg_refine,
                  radial_sampling        = args.radial_sampling_num,
                  train                  = False).cuda()

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
        # Strip 'module.' prefix if present
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

    for idx in tqdm(range(len(test_entries)), desc='Processing surround view images'):
        if args.sjtu_test:
            # Load images
            prev_images_undistorted = {}
            curr_images_undistorted = {}
            # K_undist_prev = {}
            # K_undist_curr = {}
            # roi_undist_prev = {}
            # roi_undist_curr = {}
            for ch in camera_channels:
                prev_path = test_entries[idx]['prev_camera_data'][ch]['filename']
                prev_images_undistorted[ch] = cv2.imread(prev_path, cv2.IMREAD_COLOR)
                prev_images_undistorted[ch], K_prev, roi_prev = undistort_image(prev_images_undistorted[ch], 
                                                                                test_entries[idx]['sensor_metas_prev'][ch]['K'],
                                                                                test_entries[idx]['sensor_metas_prev'][ch]['dist'])
                prev_images_undistorted[ch] = Image.fromarray(cv2.cvtColor(prev_images_undistorted[ch], cv2.COLOR_BGR2RGB))
                # K_undist_prev[ch] = K_prev
                # roi_undist_prev[ch] = roi_prev
                test_entries[idx]['sensor_metas_prev'][ch]['K_undist'] = K_prev
                test_entries[idx]['sensor_metas_prev'][ch]['old_ext'] = old_extrinsics_[ch]

                curr_path = test_entries[idx]['curr_camera_data'][ch]['filename']
                curr_images_undistorted[ch] = cv2.imread(curr_path, cv2.IMREAD_COLOR)
                curr_images_undistorted[ch], K_curr, roi_curr = undistort_image(curr_images_undistorted[ch],
                                                                                test_entries[idx]['sensor_metas_curr'][ch]['K'],
                                                                                test_entries[idx]['sensor_metas_curr'][ch]['dist'])
                test_entries[idx]['sensor_metas_curr'][ch]['K_undist'] = K_curr
                test_entries[idx]['sensor_metas_curr'][ch]['old_ext'] = old_extrinsics_[ch]
                curr_images_undistorted[ch] = Image.fromarray(cv2.cvtColor(curr_images_undistorted[ch], cv2.COLOR_BGR2RGB))

            # augment images
            orig_size = next(iter(prev_images_undistorted.values())).size  # (W, H)
            affine_params = augmentor.sample_params(orig_size)
            augmented_prev, _ = augmentor(prev_images_undistorted.copy(), affine_params)
            augmented_curr, _ = augmentor(curr_images_undistorted.copy(), affine_params)
            affine_matrix = augmentor.get_affine_matrix(affine_params)
            
            # visualize RGB images
            concat_prev = np.concatenate([augmented_prev[ch] for ch in camera_channels], axis=1)
            concat_prev = concat_prev.astype(np.uint8)
            concat_prev_img = Image.fromarray(concat_prev)
            concat_prev_img.save(os.path.join(output_dir, f"concat_prev_{idx}.png"))

            # Convert to tensors and stack
            prev_tensors = [torch.from_numpy(augmented_prev[ch]).permute(2,0,1).float() for ch in camera_channels]
            curr_tensors = [torch.from_numpy(augmented_curr[ch]).permute(2,0,1).float() for ch in camera_channels]

            prev_batch = torch.stack(prev_tensors, dim=0).unsqueeze(0).to(device)
            curr_batch = torch.stack(curr_tensors, dim=0).unsqueeze(0).to(device)
            affine_matrix = torch.from_numpy(affine_matrix).unsqueeze(0).to(device)

            # # Build spherical coordinates
            # # Get UV indices for previous and current images
            # H_sph, W_sph = prev_batch.shape[-2:]
            # H_sph, W_sph = H_sph // 4, W_sph // 4 
            # W_sph = W_sph * 6  # 6 cameras concatenated horizontally
            # R = args.radial_sampling_num
            # # xyz 按照 nuScenes 的坐标系定义构建：x向右，y向前，z向上
            # # 需要修改成SJTU的坐标系定义        ：x向左，y向后，z向上
            # # sjtu: 25° up, -25° down
            # _, xyz = build_spherical_voxels(
            #         H=H_sph, W=W_sph, R=R,
            #         r_min=5.0, r_max=50.0,
            #         fov_up_deg=15.0, fov_down_deg=-15.0,
            #     )
            # X, Y, Z = xyz[...,0], xyz[...,1], xyz[...,2]
            # X_new = -X
            # Y_new = -Y
            # Z_new =  Z
            # xyz = np.stack([X_new, Y_new, Z_new], axis=-1)

            # raw_img_size = (orig_size[1], orig_size[0])  # (H, W)
            # idx_uv_prev_raw = project_voxel_to_camera(
            #         xyz=xyz,
            #         sensor_metas=test_entries[idx]['sensor_metas_prev'],
            #         camera_channels=camera_channels,
            #         raw_img_size=raw_img_size,
            #         min_dist=1.0,
            #         sjtu=args.sjtu_test
            #     )
            # idx_uv_curr_raw = project_voxel_to_camera(
            #         xyz=xyz,
            #         sensor_metas=test_entries[idx]['sensor_metas_curr'],
            #         camera_channels=camera_channels,
            #         raw_img_size=raw_img_size,
            #         min_dist=1.0,
            #         sjtu=args.sjtu_test
            #     )
            

            # idx_uv_curr_raw = torch.from_numpy(idx_uv_curr_raw).unsqueeze(0).to(device)
            # idx_uv_prev_raw = torch.from_numpy(idx_uv_prev_raw).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                scale_pred    = model.forward(
                        img_prev         = prev_batch,
                        img_curr         = curr_batch,
                        # affine_matrix    = affine_matrix,
                        # idx_uv_prev      = idx_uv_prev_raw,
                        # idx_uv_curr      = idx_uv_curr_raw,
                        attn_type        = args.attn_type,
                        attn_splits_list = args.attn_splits_list,
                        corr_radius_list = args.corr_radius_list,
                        prop_radius_list = args.prop_radius_list,
                        num_reg_refine   = args.num_reg_refine,
                        testing          = False
                    )
                
            # Visualization
            scale_prediction_array = scale_pred[0].squeeze(0).cpu().numpy()
            scale_prediction_mask = (scale_prediction_array > 0.3) & (scale_prediction_array < 3.0)
            normalized_pred_scale_image = visual_scale_map_range_image(scale_prediction_array, scale_prediction_mask)

            # risk_prediction_array = risk_pred[0].squeeze(0).cpu().numpy()
            # normalized_pred_risk_image = visual_risk_score_map_range_image(risk_prediction_array)

            # save prediction as .npy files
            # for collision map generation
            # if args.save_pred_npy:
            #     # 添加以时间戳命名的子目录
            #     pred_npy_subdir = os.path.join(args.pred_npy_dir, f"pred_npy_{time_stamp}")
            #     os.makedirs(pred_npy_subdir, exist_ok=True)
            #     pred_data = {
            #         "scale_pred": scale_prediction_array,
            #         "risk_pred": risk_prediction_array
            #     }
            #     np.save(os.path.join(pred_npy_subdir, f"pred_{idx}.npy"), pred_data)

            # Save visuals
            plt.imsave(os.path.join(output_dir, f"pred_scale_{idx}.png"),
                    -normalized_pred_scale_image, cmap='seismic', vmin=-1, vmax=1)
            # plt.imsave(os.path.join(output_dir, f"pred_risk_{idx}.png"),
            #         normalized_pred_risk_image, cmap='seismic', vmin=-1, vmax=1)

            # Cleanup
            # del prev_batch, curr_batch, scale_pred, risk_pred
            torch.cuda.empty_cache()
        
        # test on nuScenes dataset
        else: 
            if test_entries[idx]['scene_indice'] != '10':
                continue
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
            
            # visualize RGB images
            concat_prev = np.concatenate([augmented_prev[ch] for ch in camera_channels], axis=1)
            concat_prev = concat_prev.astype(np.uint8)
            concat_prev_img = Image.fromarray(concat_prev)
            concat_prev_img.save(os.path.join(output_dir, f"concat_prev_{idx}.png"))

            # Convert to tensors and stack
            prev_tensors = [torch.from_numpy(augmented_prev[ch]).permute(2,0,1).float() for ch in camera_channels]
            curr_tensors = [torch.from_numpy(augmented_curr[ch]).permute(2,0,1).float() for ch in camera_channels]

            prev_batch = torch.stack(prev_tensors, dim=0).unsqueeze(0).to(device)
            curr_batch = torch.stack(curr_tensors, dim=0).unsqueeze(0).to(device)
            affine_matrix = torch.from_numpy(affine_matrix).unsqueeze(0).to(device)

            # # Build spherical coordinates
            # # Get UV indices for previous and current images
            # H_sph, W_sph = prev_batch.shape[-2:]
            # H_sph, W_sph = H_sph // 4, W_sph // 4 
            # W_sph = W_sph * 6  # 6 cameras concatenated horizontally
            # R = args.radial_sampling_num
            # _, xyz = build_spherical_voxels(
            #         H=H_sph, W=W_sph, R=R,
            #         r_min=5.0, r_max=50.0,
            #         fov_up_deg=8.0, fov_down_deg=-15.0,
            #     )

            # raw_img_size = (orig_size[1], orig_size[0])  # (H, W)
            # idx_uv_prev_raw = project_voxel_to_camera(
            #         xyz=xyz,
            #         sensor_metas=test_entries[idx]['sensor_metas_prev'],
            #         camera_channels=camera_channels,
            #         raw_img_size=raw_img_size,
            #         min_dist=1.0
            #     )
            # idx_uv_curr_raw = project_voxel_to_camera(
            #         xyz=xyz,
            #         sensor_metas=test_entries[idx]['sensor_metas_curr'],
            #         camera_channels=camera_channels,
            #         raw_img_size=raw_img_size,
            #         min_dist=1.0
            #     )
            
            
            # idx_uv_curr_raw = torch.from_numpy(idx_uv_curr_raw).unsqueeze(0).to(device)
            # idx_uv_prev_raw = torch.from_numpy(idx_uv_prev_raw).unsqueeze(0).to(device)

            if test_entries[idx]['gt_map_path'] is not None:
                # Load ground-truth maps
                gt_item = np.load(os.path.join(test_entries[idx]['gt_map_path'], 'range_image.npy'), allow_pickle=True).item()
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

            # Inference
            with torch.no_grad():
                scale_pred               = model.forward(
                        img_prev         = prev_batch,
                        img_curr         = curr_batch,
                        # affine_matrix    = affine_matrix,
                        # idx_uv_prev      = idx_uv_prev_raw,
                        # idx_uv_curr      = idx_uv_curr_raw,
                        attn_type        = args.attn_type,
                        attn_splits_list = args.attn_splits_list,
                        corr_radius_list = args.corr_radius_list,
                        prop_radius_list = args.prop_radius_list,
                        num_reg_refine   = args.num_reg_refine,
                        testing          = False
                    )
            # total_params = sum(p.numel() for p in model.parameters())
            # # 可训练参数量
            # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # print(f"Total params:      {total_params:,}")
            # print(f"Trainable params:  {trainable_params:,}")

            # Visualization
            gt_scale_array = gt_scale_tensor[0,0].cpu().numpy()
            gt_scale_mask_array = gt_scale_tensor[0,1].cpu().bool().numpy()
            normalized_gt_scale_image = visual_scale_map_range_image(gt_scale_array, gt_scale_mask_array)

            scale_prediction_array = scale_pred[0].squeeze(0).cpu().numpy()
            scale_prediction_mask = (scale_prediction_array > 0.3) & (scale_prediction_array < 3.0)
            normalized_pred_scale_image = visual_scale_map_range_image(scale_prediction_array, scale_prediction_mask)

            # gt_risk_array = gt_risk_tensor[0,0].cpu().numpy()
            # normalized_gt_risk_image = visual_risk_score_map_range_image(gt_risk_array)

            # risk_prediction_array = risk_pred[0].squeeze(0).cpu().numpy()
            # normalized_pred_risk_image = visual_risk_score_map_range_image(risk_prediction_array)

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
                np.save(os.path.join(pred_npy_subdir, f"pred_{idx}.npy"), pred_data)

            # Save visuals
            plt.imsave(os.path.join(output_dir, f"pred_scale_{idx}.png"),
                    -normalized_pred_scale_image, cmap='seismic', vmin=-1, vmax=1)
            plt.imsave(os.path.join(output_dir, f"gt_scale_{idx}.png"),
                    -normalized_gt_scale_image, cmap='seismic', vmin=-1, vmax=1)
            # plt.imsave(os.path.join(output_dir, f"pred_risk_{idx}.png"),
            #         normalized_pred_risk_image, cmap='seismic', vmin=-1, vmax=1)
            # plt.imsave(os.path.join(output_dir, f"gt_risk_{idx}.png"),
            #         normalized_gt_risk_image, cmap='seismic', vmin=-1, vmax=1)

            # Cleanup
            # del prev_batch, curr_batch, scale_pred, risk_pred
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()