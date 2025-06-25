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

# Distributed & misc
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')
parser.add_argument('--count_time', action='store_true')
parser.add_argument('--debug', action='store_true')

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

def main():
    # Load model
    model = FpTTC(num_scales=args.num_scales,
                  feature_channels=args.feature_channels,
                  upsample_factor=args.upsample_factor,
                  num_head=args.num_head,
                  ffn_dim_expansion=args.ffn_dim_expansion,
                  num_transformer_layers=args.num_transformer_layers,
                  reg_refine=args.reg_refine,
                  train=False).to(device)

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
        # if test_entries[idx]['scene_indice'] != '3':
        #     continue
        # # Before processing images, verify that all required files exist; if any are missing, skip this sample.
        # all_exist = True
        # for ch in camera_channels:
        #     prev_img_path = os.path.join('./Datasets/nuscenes', test_entries[idx]['prev_camera_data'][ch]['filename'])
        #     curr_img_path = os.path.join('./Datasets/nuscenes', test_entries[idx]['curr_camera_data'][ch]['filename'])
        #     if not (os.path.exists(prev_img_path) and os.path.exists(curr_img_path)):
        #         all_exist = False
        #         break
        # if not all_exist:
        #     continue
        # Load images
        prev_images = {}
        curr_images = {}
        for ch in camera_channels:
            prev_path = os.path.join('./Datasets/nuscenes', test_entries[idx]['prev_camera_data'][ch]['filename'])
            curr_path = os.path.join('./Datasets/nuscenes', test_entries[idx]['curr_camera_data'][ch]['filename'])
            prev_images[ch] = Image.open(prev_path)
            curr_images[ch] = Image.open(curr_path)

        # Apply augmentor for resize+crop
        augmented_prev = augmentor(prev_images)
        augmented_curr = augmentor(curr_images)
        
        # Prev images
        concat_prev = np.concatenate([augmented_prev[ch] for ch in camera_channels], axis=1)
        concat_prev = concat_prev.astype(np.uint8)
        concat_prev_img = Image.fromarray(concat_prev)
        concat_prev_img.save(os.path.join(output_dir, f"concat_prev_{idx}.png"))

        # Convert to tensors and stack
        prev_tensors = [torch.from_numpy(augmented_prev[ch]).permute(2,0,1).float() for ch in camera_channels]
        curr_tensors = [torch.from_numpy(augmented_curr[ch]).permute(2,0,1).float() for ch in camera_channels]

        prev_batch = torch.stack(prev_tensors, dim=0).unsqueeze(0).to(device)
        curr_batch = torch.stack(curr_tensors, dim=0).unsqueeze(0).to(device)
        sensor_meta = test_entries[idx]['sensor_metas']

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
            scale_pred, risk_pred, _, _ = model.forward_with_loss(
                prev_batch, curr_batch, sensor_meta,
                gt_scale_tensor, gt_risk_tensor,
                attn_type=args.attn_type,
                attn_splits_list=args.attn_splits_list,
                corr_radius_list=args.corr_radius_list,
                prop_radius_list=args.prop_radius_list,
                num_reg_refine=args.num_reg_refine
            )

        # Visualization
        gt_scale_array = gt_scale_tensor[0,0].cpu().numpy()
        gt_scale_mask_array = gt_scale_tensor[0,1].cpu().bool().numpy()
        normalized_gt_scale_image = visual_scale_map_range_image(gt_scale_array, gt_scale_mask_array)

        scale_prediction_array = scale_pred[0].squeeze(0).cpu().numpy()
        scale_prediction_mask = (scale_prediction_array > 0.3) & (scale_prediction_array < 3.0)
        normalized_pred_scale_image = visual_scale_map_range_image(scale_prediction_array, scale_prediction_mask)

        gt_risk_array = gt_risk_tensor[0,0].cpu().numpy()
        normalized_gt_risk_image = visual_risk_score_map_range_image(gt_risk_array)

        risk_prediction_array = risk_pred[0].squeeze(0).cpu().numpy()
        normalized_pred_risk_image = visual_risk_score_map_range_image(risk_prediction_array)

        # save prediction as .npy files
        # for collision map generation
        if args.save_pred_npy:
            os.makedirs(args.pred_npy_dir, exist_ok=True)
            pred_data = {
            "scale_pred": scale_prediction_array,
            "risk_pred": risk_prediction_array
            }
            np.save(os.path.join(args.pred_npy_dir, f"pred_{idx}.npy"), pred_data)

        # Save visuals
        plt.imsave(os.path.join(output_dir, f"pred_scale_{idx}.png"),
                   -normalized_pred_scale_image, cmap='seismic', vmin=-1, vmax=1)
        plt.imsave(os.path.join(output_dir, f"gt_scale_{idx}.png"),
                   -normalized_gt_scale_image, cmap='seismic', vmin=-1, vmax=1)
        plt.imsave(os.path.join(output_dir, f"pred_risk_{idx}.png"),
                   normalized_pred_risk_image, cmap='seismic', vmin=-1, vmax=1)
        plt.imsave(os.path.join(output_dir, f"gt_risk_{idx}.png"),
                   normalized_gt_risk_image, cmap='seismic', vmin=-1, vmax=1)

        # Cleanup
        del prev_batch, curr_batch, scale_pred, risk_pred
        torch.cuda.empty_cache()

        with open(os.path.join(output_dir, "processed_indices.txt"), "a") as f:
            f.write(f"{idx}\n")

if __name__ == "__main__":
    main()