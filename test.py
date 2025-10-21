import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)

import numpy as np

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import argparse
import datetime
import pickle
import time
import albumentations as A

from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from fpttc.fp_ttc import FpTTC
from utils.draw import scale2rgb, orientation2rgb
from dataloader.utils.augmentor import NuscRangeImageAugmentor
from dataloader.dataset import build_frame_mapping, build_frame_mapping_fast
from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from utils.loss import get_loss_scale_map, get_loss_risk_score_map, compute_errs_ttc_from_scale, eval_orientation_and_highrisk_stats, reduce_metrics_across_frames

parser = argparse.ArgumentParser()

torch.backends.cuda.matmul.allow_tf32 = True          # 让 TF32 生效
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True                 # 固定输入尺寸时很有效

FAST_DEPTH_INPUT = 192          # 224→192 常见性价比更高
FAST_PIXEL_STRIDE = 4           # 原来是 2，进一步减半 CPU 开销
FAST_RANGE_H, FAST_RANGE_W = 32, 384   # 40x480 → 32x384

# bfloat16 精度略低，速度更快；float16 精度稍高，但速度稍慢；时间约差 30~40 ms
AMP_DTYPE = torch.bfloat16      

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
def main():
    # Load model
    model = FpTTC(num_scales             = args.num_scales,
                  feature_channels       = args.feature_channels,
                  upsample_factor        = args.upsample_factor,
                  num_head               = args.num_head,
                  ffn_dim_expansion      = args.ffn_dim_expansion,
                  num_transformer_layers = args.num_transformer_layers,
                  reg_refine             = args.reg_refine).to('cuda', memory_format=torch.channels_last).eval()

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
        # missing, unexpected = model.load_state_dict(processed_state_dict, strict=False)
        # print("missing:", missing)        # 会看到 init_scale / init_risk / corr_enc_shared 相关
        # print("unexpected:", unexpected)  # 会看到旧的 conv_corr / conv_corr_risk 相关

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

    encoder = 'vits' # or 'vits', 'vitb'
    dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80 # 20 for indoor model, 80 for outdoor model

    depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(f'pretrained/new/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    depth_model.to('cuda', memory_format=torch.channels_last).eval()
    
    if args.sjtu_test:

        def load_rectified(ch: str, frame_type: str, idx: int, test_entries: list):
            path = test_entries[idx][f'{frame_type}_camera_data'][ch]['filename']
            img  = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
            return ch, img

        def make_sjtu_albu_transform(params: dict, crop_size: Tuple[int, int]):
            resize_w, resize_h = params['resize']    # (W, H)
            crop_x, crop_y     = params['crop']
            crop_h, crop_w     = crop_size
            tfms = [
                A.Resize(height=resize_h, width=resize_w, interpolation=cv2.INTER_LINEAR),
                A.Crop(x_min=crop_x, y_min=crop_y, x_max=crop_x + crop_w, y_max=crop_y + crop_h),
            ]
            if params.get('flip_h'): tfms.append(A.HorizontalFlip(p=1.0))
            if params.get('flip_v'): tfms.append(A.VerticalFlip(p=1.0))
            if params.get('rotate'):
                tfms.append(A.Rotate(limit=(params['angle'], params['angle']),
                                     p=1.0, border_mode=cv2.BORDER_CONSTANT))
            return A.Compose(tfms, p=1.0)

        def stack_imgs_to_tensor(imgs: Dict[str, np.ndarray],
                                 camera_channels: List[str],
                                 device: torch.device) -> torch.Tensor:
            # imgs[ch] = (H,W,3) BGR uint8
            tensors = [torch.from_numpy(imgs[ch]).permute(2, 0, 1).float() for ch in camera_channels]
            return torch.stack(tensors, dim=0).unsqueeze(0).to(device)

        pool = ThreadPoolExecutor(max_workers=len(camera_channels) * 2)

        for idx in tqdm(range(len(test_entries)), desc='Processing surround view'):
            # if idx < 4400 or idx > 4600:
            #     continue

            start = time.perf_counter()

            # -------- 1) 并行读取（去畸变图像） --------
            t0 = time.perf_counter()
            prev_futs = [pool.submit(load_rectified, ch, 'prev', idx, test_entries) for ch in camera_channels]
            curr_futs = [pool.submit(load_rectified, ch, 'curr', idx, test_entries) for ch in camera_channels]
            prev_raw = {ch: img for ch, img in (f.result() for f in prev_futs)}
            curr_raw = {ch: img for ch, img in (f.result() for f in curr_futs)}
            t1 = time.perf_counter()

            # -------- 2) 采样同一套仿射变换，然后逐视角应用 --------
            orig_h, orig_w = next(iter(prev_raw.values())).shape[:2]
            params   = augmentor.sample_params_sjtu((orig_w, orig_h))   # (W, H)
            alb_tf   = make_sjtu_albu_transform(params, crop_size=augmentor.crop_size)
            augmented_prev = {ch: alb_tf(image=prev_raw[ch])['image'] for ch in camera_channels}
            augmented_curr = {ch: alb_tf(image=curr_raw[ch])['image'] for ch in camera_channels}
            affine_matrix  = augmentor.get_affine_matrix(params)
            t2 = time.perf_counter()

            # -------- 3) 打包为 batch tensor --------
            prev_batch = stack_imgs_to_tensor(augmented_prev, camera_channels, device)
            curr_batch = stack_imgs_to_tensor(augmented_curr, camera_channels, device)
            
            t3 = time.perf_counter()
            print(f"Image loading and augmentation took {(t3 - t0)*1000:.2f} ms "
                  f"[io={ (t1-t0)*1000:.2f} | aug={ (t2-t1)*1000:.2f} | to_tensor={ (t3-t2)*1000:.2f}]")

            # ---- DepthAnythingV2 ----
            end1 = time.perf_counter()
            prev_bgr_list = [augmented_prev[ch] for ch in camera_channels]
            curr_bgr_list = [augmented_curr[ch] for ch in camera_channels]
            imgs_bgr_12 = prev_bgr_list + curr_bgr_list

            torch.cuda.nvtx.range_push("DepthAnything")
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    depth_list_12 = depth_model.infer_images(imgs_bgr_12, input_size=FAST_DEPTH_INPUT)
            torch.cuda.nvtx.range_pop()

            end2 = time.perf_counter()
            print(f"DepthAnythingV2 inference took {(end2 - end1)*1000:.2f} ms")

            prev_depth_list = depth_list_12[:len(camera_channels)]
            curr_depth_list = depth_list_12[len(camera_channels):]
            prev_depth_pred_map = {ch: d for ch, d in zip(camera_channels, prev_depth_list)}
            curr_depth_pred_map = {ch: d for ch, d in zip(camera_channels, curr_depth_list)}

            def stack_depth_batch(depth_dict):
                t = torch.stack([torch.from_numpy(depth_dict[ch]) for ch in camera_channels], dim=0)\
                        .unsqueeze(0).unsqueeze(2)  # [1,V,1,H,W]
                return t.pin_memory()

            prev_depths_pred_batch = stack_depth_batch(prev_depth_pred_map).to(device, non_blocking=True)
            curr_depths_pred_batch = stack_depth_batch(curr_depth_pred_map).to(device, non_blocking=True)

            # ---- range 投影（注意：test_entries 里 meta['K'] 已是 K_ud，再无畸变） ----
            torch.cuda.nvtx.range_push("build_mapping")
            proj_range_prev, proj_pix_prev = build_frame_mapping_fast(
                test_entries, 'sjtu', 'prev', prev_depth_pred_map,
                affine_matrix, idx, H_r=FAST_RANGE_H, W_r=FAST_RANGE_W,
                visualize=False, pixel_stride=FAST_PIXEL_STRIDE
            )
            proj_range_curr, proj_pix_curr = build_frame_mapping_fast(
                test_entries, 'sjtu', 'curr', curr_depth_pred_map,
                affine_matrix, idx, H_r=FAST_RANGE_H, W_r=FAST_RANGE_W,
                visualize=False, pixel_stride=FAST_PIXEL_STRIDE
            )
            torch.cuda.nvtx.range_pop()

            proj_pix_prev_batch = torch.from_numpy(proj_pix_prev.astype(np.int64)).unsqueeze(0).pin_memory().to(device, non_blocking=True)
            proj_pix_curr_batch = torch.from_numpy(proj_pix_curr.astype(np.int64)).unsqueeze(0).pin_memory().to(device, non_blocking=True)

            end3 = time.perf_counter()
            print(f"Building range-image mapping took {(end3 - end2)*1000:.2f} ms")
            # ---- 模型前向 ----
            torch.cuda.nvtx.range_push("FpTTC.forward")
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=AMP_DTYPE):
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
                )
            torch.cuda.nvtx.range_pop()
                
            end4 = time.perf_counter()
            print(f"Inference took {(end4 - end3)*1000:.2f} ms")
            print(f"Total time for idx {idx} : {(end4 - start)*1000:.2f} ms")
            # Visualization
            # visualize RGB images
            concat_prev = np.concatenate([augmented_prev[ch] for ch in camera_channels], axis=1)
            concat_prev = concat_prev.astype(np.uint8)
            concat_prev = cv2.cvtColor(concat_prev, cv2.COLOR_BGR2RGB)
            concat_prev_img = Image.fromarray(concat_prev)
            concat_prev_img.save(os.path.join(output_dir, f"concat_prev_{idx}.png"))

            scale_prediction_array = scale_pred[0].squeeze(0).cpu().numpy()

            risk_prediction_array = risk_pred[0].squeeze(0).cpu().numpy()

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

        pool.shutdown(wait=True)

    else:
        VIS = True
        mid_err = 0.0
        mid_err_pct = {"Err-1": 0.0, "Err-2": 0.0, "Err-5": 0.0}
        orien_status = []
        count = 0
        for idx in tqdm(range(len(test_entries)), desc='Processing surround view images'):
            if test_entries[idx]['scene_indice'] != '10':
                continue
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
            proj_range_prev, proj_pix_prev, sensor_metas_prev = build_frame_mapping(test_entries, 'nusc', 'prev', depth_pred_prev, 
                                                                 affine_matrix, idx, H_r=40, W_r=480)
            proj_range_curr, proj_pix_curr, sensor_metas_curr = build_frame_mapping(test_entries, 'nusc', 'curr', depth_pred_curr,
                                                                 affine_matrix, idx, H_r=40, W_r=480)
            sensor_metas = {'prev': sensor_metas_prev, 'curr': sensor_metas_curr}
            for frame_key in sensor_metas:
                for channel in sensor_metas[frame_key]:
                    for key in sensor_metas[frame_key][channel]:
                        sensor_metas[frame_key][channel][key] = torch.from_numpy(sensor_metas[frame_key][channel][key].astype(np.float32)).unsqueeze(0).to(device)
            # 转换为 tensor
            proj_pix_prev_tensor = torch.from_numpy(proj_pix_prev.astype(np.int64))   # (M, 3)
            proj_pix_curr_tensor = torch.from_numpy(proj_pix_curr.astype(np.int64))   # (M, 3)
            # 打包 batch
            proj_pix_prev_batch = proj_pix_prev_tensor.unsqueeze(0).to(device)
            proj_pix_curr_batch = proj_pix_curr_tensor.unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                scale_pred, risk_pred = model.forward(
                        img_prev         = prev_batch,
                        img_curr         = curr_batch,
                        depth_prev       = prev_depths_pred_batch,
                        depth_curr       = curr_depths_pred_batch,
                        proj_pix_prev    = proj_pix_prev_batch,
                        proj_pix_curr    = proj_pix_curr_batch,
                        sensor_metas     = sensor_metas,
                        attn_type        = args.attn_type,
                        attn_splits_list = args.attn_splits_list,
                        corr_radius_list = args.corr_radius_list,
                        prop_radius_list = args.prop_radius_list,
                        num_reg_refine   = args.num_reg_refine,
                        scale_only       = True,
                    )
            loss = get_loss_scale_map(scale_pred, gt_scale_tensor)
            mid_err += loss.item()
            count += 1

            # 计算 Error -1 -2 -5
            errs = compute_errs_ttc_from_scale(scale_pred[0,0], gt_scale_tensor[0,0], gt_scale_tensor[0,1])
            for k in mid_err_pct.keys():
                if errs[k] is not None:
                    mid_err_pct[k] += errs[k]
            
            # # Orientation & High-Risk evaluation
            # status = eval_orientation_and_highrisk_stats(
            #     ori_pred = risk_pred[0,0], 
            #     ori_gt   = gt_risk_tensor[0,0], 
            #     scale_pred = scale_pred[0,0],
            #     scale_gt   = gt_scale_tensor[0,0],
            #     valid_mask = gt_risk_tensor[0,1], 
            #     )
            # orien_status.append(status)

            if VIS: 
                # # vis ground-truth and prediction scale
                # gt_scale_array = gt_scale_tensor[0,0].cpu().numpy()
                # gt_scale_vis = np.clip(gt_scale_array, 0.0, 2.0)
                # gt_scale_vis = scale2rgb(gt_scale_vis)
                # gt_scale_vis = gt_scale_vis * 255.0
                # cv2.imwrite(os.path.join(output_dir, f"gt_scale_{idx}.png"), gt_scale_vis)

                scale_prediction_array = scale_pred[0].squeeze(0).cpu().numpy()
                scale_vis = np.clip(scale_prediction_array, 0.0, 2.0)
                vis = scale2rgb(scale_vis)
                vis = vis * 255.0
                cv2.imwrite(os.path.join(output_dir, f"pred_scale_{idx}.png"), vis)

                # # vis ground-truth and prediction orientation
                # gt_risk_array = gt_risk_tensor[0,0].cpu().numpy()
                # gt_risk_vis = np.clip(gt_risk_array, 0.0, np.pi)
                # gt_risk_vis = orientation2rgb(gt_risk_vis)
                # gt_risk_vis = gt_risk_vis * 255.0
                # vis_bgr = cv2.cvtColor(gt_risk_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
                # cv2.imwrite(os.path.join(output_dir, f"gt_risk_{idx}.png"), vis_bgr)

                # risk_prediction_array = risk_pred[0].squeeze(0).cpu().numpy()
                # risk_vis = np.clip(risk_prediction_array, 0.0, np.pi)
                # risk_vis = orientation2rgb(risk_vis)
                # risk_vis = risk_vis * 255.0
                # risk_vis = cv2.cvtColor(risk_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
                # cv2.imwrite(os.path.join(output_dir, f"pred_risk_{idx}.png"), risk_vis)

                # # 将 augmented_prev 中的图像按照channel顺序拼接，并保存为 concat_prev_{idx}.png
                # concat_prev = np.concatenate([augmented_prev[ch] for ch in camera_channels], axis=1)
                # concat_prev = concat_prev.astype(np.uint8)
                # concat_prev_img = Image.fromarray(concat_prev)
                # concat_prev_img.save(os.path.join(output_dir, f"concat_prev_{idx}.png"))

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

            # Cleanup
            del prev_batch, curr_batch, scale_pred, risk_pred
            torch.cuda.empty_cache()
        # MiD Error
        print(f"Average scale MAE on scene {scene_indice}: {mid_err/count:.6f}")
        # MiD Error -1 -2 -5
        for k in mid_err_pct.keys():
            print(f"Error-{k} (%): {mid_err_pct[k]/count*100:.2f}")
        # Orientation & High-Risk metrics
        # orien_final_metrics = reduce_metrics_across_frames(orien_status, beta=2.0)
        # print("Orientation and High-Risk Metrics:", orien_final_metrics)
        

if __name__ == "__main__":
    main()