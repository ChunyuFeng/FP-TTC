from PIL import Image
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import datetime
from glob import glob
from fpttc.fp_ttc import FpTTC
from utils.trainer import TTCTrainer
from utils.draw import disp2rgb_normalized, flow_uv_to_colors, flow_to_image, visual_scale_map_range_image
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                    help='where to save the training log and models')
parser.add_argument('--stage', default='chairs', type=str,
                    help='training stage on different datasets')
parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                    help='validation datasets')
parser.add_argument('--max_flow', default=400, type=int,
                    help='exclude very large motions during training')
parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                    help='image size for training')
parser.add_argument('--padding_factor', default=16, type=int,
                    help='the input should be divisible by padding_factor, otherwise do padding or resizing')

# evaluation
parser.add_argument('--eval', action='store_true',
                    help='evaluation after training done')
parser.add_argument('--save_eval_to_file', action='store_true')
parser.add_argument('--evaluate_matched_unmatched', action='store_true')
parser.add_argument('--val_things_clean_only', action='store_true')
parser.add_argument('--with_speed_metric', action='store_true',
                    help='with speed methic when evaluation')

# training
parser.add_argument('--load_flow_param', action='store_true')
parser.add_argument('--epoch', default=400, type=int)
parser.add_argument('--lr', default=4e-4, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--grad_clip', default=1.0, type=float)
parser.add_argument('--num_steps', default=100000, type=int)
parser.add_argument('--seed', default=326, type=int)
parser.add_argument('--summary_freq', default=100, type=int)
parser.add_argument('--val_freq', default=10000, type=int)
parser.add_argument('--save_ckpt_freq', default=10000, type=int)
parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

# resume pretrained model or resume training
parser.add_argument('--resume', default=None, type=str,
                    help='resume from pretrained model or resume from unexpectedly terminated training')
parser.add_argument('--strict_resume', action='store_true',
                    help='strict resume while loading pretrained weights')
parser.add_argument('--no_resume_optimizer', action='store_true')

# model: learnable parameters
parser.add_argument('--num_scales', default=1, type=int,
                    help='feature scales: 1/8 or 1/8 + 1/4')
parser.add_argument('--feature_channels', default=128, type=int)
parser.add_argument('--upsample_factor', default=8, type=int)
parser.add_argument('--num_head', default=1, type=int)
parser.add_argument('--ffn_dim_expansion', default=4, type=int)
parser.add_argument('--num_transformer_layers', default=6, type=int)
parser.add_argument('--reg_refine', action='store_true',
                    help='optional task-specific local regression refinement')
parser.add_argument('--parallel', action='store_true',
                    help='optional task-specific local regression refinement')
parser.add_argument('--load_opt', action='store_true',
                    help='optional task-specific local regression refinement')

# model: parameter-free
parser.add_argument('--attn_type', default='swin', type=str,
                    help='attention function')
parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                    help='number of splits in attention')
parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                    help='correlation radius for matching, -1 indicates global matching')
parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                    help='self-attention radius for propagation, -1 indicates global attention')
parser.add_argument('--num_reg_refine', default=1, type=int,
                    help='number of additional local regression refinement')

# loss
parser.add_argument('--gamma', default=0.9, type=float,
                    help='exponential weighting')

# predict on sintel and kitti test set for submission
parser.add_argument('--kittidataset', default='/mnt/pool2/lcl/data/data_scene_flow/training/', type=str)
parser.add_argument('--drivingdataset', default='/mnt/pool2/lcl/data/Driving/', type=str)
parser.add_argument('--submission', action='store_true',
                    help='submission to sintel or kitti test sets')
parser.add_argument('--output_path', default='output', type=str,
                    help='where to save the prediction results')
parser.add_argument('--save_vis_flow', action='store_true',
                    help='visualize flow prediction as .png image')
parser.add_argument('--no_save_flo', action='store_true',
                    help='not save flow as .flo if only visualization is needed')

# inference on images or videos
parser.add_argument('--inference_dir', default=None, type=str)
parser.add_argument('--inference_video', default=None, type=str)
parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                    help='can specify the inference size for the input to the network')
parser.add_argument('--save_flo_flow', action='store_true')
parser.add_argument('--pred_bidir_flow', action='store_true',
                    help='predict bidirectional flow')
parser.add_argument('--pred_bwd_flow', action='store_true',
                    help='predict backward flow only')
parser.add_argument('--fwd_bwd_check', action='store_true',
                    help='forward backward consistency check with bidirection flow')
parser.add_argument('--save_video', action='store_true')
parser.add_argument('--concat_flow_img', action='store_true')

# distributed training
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

# misc
parser.add_argument('--count_time', action='store_true',
                    help='measure the inference time')

parser.add_argument('--debug', action='store_true')

parser.add_argument('--train_location', type=str, default='local', choices=['local', 'remote-eden'],
                    help='Specify the training location. If "local", data will be stored in the local directory; if "remote", data will be stored in the remote specific path.')


args = parser.parse_args()
torch.cuda.set_device(0)
device = torch.device("cuda")


def resize_and_crop(img, width, height):
    """将图像缩放并裁剪到指定大小"""
    w, h = img.size
    crop_h, crop_w = height, width
    resize = max(crop_h / h, crop_w / w)

    resize_h, resize_w = int(h * resize), int(w * resize)
    crop_h_start = (resize_h - crop_h) // 2
    crop_w_start = (resize_w - crop_w) // 2
    crop = (crop_w_start, crop_h_start, crop_w_start + crop_w, crop_h_start + crop_h)


    img = img.resize((resize_w, resize_h), Image.BILINEAR)

    img = img.crop(crop)

    return img

def main():


    # from utils.unimatch.unimatch import UniMatch
    model_loaded = FpTTC(num_scales=args.num_scales,
                            feature_channels=args.feature_channels,
                            upsample_factor=args.upsample_factor,
                            num_head=args.num_head,
                            ffn_dim_expansion=args.ffn_dim_expansion,
                            num_transformer_layers=args.num_transformer_layers,
                            reg_refine=args.reg_refine,
                            train=False).to(device)
    num_params = sum(p.numel() for p in model_loaded.parameters())
    print('Number of params:', num_params)


    if args.resume is not None:
        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        if 'net' in checkpoint:
            model_loaded.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
        elif 'state_dict' in checkpoint:
            model_loaded.load_state_dict(checkpoint['state_dict'])
        else:
            model_loaded.load_state_dict(checkpoint)


    time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")

    model_loaded.eval()
    out_dir = "./test/%s_selfcon_ttc"%(time_stamp)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # # path1, path2 = 'test_img/2341.jpg', 'test_img/2344.jpg'
    # inference_dir = args.inference_dir
    # filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))
    # # print(filenames)
    # print('%d images found' % len(filenames))

    camera_lut_path = '/mnt/fpttc_data/TVT_infos/nusc_range_image_train_infos_160_1920.pkl'
    # camera_lut_path = './Datasets/nuscenes/camera_lut_sorted.pkl'
    # 读取 camera_lut.pkl
    with open(camera_lut_path, 'rb') as file:
        camera_lut = pickle.load(file)

    image_list= []
    scale_map_list = []
    infos = camera_lut['infos']
    for i in range(len(infos) - 1):
        current_info = infos[i]
        next_info = infos[i + 1]

        # 25 ms < timestamp_diff < 125 ms
        if abs(next_info['timestamp'] - current_info['timestamp']) / 1e3 < 25 or abs(
                next_info['timestamp'] - current_info['timestamp']) / 1e3 > 125:
            continue
        else:
            image_list.append([current_info['original_imgs_path'], next_info['original_imgs_path']])
            scale_map_list.append([current_info['scale_map_path'], next_info['scale_map_path']])

    surr_view_imgs1 = {}
    surr_view_imgs2 = {}
    image_path_prefix = '/home/chunyu/WorkSpace/BugStudio/FP-TTC/Datasets/nuscenes/'
    camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                       'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

    for i in tqdm(range(len(image_list)), desc='Processing'):
        surr_view_imgs_path1, surr_view_imgs_path2 = image_list[i]
        scale_map_path1, scale_map_path2 = scale_map_list[i]
        for channel in camera_channels:
            if not surr_view_imgs_path1[channel].startswith(image_path_prefix):
                surr_view_imgs_path1[channel] = image_path_prefix + surr_view_imgs_path1[channel]
            if not surr_view_imgs_path2[channel].startswith(image_path_prefix):
                surr_view_imgs_path2[channel] = image_path_prefix + surr_view_imgs_path2[channel]
            surr_view_imgs1[channel] = Image.open(surr_view_imgs_path1[channel]).convert('RGB')
            surr_view_imgs1[channel] = resize_and_crop(surr_view_imgs1[channel], 320, 160)
            surr_view_imgs1[channel] = np.array(surr_view_imgs1[channel])
            surr_view_imgs2[channel] = Image.open(surr_view_imgs_path2[channel]).convert('RGB')
            surr_view_imgs2[channel] = resize_and_crop(surr_view_imgs2[channel], 320, 160)
            surr_view_imgs2[channel] = np.array(surr_view_imgs2[channel])

        for channel in camera_channels:
            surr_view_imgs1[channel] = torch.from_numpy(surr_view_imgs1[channel]).permute(2, 0, 1).float().unsqueeze(0).to(device)
            surr_view_imgs2[channel] = torch.from_numpy(surr_view_imgs2[channel]).permute(2, 0, 1).float().unsqueeze(0).to(device)

        prev_surr_view_imgs_tensor = torch.stack([surr_view_imgs1[channel] for channel in camera_channels], dim=1)
        curr_surr_view_imgs_tensor = torch.stack([surr_view_imgs2[channel] for channel in camera_channels], dim=1)

        # 服务器上的路径前缀和本地路径前缀不一样，需要进行替换
        local_scale_map_path_prefix = '/mnt/fpttc_data/scale_map/'
        eden_scale_map_path_prefix = '/mnt/pool/fcy/FP-TTC/Datasets/scale_map/'
        if args.train_location == 'local':
            print('Training on Odyssey...')
        elif args.train_location == 'remote-eden':
            for info in infos:
                if info['scale_map_path'].startswith(local_scale_map_path_prefix):
                    info['scale_map_path'] = info['scale_map_path'].replace(local_scale_map_path_prefix,
                                                                            eden_scale_map_path_prefix)
        else:
            raise ValueError('Invalid train_location: ', args.train_location)

        gt_scale = np.load(scale_map_path1)
        gt_scale = torch.from_numpy(gt_scale).float()
        mask = gt_scale > 0

        scale, flow, _ = model_loaded(prev_surr_view_imgs_tensor, curr_surr_view_imgs_tensor,
                                      attn_type=args.attn_type,
                                      attn_splits_list=args.attn_splits_list,
                                      corr_radius_list=args.corr_radius_list,
                                      prop_radius_list=args.prop_radius_list,
                                      num_reg_refine=args.num_reg_refine,
                                      testing=False)

        # visualization
        gt_scale_np = gt_scale.detach().squeeze(0).cpu().numpy()
        gt_valid_mask = gt_scale_np > 0
        normalized_gt = visual_scale_map_range_image(gt_scale_np, gt_valid_mask)

        scale_np = scale[0].detach().squeeze(0).cpu().numpy()
        pred_valid_mask = scale_np > 0
        normalized_pred = visual_scale_map_range_image(scale_np, pred_valid_mask)

        plt.imsave(os.path.join(out_dir, f"gt{i}.jpg"), normalized_gt, cmap='seismic', vmin=-1, vmax=1)
        plt.imsave(os.path.join(out_dir, f"pred{i}.jpg"), normalized_pred, cmap='seismic', vmin=-1, vmax=1)

if __name__ == "__main__":
    main()