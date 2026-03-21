#!/usr/bin/env python3
import os
import cv2
import numpy as np
import math
import argparse
import pickle
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from PIL import Image
import datetime
import random

from dataloader.utils.augmentor import NuscRangeImageAugmentor

from utils.draw import (
    visual_scale_map_range_image,        # 若不再需要，可移除
    visual_risk_score_map_range_image,   # 若不再需要，可移除
    scale2rgb,
    orientation2rgb
)
from utils.nusc_paths import resolve_nusc_path
from collision_utils import (
    get_second_grad,
    get_ttc_var,
    get_grid_ttc,
    inverse_range_projection,
    project_lidar_to_surround_view_img
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collision risk detection from scale predictions (OpenCV-only visualization)"
    )
    # Dataset settings
    parser.add_argument("--version", type=str, default="v1.0-trainval",
                        help="NuScenes dataset version")
    parser.add_argument("--dataroot", type=str, default="./Datasets/nuscenes",
                        help="Path to NuScenes dataroot")
    # Timing and thresholds
    parser.add_argument("--fps", type=int, default=12,
                        help="Frame rate for delta_t calculation")
    parser.add_argument("--approach_threshold", type=float, default=1.0,
                        help="Scale threshold for valid region")
    parser.add_argument("--risk_time_threshold", type=float, default=3.0,
                        help="Max collision time to consider risk (seconds)")
    parser.add_argument("--risk_pred_threshold", type=float, default=0.0,
                        help="Only keep collision points whose risk > this threshold")
    # Feature extraction parameters
    parser.add_argument("--grid_size", type=int, default=3,
                        help="Size of grid cell for variance and grid TTC")
    parser.add_argument("--risk_rate", type=float, default=0.28,
                        help="Proportion of mean gradient for valid_grad_mask")
    parser.add_argument("--kp_detector_threshold", type=int, default=2,
                        help="FAST keypoint detector threshold")
    parser.add_argument("--grad_step", type=int, default=2,
                        help="Stride for second-order gradient computation")
    parser.add_argument('--image_size', default=[160, 320], type=int, nargs='+',
                        help='[height, width] for resize operations')
    # Modes
    parser.add_argument("--keyboard", action="store_true",
                        help="Use keyboard mode (cv2.imshow + waitKey)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to disk")
    # Paths
    parser.add_argument("--pred_npy_dir", type=str, required=True,
                        help="Directory of predicted .npy files, including pred scale and risk")
    parser.add_argument('--test_info_path', default='./Datasets/nuscenes/2_trainval_test_infos/nusc_trainval_infos_160_1920.pkl',
                        type=str,
                        help='Path to test info file (e.g., nusc_trainval_infos_160_1920.pkl)')
    parser.add_argument("--vis_dir", type=str, required=True,
                        help="Output directory for overlays")
    parser.add_argument("--sjtu_test", action="store_true",
                        help="Use SJTU test set, which has no ground truth data")
    parser.add_argument("--sjtu_surround_view_path", type=str, default='./Datasets/sjtu_surround_view',
                        help="Path to SJTU surround view images")
    parser.add_argument("--sjtu_scene_indice", type=int, default=0,
                        help="Scene index for SJTU test set, default is 0")
    return parser.parse_args()


def compute_and_save_masks(scale_map, risk_map, delta_t, ttc_thresh, risk_thresh, save_dir, idx):
    """
    计算两个 mask 后，将它们以二值图形式保存到磁盘（OpenCV）：
      mask_ttc： ttc = delta_t/(1-scale_map) < ttc_thresh
      mask_risk：risk_map > risk_thresh
    """
    eps = 1e-5
    ttc       = delta_t / (1.0 - scale_map + eps)
    mask_ttc  = (ttc       < ttc_thresh)
    mask_risk = (risk_map  > risk_thresh)

    ttc_img  = (mask_ttc .astype(np.uint8) * 255)
    risk_img = (mask_risk.astype(np.uint8) * 255)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"mask_ttc_{idx}.png"),  ttc_img)
    cv2.imwrite(os.path.join(save_dir, f"mask_risk_{idx}.png"), risk_img)
    return mask_ttc, mask_risk


def extract_small_regions(args, scale_map, risk_map, delta_t, ttc_thresh):
    """
    把所有 scale<阈值 的连通域都当碰撞点，返回 [x, y, ttc, scale, risk]
    """
    thresh_mask = (scale_map < args.approach_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_closed = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_closed, connectivity=8
    )
    out = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        cx, cy = int(cx), int(cy)
        ttc = delta_t / (1 - scale_map[cy, cx] + 1e-5)
        if 0 < ttc < ttc_thresh:
            out.append([cx, cy, ttc,
                        scale_map[cy, cx],
                        risk_map[cy, cx]])
    return out


def load_test_infos(test_info_path):
    with open(test_info_path, 'rb') as f:
        return pickle.load(f)


def get_pred_npy_files(pred_dir):
    files = [f for f in os.listdir(pred_dir) if f.endswith('.npy')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    return files


def load_prediction(pred_path):
    data = np.load(pred_path, allow_pickle=True).item()
    return data['scale_pred'], data['risk_pred']


def preprocess_scale_map(scale_map, grid_size, approach_threshold):
    """
    裁剪 + 归一化 scale_map（用于特征点检测）
    """
    H, W = scale_map.shape
    proc_h = (math.ceil(H / grid_size) - 1) * grid_size
    proc_w = (math.ceil(W / grid_size) - 1) * grid_size
    cropped = scale_map[:proc_h, :proc_w]
    min_s = np.min(cropped)
    norm_scale = (cropped - min_s) / (approach_threshold - min_s) * 255.0
    return cropped, norm_scale.astype(np.uint8)


def compute_valid_masks(scale_map, grid_size, grad_step, risk_rate, approach_threshold):
    """
    计算有效梯度 mask 和网格级有效 map
    """
    variance = get_ttc_var(scale_map, grid_len=grid_size)
    gy, gx, gy2, gx2 = get_second_grad(scale_map, stride=grad_step)

    mask_grad    = np.abs(gy) < (risk_rate * np.mean(np.abs(gy)))
    mask_thresh1 = scale_map[grad_step:, :] < approach_threshold
    mask_thresh2 = scale_map[1:-1, :]       < approach_threshold

    valid_grad = np.logical_and(mask_grad, mask_thresh1)
    valid_grad = np.logical_and(valid_grad, mask_thresh2)

    grid_map, _ = get_grid_ttc(scale_map, variance, grid_len=grid_size)
    grid_map = grid_map.astype(bool)

    return valid_grad, grid_map


def extract_candidate_points(scale_map, norm_scale, valid_grad_mask, grid_map, grid_size, grad_step, thresh, kp_thresh):
    """
    FAST + 网格候选点
    """
    fast = cv2.FastFeatureDetector_create(6400)
    fast.setThreshold(kp_thresh)
    kps = fast.detect(norm_scale, None)

    H, W = scale_map.shape
    proc_h, proc_w = H, W
    candidates = []
    dense_map = np.zeros_like(valid_grad_mask, dtype=np.uint8)
    invalid = []

    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if not (0.3*proc_h < y < 0.8*proc_h and 0.1*proc_w < x < 0.9*proc_w):
            continue
        area = 9
        y0, y1 = max(y-area//2,0), min(y+area//2+1,proc_h)
        x0, x1 = max(x-area//2,0), min(x+area//2+1,proc_w)
        window = valid_grad_mask[y0:y1, x0:x1]
        dense_map[y0:y1, x0:x1] = 1
        if window.sum()>3:
            orig = scale_map[y0:y1, x0:x1]
            if orig.min()<thresh and orig.max()<1.04:
                candidates.append([x, y, norm_scale[y0:y1, x0:x1].min()])
        else:
            invalid.append([x,y])

    for x,y in invalid:
        if dense_map[y,x]==0:
            continue
        area=7
        y0,y1 = max(y-area//2,0), min(y+area//2+1,proc_h)
        x0,x1 = max(x-area//2,0), min(x+area//2+1,proc_w)
        orig = scale_map[y0:y1, x0:x1]
        if orig.min()<thresh and orig.max()<1.04:
            candidates.append([x,y,np.min(norm_scale[y0:y1, x0:x1])])

    rows, cols = grid_map.shape
    for i in range(rows):
        for j in range(cols):
            if grid_map[i,j]:
                cy = int((2*i+1)*grid_size/2)
                cx = int((2*j+1)*grid_size/2)
                if not valid_grad_mask[cy, cx] or cy<0.4*proc_h:
                    continue
                if scale_map[cy, cx]<thresh:
                    local_norm = np.mean(
                        norm_scale[i*grid_size:(i+1)*grid_size,
                                   j*grid_size:(j+1)*grid_size]
                    )
                    candidates.append([cx,cy,local_norm])
    return np.array(candidates)


def cluster_and_compute(scale_map, risk_map, candidates, delta_t, risk_time_threshold):
    """
    两阶段 DBSCAN 聚类并计算碰撞时间
    返回 collision_points: [x, y, ttc, scale, risk]
    """
    collisions = []
    if candidates.size==0:
        return collisions

    clu1 = DBSCAN(eps=50, min_samples=5).fit(candidates)
    clusters1 = {}
    for idx,label in enumerate(clu1.labels_):
        if label<0: continue
        clusters1.setdefault(label,[]).append(candidates[idx])

    sec_centers, sec_regions = [], []
    for pts in clusters1.values():
        arr = np.array(pts)[:,:2].astype(int)
        (cx,cy), r = cv2.minEnclosingCircle(arr)
        ct = delta_t/(1-scale_map[int(cy),int(cx)]+1e-5)
        if r<6 and arr.shape[0]<4:
            continue
        sec_centers.append([cx,cy,ct*200])
        sec_regions.append(arr)
    sec_centers = np.array(sec_centers)
    if sec_centers.size==0:
        return collisions

    clu2 = DBSCAN(eps=100, min_samples=3).fit(sec_centers)
    regions = {}
    for idx,label in enumerate(clu2.labels_):
        regions.setdefault(label,[]).append(sec_regions[idx])

    for label, regs in regions.items():
        if label<0:
            for reg in regs:
                (cx,cy),_ = cv2.minEnclosingCircle(reg)
                ct = delta_t/(1-scale_map[int(cy),int(cx)]+1e-5)
                if 0<ct<risk_time_threshold:
                    collisions.append([int(cx),int(cy),ct,scale_map[int(cy),int(cx)],risk_map[int(cy),int(cx)]])
        else:
            merged = np.vstack(regs)
            cts = []
            for p in merged:
                y,x = int(p[1]),int(p[0])
                cts.append(delta_t/(1-scale_map[y,x]+1e-5))
            ctmin = min(cts)
            (cx,cy),_ = cv2.minEnclosingCircle(merged)
            if 0<ctmin<risk_time_threshold:
                collisions.append([int(cx),int(cy),ctmin,scale_map[int(cy),int(cx)],risk_map[int(cy),int(cx)]])
    return collisions


def filter_by_risk(collisions, risk_pred_threshold):
    """
    根据 risk_pred_map 过滤碰撞点
    """
    return [pt for pt in collisions if pt[-1] < risk_pred_threshold]


def render_range_image_bgr(map_to_vis, collisions, kind='scale'):
    """
    生成 BGR 彩图（用 scale2rgb / orientation2rgb 上色），并用 OpenCV 叠加圆点与文本
    """
    if kind == 'scale':
        scale = np.clip(map_to_vis, 0.0, 2.0)
        rgb  = scale2rgb(scale)  # (H,W,3), float32, 0~1

        # # 保留原先 mask 逻辑（可选）
        # mask = (map_to_vis > 0.3) & (map_to_vis < 3.0)
        # rgb  = np.where(mask[..., None], rgb, 0.0)

    elif kind == 'risk':
        # 原 risk 可视范围 [-pi/2, +pi/2]，线性映射到 [0,2]
        orien = np.clip(map_to_vis, 0.0, np.pi)
        rgb  = orientation2rgb(orien)
    else:
        raise ValueError("kind must be 'scale' or 'risk'")

    bgr = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    # bgr    = rgb_u8[..., ::-1].copy()  # RGB -> BGR

    # 叠加碰撞点与文本（黑色）
    for x, y, ttc, _, risk in collisions:
        cv2.circle(bgr, (int(x), int(y)), 5, (0, 0, 0), -1)
        cv2.putText(bgr, f"t={ttc:.2f}", (int(x)+6, int(y)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"r={risk:.2f}", (int(x)+6, int(y)+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return bgr


def overlay_surround_views_sjtu(collisions, test_info, args, idx):
    """
    在 SJTU 环视图上叠加碰撞点并保存（OpenCV）
    """
    surround_view_img_files = [f for f in os.listdir(args.sjtu_surround_view_path) if f.endswith('.png')]
    scene_idx = args.sjtu_scene_indice
    target_filename = f"scene_{scene_idx}_concat_prev_{idx}.png"
    if target_filename not in surround_view_img_files:
        print(f"Warning: {target_filename} not found in {args.sjtu_surround_view_path}. Skipping overlay.")
        return
    img_path = os.path.join(args.sjtu_surround_view_path, target_filename)

    cv_img = cv2.imread(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_h = int(30 * font_scale)

    for x, y, ttc, _, risk in collisions:
        cv2.circle(cv_img, (x, y), 6, (255, 0, 0), -1)
        cv2.putText(cv_img, f"t={ttc:.2f}", (x + 6, y - 6),
                    font, font_scale, (255, 0, 0), thickness, lineType=cv2.LINE_AA)
        cv2.putText(cv_img, f"r={risk:.2f}", (x + 6, y - 6 + line_h),
                    font, font_scale, (255, 0, 0), thickness, lineType=cv2.LINE_AA)

    out_path = os.path.join(args.vis_dir, f"surround_sjtu_{idx}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv_img)


def overlay_surround_views(collisions, test_info, args, idx):
    """
    在环视图上叠加碰撞点、裁剪、拼接并保存
    """
    gt = np.load(
        resolve_nusc_path(test_info['gt_map_path'], args.dataroot) / 'range_image_prev.npy',
        allow_pickle=True
    ).item()
    depth_map = gt['depth']
    metas     = test_info['sensor_metas_prev']
    prev_cam  = test_info['prev_camera_data']

    empty_pts = np.zeros((0, 3))
    proj_all = project_lidar_to_surround_view_img(
        empty_pts, prev_cam, metas, min_dist=1.0
    )
    surround_imgs = {}
    for ch, data in proj_all.items():
        img_np = np.array(data['original_img'])
        surround_imgs[ch] = Image.fromarray(img_np)

    if collisions and depth_map is not None:
        for x, y, ttc, _, risk in collisions:
            d = depth_map[y, x]
            if d < 0:
                continue
            xyz = inverse_range_projection(
                x, y, d,
                H=depth_map.shape[0],
                W=depth_map.shape[1],
                fov_up=8.0, fov_down=-15.0
            )
            proj_pts = project_lidar_to_surround_view_img(
                np.array([xyz]), prev_cam, metas, min_dist=1.0
            )
            for ch, data in proj_pts.items():
                pts2d = data['points'][:2, :].astype(int)
                if pts2d.shape[1] == 0:
                    continue

                pil_img = surround_imgs[ch]
                cv_img  = np.array(pil_img)

                xp, yp = pts2d[:, 0]
                cv2.circle(cv_img, (xp, yp), 30, (255, 0, 0), -1)
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                thickness  = 5
                line_h     = int(30 * font_scale)

                cv2.putText(cv_img, f"t={ttc:.2f}", (xp + 30, yp - 30),
                            font, font_scale, (255, 0, 0), thickness, lineType=cv2.LINE_AA)
                cv2.putText(cv_img, f"r={risk:.2f}", (xp + 30, yp - 30 + line_h),
                            font, font_scale, (255, 0, 0), thickness, lineType=cv2.LINE_AA)

                surround_imgs[ch] = Image.fromarray(cv_img)

    out_path = os.path.join(args.vis_dir, f"surround_{idx}.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resize_param = [args.image_size[0], args.image_size[1]*6]
    combine_surround_views(surround_imgs, resize_param, out_path)


def combine_surround_views(surround_imgs, resize_param, out_path):
    """
    surround_imgs: dict[str, PIL.Image]，6 路原始环视图
    out_path: 最终拼接图保存路径
    """
    camera_channels = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
    ]

    img_width_raw = surround_imgs[camera_channels[0]].size[0]
    range_image_width = resize_param[1]

    fx = (range_image_width / 6.0) / img_width_raw
    fy = fx

    resized_list = []
    for ch in camera_channels:
        img = surround_imgs[ch]
        arr = np.array(img)
        resized_arr = cv2.resize(arr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        resized_img = Image.fromarray(resized_arr)
        resized_list.append(resized_img)

    widths, heights = zip(*(im.size for im in resized_list))
    total_w = sum(widths)
    max_h = max(heights)

    canvas = Image.new('RGB', (total_w, max_h))
    x_offset = 0
    for im in resized_list:
        canvas.paste(im, (x_offset, 0))
        x_offset += im.width

    os.makedirs(os.path.dirname(out_path), exist_ok=True
               ) if os.path.dirname(out_path) else None
    canvas.save(out_path)


def main():
    args = parse_args()

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.vis_dir = os.path.join(args.vis_dir, f'collision_{ts}_ttc{args.risk_time_threshold}_risk{args.risk_pred_threshold}')
    os.makedirs(args.vis_dir, exist_ok=True)

    resize_height, resize_width = args.image_size
    augmentor = NuscRangeImageAugmentor(crop_size=(resize_height, resize_width),
                                        do_flip=False,
                                        rotate=False)

    delta_t = 1.0 / args.fps
    test_infos = load_test_infos(args.test_info_path)
    files = get_pred_npy_files(args.pred_npy_dir)

    idx_offset = int(files[0].split('_')[-1].split('.')[0]) if files else 0

    win_name = "range_image_vis"
    if args.keyboard or not args.save:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    for idx0, fn in enumerate(tqdm(files)):
        idx = idx0 + idx_offset

        # if idx < 101 or idx > 449:
        #     continue

        pred_path = os.path.join(args.pred_npy_dir, fn)
        scale_map, risk_map = load_prediction(pred_path)

        scale_thresh = 1 - delta_t / args.risk_time_threshold
        scale_map_filtered = np.where(scale_map < scale_thresh, scale_map, 0.999)

        crop_scale, norm_scale = preprocess_scale_map(
            scale_map_filtered, args.grid_size, args.approach_threshold
        )
        valid_grad, grid_map = compute_valid_masks(
            crop_scale, args.grid_size, args.grad_step,
            args.risk_rate, args.approach_threshold
        )
        candidates = extract_candidate_points(
            crop_scale, norm_scale, valid_grad, grid_map,
            args.grid_size, args.grad_step,
            args.approach_threshold, args.kp_detector_threshold
        )
        collisions = cluster_and_compute(
            crop_scale, risk_map, candidates, delta_t, args.risk_time_threshold
        )

        regions = extract_small_regions(
            args, crop_scale, risk_map, delta_t, args.risk_time_threshold
        )
        for r in regions:
            collisions.append(r)

        collisions = filter_by_risk(collisions, args.risk_pred_threshold)

        # ----- 生成并保存/显示：pred scale -----
        bgr = render_range_image_bgr(scale_map, collisions, kind='scale')
        if args.save:
            out_colli_scale_pred = os.path.join(args.vis_dir, f'colli_scale_pred_{idx}.png')
            cv2.imwrite(out_colli_scale_pred, bgr)
        if args.keyboard:
            cv2.imshow(win_name, bgr)
            key = cv2.waitKey(0)
            if key == 27:  # ESC 退出
                break
        elif not args.save:
            cv2.imshow(win_name, bgr)
            cv2.waitKey(1)

        if not args.sjtu_test:
            # ----- 环视图叠加 -----
            overlay_surround_views(collisions, test_infos[idx], args, idx)

            # ----- gt scale -----
            gt = np.load(
                resolve_nusc_path(test_infos[idx]['gt_map_path'], args.dataroot) / 'range_image_prev.npy',
                allow_pickle=True
            ).item()
            gt_scale_map = gt['scale']
            bgr = render_range_image_bgr(gt_scale_map, collisions, kind='scale')
            if args.save:
                out_colli_scale_gt = os.path.join(args.vis_dir, f'colli_scale_gt_{idx}.png')
                cv2.imwrite(out_colli_scale_gt, bgr)

            # ----- gt risk -----
            gt = np.load(
                resolve_nusc_path(test_infos[idx]['gt_map_path'], args.dataroot) / 'range_image_prev.npy',
                allow_pickle=True
            ).item()
            gt_risk_map = gt['risk_score']
            bgr = render_range_image_bgr(gt_risk_map, collisions, kind='risk')
            if args.save:
                out_colli_risk_gt = os.path.join(args.vis_dir, f'colli_risk_gt_{idx}.png')
                cv2.imwrite(out_colli_risk_gt, bgr)

            # ----- pred risk -----
            bgr = render_range_image_bgr(risk_map, collisions, kind='risk')
            if args.save:
                out_colli_risk_pred = os.path.join(args.vis_dir, f'colli_risk_pred_{idx}.png')
                cv2.imwrite(out_colli_risk_pred, bgr)
        else:
            # SJTU: 仅保存 pred scale / pred risk；环视图如需开启可用 overlay_surround_views_sjtu()
            if args.save:
                bgr = render_range_image_bgr(risk_map, collisions, kind='risk')
                out_colli_risk_pred = os.path.join(args.vis_dir, f'colli_risk_pred_{idx}.png')
                cv2.imwrite(out_colli_risk_pred, bgr)
            # overlay_surround_views_sjtu(collisions, test_infos[idx], args, idx)

    if args.keyboard or not args.save:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
