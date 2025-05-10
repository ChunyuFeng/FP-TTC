#!/usr/bin/env python3
import os
import cv2
import numpy as np
import math
import argparse
import pickle
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from PIL import Image
import datetime

from utils.draw import (
    visual_scale_map_range_image,
    visual_risk_score_map_range_image
)
from collision_utils import (
    get_second_grad,
    get_ttc_var,
    get_grid_ttc,
    inverse_range_projection,
    project_lidar_to_surround_view_img
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collision risk detection from scale predictions"
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
                        help="Wait for keyboard input each frame")
    parser.add_argument("--save", action="store_true",
                        help="Save results instead of interactive display")
    # Paths
    parser.add_argument("--pred_npy_dir", type=str, required=True,
                        help="Directory of predicted .npy files, including pred scale and risk")
    parser.add_argument('--test_info_path', default='./Datasets/nuscenes/2_trainval_test_infos/nusc_trainval_infos_160_1920.pkl',
                        type=str,
                        help='Path to test info file (e.g., nusc_trainval_infos_160_1920.pkl)')
    parser.add_argument("--vis_dir", type=str, required=True,
                        help="Output directory for RGB overlays")
    return parser.parse_args()


def init_visualization(save_mode, keyboard_mode):
    """
    初始化 Matplotlib 可视化窗口
    """
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis('off')
    return fig, ax


def load_test_infos(test_info_path):
    """
    加载 test_infos 数据
    """
    with open(test_info_path, 'rb') as f:
        return pickle.load(f)


def get_pred_npy_files(pred_dir):
    """
    获取并排序预测 .npy 文件列表
    """
    files = [f for f in os.listdir(pred_dir) if f.endswith('.npy')]
    files.sort(key=lambda x: int(x[5:-4]))
    return files


def load_prediction(pred_path):
    """
    加载单个预测文件，返回 scale 和 risk map
    """
    data = np.load(pred_path, allow_pickle=True).item()
    return data['scale_pred'], data['risk_pred']


def preprocess_scale_map(scale_map, grid_size, approach_threshold):
    """
    裁剪和归一化 scale_map
    返回裁剪后的 scale_map 和归一化结果 norm_scale
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
    计算有效梯度 mask 和网格级有效 map，返回：
      - valid_grad: 与 scale_map[grad_step:-1, :] 对齐的布尔数组（像素级）
      - grid_map: 与 scale_map 形状缩小 grid_size 倍后的网格级布尔数组
    """
    # 1. 计算局部方差（供网格TTC使用）
    variance = get_ttc_var(scale_map, grid_len=grid_size)

    # 2. 计算二阶梯度（与 scale_map[grad_step:, :] 对齐）
    gy, gx, gy2, gx2 = get_second_grad(scale_map, stride=grad_step)

    # 3. 梯度 & 阈值筛选（保持和原脚本对齐的切片）
    mask_grad    = np.abs(gy) < (risk_rate * np.mean(np.abs(gy)))
    mask_thresh1 = scale_map[grad_step:, :] < approach_threshold
    mask_thresh2 = scale_map[1:-1, :]   < approach_threshold

    valid_grad = np.logical_and(mask_grad, mask_thresh1)
    valid_grad = np.logical_and(valid_grad, mask_thresh2)

    # 4. 网格级TTC——获得 (grid_valid_map, grid_valid_mask)
    grid_map, _ = get_grid_ttc(scale_map, variance, grid_len=grid_size)

    # 5. 只保留布尔类型，后面按网格迭代使用
    grid_map = grid_map.astype(bool)

    return valid_grad, grid_map


def extract_candidate_points(scale_map, norm_scale, valid_grad_mask, grid_map, grid_size, grad_step, thresh, kp_thresh):
    """
    提取 FAST 关键点并生成候选碰撞点
    返回候选点列表 (x, y, norm_scale_value)
    """
    # FAST 检测
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
        if not (0.4*proc_h < y < 0.9*proc_h and 0.2*proc_w < x < 0.8*proc_w):
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

    # 重新检查 invalid
    for x,y in invalid:
        if dense_map[y,x]==0:
            continue
        area=7
        y0,y1 = max(y-area//2,0), min(y+area//2+1,proc_h)
        x0,x1 = max(x-area//2,0), min(x+area//2+1,proc_w)
        orig = scale_map[y0:y1, x0:x1]
        if orig.min()<thresh and orig.max()<1.04:
            candidates.append([x,y,np.min(norm_scale[y0:y1, x0:x1])])

    # 网格候选点
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
    对候选点进行两阶段 DBSCAN 聚类并计算碰撞时间
    返回 collision_points 列表 [x, y, time_to_collision, scale]
    """
    collisions = []
    if candidates.size==0:
        return collisions

    # 第一阶段
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

    # 第二阶段
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
    return [pt for pt in collisions if pt[-1] > risk_pred_threshold]


def visualize_range_image(ax, map_to_vis, collisions, type='scale'):
    """
    在 range image 上绘制碰撞点
    - map_to_vis: 要可视化的 range image，与 type 对应
    - collisions: 碰撞点列表 [x, y, time_to_collision, scale, risk]
    - type: 'scale' 或 'risk'，决定使用的可视化方法
    """
    if type == 'scale':
        mask = (map_to_vis > 0.3) & (map_to_vis < 3.0) 
        disp = visual_scale_map_range_image(map_to_vis, mask)
        ax.imshow(-disp, cmap='seismic', vmin=-1, vmax=1)
    elif type == 'risk':
        disp = visual_risk_score_map_range_image(map_to_vis)
        ax.imshow(disp, cmap='seismic', vmin=-1, vmax=1)
    else:
        raise ValueError("Invalid type. Use 'scale' or 'risk'.")
    # 绘制碰撞点，并添加文本标签：ttc 和 risk
    for x,y,ttc,_,risk in collisions:
        circ = plt.Circle((x,y),5,color='black',fill=True)
        ax.add_patch(circ)
        ax.text(x+6, y-6,
                f"t={ttc:.2f}\nr={risk:.2f}",
                color='black', fontsize=12,
                va='top', ha='left')

def save_range_image(fig, ax, out_path, scale_map):
    """
    保存 range image 可视化结果
    """
    dpi = 100
    fig.set_size_inches(scale_map.shape[1]/dpi, scale_map.shape[0]/dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    ax.clear()


def overlay_surround_views(collisions, test_info, args, idx):
    """
    在环视图上叠加碰撞点、裁剪、拼接并保存；
    即使 collisions 为空，也会输出原始环视图。
    """
    # 读取深度真值和相机元数据
    gt = np.load(
        os.path.join(test_info['gt_map_path'], 'range_image.npy'),
        allow_pickle=True
    ).item()
    depth_map = gt['depth']
    metas     = test_info['sensor_metas']
    prev_cam  = test_info['prev_camera_data']

    # 1. 先获取所有原始环视图（没有任何点），并转为 PIL.Image
    empty_pts = np.zeros((0, 3))
    proj_all = project_lidar_to_surround_view_img(
        empty_pts, prev_cam, metas, min_dist=1.0
    )
    surround_imgs = {}
    for ch, data in proj_all.items():
        img_np = np.array(data['original_img'])
        surround_imgs[ch] = Image.fromarray(img_np)

    # 2. 如果有碰撞点，则在对应视图上叠加红圈和文字
    if collisions and depth_map is not None:
        for x, y, ttc, _, risk in collisions:
            d = depth_map[y, x]
            if d < 0:
                continue
            xyz = inverse_range_projection(
                x, y, d,
                H=depth_map.shape[0],
                W=depth_map.shape[1],
                fov_up=10.0, fov_down=-30.0
            )
            proj_pts = project_lidar_to_surround_view_img(
                np.array([xyz]), prev_cam, metas, min_dist=1.0
            )
            for ch, data in proj_pts.items():
                pts2d = data['points'][:2, :].astype(int)
                if pts2d.shape[1] == 0:
                    continue

                # 从 PIL 转为 OpenCV 数组，叠加标记，再转回 PIL
                pil_img = surround_imgs[ch]
                cv_img  = np.array(pil_img)

                xp, yp = pts2d[:, 0]
                cv2.circle(cv_img, (xp, yp), 30, (255, 0, 0), -1)
                font      = cv2.FONT_HERSHEY_SIMPLEX
                font_scale= 3
                thickness = 5
                line_h    = int(30 * font_scale)  # 行高，根据 font_scale 调整

                # 第一行：t
                cv2.putText(
                    cv_img,
                    f"t={ttc:.2f}",
                    (xp + 30, yp - 30),
                    font, font_scale,
                    (255, 0, 0), thickness,
                    lineType=cv2.LINE_AA
                )
                # 第二行：r，y 坐标往下移动一个行高
                cv2.putText(
                    cv_img,
                    f"r={risk:.2f}",
                    (xp + 30, yp - 30 + line_h),
                    font, font_scale,
                    (255, 0, 0), thickness,
                    lineType=cv2.LINE_AA
                )

                surround_imgs[ch] = Image.fromarray(cv_img)

    # 3. 裁剪、拼接并保存最终合图
    out_path = os.path.join(args.vis_dir, f"surround_{idx}.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resize_param = [args.image_size[0], args.image_size[1]*6]
    combine_surround_views(surround_imgs, resize_param, out_path)


def combine_surround_views(surround_imgs, resize_param, out_path):
    """
    surround_imgs: dict[str, PIL.Image]，6 路原始环视图
    augmentor: 带有 resize_and_crop(self, img) 方法的实例
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

    # 1. 对每一路做 resize
    resized_list = []
    for ch in camera_channels:
        img = surround_imgs[ch]
        arr = np.array(img)
        resized_arr = cv2.resize(arr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        resized_img = Image.fromarray(resized_arr)
        resized_list.append(resized_img)

    # 2. 计算拼接后画布大小
    widths, heights = zip(*(im.size for im in resized_list))
    total_w = sum(widths)
    max_h = max(heights)

    # 3. 新建一个 RGB 画布，把每张图贴上去
    canvas = Image.new('RGB', (total_w, max_h))
    x_offset = 0
    for im in resized_list:
        canvas.paste(im, (x_offset, 0))
        x_offset += im.width

    # 4. 保存
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)

def main():
    args = parse_args()

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.vis_dir = os.path.join(args.vis_dir, f'collision_{ts}_ttc{args.risk_time_threshold}_risk{args.risk_pred_threshold}')
    os.makedirs(args.vis_dir, exist_ok=True)

    delta_t = 1.0 / args.fps
    test_infos = load_test_infos(args.test_info_path)
    files = get_pred_npy_files(args.pred_npy_dir)
    # results = {}
    fig, ax = init_visualization(args.save, args.keyboard)

    for idx, fn in enumerate(tqdm(files)):
        pred_path = os.path.join(args.pred_npy_dir, fn)
        scale_map, risk_map = load_prediction(pred_path)
        crop_scale, norm_scale = preprocess_scale_map(
            scale_map, args.grid_size, args.approach_threshold
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
        collisions = filter_by_risk(collisions, args.risk_pred_threshold)

        # 可视化 range image
        ax.clear()
        visualize_range_image(ax, scale_map, collisions, type='scale')
        # 交互式显示
        if args.keyboard:
            plt.title(fn)
            plt.draw()
            plt.waitforbuttonpress()
            ax.clear()
        # 保存可视化结果
        elif args.save:
            # 1. 保存 pred range image 可视化结果
            out_colli_scale_pred = os.path.join(args.vis_dir, f'colli_scale_pred_{idx}.png')
            save_range_image(fig, ax, out_colli_scale_pred, scale_map)
            # 2. 保存环视图叠加结果
            overlay_surround_views(collisions, test_infos[idx], args, idx)
            # 3. 保存 gt scale range image 可视化结果
            out_colli_scale_gt = os.path.join(args.vis_dir, f'colli_scale_gt_{idx}.png')
            gt = np.load(
                os.path.join(test_infos[idx]['gt_map_path'], 'range_image.npy'),
                allow_pickle=True
            ).item()
            gt_scale_map = gt['scale']
            ax.clear()
            visualize_range_image(ax, gt_scale_map, collisions, type='scale')
            save_range_image(fig, ax, out_colli_scale_gt, gt_scale_map)
            # 4. 保存 gt risk range image 可视化结果
            out_colli_risk_gt = os.path.join(args.vis_dir, f'colli_risk_gt_{idx}.png')
            gt = np.load(
                os.path.join(test_infos[idx]['gt_map_path'], 'range_image.npy'),
                allow_pickle=True
            ).item()
            gt_risk_map = gt['risk_score']
            ax.clear()
            visualize_range_image(ax, gt_risk_map, collisions, type='risk')
            save_range_image(fig, ax, out_colli_risk_gt, gt_risk_map)
            # 5. 保存 pred risk range image 可视化结果
            out_colli_risk_pred = os.path.join(args.vis_dir, f'colli_risk_pred_{idx}.png')
            ax.clear()
            visualize_range_image(ax, risk_map, collisions, type='risk')
            save_range_image(fig, ax, out_colli_risk_pred, risk_map) 
        # 逐帧自动显示结果
        else:
            plt.draw()
            plt.pause(0.01)
            ax.clear()


if __name__ == '__main__':
    main()
