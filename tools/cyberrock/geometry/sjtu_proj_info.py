#!/usr/bin/env python3
import os
import sys
import bisect
import pickle
import argparse
import yaml
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from tools.cyberrock.sjtu_test_info import undistort_image
from fpttc.scale_net.utils.spherical import build_spherical_voxels, project_voxel_to_camera
from PIL import ImageDraw

def load_camera_projections(yaml_path):
    """
    读取包含多相机内外参以及畸变参数的 YAML 文件，返回每个相机的投影矩阵 P（3x4）及畸变系数。

    Args:
        yaml_path (str): YAML 文件路径。

    Returns:
        dict: 键为相机名称，值为包含以下键的字典：
              - 'K': 3x3 相机内参矩阵
              - 'dist': 畸变系数数组（如果未定义则为空数组）
              - 'has_distortion': 布尔，是否定义了畸变系数
              - 'R': 3x3 旋转矩阵
              - 't': 3x1 平移向量
              - 'P': 3x4 投影矩阵
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    calib = {}
    cameras = config.get('cameras', {})

    for cam_name, cam_cfg in cameras.items():
        # 内参
        intr = cam_cfg['intrinsics']
        fx, fy = intr['fx'], intr['fy']
        cx, cy = intr['cx'], intr['cy']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

        # 判断是否包含畸变参数
        if 'distortion' in intr and intr['distortion']:
            dist = np.array(intr['distortion'], dtype=float)
            has_dist = True
        else:
            dist = np.array([], dtype=float)
            has_dist = False
            print(f"[Warning] Camera '{cam_name}' missing distortion params in YAML.", file=sys.stderr)

        # 外参
        ext = cam_cfg['extrinsics']
        t = np.array(ext['translation'], dtype=float)
        q = ext['rotation']  # 四元数 [x, y, z, w]
        R = Rotation.from_quat(q).as_matrix()

        # 投影矩阵 P = K * [R | t]
        RT = np.hstack((R, t.reshape(3, 1)))  # 3x4 矩阵
        P = K.dot(RT)

        calib[cam_name] = {
            'K':              K,
            'dist':           dist,
            'has_distortion': has_dist,
            'R':              R,
            't':              t,
            'P':              P,
        }

    return calib

def build_sensor_index(base_dir):
    """
    遍历 base_dir 下每个子文件夹（channel），
    收集所有 .jpg 和 .npy 文件，按文件名中的 __timestamp 排序，
    返回结构：
      { channel: { 'filename': [...], 'timestamp': [...] }, ... }
    """
    if not os.path.isdir(base_dir):
        print(f"ERROR: 目录不存在: {base_dir}", file=sys.stderr)
        sys.exit(1)

    data_dict = {}
    for channel in os.listdir(base_dir):
        ch_dir = os.path.join(base_dir, channel)
        if not os.path.isdir(ch_dir):
            continue

        items = []
        for fn in os.listdir(ch_dir):
            ext = fn.lower().rsplit('.',1)[-1]
            if ext not in ('jpg','npy'):
                continue
            # 文件名形如 CHANNEL__123456789.npy 或 .jpg
            try:
                ts = int(fn.rsplit('__',1)[1].split('.')[0])
            except (IndexError, ValueError):
                continue
            items.append((ts, os.path.join(ch_dir, fn)))

        if not items:
            continue
        items.sort(key=lambda x: x[0])
        data_dict[channel] = {
            'filename':  [path for ts, path in items],
            'timestamp': [ts   for ts, path in items]
        }
    
    delta_ts = data_dict['LIDAR_TOP']['timestamp'][0] - data_dict['CAM_FRONT']['timestamp'][0]
    data_dict['LIDAR_TOP']['timestamp'] = [
        ts - delta_ts for ts in data_dict['LIDAR_TOP']['timestamp']
    ]

    return data_dict

def sync_sensor_dict(data_dict, hz=10):
    """
    按照指定频率同步所有 channel 的索引，
    返回和 data_dict 结构一样，但每个 channel 列表长度一致且已对齐。
    """
    channels = sorted(data_dict.keys())
    start_us = max(data_dict[ch]['timestamp'][0]     for ch in channels)
    end_us   = min(data_dict[ch]['timestamp'][-1]    for ch in channels)
    interval = int(1e6 / hz)
    target_ts = list(range(start_us, end_us+1, interval))

    synced = {ch: {'filename': [], 'timestamp': []} for ch in channels}
    for t in target_ts:
        for ch in channels:
            ts_list = data_dict[ch]['timestamp']
            fn_list = data_dict[ch]['filename']
            idx = bisect.bisect_left(ts_list, t)
            if idx == 0:
                pick = 0
            elif idx >= len(ts_list):
                pick = len(ts_list) - 1
            else:
                before, after = ts_list[idx-1], ts_list[idx]
                pick = idx-1 if abs(before-t) <= abs(after-t) else idx
            synced[ch]['timestamp'].append(ts_list[pick])
            synced[ch]['filename'].append(fn_list[pick])
    return synced

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',    required=True,
                        help='包含各通道（含 LIDAR_TOP）子文件夹的根目录')
    parser.add_argument('--yaml_path',   required=True,
                        help='相机内外参 YAML')
    parser.add_argument('--pkl_save_path', required=True,
                        help='输出 pkl 所在目录')
    parser.add_argument('--vis_save_path', required=True,
                        help='可视化结果所在目录')
    parser.add_argument('--hz', type=int, default=10,
                        help='同步帧率 (Hz)')
    args = parser.parse_args()

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

    # 1) 构建并同步索引
    data_dict = build_sensor_index(args.base_dir)
    synced    = sync_sensor_dict(data_dict, hz=args.hz)

    # 2) 读取相机标定
    calib = load_camera_projections(args.yaml_path)

    # 3) 打印对齐精度
    max_diffs = []
    chs = sorted(synced.keys())
    n = len(synced[chs[0]]['timestamp'])
    for i in range(n):
        ts_group = [synced[ch]['timestamp'][i] for ch in chs]
        # 这里只统计相机间差异，也可以一并加上 LiDAR
        max_diffs.append(max(ts_group) - min(ts_group))
    print(f"相机最大时间戳差: 最大 {max(max_diffs)} μs, 平均 {sum(max_diffs)/n:.1f} μs")
    
    # lidar 点云投影到对应帧的相机上
    camera_channels = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
        ]
    # 读取第一个 LiDAR TOP 的 .npy 点云文件
    lidar_path = synced['LIDAR_TOP']['filename'][0]
    lidar_pcd = np.load(lidar_path)
    images_undistorted = {}
    images = {}
    sensor_metas = {}
    for channel in camera_channels:
        image_path = synced[channel]['filename'][0]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images_undistorted[channel], K_undist, _ = undistort_image(
            image, calib[channel]['K'], calib[channel]['dist']
            )
        images_undistorted[channel] = Image.fromarray(cv2.cvtColor(images_undistorted[channel], 
                                                                   cv2.COLOR_BGR2RGB))
        images[channel] = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        sensor_metas[channel] = {
            'K_undist': K_undist,
            'K': calib[channel]['K'],
            'R': calib[channel]['R'],
            't': calib[channel]['t'],
            'P': calib[channel]['P'],
        }

    ones = np.ones((lidar_pcd.shape[0], 1), dtype=np.float32)
    pts_hom = np.concatenate([lidar_pcd, ones], axis=1).T  # (4, N)
    for channel in camera_channels:
        # RT = np.hstack([sensor_metas[channel]['R'], sensor_metas[channel]['t'].reshape(3, 1)])
        RT = old_extrinsics_[channel]  # 使用旧外参
        P = sensor_metas[channel]['K_undist'].dot(RT)
        uvw = P @ pts_hom
        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]
        depth = uvw[2]
        w_cam, h_cam = images_undistorted[channel].size
        # 过滤掉超出图像范围的点
        valid_mask = (depth > 1.0) & (u >= 0) & (u < w_cam) & (v >= 0) & (v < h_cam)
        # 将投影到图像平面的点转换为整数像素坐标
        us = u[valid_mask].astype(int)
        vs = v[valid_mask].astype(int)

        # 在 undistorted PIL 图像上绘制红点
        img = images_undistorted[channel].copy()
        draw = ImageDraw.Draw(img)
        r = 2  # 点半径
        for x, y in zip(us, vs):
            draw.ellipse((x - r, y - r, x + r, y + r), fill='red')

        # 保存结果图像到指定目录
        save_path = os.path.join(args.vis_save_path, f"0_{channel}_undist_proj.png")
        img.save(save_path)

        P_ = calib[channel]['P']
        uvw_ = P_ @ pts_hom
        u_ = uvw_[0] / uvw_[2]
        v_ = uvw_[1] / uvw_[2]
        depth_ = uvw_[2]
        valid_mask_ = (depth_ > 1.0) & (u_ >= 0) & (u_ < w_cam) & (v_ >= 0) & (v_ < h_cam)
        us_ = u_[valid_mask_].astype(int)
        vs_ = v_[valid_mask_].astype(int)
        # 在原始 PIL 图像上绘制红点
        img_orig = images[channel].copy()
        draw_orig = ImageDraw.Draw(img_orig)
        for x, y in zip(us_, vs_):
            draw_orig.ellipse((x - r, y - r, x + r, y + r), fill='red')
        # 保存原始图像投影结果
        save_path_orig = os.path.join(args.vis_save_path, f"1_{channel}_orig_proj.png")
        img_orig.save(save_path_orig)



    print(f"投影结果已保存到 {args.vis_save_path} 目录下。")



    # _, xyz = build_spherical_voxels(
    #     H=H_sph, W=W_sph, R=R,
    #     r_min=5.0, r_max=50.0,
    #     fov_up_deg=8, fov_down_deg=-15,
    # )
    # X, Y, Z = xyz[...,0], xyz[...,1], xyz[...,2]
    # X_new = -X
    # Y_new = -Y
    # Z_new =  Z
    # xyz = np.stack([X_new, Y_new, Z_new], axis=-1)

    # raw_img_size = next(iter(images_undistorted.values())).size
    # idx_uv = project_voxel_to_camera(
    #     xyz=xyz,
    #     sensor_metas=sensor_metas,
    #     camera_channels=camera_channels,
    #     raw_img_size=raw_img_size,
    #     min_dist=1.0,
    #     sjtu=True
    # )

    # # 4) 生成测试条目
    # infos = []
    # for i in range(1, n):
    #     prev_cam = {}
    #     curr_cam = {}
    #     for ch in chs:
    #         entry_prev = {
    #             'filename':  synced[ch]['filename'][i-1],
    #             'timestamp': synced[ch]['timestamp'][i-1]
    #         }
    #         entry_curr = {
    #             'filename':  synced[ch]['filename'][i],
    #             'timestamp': synced[ch]['timestamp'][i]
    #         }
    #         if ch == 'LIDAR_TOP':
    #             infos.append  # we'll fill lidar separately below
    #         else:
    #             prev_cam[ch] = entry_prev
    #             curr_cam[ch] = entry_curr

    #     info = {
    #         'prev_camera_data': prev_cam,
    #         'curr_camera_data': curr_cam,
    #         'prev_lidar_data':  {
    #             'filename':  synced['LIDAR_TOP']['filename'][i-1],
    #             'timestamp': synced['LIDAR_TOP']['timestamp'][i-1]
    #         },
    #         'curr_lidar_data':  {
    #             'filename':  synced['LIDAR_TOP']['filename'][i],
    #             'timestamp': synced['LIDAR_TOP']['timestamp'][i]
    #         },
    #         'sensor_metas_prev': calib,
    #         'sensor_metas_curr': calib,
    #         'gt_map_path':       None,
    #         'scene_flow_path':   None,
    #         'scene_indice':      None,
    #     }
    #     infos.append(info)

    # # 5) 保存 pkl
    # os.makedirs(args.pkl_save_path, exist_ok=True)
    # pkl_path = os.path.join(
    #     args.pkl_save_path,
    #     'sjtu_test_infos_synced.pkl'
    # )
    # with open(pkl_path, 'wb') as f:
    #     pickle.dump(infos, f)

    # print(f"Saved {len(infos)} entries to {pkl_path}")

if __name__ == '__main__':
    main()
