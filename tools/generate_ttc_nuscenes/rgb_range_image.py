import argparse
from tqdm import tqdm
import os
import numpy as np
import pickle
from pathlib import Path
from nuscenes.utils.geometry_utils import view_points
import argparse
from pyquaternion import Quaternion
from PIL import Image
import cv2
import glob

from dataloader.utils.augmentor import NuscRangeImageAugmentor
from utils.nusc_paths import infer_nusc_dataset_root, resolve_nusc_path

augmentor = NuscRangeImageAugmentor(crop_size=(160, 320),
                                    do_flip=False,
                                    rotate=False)

def create_video_from_images(save_path, fps=10):
    # 拼接所有保存的图像
    image_files = sorted(
        glob.glob(os.path.join(save_path, "rgb_range_image_*.png")),
        key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
    )
    if not image_files:
        print("没有找到图像文件用于创建视频。")
        return

    # 获取视频参数
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape
    video_path = os.path.join(save_path, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img_file in image_files:
        frame = cv2.imread(img_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"视频拼接完成，保存到: {video_path}")

def fill_line_holes(rgb_range,
                    hole_value=255,
                    horiz_length=31,
                    vert_length=31,
                    inpaint_radius=1):
    """
    专门填补横向和纵向的线状孔洞：
      1) 水平闭运算：kernel = (horiz_length x 1)
      2) 垂直闭运算：kernel = (1 x vert_length)
      3) 计算哪些孔洞被“闭合”了，生成 fill_mask
      4) 用 inpaint 仅修复这些位置

    参数
    ----
    rgb_range : np.ndarray, (H,W,3), uint8
        需修补的 RGB_range_image，孔洞像素 == hole_value。
    hole_value : int
        孔洞像素值，默认 255（白）。
    horiz_length : int
        水平线型结构元长度（要大于横洞最大长度）。
    vert_length : int
        垂直线型结构元长度。
    inpaint_radius : float
        cv2.inpaint 半径。
    """
    H, W = rgb_range.shape[:2]

    # 1. 构造孔洞 mask（1=孔洞，0=有效）
    hole_mask = (
        (rgb_range[:,:,0] == hole_value) &
        (rgb_range[:,:,1] == hole_value) &
        (rgb_range[:,:,2] == hole_value)
    ).astype(np.uint8)

    # 2. 反转得到有效像素 mask（1=有效，0=孔洞）
    valid_mask = 1 - hole_mask

    # 3. 水平闭运算：填平横向细缝
    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horiz_length, 1)
    )
    closed_h = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, horiz_kernel)

    # 4. 垂直闭运算：填平纵向细缝
    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vert_length)
    )
    closed_v = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, vert_kernel)

    # 5. 哪些位置原来是孔洞（valid_mask==0），
    #    而闭运算后变成有效（closed_h/v==1）——这些就是被“闭合”的线状空洞
    fill_h = (valid_mask == 0) & (closed_h == 1)
    fill_v = (valid_mask == 0) & (closed_v == 1)
    fill_mask = ((fill_h | fill_v).astype(np.uint8)) * 255  # 0/255 单通道

    # 6. 只对这些线状孔洞做 inpainting 修复
    filled = cv2.inpaint(
        rgb_range, fill_mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA
    )

    return filled

def fill_holes_with_closing(rgb_range, 
                            hole_value=255,
                            kernel_size=3,
                            inpaint_radius=3):
    """
    只对孔洞做闭运算并填补，不改变原图其它像素。

    参数
    ----
    rgb_range : np.ndarray, shape (H,W,3), dtype=uint8  
        带孔洞的彩色 range image，孔洞像素值 == hole_value。
    hole_value : int  
        用来表示孔洞的像素值（通常是255白色）。
    kernel_size : int  
        闭运算结构元尺寸（建议奇数，如5或7）。
    inpaint_radius : float  
        cv2.inpaint 中的 radius 参数，控制修复邻域大小。

    返回
    ----
    filled : np.ndarray, same as rgb_range  
        孔洞被修复后的图像，其它像素完好保留。
    """
    # 1. 构建单通道孔洞 mask (0/1)
    hole_mask = (
        (rgb_range[:, :, 0] == hole_value) &
        (rgb_range[:, :, 1] == hole_value) &
        (rgb_range[:, :, 2] == hole_value)
    ).astype(np.uint8)
    valid_mask = 1 - hole_mask  # 1表示有效像素，0表示孔洞

    # 2. 对孔洞 mask 做闭运算 (先膨胀再腐蚀)，得到闭合后的 mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                       (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(valid_mask, 
                                   cv2.MORPH_CLOSE, 
                                   kernel)

    # 3. 只有那些被“闭”上去的像素才真正需要填补
    #    (即在闭运算后变成非孔洞的位置)
    fill_mask = ((valid_mask == 0) & (closed_mask == 1)).astype(np.uint8) * 255
    # fill_mask 是单通道，0表示无需修复，255表示要修复

    # 4. 用 inpaint 只在 fill_mask 指定的位置做颜色修复
    filled = cv2.inpaint(rgb_range, 
                         fill_mask, 
                         inpaint_radius, 
                         flags=cv2.INPAINT_TELEA)
    return filled

def morphological_closing(rgb_range, kernel_size=5):
    """
    对 rgb_range 做一次形态学闭运算（先膨胀再腐蚀），填补小孔洞：
      kernel_size: 结构元大小，建议为奇数。
    """
    # 1. 构造一个椭圆形结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 2. 闭运算
    closed = cv2.morphologyEx(rgb_range, cv2.MORPH_CLOSE, kernel)
    return closed

def project_to_rgb_range_image0(proj_xyz,
                               proj_mask,
                               sensor_metas,
                               camera_data,
                               min_dist=1.0):
    """
    将 proj_xyz 投影回各环视相机，采样它们的 RGB，并组合成一个 [H,W,3] 的 RGB range image。
    没有投影点的像素保持黑色。
    """
    H, W = proj_mask.shape
    # 1. 先分解所有有效像素的位置
    mask_flat = proj_mask.reshape(-1).astype(bool)    # 长度 H*W
    idxs_flat = np.nonzero(mask_flat)[0]              # 扁平索引
    ys = idxs_flat // W
    xs = idxs_flat %  W
    pts_lidar = proj_xyz[ys, xs]                        # [N,3]

    # 准备输出 RGB range image
    rgb_range = np.ones((H, W, 3), dtype=np.uint8) * 255     # 白色背景
    # rgb_range = np.zeros((H, W, 3), dtype=np.uint8)          # 黑色背景

    # 2. 预取 LiDAR → Global 的变换
    lidar_cs   = sensor_metas['lidar']['calibrated_sensor']
    lidar_pose = sensor_metas['lidar']['ego_pose']
    R_s2e = Quaternion(lidar_cs['rotation']).rotation_matrix
    t_s2e = np.array(lidar_cs['translation'])
    R_e2g = Quaternion(lidar_pose['rotation']).rotation_matrix
    t_e2g = np.array(lidar_pose['translation'])

    # 3. 对每路环视相机进行反变换／投影／采样
    for channel, cam_info in camera_data.items():
        cam_cs   = sensor_metas['camera']['calibrated_sensor'][channel]
        cam_pose = sensor_metas['camera']['ego_pose'][channel]

        # 加载相机图像并转为 numpy array
        im = Image.open(resolve_nusc_path(cam_info['filename']))
        im_arr = np.array(im)  # H_cam x W_cam x 3

        # 相机外参（反向变换）
        R_e2g_cam = Quaternion(cam_pose['rotation']).rotation_matrix
        t_e2g_cam = np.array(cam_pose['translation'])
        R_c2e     = Quaternion(cam_cs['rotation']).rotation_matrix
        t_c2e     = np.array(cam_cs['translation'])

        # 把 LiDAR 点先变换到 Global
        pts = pts_lidar.copy()
        pts = (R_s2e @ pts.T).T + t_s2e
        pts = (R_e2g @ pts.T).T + t_e2g

        # Global → Ego_cam
        pts = pts - t_e2g_cam
        pts = (R_e2g_cam.T @ pts.T).T

        # Ego_cam → Camera_sensor
        pts = pts - t_c2e
        pts = (R_c2e.T  @ pts.T).T

        # 深度过滤
        depths = pts[:, 2]
        valid = depths > min_dist

        # 投影到像素
        intrinsic = np.array(cam_cs['camera_intrinsic'])
        uv = view_points(pts.T[:3, :], intrinsic, normalize=True)  # [2, N]
        u = uv[0].astype(np.int32)
        v = uv[1].astype(np.int32)

        # 再次裁剪到相机图像范围内
        h_cam, w_cam = im_arr.shape[:2]
        valid &= (u >= 0) & (u < w_cam) & (v >= 0) & (v < h_cam)

        # 最终有效点的索引
        sel = np.nonzero(valid)[0]

        # 4. 从相机图像采样 RGB，写回 range image
        u_sel = u[sel]
        v_sel = v[sel]
        # 对应在 range image 上的坐标
        ys_sel = ys[sel]
        xs_sel = xs[sel]

        colors = im_arr[v_sel, u_sel]  # [M,3] uint8
        rgb_range[ys_sel, xs_sel] = colors
    
    # 5. 填充未采样的区域
    # rgb_range = fill_only_small_holes(rgb_range, hole_value=255, max_area=100, inpaint_radius=1)
    # rgb_range = morphological_closing(rgb_range, kernel_size=2)
    rgb_range = fill_holes_with_closing(rgb_range, hole_value=255, kernel_size=3, inpaint_radius=3)
    rgb_range = fill_line_holes(rgb_range, hole_value=255,
                                horiz_length=31, vert_length=31, inpaint_radius=3)

    return rgb_range

    
def make_homog(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """构造 4×4 齐次变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def build_lidar_to_camera_projection(
    sensor_metas: dict,
    cam_cs: dict,
    cam_pose: dict
):
    """
    构造 LiDAR(S) -> Camera(C) 的投影矩阵 P（3x4）以及合并变换的 R_tot 和 t_tot。
    """
    # 1) LiDAR → Ego
    lidar_cs   = sensor_metas['lidar']['calibrated_sensor']
    lidar_pose = sensor_metas['lidar']['ego_pose']
    R_s2e = Quaternion(lidar_cs['rotation']).rotation_matrix
    t_s2e = np.array(lidar_cs['translation'])
    T_s2e = make_homog(R_s2e, t_s2e)

    # 2) Ego → Global
    R_e2g = Quaternion(lidar_pose['rotation']).rotation_matrix
    t_e2g = np.array(lidar_pose['translation'])
    T_e2g = make_homog(R_e2g, t_e2g)

    # 3) Global → Ego_Cam (取逆)
    R_e2g_cam = Quaternion(cam_pose['rotation']).rotation_matrix
    t_e2g_cam = np.array(cam_pose['translation'])
    T_g2ecam  = np.linalg.inv(make_homog(R_e2g_cam, t_e2g_cam))

    # 4) Ego_Cam → Cam_Sensor (取逆)
    R_c2e = Quaternion(cam_cs['rotation']).rotation_matrix
    t_c2e = np.array(cam_cs['translation'])
    T_ecam2c = np.linalg.inv(make_homog(R_c2e, t_c2e))

    # 5) 合并到 LiDAR → Cam_Sensor
    T_s2c = T_ecam2c @ T_g2ecam @ T_e2g @ T_s2e

    # 6) 拆出 R_tot, t_tot
    R_tot = T_s2c[:3, :3]
    t_tot = T_s2c[:3,  3]

    # 7) 加上内参，得到 P = K [R|t]
    K = np.array(cam_cs['camera_intrinsic'])
    P = K @ T_s2c[:3, :]

    return P, R_tot, t_tot


def project_to_rgb_range_image(proj_xyz,
                               proj_mask,
                               sensor_metas,
                               camera_data,
                               min_dist=1.0):  
    H, W = proj_mask.shape
    mask_flat = proj_mask.reshape(-1).astype(bool)
    idxs_flat = np.nonzero(mask_flat)[0]
    ys = idxs_flat // W
    xs = idxs_flat %  W
    pts_lidar = proj_xyz[ys, xs]          # (N,3)

    rgb_range = np.ones((H, W, 3), dtype=np.uint8) * 255

    prev_images = {}
    for channel, cam_info in camera_data.items():
        # 预加载相机图像
        im = Image.open(os.path.join('./Datasets/nuscenes', cam_info['filename']))
        prev_images[channel] = im

    # 预处理图像
    prev_images_aug, affine_matrices = augmentor(prev_images)

    for channel, cam_info in camera_data.items():
        # 加载图像
        # im = Image.open(os.path.join('./Datasets/nuscenes', cam_info['filename']))
        # im_arr = np.array(im)
        im_arr = prev_images_aug[channel]  # 预处理后的图像
        h_cam, w_cam = im_arr.shape[:2]

        # 相机外参
        cam_cs   = sensor_metas['camera']['calibrated_sensor'][channel]
        cam_pose = sensor_metas['camera']['ego_pose'][channel]

        # 构造投影矩阵与合并变换
        P, R_tot, t_tot = build_lidar_to_camera_projection(sensor_metas, cam_cs, cam_pose)

        # 用 R_tot, t_tot 计算每个点在相机坐标系下的深度
        pts_cam = (R_tot @ pts_lidar.T) + t_tot[:, None]  # shape=(3,N)
        depths  = pts_cam[2, :]

        P = affine_matrices @ P  # 应用预处理的仿射变换

        # # 一次性投影到像素平面
        # uv = view_points(pts_lidar.T, P, normalize=True)  # shape=(3,N)
        # u = uv[0].astype(np.int32)
        # v = uv[1].astype(np.int32)
        # 手动投影：齐次坐标
        N = pts_lidar.shape[0]
        pts_h = np.concatenate([pts_lidar, np.ones((N, 1))], axis=1).T  # 4×N
        uvw = (P @ pts_h)  # 3×N
        u = (uvw[0] / uvw[2]).astype(np.int32)
        v = (uvw[1] / uvw[2]).astype(np.int32)

        # 过滤
        valid = (depths > min_dist) & (u >= 0) & (u < w_cam) & (v >= 0) & (v < h_cam)
        sel = np.nonzero(valid)[0]

        # 写色
        ys_sel = ys[sel]
        xs_sel = xs[sel]
        rgb_range[ys_sel, xs_sel] = im_arr[v[sel], u[sel]]

    # 填洞
    rgb_range = fill_holes_with_closing(rgb_range, hole_value=255, kernel_size=3, inpaint_radius=3)
    rgb_range = fill_line_holes       (rgb_range, hole_value=255,
                                       horiz_length=31, vert_length=31, inpaint_radius=3)

    return rgb_range

def main(args):
    dataset_root = infer_nusc_dataset_root(args.pkl_path)

    try:
        with open(args.pkl_path, "rb") as f:
            # 尝试加载 pkl 文件
            trainval_test_info = pickle.load(f)
        print("pkl 文件加载成功!")
    except Exception as e:
        print("加载 pkl 文件失败:")
        print(e)

    for idx, info in enumerate(tqdm(trainval_test_info, desc="Processing info")):
        gt_map_path = resolve_nusc_path(info['gt_map_path'], dataset_root) / 'range_image.npy'
        gt_map = np.load(gt_map_path,
                         allow_pickle=True).item()
        
        rgb_range_image = project_to_rgb_range_image(
            proj_xyz=gt_map['xyz'],
            proj_mask=gt_map['mask'],
            sensor_metas=info['sensor_metas'],
            camera_data=info['prev_camera_data'],
            min_dist=1.0
        )
        
        # curr_rgb_range_image = project_to_rgb_range_image(
        #     proj_xyz=gt_map['xyz'],
        #     proj_mask=gt_map['mask'],
        #     sensor_metas=info['sensor_metas'],
        #     camera_data=info['curr_camera_data'],
        #     min_dist=1.0
        # )

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_path = os.path.join(args.save_path, f'rgb_range_image_{idx}.png')
        rgb_range_image = Image.fromarray(rgb_range_image.astype(np.uint8))
        rgb_range_image.save(save_path, format='PNG')

    # 创建视频
    create_video_from_images(args.save_path, fps=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", help="Path to the pkl file")
    parser.add_argument("--save_path", help="Path to save the output RGB range image")
    args = parser.parse_args()
    main(args)
