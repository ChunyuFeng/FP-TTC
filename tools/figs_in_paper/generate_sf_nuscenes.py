import os
import torch
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation
from tqdm import trange
import open3d.visualization.gui as gui
from open3d.visualization import O3DVisualizer

import numpy as np

def se3_from_RT(R: np.ndarray, t: np.ndarray):
    """R(3,3), t(3,) -> T(4,4)"""
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def T_world_from_lidar(calibrated_sensor, ego_pose):
    """
    NuScenes：lidar->ego->world
    calibrated_sensor['rotation'] / ['translation'] : lidar->ego
    ego_pose['rotation'] / ['translation']         : ego->world
    返回: T_w_l = T_w_e * T_e_l
    """
    R_el = Quaternion(calibrated_sensor['rotation']).rotation_matrix
    t_el = np.array(calibrated_sensor['translation'], dtype=np.float64)
    T_e_l = se3_from_RT(R_el, t_el)

    R_we = Quaternion(ego_pose['rotation']).rotation_matrix
    t_we = np.array(ego_pose['translation'], dtype=np.float64)
    T_w_e = se3_from_RT(R_we, t_we)

    return T_w_e @ T_e_l


def load_pcd_from_file(pcd_file_path):

    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points = np.asarray(pcd.points, dtype=np.float32)  # (N,3)

    # rotate from (x->right, y->up, z->back) to (x->right, y->front, z->up)
    R = Rotation.from_euler('x', 90, degrees=True).as_matrix()  # (3,3)
    points_rotated = points @ R.T
    # (x->right, y->back, z->up)
    points_rotated[:, 1] = -points_rotated[:, 1]
    # normalize the points to [0,1]³
    mins = points_rotated.min(axis=0)  # (3,)
    maxs = points_rotated.max(axis=0)  # (3,)
    extents = maxs - mins             # (3,)
    points_norm = (points_rotated - mins) / extents

    # move the origin of the points to the center of the bounding box
    box_center = (0.5, 0.5, 0)
    points_norm -= box_center  # 将点云坐标转换到以aabb中心为原点的坐标系

    return points_norm



def visualize_prev_frame(filtered_prev_object_points_list,
                         filtered_prev_object_boxes_list):
    # 1. 准备几何体列表
    vis_geoms = []
    for idx, (pts, box) in enumerate(zip(filtered_prev_object_points_list,
                                         filtered_prev_object_boxes_list)):
        # 1.1 点云
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(pts[:, :3])
        pcd_obj.paint_uniform_color(np.random.uniform(0, 1, size=3))
        vis_geoms.append((f"pcd_{idx}", pcd_obj))

        # 1.2 线框
        corners = box.corners().T
        lines = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]
        colors = [[1,0,0] for _ in lines]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis_geoms.append((f"bbox_{idx}", line_set))

    # 2. 初始化 GUI
    gui.Application.instance.initialize()

    # 3. 创建 CPU GUI 可视化窗口
    vis = O3DVisualizer("SceneFlow 上一帧预览", 1024, 768)
    vis.show_settings = True

    # 4. 添加所有几何体（注意传入 name, geometry）
    for name, geom in vis_geoms:
        vis.add_geometry(name, geom)

    # 5. 在每个 bbox 顶面中心加文字
    for idx, box in enumerate(filtered_prev_object_boxes_list):
        corners = box.corners().T
        top_center = corners[4:8].mean(axis=0)
        top_center[2] += 0.1
        vis.add_3d_label(top_center, box.name)

    # 6. 把窗口注册到 Application 并运行
    gui.Application.instance.add_window(vis)
    gui.Application.instance.run()


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities

def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)

def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """
    import open3d as o3d

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances



def lidar_to_world_to_lidar(pc,lidar_calibrated_sensor,lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose):

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc


def main(nusc, val_list, indice, args):

    save_path = args.save_path
    data_root = args.dataroot
    # learning_map = nuscenesyaml['learning_map']

    my_scene = nusc.scene[indice]
    sensor = 'LIDAR_TOP'

    if args.split == 'train':
        if my_scene['token'] in val_list:
            return
    elif args.split == 'val':
        if my_scene['token'] not in val_list:
            return
    elif args.split == 'all':
        pass
    else:
        raise NotImplementedError

    # load the first sample to start
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    lidar_data = nusc.get('sample_data', my_sample['data'][sensor])
    lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    # collect LiDAR sequence
    dict_list = []

    while True:
        ############################# get boxes ##########################
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_data['token'])
        boxes_token = [box.token for box in boxes]
        object_tokens = [nusc.get('sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
        object_category = [nusc.get('sample_annotation', box_token)['category_name'] for box_token in boxes_token]

        ############################# get object categories ##########################
        # converted_object_category = []
        # for category in object_category:
        #     for (j, label) in enumerate(nuscenesyaml['labels']):
        #         if category == nuscenesyaml['labels'][label]:
        #             converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        # gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        # gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1 # Slightly expand the bbox to wrap all object points

        ############################# get LiDAR points with semantics ##########################
        pc_file_name = lidar_data['filename'] # load LiDAR names
        pc0 = np.fromfile(os.path.join(data_root, pc_file_name),
                          dtype=np.float32,
                          count=-1).reshape(-1, 5)[..., :4]
        # if lidar_data['is_key_frame']: # only key frame has semantic annotations
        #     lidar_sd_token = lidar_data['token']
        #     lidarseg_labels_filename = os.path.join(nusc.dataroot,
        #                                             nusc.get('lidarseg', lidar_sd_token)['filename'])

        #     points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        #     points_label = np.vectorize(learning_map.__getitem__)(points_label)

        #     pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)

        ############################# cut out movable object points and masks ##########################
        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                                              torch.from_numpy(gt_bbox_3d[np.newaxis, :]))
        object_points_list = []
        j = 0
        while j < points_in_boxes.shape[-1]:
            object_points_mask = points_in_boxes[0][:,j].bool()
            object_points = pc0[object_points_mask]
            object_points_list.append(object_points)
            j = j + 1

        moving_mask = torch.ones_like(points_in_boxes)
        points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
        points_mask = ~(points_in_boxes[0])

        ############################# get point mask of the vehicle itself ##########################
        ego_range = [3.0, 3.0, 3.0] # remove points within 3m of the vehicle itself
        oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > ego_range[0]) |
                                        (np.abs(pc0[:, 1]) > ego_range[1]) |
                                        (np.abs(pc0[:, 2]) > ego_range[2]))

        ############################# get static scene segment ##########################
        points_mask = points_mask & oneself_mask
        pc = pc0[points_mask]

        ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
        lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_pc = lidar_to_world_to_lidar(pc.copy(), lidar_calibrated_sensor.copy(), lidar_ego_pose.copy(),
                                           lidar_calibrated_sensor0,
                                           lidar_ego_pose0)
        ################## record Non-key frame information into a dict  ########################
        dict = {"object_tokens": object_tokens,
                "object_points_list": object_points_list,
                "lidar_pc": lidar_pc.points,
                "lidar_ego_pose": lidar_ego_pose,
                "lidar_calibrated_sensor": lidar_calibrated_sensor,
                "lidar_token": lidar_data['token'],
                "is_key_frame": lidar_data['is_key_frame'],
                "gt_bbox_3d": gt_bbox_3d,
                # "converted_object_category": converted_object_category,
                "pc_file_name": pc_file_name.split('/')[-1],
                "object_category": object_category}

        dict_list.append(dict)
        ################## go to next frame of the sequence  ########################
        next_token = lidar_data['next']
        if next_token != '':
            lidar_data = nusc.get('sample_data', next_token)
        else:
            break

    ################## concatenate all static scene segments (including non-key frames)  ########################
    lidar_pc_list = [dict['lidar_pc'] for dict in dict_list]
    lidar_pc = np.concatenate(lidar_pc_list, axis=1).T

    ################## concatenate all object segments in the scene (including non-key frames)  ########################
    object_token_zoo = []
    # object_semantic = []
    for dict in dict_list:
        for i,object_token in enumerate(dict['object_tokens']):
            if object_token not in object_token_zoo:
                if (dict['object_points_list'][i].shape[0] > 0):
                    object_token_zoo.append(object_token)
                    # object_semantic.append(dict['converted_object_category'][i])
                else:
                    continue

    # convert the absolute coordinates of the object point cloud to the coordinates relative to the bbox
    object_points_dict = {}  
    for query_object_token in object_token_zoo:
        object_points_dict[query_object_token] = []
        for dict in dict_list:
            for i, object_token in enumerate(dict['object_tokens']):
                if query_object_token == object_token:
                    object_points = dict['object_points_list'][i]
                    if object_points.shape[0] > 0:
                        object_points = object_points[:,:3] - dict['gt_bbox_3d'][i][:3]
                        rots = dict['gt_bbox_3d'][i][6]
                        Rot = Rotation.from_euler('z', -rots, degrees=False)
                        rotated_object_points = Rot.apply(object_points)
                        object_points_dict[query_object_token].append(rotated_object_points)
                else:
                    continue
        object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token],
                                                                axis=0)

    object_points_xyz = []
    for key in object_points_dict.keys():
        point_cloud = object_points_dict[key]
        object_points_xyz.append(point_cloud[:,:3])

    for i in trange(1, len(dict_list), desc="Processing frames"):
        prev_dict = dict_list[i-1]
        curr_dict = dict_list[i]

        # 1) 读取当前帧原始点云（含强度），我们只用 xyz
        pc_curr_raw = np.fromfile(
            os.path.join(data_root, curr_dict['pc_file_name']),
            dtype=np.float32, count=-1
        ).reshape(-1, 5)[..., :4]
        pc_curr_xyz = pc_curr_raw[:, :3].copy()

        # 2) 计算当前帧 3D 框（用于给点打“目标掩码”）
        lidar_path, curr_boxes, _ = nusc.get_sample_data(curr_dict['lidar_token'])
        curr_locs = np.array([b.center for b in curr_boxes]).reshape(-1, 3)
        curr_dims = np.array([b.wlh    for b in curr_boxes]).reshape(-1, 3)
        curr_rots = np.array([b.orientation.yaw_pitch_roll[0] for b in curr_boxes]).reshape(-1,1)
        curr_bbox_3d = np.concatenate([curr_locs, curr_dims, curr_rots], axis=1).astype(np.float32)
        curr_bbox_3d[:, 6] += np.pi/2.
        curr_bbox_3d[:, 2] -= curr_dims[:, 2] / 2.

        # 3) 当前帧对每个点：是否落在任一框内（points_in_boxes_cpu 接受 (B,N,3),(B,M,7)）
        pinb = points_in_boxes_cpu(
            torch.from_numpy(pc_curr_xyz[np.newaxis, :, :]),
            torch.from_numpy(curr_bbox_3d[np.newaxis, :])
        )[0]  # (M, N)  bool 但在torch里是0/1
        any_obj_mask = (torch.sum(pinb, dim=0) > 0)  # (N,)

        # 4) 过滤掉车体附近点（和你原逻辑一致）
        ego_range = [3.0, 3.0, 3.0]
        oneself_mask = torch.from_numpy(
            (np.abs(pc_curr_xyz[:, 0]) > ego_range[0]) |
            (np.abs(pc_curr_xyz[:, 1]) > ego_range[1]) |
            (np.abs(pc_curr_xyz[:, 2]) > ego_range[2])
        )

        # 5) 静态点 = 不在任何框 且 不在车体附近
        static_mask = (~any_obj_mask) & oneself_mask
        static_idx = static_mask.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()

        # 6) 构建“可用于配对”的目标 token 交集（上一帧 & 当前帧都存在）
        intersection_tokens = list(set(prev_dict['object_tokens']).intersection(curr_dict['object_tokens']))
        # 为当前帧生成：token -> box 索引
        token2idx_curr = {tok: j for j, tok in enumerate(curr_dict['object_tokens'])}
        token2idx_prev = {tok: j for j, tok in enumerate(prev_dict['object_tokens'])}

        # 7) 为当前帧点逐目标建掩码（列表对齐 token 顺序，保证可重复拼接时的一致性）
        per_obj_indices = []
        for tok in intersection_tokens:
            j = token2idx_curr[tok]
            mask_j = (pinb[j] > 0) & oneself_mask
            inds_j = mask_j.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            if inds_j.size > 0:
                per_obj_indices.append((tok, inds_j))

        # 8) 组装 pc_curr：静态点 + （按 intersection_tokens 顺序的）各目标点
        pc_curr_parts = [pc_curr_xyz[static_idx]]
        for tok, inds in per_obj_indices:
            pc_curr_parts.append(pc_curr_xyz[inds])
        pc_curr = np.concatenate(pc_curr_parts, axis=0)

        # 9) 计算雷达间相对位姿 T_{L_{t-1} <- L_t}，用于静态点回投
        T_w_l_prev = T_world_from_lidar(prev_dict['lidar_calibrated_sensor'], prev_dict['lidar_ego_pose'])
        T_w_l_curr = T_world_from_lidar(curr_dict['lidar_calibrated_sensor'], curr_dict['lidar_ego_pose'])
        T_lprev_lcurr = np.linalg.inv(T_w_l_prev) @ T_w_l_curr  # 4x4

        R_lprev_lcurr = T_lprev_lcurr[:3, :3]
        t_lprev_lcurr = T_lprev_lcurr[:3, 3]

        # 10) 静态点：直接用 T_{L_{t-1} <- L_t} 回投
        pc_prev_static = (pc_curr_xyz[static_idx] @ R_lprev_lcurr.T) + t_lprev_lcurr

        # 11) 动态目标：用“目标局部”桥接
        #     p_curr(在L_t) -> x(在obj局部, 使用当前框) -> p_prev(在L_{t-1}, 使用上一帧该目标的框)
        # 先备好上一帧与当前帧的 box 姿态（均在各自雷达坐标）
        #   yaw 是 z 轴旋转；我们已经做了与原逻辑一致的 +pi/2 & z下移
        def box_pose_from_bbox_row(row):
            loc = row[:3]
            yaw = row[6]
            Rz = Rotation.from_euler('z', yaw, degrees=False).as_matrix()
            t = loc
            return Rz, t

        # 上一帧 box
        lidar_path, prev_boxes, _ = nusc.get_sample_data(prev_dict['lidar_token'])
        prev_locs = np.array([b.center for b in prev_boxes]).reshape(-1, 3)
        prev_dims = np.array([b.wlh    for b in prev_boxes]).reshape(-1, 3)
        prev_rots = np.array([b.orientation.yaw_pitch_roll[0] for b in prev_boxes]).reshape(-1,1)
        prev_bbox_3d = np.concatenate([prev_locs, prev_dims, prev_rots], axis=1).astype(np.float32)
        prev_bbox_3d[:, 6] += np.pi/2.
        prev_bbox_3d[:, 2] -= prev_dims[:, 2] / 2.

        # 为上一帧生成：token -> box 姿态
        prev_box_pose = {}
        for tok in intersection_tokens:
            j_prev = token2idx_prev[tok]
            R_prev, t_prev = box_pose_from_bbox_row(prev_bbox_3d[j_prev])
            prev_box_pose[tok] = (R_prev, t_prev)

        # 为当前帧生成：token -> box 姿态
        curr_box_pose = {}
        for tok in intersection_tokens:
            j_curr = token2idx_curr[tok]
            R_curr, t_curr = box_pose_from_bbox_row(curr_bbox_3d[j_curr])
            curr_box_pose[tok] = (R_curr, t_curr)

        # 12) 逐目标把当前帧点回投到上一帧
        pc_prev_parts = [pc_prev_static]  # 与 pc_curr_parts 对齐：先静态，再各目标
        for (tok, inds) in per_obj_indices:
            pts_curr = pc_curr_xyz[inds]  # (k,3) 在 L_t
            R_c, t_c = curr_box_pose[tok]
            R_p, t_p = prev_box_pose[tok]

            # obj 局部坐标：x = R_c^T (p_curr - t_c)
            x_local = (pts_curr - t_c) @ R_c.T
            # 回到上一帧雷达：p_prev = R_p x + t_p
            pts_prev = (x_local @ R_p.T) + t_p
            pc_prev_parts.append(pts_prev)

        pc_prev = np.concatenate(pc_prev_parts, axis=0)

        # 13) 空间范围裁剪（与原逻辑一致，保证一一对应：对 pc_curr 与 pc_prev 同样索引）
        range_mask_curr = (
            (np.abs(pc_curr[:,0]) < 50) &
            (np.abs(pc_curr[:,1]) < 50.0) &
            (pc_curr[:,2] > -5.0) & (pc_curr[:,2] < 3.0)
        )
        range_mask_prev = (
            (np.abs(pc_prev[:,0]) < 50) &
            (np.abs(pc_prev[:,1]) < 50.0) &
            (pc_prev[:,2] > -5.0) & (pc_prev[:,2] < 3.0)
        )
        keep = range_mask_curr & range_mask_prev
        pc_curr = pc_curr[keep]
        pc_prev = pc_prev[keep]

        # 这里 pc_prev 与 pc_curr 数量相同、顺序一一对应（先静态、再按 token 顺序逐目标点）

        # 14) 保存
        pc_file_name_folder = curr_dict['pc_file_name'].replace('.pcd.bin', '')
        pc_file_name_folder = f"scene_{indice}_{pc_file_name_folder}"
        dirs = os.path.join(save_path, 'scene_flow_single_frames/', pc_file_name_folder)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        np.save(os.path.join(dirs, 'pc_prev.npy'), pc_prev)
        np.save(os.path.join(dirs, 'pc_curr.npy'), pc_curr)


def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:,:3])
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    parse.add_argument('--dataset', type=str, default='nuscenes')
    parse.add_argument('--split', type=str, default='train')
    parse.add_argument('--save_path', type=str, default='./Datasets/nuscenes/0_scene_flow')
    parse.add_argument('--start', type=int, default=0)
    parse.add_argument('--end', type=int, default=2)
    parse.add_argument('--dataroot', type=str, default='./Datasets/nuscenes/')
    parse.add_argument('--nusc_val_list', type=str, default='./tools/generate_scene_flow_nuscenes/nuscenes_val_list.txt')
    args=parse.parse_args()

    if args.dataset=='nuscenes':
        val_list = []
        with open(args.nusc_val_list, 'r') as file:
            for item in file:
                val_list.append(item[:-1])
        file.close()

        nusc = NuScenes(version='v1.0-trainval',
                        dataroot=args.dataroot,
                        verbose=True)
        train_scenes = splits.train
        val_scenes = splits.val
    else:
        raise NotImplementedError

    for i in range(args.start,args.end):
        print('processing sequecne:', i)
        main(nusc, val_list, indice=i, args=args)
