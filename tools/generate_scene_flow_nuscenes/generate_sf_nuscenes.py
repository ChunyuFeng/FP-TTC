import os

import torch
import numpy as np
import open3d as o3d
from tqdm import trange
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import LidarPointCloud
from mmcv.ops.points_in_boxes import points_in_boxes_cpu
from scipy.spatial.transform import Rotation
from collections import defaultdict

def preprocess_for_poisson(points_xyz: np.ndarray,
                           normals_knn: int = 30,
                           voxel_down: float = None,
                           towards_origin: bool = True):
    """
    points_xyz: (N,3) 单帧点云（已在各自 LiDAR 坐标系）
    towards_origin=True 时将法向朝向 (0,0,0)，适合 LiDAR 场景
    """
    pts3 = np.asarray(points_xyz, dtype=np.float32)
    pts3 = np.ascontiguousarray(pts3[:, :3])  # 只取 x,y,z

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3.astype(np.float32))
    if voxel_down is not None and voxel_down > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_down)
    
    # 法向估计
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normals_knn)
    )
    # 对于 per-frame LiDAR，朝传感器原点定向即可；（不要用 consistent_tangent_plane，以免过平滑）
    if towards_origin:
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    return pcd

def poisson_from_points(points_xyz: np.ndarray,
                        depth: int = 9,
                        n_threads: int = 8,
                        min_density_q: float = None):
    """
    返回: mesh( TriangleMesh ), densities(np.ndarray)
    min_density_q: 分位阈值去噪（如 0.02 表示去掉最低 2% 密度的顶点）
    """
    pcd = preprocess_for_poisson(points_xyz, normals_knn=30, voxel_down=None, towards_origin=True)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=n_threads
    )
    # 低密度裁剪（去除边缘飞片）
    if min_density_q is not None and 0.0 < min_density_q < 1.0:
        mask = densities < np.quantile(densities, min_density_q)
        mesh.remove_vertices_by_mask(mask)

    # 基础清理 & 法向
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh, densities

def orient_normals_outward(pcd):
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    c = pts.mean(axis=0, keepdims=True)
    # 令法向尽量指向“远离质心”
    flip = ((nrm * (pts - c)).sum(axis=1) < 0)
    nrm[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    return pcd

def build_psr_samples(points_local,
                      depth=10,
                      min_density_q=0.08,
                      n_samples=20000,
                      voxel_down=None,
                      normals_knn=30):
    p = preprocess_for_poisson(points_local,
                               normals_knn=normals_knn,
                               voxel_down=voxel_down,
                               towards_origin=False)
    print(f"Number of points before Poisson reconstruction: {len(p.points)}")
    p = orient_normals_outward(p)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        p, depth=depth, n_threads=os.cpu_count()
    )
    if min_density_q is not None:
        mask = dens < np.quantile(dens, min_density_q)
        mesh.remove_vertices_by_mask(mask)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices(); mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    samp = mesh.sample_points_poisson_disk(number_of_points=n_samples)
    return np.asarray(samp.points, dtype=np.float32)

def estimate_n_samples_from_wlh(wlh,
                                base_density=800.0,   # 每 m^2 的点数（经验值）
                                min_pts=1500,
                                max_pts=40000):
    """
    根据 bbox 尺寸 w,l,h 估算需要的采样点数。
    采用表面积近似: S ~= w*l + l*h + w*h (与建议一致)
    """
    w, l, h = map(float, wlh)
    S = max(1e-6, w*l + l*h + w*h)        # m^2
    N = int(np.clip(base_density * S, min_pts, max_pts))
    return N

def estimate_psr_depth_from_wlh(wlh):
    """
    依据尺度粗调 PSR 的八叉树深度（越大越细，耗时越高）。
    可按需再调阈值。
    """
    max_dim = float(np.max(wlh))
    if max_dim < 1.0:   # traffic cone / 行人等小目标
        return 9
    elif max_dim < 2.5: # 轿车/摩托/自行车等
        return 10
    elif max_dim < 5.0: # 卡车/巴士等较大目标
        return 11
    else:
        return 12

def pick_voxel_from_wlh(wlh, frac=0.02, vmin=0.005, vmax=0.05):
    min_edge = float(np.min(wlh))
    return float(np.clip(frac * max(min_edge, 1e-6), vmin, vmax)) 
   
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

def lidar_to_world_to_lidar(pc,
                            lidar_calibrated_sensor, lidar_ego_pose,
                            cam_calibrated_sensor, cam_ego_pose):

    """
    将点从 'source LiDAR' 坐标系 -> 世界坐标 -> 'target LiDAR' 坐标系。
    输入 pc 可以是 (N,3)/(N,4)/(3,N)/(4,N)，本函数会归一成 4xN。
    第四维（反射强度）若缺失则补 0。
    """
    pts = np.asarray(pc)
    if pts.size == 0:
        # 返回一个空的点云对象，避免后续流程炸掉
        return LidarPointCloud(np.zeros((4, 0), dtype=np.float32))

    # 统一 dtype
    dtype = np.float32
    pts = pts.astype(dtype, copy=False)

    # 归一到 4 x N
    if pts.ndim != 2:
        raise ValueError(f"pc must be 2D, got shape {pts.shape}")

    if pts.shape[0] in (3, 4):        # 已经是 (3/4, N)
        pc4xN = pts
        if pc4xN.shape[0] == 3:
            pc4xN = np.vstack([pc4xN, np.zeros((1, pc4xN.shape[1]), dtype=dtype)])
    elif pts.shape[1] in (3, 4):       # 是 (N, 3/4)
        if pts.shape[1] == 3:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=dtype)])
        pc4xN = pts.T
    else:
        raise ValueError(f"Unrecognized pc shape {pts.shape}, expect (N,3/4) or (3/4,N)")

    # 用 nuScenes 的 LidarPointCloud 做位姿变换
    pc = LidarPointCloud(pc4xN)

    # source LiDAR -> ego -> world
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    # world -> target ego -> target LiDAR
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc

def main(nusc, val_list, indice, args):

    save_path = args.save_path
    data_root = args.dataroot
  
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
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] + 0.2  # Move the bbox slightly up in the z direction to avoid ground points
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.2 # Slightly expand the bbox to wrap all object points

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
        frame_rec = {"object_tokens": object_tokens,
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
        ################## record semantic information into the dict if it's a key frame  ########################
        # if lidar_data['is_key_frame']:
        #     pc_with_semantic = pc_with_semantic[points_mask]
        #     lidar_pc_with_semantic = lidar_to_world_to_lidar(pc_with_semantic.copy(),
        #                                                      lidar_calibrated_sensor.copy(),
        #                                                      lidar_ego_pose.copy(),
        #                                                      lidar_calibrated_sensor0,
        #                                                      lidar_ego_pose0)
        #     dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points

        dict_list.append(frame_rec)
        ################## go to next frame of the sequence  ########################
        next_token = lidar_data['next']
        if next_token != '':
            lidar_data = nusc.get('sample_data', next_token)
        else:
            break
    
    # === 为每个实例统计一个稳健的尺寸（w,l,h）的中位数 ===
    
    token2dims_list = defaultdict(list)

    for fr in dict_list:
        for tok, bbox in zip(fr['object_tokens'], fr['gt_bbox_3d']):
            token2dims_list[tok].append(bbox[3:6])  # gt_bbox_3d: [cx,cy,cz, w,l,h, yaw]

    token2dims = {}
    for tok, arrs in token2dims_list.items():
        arrs = np.asarray(arrs, dtype=np.float32)
        token2dims[tok] = np.median(arrs, axis=0)   # (3,)

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
    object_token2samples_local = {}
    for token, pts_local in object_points_dict.items():
        if pts_local.shape[0] < 200:
            continue

        wlh = token2dims.get(token, np.array([2.0, 4.0, 1.6], dtype=np.float32))
        n_samp  = estimate_n_samples_from_wlh(wlh, base_density=800.0, min_pts=1500, max_pts=40000)
        d_level = estimate_psr_depth_from_wlh(wlh)

        # 自适应体素：最小边的 2%，并限制在 [5mm, 5cm]
        voxel_down = pick_voxel_from_wlh(wlh, frac=0.02, vmin=0.005, vmax=0.05)

        object_token2samples_local[token] = build_psr_samples(
            pts_local,
            depth=d_level,
            min_density_q=0.08,
            n_samples=n_samp,
            voxel_down=voxel_down,
            normals_knn=30
        )


    # 2) 为 static 背景生成规范点集（在第一帧LiDAR坐标聚合后的 lidar_pc 上做）
    static_samples_local = build_psr_samples(lidar_pc, 
                                             depth=12, 
                                             min_density_q=0.10, 
                                             n_samples=500000)

    for i in trange(1, len(dict_list), desc="Processing frames"):

        prev_dict = dict_list[i-1]
        curr_dict = dict_list[i]

        ################## convert the static scene to previous coordinate system ##############
        lidar_calibrated_sensor = prev_dict['lidar_calibrated_sensor']
        lidar_ego_pose = prev_dict['lidar_ego_pose']
        prev_lidar_pc = lidar_to_world_to_lidar(static_samples_local.copy(),
                                                lidar_calibrated_sensor0.copy(),
                                                lidar_ego_pose0.copy(),
                                                lidar_calibrated_sensor,
                                                lidar_ego_pose)

        prev_point_cloud = prev_lidar_pc.points.T[:, :3]

        ################## convert the static scene to current coordinate system ##############
        lidar_calibrated_sensor = curr_dict['lidar_calibrated_sensor']
        lidar_ego_pose = curr_dict['lidar_ego_pose']
        curr_lidar_pc = lidar_to_world_to_lidar(static_samples_local.copy(),
                                                lidar_calibrated_sensor0.copy(),
                                                lidar_ego_pose0.copy(),
                                                lidar_calibrated_sensor,
                                                lidar_ego_pose)

        curr_point_cloud = curr_lidar_pc.points.T[:, :3]

        ################## load bboxes of previous frame ##############
        lidar_path, prev_boxes, _ = nusc.get_sample_data(prev_dict['lidar_token'])
        # 记录 prev 的几何与姿态（注意与上文一致的 yaw/z 修正）
        prev_locs = np.array([b.center for b in prev_boxes], dtype=np.float32)
        prev_dims = np.array([b.wlh for b in prev_boxes], dtype=np.float32)
        prev_rots = np.array([b.orientation.yaw_pitch_roll[0] for b in prev_boxes], dtype=np.float32)
        prev_rots = prev_rots + np.pi / 2.0   
        prev_locs[:, 2] = prev_locs[:, 2] - prev_dims[:, 2] / 2.0

        ################## load bboxes of current frame ##############
        lidar_path, curr_boxes, _ = nusc.get_sample_data(curr_dict['lidar_token'])
        curr_locs = np.array([b.center for b in curr_boxes], dtype=np.float32)
        curr_dims = np.array([b.wlh for b in curr_boxes], dtype=np.float32)
        curr_rots = np.array([b.orientation.yaw_pitch_roll[0] for b in curr_boxes], dtype=np.float32)
        curr_rots = curr_rots + np.pi / 2.0
        curr_locs[:, 2] = curr_locs[:, 2] - curr_dims[:, 2] / 2.0

        # === 建立 "instance_token -> 索引/位姿/box" 的映射 ===
        # prev/curr 的 instance token 顺序分别与 prev_dict['object_tokens']/curr_dict['object_tokens'] 对齐
        prev_inst_tokens = prev_dict['object_tokens']  # 已在采集阶段按 boxes 顺序生成
        curr_inst_tokens = curr_dict['object_tokens']

        prev_tok2idx = {tok: idx for idx, tok in enumerate(prev_inst_tokens)}
        curr_tok2idx = {tok: idx for idx, tok in enumerate(curr_inst_tokens)}

        # === 求交集，并以 prev 的顺序确定稳定顺序，同时要求该实例在 PSR 采样字典里 ===
        common_tokens = [
            tok for tok in prev_inst_tokens
            if (tok in curr_tok2idx) and (tok in object_token2samples_local) and (tok in prev_tok2idx)
        ]

        # === 按相同顺序构造 prev/curr 的实例点与 box（保证一一对应、长度一致）===
        prev_object_points_list, prev_boxes_sel = [], []
        curr_object_points_list, curr_boxes_sel = [], []

        for tok in common_tokens:
            j_prev = prev_tok2idx[tok]
            j_curr = curr_tok2idx[tok]
            pts_local = object_token2samples_local[tok]          # 规范坐标的一次性采样

            # prev 帧：Rz(rot) * pts + loc
            Rp = Rotation.from_euler('z', float(prev_rots[j_prev]), degrees=False)
            prev_object_points_list.append(Rp.apply(pts_local) + prev_locs[j_prev])
            prev_boxes_sel.append(prev_boxes[j_prev])

            # curr 帧
            Rc = Rotation.from_euler('z', float(curr_rots[j_curr]), degrees=False)
            curr_object_points_list.append(Rc.apply(pts_local) + curr_locs[j_curr])
            curr_boxes_sel.append(curr_boxes[j_curr])

        ################## concatenate static scene segments and object points  ########################

        if len(prev_object_points_list) > 0:
            prev_scene_points = np.concatenate([prev_point_cloud, np.concatenate(prev_object_points_list, axis=0)], axis=0)
        else:
            prev_scene_points = prev_point_cloud

        if len(curr_object_points_list) > 0:
            curr_scene_points = np.concatenate([curr_point_cloud, np.concatenate(curr_object_points_list, axis=0)], axis=0)
        else:
            curr_scene_points = curr_point_cloud

        ################## remain points with a spatial range ##############
        range_mask = (
            (np.abs(prev_scene_points[:, 0]) < 50.0) &
            (np.abs(prev_scene_points[:, 1]) < 50.0) &
            (prev_scene_points[:, 2] > -5.0) &
            (prev_scene_points[:, 2] <  3.0)
        )
        prev_scene_points = prev_scene_points[range_mask]
        curr_scene_points = curr_scene_points[range_mask]

        # ################## visualize the current scene points  ########################
        # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curr_scene_points))],
        #                                   window_name="Current Scene Points Visualization")


        ################## save the scene points and object points  ########################
        pc_file_name_folder = curr_dict['pc_file_name'].replace('.pcd.bin', '')
        pc_file_name_folder = f"scene_{indice}_{pc_file_name_folder}"
        dirs = os.path.join(save_path, 'scene_flow_all_frames_poisson/', pc_file_name_folder)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        np.save(os.path.join(dirs, 'pc_prev.npy'), prev_scene_points)
        np.save(os.path.join(dirs, 'pc_curr.npy'), curr_scene_points)

        i = i + 1
        continue


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
