import os
import sys
import pdb
import time
import yaml
import torch
import mmcv
import numpy as np
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

# import open3d.visualization.gui as gui
# from open3d.visualization import O3DVisualizer

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



# def visualize_prev_frame(filtered_prev_object_points_list,
#                          filtered_prev_object_boxes_list):
#     # 1. 准备几何体列表
#     vis_geoms = []
#     for idx, (pts, box) in enumerate(zip(filtered_prev_object_points_list,
#                                          filtered_prev_object_boxes_list)):
#         # 1.1 点云
#         pcd_obj = o3d.geometry.PointCloud()
#         pcd_obj.points = o3d.utility.Vector3dVector(pts[:, :3])
#         pcd_obj.paint_uniform_color(np.random.uniform(0, 1, size=3))
#         vis_geoms.append((f"pcd_{idx}", pcd_obj))

#         # 1.2 线框
#         corners = box.corners().T
#         lines = [
#             [0,1],[1,2],[2,3],[3,0],
#             [4,5],[5,6],[6,7],[7,4],
#             [0,4],[1,5],[2,6],[3,7]
#         ]
#         colors = [[1,0,0] for _ in lines]
#         line_set = o3d.geometry.LineSet(
#             points=o3d.utility.Vector3dVector(corners),
#             lines=o3d.utility.Vector2iVector(lines)
#         )
#         line_set.colors = o3d.utility.Vector3dVector(colors)
#         vis_geoms.append((f"bbox_{idx}", line_set))

#     # 2. 初始化 GUI
#     gui.Application.instance.initialize()

#     # 3. 创建 CPU GUI 可视化窗口
#     vis = O3DVisualizer("SceneFlow 上一帧预览", 1024, 768)
#     vis.show_settings = True

#     # 4. 添加所有几何体（注意传入 name, geometry）
#     for name, geom in vis_geoms:
#         vis.add_geometry(name, geom)

#     # 5. 在每个 bbox 顶面中心加文字
#     for idx, box in enumerate(filtered_prev_object_boxes_list):
#         corners = box.corners().T
#         top_center = corners[4:8].mean(axis=0)
#         top_center[2] += 0.1
#         vis.add_3d_label(top_center, box.name)

#     # 6. 把窗口注册到 Application 并运行
#     gui.Application.instance.add_window(vis)
#     gui.Application.instance.run()


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


def dedup_static_ground_layers(points_xyz,
                               grid_size,
                               ground_z_max,
                               min_points_per_cell,
                               cell_z_span_max):
    keep_mask = np.ones(points_xyz.shape[0], dtype=bool)

    if points_xyz.shape[0] == 0:
        return keep_mask, {
            'ground_candidate_count': 0,
            'collapsed_cell_count': 0,
            'removed_point_count': 0,
        }

    ground_candidate_indices = np.flatnonzero(points_xyz[:, 2] < ground_z_max)
    if ground_candidate_indices.size == 0:
        return keep_mask, {
            'ground_candidate_count': 0,
            'collapsed_cell_count': 0,
            'removed_point_count': 0,
        }

    ground_points = points_xyz[ground_candidate_indices]
    grid_x = np.floor(ground_points[:, 0] / grid_size).astype(np.int32)
    grid_y = np.floor(ground_points[:, 1] / grid_size).astype(np.int32)
    grid_coords = np.stack([grid_x, grid_y], axis=1)

    _, inverse = np.unique(grid_coords, axis=0, return_inverse=True)
    sort_order = np.argsort(inverse, kind='mergesort')
    sorted_group_ids = inverse[sort_order]
    sorted_ground_indices = ground_candidate_indices[sort_order]
    sorted_ground_z = ground_points[sort_order, 2]

    group_starts = np.flatnonzero(np.r_[True, sorted_group_ids[1:] != sorted_group_ids[:-1]])
    group_counts = np.diff(np.r_[group_starts, sorted_group_ids.size])
    group_mins = np.minimum.reduceat(sorted_ground_z, group_starts)
    group_maxs = np.maximum.reduceat(sorted_ground_z, group_starts)
    group_spans = group_maxs - group_mins

    collapsed_cell_count = 0
    removed_point_count = 0

    for start, count, span in zip(group_starts, group_counts, group_spans):
        if count < min_points_per_cell or span > cell_z_span_max:
            continue

        group_slice = slice(start, start + count)
        group_indices = sorted_ground_indices[group_slice]
        group_z = sorted_ground_z[group_slice]
        median_z = np.median(group_z)
        representative_offset = int(np.argmin(np.abs(group_z - median_z)))
        representative_index = group_indices[representative_offset]

        keep_mask[group_indices] = False
        keep_mask[representative_index] = True

        collapsed_cell_count += 1
        removed_point_count += count - 1

    return keep_mask, {
        'ground_candidate_count': int(ground_candidate_indices.size),
        'collapsed_cell_count': int(collapsed_cell_count),
        'removed_point_count': int(removed_point_count),
    }


def get_bbox_family(category_name):
    if category_name.startswith('vehicle.') and category_name not in {'vehicle.bicycle', 'vehicle.motorcycle'}:
        return 'vehicle'
    if category_name in {'vehicle.bicycle', 'vehicle.motorcycle'}:
        return 'two_wheeler'
    if category_name.startswith('human.pedestrian.'):
        return 'pedestrian'
    if category_name.startswith('movable_object.'):
        return 'small_object'
    return 'fallback'


def build_separation_boxes(base_gt_bbox_3d,
                           object_categories,
                           bbox_adjustment_mode,
                           global_bbox_dilation,
                           global_bbox_shift):
    separation_gt_bbox_3d = base_gt_bbox_3d.copy()

    if bbox_adjustment_mode == 'global':
        separation_gt_bbox_3d[:, :3] += global_bbox_shift
        separation_gt_bbox_3d[:, 3:6] += global_bbox_dilation
        return separation_gt_bbox_3d

    if bbox_adjustment_mode != 'class_aware':
        raise ValueError(f"Unsupported bbox_adjustment_mode={bbox_adjustment_mode}")

    family_ratio = {
        'vehicle': np.asarray([0.25, 0.15, 0.20], dtype=np.float32),
        'two_wheeler': np.asarray([0.15, 0.12, 0.15], dtype=np.float32),
        'pedestrian': np.asarray([0.08, 0.08, 0.10], dtype=np.float32),
        'small_object': np.asarray([0.10, 0.10, 0.10], dtype=np.float32),
        'fallback': np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }
    family_cap = {
        'vehicle': np.asarray([0.50, 0.50, 0.30], dtype=np.float32),
        'two_wheeler': np.asarray([0.12, 0.12, 0.12], dtype=np.float32),
        'pedestrian': np.asarray([0.05, 0.05, 0.10], dtype=np.float32),
        'small_object': np.asarray([0.05, 0.05, 0.08], dtype=np.float32),
        'fallback': np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }

    dims = separation_gt_bbox_3d[:, 3:6]
    effective_dilation = np.zeros_like(dims, dtype=np.float32)
    effective_shift = np.zeros((separation_gt_bbox_3d.shape[0], 3), dtype=np.float32)

    for i, category_name in enumerate(object_categories):
        family = get_bbox_family(category_name)
        effective_dilation[i] = np.minimum(family_cap[family], family_ratio[family] * dims[i])

        height = float(dims[i, 2])
        if family == 'vehicle':
            effective_shift[i, 2] = min(0.15, 0.10 * height)
        elif family == 'two_wheeler':
            effective_shift[i, 2] = min(0.05, 0.05 * height)

    separation_gt_bbox_3d[:, :3] += effective_shift
    separation_gt_bbox_3d[:, 3:6] += effective_dilation
    return separation_gt_bbox_3d


def main(nusc, val_list, indice, args):

    save_path = args.save_path
    data_root = args.dataroot
    bbox_dilation = np.asarray(args.bbox_dilation, dtype=np.float32)
    bbox_shift = np.asarray(args.bbox_shift, dtype=np.float32)
    # learning_map = nuscenesyaml['learning_map']

    my_scene = nusc.scene[indice]
    sensor = 'LIDAR_TOP'

    if not args.scene_list_file:
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
        separation_gt_bbox_3d = build_separation_boxes(
            gt_bbox_3d,
            object_category,
            args.bbox_adjustment_mode,
            bbox_dilation,
            bbox_shift,
        )

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
                                              torch.from_numpy(separation_gt_bbox_3d[np.newaxis, :]))
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
        self_range = [3.0, 3.0, 3.0] # remove points within 3m of the vehicle itself
        oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > self_range[0]) |
                                        (np.abs(pc0[:, 1]) > self_range[1]) |
                                        (np.abs(pc0[:, 2]) > self_range[2]))

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
        ################## record semantic information into the dict if it's a key frame  ########################
        # if lidar_data['is_key_frame']:
        #     pc_with_semantic = pc_with_semantic[points_mask]
        #     lidar_pc_with_semantic = lidar_to_world_to_lidar(pc_with_semantic.copy(),
        #                                                      lidar_calibrated_sensor.copy(),
        #                                                      lidar_ego_pose.copy(),
        #                                                      lidar_calibrated_sensor0,
        #                                                      lidar_ego_pose0)
        #     dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points

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
    # ################################### load 3d model ######################################
    # car_points_norm = load_pcd_from_file("Datasets/nuscenes/0_scene_flow/object_3d_model/car.pcd")
    # bus_points_norm = load_pcd_from_file("Datasets/nuscenes/0_scene_flow/object_3d_model/bus.pcd")
    # human_points_norm = load_pcd_from_file("Datasets/nuscenes/0_scene_flow/object_3d_model/human.pcd")
    # trafficcone_points_norm = load_pcd_from_file("Datasets/nuscenes/0_scene_flow/object_3d_model/trafficcone.pcd")
    # truck_points_norm = load_pcd_from_file("Datasets/nuscenes/0_scene_flow/object_3d_model/truck.pcd")

    # token2category = {}
    # token2bbox = {}
    # for frame in dict_list:
    #     for token, category, gt_bbox_3d in zip(frame['object_tokens'], frame['object_category'], frame['gt_bbox_3d']):
    #         if token not in token2category:
    #             token2category[token] = category
    #             token2bbox[token] = gt_bbox_3d
    # ############################################################################################

    # convert the absolute coordinates of the object point cloud to the coordinates relative to the bbox
    object_points_dict = {}  
    for query_object_token in object_token_zoo:

        # ########################### using 3D model points for some categories ##########################
        # # replace the car points with the 3D model car points
        # if token2category.get(query_object_token) == 'vehicle.car':
        #     # load the w-l-h of the bounding box
        #     wlh = token2bbox[query_object_token][3:6]
        #     object_points_dict[query_object_token] = car_points_norm.copy()
        #     # scale the car points to the bounding box size
        #     object_points_dict[query_object_token] *= wlh / np.array([1.0, 1.0, 1.0])
        #     continue
        # # replace the truck points with the 3D model truck points
        # if token2category.get(query_object_token) == 'vehicle.truck':
        #     # load the w-l-h of the bounding box
        #     wlh = token2bbox[query_object_token][3:6]
        #     object_points_dict[query_object_token] = truck_points_norm.copy()
        #     # scale the truck points to the bounding box size
        #     object_points_dict[query_object_token] *= wlh / np.array([1.0, 1.0, 1.0])
        #     continue
        # # replace the human points with the 3D model human points
        # if token2category.get(query_object_token) == 'human.pedestrian.construction_worker':
        #     # load the w-l-h of the bounding box
        #     wlh = token2bbox[query_object_token][3:6]
        #     object_points_dict[query_object_token] = human_points_norm.copy()
        #     # scale the human points to the bounding box size
        #     object_points_dict[query_object_token] *= wlh / np.array([1.0, 1.0, 1.0])
        #     continue
        # if token2category.get(query_object_token) == 'vehicle.bus.rigid':
        #     # load the w-l-h of the bounding box
        #     wlh = token2bbox[query_object_token][3:6]
        #     object_points_dict[query_object_token] = bus_points_norm.copy()
        #     # scale the bus points to the bounding box size
        #     object_points_dict[query_object_token] *= wlh / np.array([1.0, 1.0, 1.0])
        #     continue
        # if token2category.get(query_object_token) == 'movable_object.trafficcone':
        #     # load the w-l-h of the bounding box
        #     wlh = token2bbox[query_object_token][3:6]
        #     object_points_dict[query_object_token] = trafficcone_points_norm.copy()
        #     # scale the traffic cone points to the bounding box size
        #     object_points_dict[query_object_token] *= wlh / np.array([1.0, 1.0, 1.0])
        #     continue
        # ############################################################################################
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

    object_points_xyz_by_token = {}
    for object_token, point_cloud in object_points_dict.items():
        object_points_xyz_by_token[object_token] = point_cloud[:, :3]

    if args.keyframe_only:
        keyframe_indices = [i for i, frame_dict in enumerate(dict_list) if frame_dict['is_key_frame']]
        frame_pairs = [(keyframe_indices[i - 1], keyframe_indices[i]) for i in range(1, len(keyframe_indices))]
        progress_desc = "Processing key frames"
        scene_flow_subdir = 'scene_flow_key_frames'
    else:
        frame_pairs = [(i - 1, i) for i in range(1, len(dict_list))]
        progress_desc = "Processing frames"
        scene_flow_subdir = 'scene_flow_all_frames'

    for prev_idx, curr_idx in tqdm(frame_pairs, desc=progress_desc):

        prev_dict = dict_list[prev_idx]
        curr_dict = dict_list[curr_idx]

        ################## concatenate static point cloud ########################
        if args.static_context_mode == 'scene':
            static_dict_list = dict_list
        elif args.static_context_mode == 'local':
            window_start = max(0, min(prev_idx, curr_idx) - args.static_context_radius)
            window_end = min(len(dict_list), max(prev_idx, curr_idx) + args.static_context_radius + 1)
            static_dict_list = dict_list[window_start:window_end]
        else:
            raise ValueError(f"Unsupported static_context_mode={args.static_context_mode}")

        lidar_pc_slice_list = [frame_dict['lidar_pc'] for frame_dict in static_dict_list]
        lidar_pc_slice = np.concatenate(lidar_pc_slice_list, axis=1).T

        if args.static_ground_dedup == 'bev':
            static_keep_mask, dedup_stats = dedup_static_ground_layers(
                lidar_pc_slice[:, :3],
                grid_size=args.static_ground_grid_size,
                ground_z_max=args.static_ground_z_max,
                min_points_per_cell=args.static_ground_min_points_per_cell,
                cell_z_span_max=args.static_ground_cell_z_span_max,
            )
            static_point_count_before = lidar_pc_slice.shape[0]
            lidar_pc_slice = lidar_pc_slice[static_keep_mask]
            static_point_count_after = lidar_pc_slice.shape[0]
            print(
                "Static ground dedup "
                f"scene={indice} prev_idx={prev_idx} curr_idx={curr_idx} "
                f"static_before={static_point_count_before} static_after={static_point_count_after} "
                f"ground_candidates={dedup_stats['ground_candidate_count']} "
                f"collapsed_cells={dedup_stats['collapsed_cell_count']} "
                f"removed_duplicates={dedup_stats['removed_point_count']}"
            )
        elif args.static_ground_dedup != 'off':
            raise ValueError(f"Unsupported static_ground_dedup={args.static_ground_dedup}")

        # ################## concatenate object points ########################
        # obj_pc_slice_list = [dict['object_points_list'] for dict in dict_list[i-N:i+N+1]]
        # obj_token_list = [dict['object_tokens'] for dict in dict_list[i-N:i+N+1]]
        # obj_gt_bbox_3d = [dict['gt_bbox_3d'] for dict in dict_list[i-N:i+N+1]]

        ################## convert the static scene to previous coordinate system ##############
        lidar_calibrated_sensor = prev_dict['lidar_calibrated_sensor']
        lidar_ego_pose = prev_dict['lidar_ego_pose']
        prev_lidar_pc = lidar_to_world_to_lidar(lidar_pc_slice.copy(),
                                             lidar_calibrated_sensor0.copy(),
                                             lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)

        prev_point_cloud = prev_lidar_pc.points.T[:, :3]

        ################## convert the static scene to current coordinate system ##############
        lidar_calibrated_sensor = curr_dict['lidar_calibrated_sensor']
        lidar_ego_pose = curr_dict['lidar_ego_pose']
        curr_lidar_pc = lidar_to_world_to_lidar(lidar_pc_slice.copy(),
                                             lidar_calibrated_sensor0.copy(),
                                             lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)

        curr_point_cloud = curr_lidar_pc.points.T[:, :3]

        ################## load bboxes of previous frame ##############
        lidar_path, prev_boxes, _ = nusc.get_sample_data(prev_dict['lidar_token'])
        locs = np.array([b.center for b in prev_boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in prev_boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in prev_boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2]
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6]
        prev_rots = gt_bbox_3d[:, 6:7]
        prev_locs = gt_bbox_3d[:, 0:3]

        ################## load bboxes of current frame ##############
        lidar_path, curr_boxes, _ = nusc.get_sample_data(curr_dict['lidar_token'])
        locs = np.array([b.center for b in curr_boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in curr_boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in curr_boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2]
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6]
        curr_rots = gt_bbox_3d[:, 6:7]
        curr_locs = gt_bbox_3d[:, 0:3]

        ################## place aligned object points into corresponding bboxes ##############
        prev_box_idx_by_token = {}
        for j, object_token in enumerate(prev_dict['object_tokens']):
            if object_token not in prev_box_idx_by_token:
                prev_box_idx_by_token[object_token] = j

        curr_box_idx_by_token = {}
        for j, object_token in enumerate(curr_dict['object_tokens']):
            if object_token not in curr_box_idx_by_token:
                curr_box_idx_by_token[object_token] = j

        common_object_tokens = []
        seen_common_tokens = set()
        for object_token in prev_dict['object_tokens']:
            if object_token in seen_common_tokens:
                continue
            if object_token not in curr_box_idx_by_token:
                continue
            if object_token not in object_points_xyz_by_token:
                continue
            common_object_tokens.append(object_token)
            seen_common_tokens.add(object_token)

        filtered_prev_object_points_list = []
        filtered_prev_object_boxes_list = []
        filtered_curr_object_points_list = []
        filtered_curr_object_boxes_list = []
        for object_token in common_object_tokens:
            prev_box_idx = prev_box_idx_by_token[object_token]
            curr_box_idx = curr_box_idx_by_token[object_token]
            canonical_object_points = object_points_xyz_by_token[object_token]

            prev_rot = Rotation.from_euler('z', prev_rots[prev_box_idx], degrees=False)
            prev_object_points = prev_rot.apply(canonical_object_points) + prev_locs[prev_box_idx]
            filtered_prev_object_points_list.append(prev_object_points)
            filtered_prev_object_boxes_list.append(prev_boxes[prev_box_idx])

            curr_rot = Rotation.from_euler('z', curr_rots[curr_box_idx], degrees=False)
            curr_object_points = curr_rot.apply(canonical_object_points) + curr_locs[curr_box_idx]
            filtered_curr_object_points_list.append(curr_object_points)
            filtered_curr_object_boxes_list.append(curr_boxes[curr_box_idx])
        
        # visualize_prev_frame(filtered_prev_object_points_list,
        #                             filtered_prev_object_boxes_list)
        

        ################## concatenate static scene segments and object points  ########################
        try:
            prev_temp = np.concatenate(filtered_prev_object_points_list)
            prev_scene_points = np.concatenate([prev_point_cloud, prev_temp])
        except:
            prev_scene_points = prev_point_cloud

        try:
            curr_temp = np.concatenate(filtered_curr_object_points_list)
            curr_scene_points = np.concatenate([curr_point_cloud, curr_temp])
        except:
            curr_scene_points = curr_point_cloud

        if prev_scene_points.shape[0] != curr_scene_points.shape[0]:
            raise RuntimeError(
                "Aligned scene point count mismatch before range cropping: "
                f"scene_indice={indice}, prev_idx={prev_idx}, curr_idx={curr_idx}, "
                f"prev_static={len(prev_point_cloud)}, curr_static={len(curr_point_cloud)}, "
                f"prev_objects={len(filtered_prev_object_points_list)}, "
                f"curr_objects={len(filtered_curr_object_points_list)}, "
                f"common_tokens={len(common_object_tokens)}, "
                f"prev_points={prev_scene_points.shape[0]}, curr_points={curr_scene_points.shape[0]}"
            )

        ################## remain points with a spatial range ##############
        prev_range_mask = (np.abs(prev_scene_points[:, 0]) < 50) & (np.abs(prev_scene_points[:, 1]) < 50.0) \
               & (prev_scene_points[:, 2] > -5.0) & (prev_scene_points[:, 2] < 3.0)

        curr_range_mask = (np.abs(curr_scene_points[:, 0]) < 50) & (np.abs(curr_scene_points[:, 1]) < 50.0) \
                & (curr_scene_points[:, 2] > -5.0) & (curr_scene_points[:, 2] < 3.0)

        intersection_points_mask = prev_range_mask & curr_range_mask

        prev_scene_points = prev_scene_points[intersection_points_mask]
        curr_scene_points = curr_scene_points[intersection_points_mask]

        ################## visualization ##################
        # point_cloud_static_vis = o3d.geometry.PointCloud()
        # point_cloud_static_vis.points = o3d.utility.Vector3dVector(prev_scene_points)
        # o3d.visualization.draw_geometries([point_cloud_static_vis])

        # ################## voxel downsampling ##################
        # voxel_size = 0.01  # Set the voxel size for downsampling
        # point_cloud_static_vis = o3d.geometry.PointCloud()
        # point_cloud_static_vis.points = o3d.utility.Vector3dVector(prev_scene_points)
        # point_cloud_static_vis = point_cloud_static_vis.voxel_down_sample(voxel_size=voxel_size)

        # ################## visualization ##################
        # o3d.visualization.draw_geometries([point_cloud_static_vis])


        ################## save the scene points and object points  ########################
        prev_scene_points = prev_scene_points.astype(np.float32, copy=False)
        curr_scene_points = curr_scene_points.astype(np.float32, copy=False)

        pc_file_name_folder = curr_dict['pc_file_name'].replace('.pcd.bin', '')
        pc_file_name_folder = f"scene_{indice}_{pc_file_name_folder}"
        dirs = os.path.join(save_path, scene_flow_subdir, pc_file_name_folder)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        np.save(os.path.join(dirs, 'pc_prev.npy'), prev_scene_points)
        np.save(os.path.join(dirs, 'pc_curr.npy'), curr_scene_points)


def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:,:3])
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original)


def load_scene_names(scene_list_file):
    scene_names = []
    with open(scene_list_file, 'r') as file:
        for line in file:
            scene_name = line.strip()
            if not scene_name or scene_name.startswith('#'):
                continue
            scene_names.append(scene_name)
    return scene_names


def resolve_scene_indices(nusc, scene_names):
    scene_index_by_name = {scene['name']: idx for idx, scene in enumerate(nusc.scene)}

    missing_scene_names = [scene_name for scene_name in scene_names if scene_name not in scene_index_by_name]
    if missing_scene_names:
        missing_str = ', '.join(missing_scene_names[:10])
        if len(missing_scene_names) > 10:
            missing_str += ', ...'
        raise ValueError(f"Unknown scene names in scene list: {missing_str}")

    selected_scene_indices = []
    seen_scene_names = set()
    duplicate_scene_names = []
    for scene_name in scene_names:
        if scene_name in seen_scene_names:
            duplicate_scene_names.append(scene_name)
            continue
        seen_scene_names.add(scene_name)
        selected_scene_indices.append(scene_index_by_name[scene_name])

    if duplicate_scene_names:
        duplicate_scene_names = sorted(set(duplicate_scene_names))
        duplicate_str = ', '.join(duplicate_scene_names[:10])
        if len(duplicate_scene_names) > 10:
            duplicate_str += ', ...'
        print(f"Ignoring duplicate scene names from scene list: {duplicate_str}")

    return selected_scene_indices


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
    parse.add_argument('--scene_list_file', type=str, default=None,
                       help='Optional text file containing one scene name per line, e.g. scene-0001.')
    parse.add_argument('--keyframe_only', action='store_true',
                       help='Only export scene flow pairs between consecutive annotated LiDAR key frames.')
    parse.add_argument('--bbox_adjustment_mode', type=str, default='global', choices=['global', 'class_aware'],
                       help='How to enlarge boxes for static/dynamic separation. class_aware ignores --bbox_dilation and --bbox_shift.')
    parse.add_argument('--bbox_dilation', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       help='Optional bbox dilation added to box w/l/h before separating dynamic and static points.')
    parse.add_argument('--bbox_shift', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       help='Optional xyz shift applied to the separation-stage boxes after converting to box format.')
    parse.add_argument('--static_context_mode', type=str, default='scene', choices=['scene', 'local'],
                       help='Static background aggregation mode: whole scene or local window around the current pair.')
    parse.add_argument('--static_context_radius', type=int, default=10,
                       help='Number of sweeps on each side when static_context_mode=local.')
    parse.add_argument('--static_ground_dedup', type=str, default='off', choices=['off', 'bev'],
                       help='Optional de-layering for near-ground static background points before prev/curr transforms.')
    parse.add_argument('--static_ground_grid_size', type=float, default=0.15,
                       help='XY grid size in meters for static ground de-layering.')
    parse.add_argument('--static_ground_z_max', type=float, default=-1.0,
                       help='Only static points below this z in the first-frame lidar coordinate are considered ground.')
    parse.add_argument('--static_ground_min_points_per_cell', type=int, default=3,
                       help='Minimum points in one BEV cell before static ground de-layering is applied.')
    parse.add_argument('--static_ground_cell_z_span_max', type=float, default=0.4,
                       help='Only collapse BEV cells whose ground candidate z span is at most this value.')
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

    # # load config
    # with open(args.config_path, 'r') as stream:
    #     config = yaml.safe_load(stream)

    # load learning map
    # label_mapping = args.label_mapping
    # with open(label_mapping, 'r') as stream:
    #     nuscenesyaml = yaml.safe_load(stream)

    if args.scene_list_file:
        scene_names = load_scene_names(args.scene_list_file)
        if not scene_names:
            raise ValueError(f"No valid scene names found in {args.scene_list_file}")

        selected_scene_indices = resolve_scene_indices(nusc, scene_names)
        print(
            f"Using custom scene list from {args.scene_list_file} with {len(selected_scene_indices)} scenes. "
            f"Ignoring --split={args.split!r} and --start/--end for scene selection."
        )
    else:
        if args.start < 0 or args.end > len(nusc.scene) or args.start >= args.end:
            raise ValueError(
                f"Invalid scene index range [{args.start}, {args.end}) for {len(nusc.scene)} available scenes."
            )
        selected_scene_indices = list(range(args.start, args.end))

    for i in selected_scene_indices:
        print(f"processing sequecne: {i} ({nusc.scene[i]['name']})")
        main(nusc, val_list, indice=i, args=args)
