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

import open3d.visualization.gui as gui
from open3d.visualization import O3DVisualizer

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

    args.scene_num -= 1
    if args.scene_num < 0:
        return

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
        range = [3.0, 3.0, 3.0] # remove points within 3m of the vehicle itself
        oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > range[0]) |
                                        (np.abs(pc0[:, 1]) > range[1]) |
                                        (np.abs(pc0[:, 2]) > range[2]))

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

    object_points_xyz = []
    for key in object_points_dict.keys():
        point_cloud = object_points_dict[key]
        object_points_xyz.append(point_cloud[:,:3])

    for i in trange(1, len(dict_list), desc="Processing frames"):

        prev_dict = dict_list[i-1]
        curr_dict = dict_list[i]

        ################## concatenate static point cloud ########################
        lidar_pc_slice_list = [dict['lidar_pc'] for dict in dict_list]
        lidar_pc_slice = np.concatenate(lidar_pc_slice_list, axis=1).T

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
        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        ################## place object points into corresponding bboxes ##############
        prev_object_points_list = []
        for j, object_token in enumerate(prev_dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token == object_token_in_zoo:
                    points = object_points_xyz[k]
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]
                    prev_object_points_list.append(points)

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
        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        ################## place the points into corresponding bboxes ##############
        curr_object_points_list = []
        for j, object_token in enumerate(curr_dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token == object_token_in_zoo:
                    points = object_points_xyz[k]
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]
                    curr_object_points_list.append(points)

        ################## get the intersection of current and next object points list ##############
        # Find intersection of object tokens between previous and current frames
        intersection_object_tokens = set(prev_dict['object_tokens']).intersection(curr_dict['object_tokens'])

        # Create masks for filtering object points
        prev_obj_mask = [token in intersection_object_tokens for token in prev_dict['object_tokens']]
        curr_obj_mask = [token in intersection_object_tokens for token in curr_dict['object_tokens']]

        # Filter prev object points based on the mask
        filtered_prev_object_points_list = []
        filtered_prev_object_boxes_list = []
        for j, prev_object_point in enumerate(prev_object_points_list):
            if prev_obj_mask[j]:
                filtered_prev_object_points_list.append(prev_object_point)
                filtered_prev_object_boxes_list.append(prev_boxes[j])

        # Filter curr object points based on the mask
        filtered_curr_object_points_list = []
        filtered_curr_object_boxes_list = []
        for j, curr_object_point in enumerate(curr_object_points_list):
            if curr_obj_mask[j]:
                filtered_curr_object_points_list.append(curr_object_point)
                filtered_curr_object_boxes_list.append(curr_boxes[j])
        
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
        pc_file_name_folder = curr_dict['pc_file_name'].replace('.pcd.bin', '')
        pc_file_name_folder = f"scene_{indice}_{pc_file_name_folder}"
        dirs = os.path.join(save_path, 'scene_flow_all_frames/', pc_file_name_folder)
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
    parse.add_argument('--scene_num', type=int, default=10)
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

    # # load config
    # with open(args.config_path, 'r') as stream:
    #     config = yaml.safe_load(stream)

    # load learning map
    # label_mapping = args.label_mapping
    # with open(label_mapping, 'r') as stream:
    #     nuscenesyaml = yaml.safe_load(stream)

    for i in range(args.start, args.end):
        print('processing sequecne:', i)
        main(nusc, val_list, indice=i, args=args)
