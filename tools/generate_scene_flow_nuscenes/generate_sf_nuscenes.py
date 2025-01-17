import os
import sys
import pdb
import time
import yaml
import torch
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation
import open3d as o3d



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

# def preprocess_cloud(
#     pcd,
#     max_nn=20,
#     normals=None,
# ):
#
#     cloud = deepcopy(pcd)
#     if normals:
#         params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
#         cloud.estimate_normals(params)
#         cloud.orient_normals_towards_camera_location()
#
#     return cloud
#
#
# def preprocess(pcd, config):
#     return preprocess_cloud(
#         pcd,
#         config['max_nn'],
#         normals=True
#     )

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


def main(nusc, val_list, indice, nuscenesyaml, args):

    save_path = args.save_path
    data_root = args.dataroot
    learning_map = nuscenesyaml['learning_map']

    my_scene = nusc.scene[indice]
    sensor = 'LIDAR_TOP'


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
        converted_object_category = []
        for category in object_category:
            for (j, label) in enumerate(nuscenesyaml['labels']):
                if category == nuscenesyaml['labels'][label]:
                    converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

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
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6]

        ############################# get LiDAR points with semantics ##########################
        pc_file_name = lidar_data['filename'] # load LiDAR names
        pc0 = np.fromfile(os.path.join(data_root, pc_file_name),
                          dtype=np.float32,
                          count=-1).reshape(-1, 5)[..., :4]
        if lidar_data['is_key_frame']: # only key frame has semantic annotations
            lidar_sd_token = lidar_data['token']
            lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                    nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)

            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)

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
                "converted_object_category": converted_object_category,
                "pc_file_name": pc_file_name.split('/')[-1]}
        ################## record semantic information into the dict if it's a key frame  ########################
        if lidar_data['is_key_frame']:
            pc_with_semantic = pc_with_semantic[points_mask]
            lidar_pc_with_semantic = lidar_to_world_to_lidar(pc_with_semantic.copy(),
                                                             lidar_calibrated_sensor.copy(),
                                                             lidar_ego_pose.copy(),
                                                             lidar_calibrated_sensor0,
                                                             lidar_ego_pose0)
            dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points

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


    ################## concatenate all object segments (including non-key frames)  ########################
    object_token_zoo = []
    object_semantic = []
    for dict in dict_list:
        for i,object_token in enumerate(dict['object_tokens']):
            if object_token not in object_token_zoo:
                if (dict['object_points_list'][i].shape[0] > 0):
                    object_token_zoo.append(object_token)
                    object_semantic.append(dict['converted_object_category'][i])
                else:
                    continue

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

    # object_points_vertice = []
    # for key in object_points_dict.keys():
    #     point_cloud = object_points_dict[key]
    #     object_points_vertice.append(point_cloud[:,:3])

    object_points_vertice = []
    for key in object_points_dict.keys():
        point_cloud = object_points_dict[key]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        voxel_size = 0.1
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        point_cloud = np.asarray(pcd_down.points)
        object_points_vertice.append(point_cloud[:, :3])


    # concatenate N frames before and after current frame
    # 2*N+1 frames in total. N has to be equal or larger than 1.
    N = 15

    i = 0
    while int(i) < len(dict_list):
        if i <= N-1:
            i += 1
            continue

        if i >= len(dict_list)-N:
            print('finish scene!')
            break

        # if i == 110:
        #     break

        current_dict = dict_list[i]
        next_dict = dict_list[i+1]

        # concatenate N frames before and after current frame
        # 2*N+1 static scene segments in total
        ################## concatenate static point cloud ########################
        lidar_pc_slice_list = [dict['lidar_pc'] for dict in dict_list[i-N:i+N+1]]
        lidar_pc_slice = np.concatenate(lidar_pc_slice_list, axis=1).T

        # ################## concatenate object points ########################
        # obj_pc_slice_list = [dict['object_points_list'] for dict in dict_list[i-N:i+N+1]]
        # obj_token_list = [dict['object_tokens'] for dict in dict_list[i-N:i+N+1]]
        # obj_gt_bbox_3d = [dict['gt_bbox_3d'] for dict in dict_list[i-N:i+N+1]]

        ################## convert the static scene to current coordinate system ##############
        lidar_calibrated_sensor = current_dict['lidar_calibrated_sensor']
        lidar_ego_pose = current_dict['lidar_ego_pose']
        current_lidar_pc = lidar_to_world_to_lidar(lidar_pc_slice.copy(),
                                             lidar_calibrated_sensor0.copy(),
                                             lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)

        current_point_cloud = current_lidar_pc.points.T[:, :3]

        ################## convert the static scene to the next coordinate system ##############
        lidar_calibrated_sensor = next_dict['lidar_calibrated_sensor']
        lidar_ego_pose = next_dict['lidar_ego_pose']
        next_lidar_pc = lidar_to_world_to_lidar(lidar_pc_slice.copy(),
                                             lidar_calibrated_sensor0.copy(),
                                             lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)

        next_point_cloud = next_lidar_pc.points.T[:, :3]

        ################## load bbox of current frame ##############
        lidar_path, boxes, _ = nusc.get_sample_data(current_dict['lidar_token'])
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2]
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6]
        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        ################## bbox placement ##############
        current_object_points_list = []
        for j, object_token in enumerate(current_dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token == object_token_in_zoo:
                    points = object_points_vertice[k]
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]
                    # if points.shape[0] >= 5:
                    #     points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                    #                                           torch.from_numpy(gt_bbox_3d[j:j + 1][np.newaxis, :]))
                    #     points = points[points_in_boxes[0, :, 0].bool()]

                    current_object_points_list.append(points)

        ################## load bbox of next frame ##############
        lidar_path, boxes, _ = nusc.get_sample_data(next_dict['lidar_token'])
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2]
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6]
        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        ################## bbox placement ##############
        next_object_points_list = []
        for j, object_token in enumerate(next_dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token == object_token_in_zoo:
                    points = object_points_vertice[k]
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]
                    # if points.shape[0] >= 5:
                    #     points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                    #                                           torch.from_numpy(gt_bbox_3d[j:j + 1][np.newaxis, :]))
                    #     points = points[points_in_boxes[0, :, 0].bool()]

                    next_object_points_list.append(points)


        # ################## concatenate object point cloud ##############
        # object_token_zoo = []
        # object_pc_dict = {}
        # for j, _ in enumerate(obj_token_list):
        #     for k, _ in enumerate(obj_token_list[j]):
        #         if obj_token_list[j][k] not in object_token_zoo:
        #             object_token_zoo.append(obj_token_list[j][k])
        #         else:
        #             continue
        # for j, object_token in enumerate(object_token_zoo):
        #     object_pc_dict[object_token] = []
        #     for k, _ in enumerate(obj_token_list):
        #         for l, _ in enumerate(obj_token_list[k]):
        #             if obj_token_list[k][l] == object_token:
        #                 object_points = obj_pc_slice_list[k][l]
        #                 if object_points.shape[0] > 0:
        #                     object_points = obj_pc_slice_list[k][l][:,:3] - obj_gt_bbox_3d[k][l][:3]
        #                     rots = obj_gt_bbox_3d[k][l][6]
        #                     Rot = Rotation.from_euler('z', -rots, degrees=False)
        #                     rotated_object_points = Rot.apply(object_points)
        #                     object_pc_dict[object_token].append(rotated_object_points)
        #             else:
        #                 continue
        #     if len(object_pc_dict[object_token]) > 0:
        #         object_pc_dict[object_token] = np.concatenate(object_pc_dict[object_token], axis=0)
        #     else:
        #         object_pc_dict[object_token] = np.array([[0, 0, 0]])

        # object_pc_vertice = []
        # for key in object_pc_dict.keys():
        #     point_cloud = object_pc_dict[key]
        #     object_pc_vertice.append(point_cloud[:,:3])

        # ################## load bbox of current frame ##############
        # lidar_path, boxes, _ = nusc.get_sample_data(current_dict['lidar_token'])
        # locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        # dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        # rots = np.array([b.orientation.yaw_pitch_roll[0]
        #                  for b in boxes]).reshape(-1, 1)
        # gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        # gt_bbox_3d[:, 6] += np.pi / 2.
        # gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        # gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2]
        # gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6]
        # rots = gt_bbox_3d[:, 6:7]
        # locs = gt_bbox_3d[:, 0:3]
        #
        # ################## bbox placement ##############
        # current_object_points_list = []
        # for j, object_token in enumerate(current_dict['object_tokens']):
        #     for k, object_token_in_zoo in enumerate(object_token_zoo):
        #         if object_token == object_token_in_zoo:
        #             points = object_pc_vertice[k]
        #             Rot = Rotation.from_euler('z', rots[j], degrees=False)
        #             rotated_object_points = Rot.apply(points)
        #             points = rotated_object_points + locs[j]
        #             current_object_points_list.append(points)
        #
        # ################## load bbox of next frame ##############
        # lidar_path, boxes, _ = nusc.get_sample_data(next_dict['lidar_token'])
        # locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        # dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        # rots = np.array([b.orientation.yaw_pitch_roll[0]
        #                  for b in boxes]).reshape(-1, 1)
        # gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        # gt_bbox_3d[:, 6] += np.pi / 2.
        # gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        # gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1
        # gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1
        # rots = gt_bbox_3d[:, 6:7]
        # locs = gt_bbox_3d[:, 0:3]
        #
        # ################## bbox placement ##############
        # next_object_points_list = []
        # for j, object_token in enumerate(next_dict['object_tokens']):
        #     for k, object_token_in_zoo in enumerate(object_token_zoo):
        #         if object_token == object_token_in_zoo:
        #             points = object_pc_vertice[k]
        #             Rot = Rotation.from_euler('z', rots[j], degrees=False)
        #             rotated_object_points = Rot.apply(points)
        #             points = rotated_object_points + locs[j]
        #             next_object_points_list.append(points)
        #
        ################## get the intersection of current and next object points list ##############
        intersection_object_tokens = list(set(current_dict['object_tokens']) & set(next_dict['object_tokens']))
        current_obj_mask = np.isin(current_dict['object_tokens'], intersection_object_tokens)
        next_obj_mask = np.isin(next_dict['object_tokens'], intersection_object_tokens)

        filtered_current_object_points_list = []
        filtered_next_object_points_list = []
        for j, current_object_point in enumerate(current_object_points_list):
            if current_obj_mask[j]:
                filtered_current_object_points_list.append(current_object_point)

        for j, next_object_point in enumerate(next_object_points_list):
            if next_obj_mask[j]:
                filtered_next_object_points_list.append(next_object_point)

        ################## concatenate static scene segments and object points  ########################
        try:
            current_temp = np.concatenate(filtered_current_object_points_list)
            current_scene_points = np.concatenate([current_point_cloud, current_temp])
        except:
            current_scene_points = current_point_cloud

        try:
            next_temp = np.concatenate(filtered_next_object_points_list)
            next_scene_points = np.concatenate([next_point_cloud, next_temp])
        except:
            next_scene_points = next_point_cloud

        ################## remain points with a spatial range ##############
        current_range_mask = (np.abs(current_scene_points[:, 0]) < 15) & (np.abs(current_scene_points[:, 1]) < 50.0) \
               & (current_scene_points[:, 2] > -5.0) & (current_scene_points[:, 2] < 3.0)

        next_range_mask = (np.abs(next_scene_points[:, 0]) < 15) & (np.abs(next_scene_points[:, 1]) < 50.0) \
                & (next_scene_points[:, 2] > -5.0) & (next_scene_points[:, 2] < 3.0)

        # current_range_mask = (np.abs(current_scene_points[:, 0]) < 50) & (np.abs(current_scene_points[:, 1]) < 50.0) \
        #                      & (current_scene_points[:, 2] > -5.0) & (current_scene_points[:, 2] < 3.0)
        #
        # next_range_mask = (np.abs(next_scene_points[:, 0]) < 50) & (np.abs(next_scene_points[:, 1]) < 50.0) \
        #                   & (next_scene_points[:, 2] > -5.0) & (next_scene_points[:, 2] < 3.0)

        intersection_points_mask = current_range_mask & next_range_mask

        current_scene_points = current_scene_points[intersection_points_mask]
        next_scene_points = next_scene_points[intersection_points_mask]

        ################## visualization ##################
        # point_cloud_static_vis = o3d.geometry.PointCloud()
        # point_cloud_static_vis.points = o3d.utility.Vector3dVector(current_scene_points)
        # o3d.visualization.draw_geometries([point_cloud_static_vis])

        ################## save the scene points and object points  ########################
        pc_file_name_folder = current_dict['pc_file_name'].replace('.pcd.bin', '')
        dirs = os.path.join(save_path, '31_scene_flow/' ,pc_file_name_folder)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        np.save(os.path.join(dirs, 'pc1.npy'), current_scene_points)
        np.save(os.path.join(dirs, 'pc3.npy'), next_scene_points)

        print(i)

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
    # parse.add_argument('--config_path', type=str, default='./tools/generate_sceneflow_nuscenes/config.yaml')
    parse.add_argument('--split', type=str, default='val')
    parse.add_argument('--save_path', type=str, default='/mnt/fpttc_data/scene_flow/multi_frame')
    parse.add_argument('--start', type=int, default=0)
    parse.add_argument('--end', type=int, default=2)
    parse.add_argument('--dataroot', type=str, default='./Datasets/nuscenes/')
    parse.add_argument('--nusc_val_list', type=str, default='./tools/generate_scene_flow_nuscenes/nuscenes_scene_flow_list.txt')
    parse.add_argument('--label_mapping', type=str, default='./tools/generate_scene_flow_nuscenes/nuscenes.yaml')
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
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)


    for i in range(args.start,args.end):
        print('processing sequecne:', i)
        main(nusc, val_list, indice=i,
             nuscenesyaml=nuscenesyaml, args=args)
