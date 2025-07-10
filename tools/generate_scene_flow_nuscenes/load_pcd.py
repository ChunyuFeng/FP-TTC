import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
# import open3d as o3d

# 1. 加载 mesh 并采样点云
pcd_car = o3d.io.read_point_cloud("Datasets/nuscenes/0_scene_flow/object_3d_model/bus.pcd")

# pcd_car = mesh_car.sample_points_uniformly(number_of_points=500000)
car_points = np.asarray(pcd_car.points)   # (1e6,3)

# # 2. 计算原始 AABB，并把点云原点从 “左下后” 角移到中心
# aabb = mesh_car.get_axis_aligned_bounding_box()
# aabb_center = aabb.get_center()           # (3,)
# car_points_centered = car_points - aabb_center

# 3. 把坐标系从 (x→右, y→上, z→后) 旋到 (x→右, y→前, z→上)
#    这是绕 X 轴 +90° 的旋转
R = Rotation.from_euler('x', 90, degrees=True).as_matrix()  # (3,3)
car_points_rot = car_points @ R.T

# 4. 把旋转后的点云再归一化到 [0,1]³
#    先找出三个维度的 min/max，再做标准化
mins = car_points_rot.min(axis=0)  # (3,)
maxs = car_points_rot.max(axis=0)  # (3,)
extents = maxs - mins             # (3,)
car_points_norm = (car_points_rot - mins) / extents

# 到这里，car_points_norm 的每一维都被线性映射到 [0,1]
# 验证一下
print("X range:", car_points_rot[:,0].min(), "~", car_points_rot[:,0].max())
print("Y range:", car_points_rot[:,1].min(), "~", car_points_rot[:,1].max())
print("Z range:", car_points_rot[:,2].min(), "~", car_points_rot[:,2].max())

import open3d as o3d


# 1. 创建坐标系 mesh，参数 size 决定轴长
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1,  # 轴长，你可以根据点云尺度调整
    origin=[0,0,0]
)

# —— 标准化前的可视化 —— #
pcd_before = o3d.geometry.PointCloud()
pcd_before.points = o3d.utility.Vector3dVector(car_points_rot)
pcd_before.paint_uniform_color([0.1, 0.1, 0.7])

o3d.visualization.draw_geometries(
    [axes, pcd_before],
    window_name="Before Normalization + Axes",
    width=800, height=600
)

# # —— 标准化后的可视化 —— #
# pcd_after = o3d.geometry.PointCloud()
# pcd_after.points = o3d.utility.Vector3dVector(car_points_norm)
# pcd_after.paint_uniform_color([0.1, 0.7, 0.1])

# o3d.visualization.draw_geometries(
#     [axes, pcd_after],
#     window_name="After Normalization + Axes",
#     width=800, height=600
# )
