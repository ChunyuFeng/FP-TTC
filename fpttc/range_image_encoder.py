import math
import time
import torch
import torch.nn as nn
import numpy as np

from pyquaternion import Quaternion
from mmcv.cnn import xavier_init, constant_init
from mmcv.utils import TORCH_VERSION, digit_version

from .scale_net.utils.affine import affine, affine_x_y
from .modules.position import PositionEmbeddingSine
from .modules.geometry import flow_wrap
from .scale_net.utils.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from .modules.attention import SelfAttnPropagation

from mmcv.runner import force_fp32, auto_fp16

# 设置一个参数，存储6通道相机的标定信息
# 用于将参考点从 3D LiDAR 坐标投影到 6 环视图
# 这个标定信息是较为粗糙的，没有考虑lidar和camera采样时间不同导致的位姿差异
# 虽然nuscenes数据集中提供了精确的标定信息，但是后续自车采集的数据中无法达到这么高的精度
# 但是在这个任务中，我们本就不要求精确地标定信息（当前设想，待实验验证）

def get_lidar2img_matrix(cam_cs_record, lidar_cs_record, orig_img_size, crop_size, feat_size):
    """
    根据相机标定和激光雷达到自车的固定外参构造 4x4 的 lidar2img 投影矩阵，
    并根据图像resize_and_crop以及后续特征图下采样的尺寸更新内参矩阵，
    使其适应特征图坐标系。

    Args:
        cam_cs_record (dict): 相机标定信息，包含内参、旋转和平移，
                              描述相机传感器坐标系到自车坐标系。
        lidar_cs_record (dict): 激光雷达到自车的固定标定信息（旋转和平移）。
        orig_img_size (tuple): 原始图像尺寸 (w, h)。
        crop_size (tuple): resize_and_crop 后的图像尺寸 (crop_h, crop_w)。
        feat_size (tuple): 最终特征图的尺寸 (feat_h, feat_w)。

    Returns:
        lidar2img (np.ndarray): 4x4 投影矩阵，将 LiDAR 坐标系下的点直接投影到特征图坐标系。
    """
    # 1. 构造 LiDAR 到自车的变换矩阵 T_lidar_to_ego
    R_lidar = Quaternion(lidar_cs_record['rotation']).rotation_matrix  # (3,3)
    t_lidar = np.array(lidar_cs_record['translation']).reshape(3, 1)
    T_lidar_to_ego = np.eye(4)
    T_lidar_to_ego[:3, :3] = R_lidar
    T_lidar_to_ego[:3, 3:4] = t_lidar

    # 2. 构造相机从自车到相机坐标系的变换矩阵 T_ego_to_cam
    # cam_cs_record 中的参数描述 T_cam_to_ego
    R_cam = Quaternion(cam_cs_record['rotation']).rotation_matrix  # (3,3)
    t_cam = np.array(cam_cs_record['translation']).reshape(3, 1)
    T_cam_to_ego = np.eye(4)
    T_cam_to_ego[:3, :3] = R_cam
    T_cam_to_ego[:3, 3:4] = t_cam
    # 求逆得到 T_ego_to_cam
    T_ego_to_cam = np.linalg.inv(T_cam_to_ego)

    # 3. 组合变换，得到 LiDAR 到相机坐标系的变换矩阵
    T_lidar_to_cam = T_ego_to_cam.dot(T_lidar_to_ego)

    # 4. 根据resize_and_crop调整内参
    # 原始图像尺寸
    w, h = orig_img_size  # 如 (1600, 900)
    crop_h, crop_w = crop_size  # 如 (800, 1600)
    # 根据resize_and_crop规则，只利用宽度计算缩放比例
    resize = crop_w / w
    resize_w = int(w * resize)
    resize_h = int(h * resize)
    # 裁剪：h方向保留底部（crop_h_start=0），w方向居中裁剪
    crop_h_start = 0
    crop_w_start = (resize_w - crop_w) // 2

    # 原始内参矩阵 K
    K = np.array(cam_cs_record['camera_intrinsic'])  # (3,3)
    # 缩放内参：先对焦距和主点按resize比例进行缩放
    K_new = np.zeros_like(K)
    K_new[0, 0] = K[0, 0] * resize
    K_new[1, 1] = K[1, 1] * resize
    K_new[0, 2] = K[0, 2] * resize - crop_w_start
    K_new[1, 2] = K[1, 2] * resize - crop_h_start
    K_new[2, 2] = 1.0

    # 5. 根据特征图尺寸对内参进一步缩放
    feat_h, feat_w = feat_size  # 如特征图尺寸 (40, 80) 或其他尺寸
    scale_x = feat_w / crop_w
    scale_y = feat_h / crop_h

    K_feat = np.zeros_like(K_new)
    K_feat[0, 0] = K_new[0, 0] * scale_x
    K_feat[1, 1] = K_new[1, 1] * scale_y
    K_feat[0, 2] = K_new[0, 2] * scale_x
    K_feat[1, 2] = K_new[1, 2] * scale_y
    K_feat[2, 2] = 1.0

    # 6. 构造新的投影矩阵 P' = [K_feat | 0]
    P = np.hstack((K_feat, np.zeros((3, 1))))
    # 计算 lidar2img：先将 LiDAR 点变换到相机坐标系，再投影到特征图平面
    lidar2img_3x4 = P.dot(T_lidar_to_cam)  # 结果为 3x4矩阵

    # 如果需要 4x4 格式，则补充一行
    lidar2img = np.eye(4)
    lidar2img[:3, :] = lidar2img_3x4

    return lidar2img

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, padding_mode="zeros"):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        )


class RangeImageEncoder(nn.Module):
    def __init__(self,
                 num_layers=2,
                 input_dim=128,
                 d_model=128,
                 nhead=4,
                 num_feature_levels=2,
                 num_level=4,
                 ):
        super(RangeImageEncoder, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.embed_dims = input_dim
        self.num_level = num_level

        self.scale_conv = nn.Sequential(
            conv(input_dim, d_model*2, padding_mode="zeros"),
            conv(d_model*2, d_model, padding_mode="zeros"),
            conv(d_model, d_model, padding_mode="zeros")
        )
        self.query_conv1 = conv(2*d_model, d_model)
        self.query_conv2 = conv(d_model, d_model)

        self.value_fs_conv = nn.Conv1d(2*input_dim, d_model, 1)

        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model/2)

        self.layers = nn.ModuleList([
            TransformerBlock(num_level=num_level,
                             d_model=d_model,
                             num_head=nhead,
                             )
            for i in range(num_layers)])

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

        self.img_metas = {
            "CAM_BACK_LEFT": {
                "cam_cs_record": {
                    "camera_intrinsic": [
                        [1256.7414812095406, 0.0, 792.1125740759628],
                        [0.0, 1256.7414812095406, 492.7757465151356],
                        [0.0, 0.0, 1.0]
                    ],
                    "rotation": [0.6924185592174665, -0.7031619420114925, -0.11648342771943819, 0.11203317912370753],
                    "translation": [1.03569100218, 0.484795032713, 1.59097014818]
                },
                "lidar_cs_record": {
                    "camera_intrinsic": [],
                    "rotation": [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817],
                    "translation": [0.943713, 0.0, 1.84023]
                }
            },
            "CAM_BACK": {
                "cam_cs_record": {
                    "camera_intrinsic": [
                        [809.2209905677063, 0.0, 829.2196003259838],
                        [0.0, 809.2209905677063, 481.77842384512485],
                        [0.0, 0.0, 1.0]
                    ],
                    "rotation": [0.5037872666382278, -0.49740249788611096, -0.4941850223835201, 0.5045496097725578],
                    "translation": [0.0283260309358, 0.00345136761476, 1.57910346144]
                },
                "lidar_cs_record": {
                    "camera_intrinsic": [],
                    "rotation": [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817],
                    "translation": [0.943713, 0.0, 1.84023]
                }
            },
            "CAM_BACK_RIGHT": {
                "cam_cs_record": {
                    "camera_intrinsic": [
                        [1259.5137405846733, 0.0, 807.2529053838625],
                        [0.0, 1259.5137405846733, 501.19579884916527],
                        [0.0, 0.0, 1.0]
                    ],
                    "rotation": [0.12280980120078765, -0.132400842670559, -0.7004305821388234, 0.690496031265798],
                    "translation": [1.0148780988, -0.480568219723, 1.56239545128]
                },
                "lidar_cs_record": {
                    "camera_intrinsic": [],
                    "rotation": [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817],
                    "translation": [0.943713, 0.0, 1.84023]
                }
            },
            "CAM_FRONT_RIGHT": {
                "cam_cs_record": {
                    "camera_intrinsic": [
                        [1260.8474446004698, 0.0, 807.968244525554],
                        [0.0, 1260.8474446004698, 495.3344268742088],
                        [0.0, 0.0, 1.0]
                    ],
                    "rotation": [0.2060347966337182, -0.2026940577919598, 0.6824507824531167, -0.6713610884174485],
                    "translation": [1.5508477543, -0.493404796419, 1.49574800619]
                },
                "lidar_cs_record": {
                    "camera_intrinsic": [],
                    "rotation": [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817],
                    "translation": [0.943713, 0.0, 1.84023]
                }
            },
            "CAM_FRONT": {
                "cam_cs_record": {
                    "camera_intrinsic": [
                        [1266.417203046554, 0.0, 816.2670197447984],
                        [0.0, 1266.417203046554, 491.50706579294757],
                        [0.0, 0.0, 1.0]
                    ],
                    "rotation": [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],
                    "translation": [1.70079118954, 0.0159456324149, 1.51095763913]
                },
                "lidar_cs_record": {
                    "camera_intrinsic": [],
                    "rotation": [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817],
                    "translation": [0.943713, 0.0, 1.84023]
                }
            },
            "CAM_FRONT_LEFT": {
                "cam_cs_record": {
                    "camera_intrinsic": [
                        [1272.5979470598488, 0.0, 826.6154927353808],
                        [0.0, 1272.5979470598488, 479.75165386361925],
                        [0.0, 0.0, 1.0]
                    ],
                    "rotation": [0.6757265034669446, -0.6736266522251881, 0.21214015046209478, -0.21122827103904068],
                    "translation": [1.52387798135, 0.494631336551, 1.50932822144]
                },
                "lidar_cs_record": {
                    "camera_intrinsic": [],
                    "rotation": [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817],
                    "translation": [0.943713, 0.0, 1.84023]
                }
            }
        }


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    '''
    query: scale0 + scale1
    value: feats0 + feats1
    '''
    def forward(self, feature0_list, feature1_list,
                query_lvl = -1, ini_query=None
                ):
        
        assert len(feature0_list)==self.num_level
        bs, _, height, width = feature1_list[query_lvl].shape

        # query
        scale_pre = feature0_list[query_lvl]
        if ini_query is None:
            query = self.scale_conv(scale_pre)  # b, c, h, w
        else:
            query = self.scale_conv(ini_query)
        query += self.pos_enc(query)    # b, c, h, w

        feat_size = feature0_list[query_lvl].shape[-2:]

        reference_points = self.get_reference_points(height, width, bs, device=query.device, dtype=query.dtype)

        reference_points_cam, cam_mask = self.point_sampling(reference_points,
                                                             self.img_metas,
                                                             orig_img_size=(1600, 900),
                                                             crop_size=(160,320),
                                                             feat_size=feat_size)

        # value
        feat0_flatten = []
        feat1_flatten = []
        spatial_shapes = []

        t0 = time.time()
        for lvl, feat in enumerate(feature0_list):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2) # (bs, c, H*W)
            feat = feat + self.level_embeds[None, lvl:lvl + 1, :].permute(0,2,1).contiguous().to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat0_flatten.append(feat)

        for lvl, feat in enumerate(feature1_list):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2) # (bs, c, H*W)
            feat = feat + self.level_embeds[None, lvl:lvl + 1, :].permute(0,2,1).contiguous().to(feat.dtype)
            feat1_flatten.append(feat)

        feat0_flatten = torch.cat(feat0_flatten, 2)   #b, c, hw+HW
        feat1_flatten = torch.cat(feat1_flatten, 2)   #b, c, hw+HW

        value = self.value_fs_conv(torch.cat([feat0_flatten, feat1_flatten], dim=1))  # b, c, hw+HW
        value = (value.permute(0,2,1).contiguous().unsqueeze(2).repeat(1, 1, self.nhead, 1))   # b, hw+HW, nhead, c

        # print(value.shape)

        bs, _, h, w = query.shape
        query = query.flatten(2).permute(0,2,1).contiguous() # b, HW, c

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat0_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        

        for i, layer in enumerate(self.layers):
            query = layer(query, value,
                            height=h,
                            width=w,
                            query_location=query_location,
                            spatial_shapes=spatial_shapes,
                            level_start_index=level_start_index,
                            )

        scale_feat = query.view(bs, height, width, self.d_model).permute(0,3,1,2).contiguous()

        return scale_feat

    # # 原有的参考点生成函数，用于生成 原图到原图的 query，不需要进行几何变换
    # @staticmethod
    # def get_reference_points(H, W, bs=1, device='cuda', dtype=torch.float):

    #     ref_y, ref_x = torch.meshgrid(
    #         torch.linspace(
    #             0.5, H - 0.5, H, dtype=dtype, device=device),
    #         torch.linspace(
    #             0.5, W - 0.5, W, dtype=dtype, device=device)
    #     )
    #     ref_y = ref_y.reshape(-1)[None] / H
    #     ref_x = ref_x.reshape(-1)[None] / W
    #     ref_2d = torch.stack((ref_x, ref_y), -1)
    #     #print(ref_2d.shape)
    #     ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
    #     return ref_2d
    
    # 生成参考点，用于生成 range image 到 6环视图的 query，需要进行几何变换
    @staticmethod
    def get_reference_points(H, W, max_range=70.0, num_points_in_ray=8,
                           fov_up=3.0, fov_down=-25.0, dim='3d', bs=1,
                           device='cuda', dtype=torch.float):
        """
        生成用于 SCA/TSA 的参考点，这里参考点基于 range image 投影。
        Args:
            H, W: range image 的高度和宽度。
            max_range: 激光雷达测量的最大范围（单位与点云一致）。
            num_points_in_ray: 每个像素在射线上均匀采样的点数。
            fov_up, fov_down: 垂直视场角（度）。
            dim: '3d' 时返回 3D 参考点（经逆投影得到 LiDAR 坐标），
                '2d' 时返回 range image 平面归一化坐标。
            bs: 批次大小。
            device, dtype: 设备和数据类型。
        Returns:
            如果 dim=='3d'，返回形状为 (bs, num_keys, num_levels, 3) 的 3D 参考点张量；
            如果 dim=='2d'，返回形状为 (bs, num_keys, 1, 2) 的归一化 range image 坐标。
        """
        # 将 fov 转换为弧度
        fov_up_rad = fov_up / 180.0 * np.pi
        fov_down_rad = fov_down / 180.0 * np.pi
        fov = abs(fov_down_rad) + abs(fov_up_rad)

        if dim == '3d':
            # 沿深度方向均匀采样候选点 (单位：实际距离)
            rs = torch.linspace(0.5, max_range - 0.5, num_points_in_ray, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_ray, H, W)
            # 对 range image 的每个像素生成归一化的横向和纵向坐标
            us = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_ray, H, W) / W
            vs = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_ray, H, W) / H

            # 计算对应的 yaw 和 pitch
            # 横向归一化坐标 -> yaw：范围 [-pi, pi]
            yaw = us * 2 * np.pi - np.pi  # shape: (num_points_in_ray, H, W)
            # 纵向归一化坐标 -> pitch：将 v=0对应 fov_up，v=1对应 -|fov_down|
            pitch = (1 - vs) * fov - abs(fov_down_rad)

            # 由 (r, yaw, pitch) 计算 3D 坐标（激光雷达坐标系）
            xs = rs * torch.cos(pitch) * torch.cos(yaw)
            ys = rs * torch.cos(pitch) * torch.sin(yaw)
            zs = rs * torch.sin(pitch)

            # 将 (num_points_in_ray, H, W, 3) 调整成与原来 get_reference_points 类似的格式
            ref_3d = torch.stack((xs, ys, zs), -1)  # shape: (num_points_in_ray, H, W, 3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (num_points_in_ray, H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  # (bs, num_points_in_ray, H*W, 3)
            return ref_3d

        elif dim == '2d':
            # 对 range image 平面，直接返回归一化的像素坐标
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)  # (1, H*W, 2)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)  # (bs, H*W, 1, 2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, img_metas, crop_size, feature_size=None):
        # 关闭 TF32 加速以确保数值精度
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # 提取每个样本的 lidar2img 变换矩阵，形状为 (B, num_cam, 4, 4)
        lidar2img = []
        # 遍历img_metas的每个key:value
        for channel, calib_info in img_metas.items():
            cam_cs_record = calib_info['cam_cs_record']
            lidar_cs_record = calib_info['lidar_cs_record']
            lidar2img_matrix = get_lidar2img_matrix(cam_cs_record,
                                                    lidar_cs_record,
                                                    orig_img_size=(1600, 900),
                                                    crop_size=crop_size,
                                                    feat_size=feature_size
                                                    )
            lidar2img.append(lidar2img_matrix)

        # for img_meta in img_metas:
        #     lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, num_cam, 4, 4)
        reference_points = reference_points.clone()
        # 此处的 reference_points 已经为 3D LiDAR 坐标，不需要用 pc_range 反归一化

        # 转换为齐次坐标
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        # 调整维度以便与多相机矩阵做矩阵乘法
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(0)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        # 将参考点从 LiDAR 坐标系投影到相机坐标系
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        # 仅保留相机前方的点（z > eps），并执行透视除法
        cam_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # 归一化到相机图像尺寸，假定 img_metas 中给定了 img_shape (height, width)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        cam_mask = (cam_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            cam_mask = torch.nan_to_num(cam_mask)
        else:
            cam_mask = cam_mask.new_tensor(np.nan_to_num(cam_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        cam_mask = cam_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        # 恢复 TF32 配置
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, cam_mask

class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=128,
                 num_head=1,
                 num_points = 8,
                 num_level=2,
                 dropout=0.1
                 ):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model
        self.num_points = num_points
        self.num_level = num_level
        self.num_head = num_head

        self.sampling_offsets = nn.Linear(d_model, num_head*num_level*num_points* 2)
        self.attention_weights = nn.Linear(d_model, num_head*num_level*num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model*num_head, d_model)
        self.dropout = nn.Dropout(dropout)

        self.MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32

        self.init_weights()


    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_head,
            dtype=torch.float32) * (2.0 * math.pi / self.num_head)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_head, 1, 1,
            2).repeat(1, self.num_level, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True


    def forward(self, query, value,
                height=None,
                width=None,
                query_location=None,
                spatial_shapes=None,
                level_start_index=None,
                ):
        '''
        query: [bs, hw, c]
        value: [bs, num_value, c]
        query_location: [bs, hw, num_level, 2]
        spatial_shapes: [2, 2]
        level_start_index: [2]叠不同时刻 BEV 特征那样只能获取定长的时序信息。


        '''

        bs, num_query, c = query.shape

        sampling_offsets = self.sampling_offsets(query)\
                .view(bs, num_query, self.num_head, self.num_level, self.num_points, 2)
        attention_weights = self.attention_weights(query)\
                .view(bs, num_query, self.num_head, self.num_level*self.num_points).softmax(-1)
        attention_weights = attention_weights\
                .view(bs, num_query, self.num_head, self.num_level, self.num_points)
    
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        # print(query_location[:, :, None, :, None, :].shape, sampling_offsets.shape, offset_normalizer[None, :, None, :].shape)
        sampling_locations = query_location[:, :, None, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, :, None, :]
        
        output = self.MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,attention_weights)
        #print(self.num_level, self.num_points)
        #print(self.num_head, output.shape)
        
        # output: (bs, num_query, c)
        output = self.output_proj(output)

        return self.dropout(output)