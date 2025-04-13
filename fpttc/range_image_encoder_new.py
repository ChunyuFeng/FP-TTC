import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyquaternion import Quaternion
from mmcv.utils import TORCH_VERSION, digit_version

from .scale_net.utils.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32

from mmcv.runner import force_fp32

from tools.collision_prediction.collision_utils import inverse_range_projection

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction_fp32, multi_scale_deformable_attn_pytorch
except ImportError:
    MultiScaleDeformableAttnFunction_fp32 = None
    multi_scale_deformable_attn_pytorch = None

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

class MSDeformableAttention3D(nn.Module):
    """基于 Deformable DETR 的 3D 多尺度可变形注意力模块。"""
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8, im2col_step=64, dropout=0.1, batch_first=True):
        super(MSDeformableAttention3D, self).__init__()
        assert embed_dims % num_heads == 0, "embed_dims 必须能被 num_heads 整除"
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.batch_first = batch_first

        # 线性层：由 query 特征预测采样偏移和注意力权重
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        # 对 value 投影，以便分成多头
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        # （可选）输出投影，可以将多头结果映射回 embed_dims（如果需要的话）
        self.output_proj = None
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, reference_points, spatial_shapes, level_start_index, **kwargs):
        """
        query: (B, Len_q, embed_dims) - 查询特征（如果 batch_first=True）。
        key: 未使用，提供接口兼容性。
        value: (B, Num_keys, embed_dims) - 跨视角图像的多尺度特征展平后序列。
        reference_points: (B, Len_q, num_levels, 2或4) - 每个查询在每个特征层上的参考点（归一化坐标）。
        spatial_shapes: (num_levels, 2) - 每个特征层的高和宽。
        level_start_index: (num_levels,) - 每个特征层起始位置的索引偏移（在展平序列中的起始索引）。
        返回值: (B, Len_q, embed_dims) - 融合后的查询特征。
        """
        # 若输入不是 batch_first，转换为 batch_first 方便处理
        if not self.batch_first:
            query = query.transpose(0, 1)  # (Len_q, B, C) -> (B, Len_q, C)
            value = value.transpose(0, 1)  # 同理处理 value
        
        B, Len_q, _ = query.shape
        # 对 value 做线性投影并reshape为多头形状
        value_proj = self.value_proj(value)            # (B, Num_keys, embed_dims)
        value_proj = value_proj.view(B, -1, self.num_heads, self.embed_dims // self.num_heads)  # (B, Num_keys, num_heads, head_dim)
        # 计算采样偏移和注意力权重
        sampling_offsets = self.sampling_offsets(query)   # (B, Len_q, num_heads*num_levels*num_points*2)
        attention_weights = self.attention_weights(query) # (B, Len_q, num_heads*num_levels*num_points)
        # 调整形状
        sampling_offsets = sampling_offsets.view(B, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = attention_weights.view(B, Len_q, self.num_heads, self.num_levels, self.num_points)
        # 对每个查询的所有采样点的权重做 softmax
        attention_weights = F.softmax(attention_weights, dim=-1)
        # 计算采样位置 sampling_locations
        if reference_points.shape[-1] == 2: # bs, Len_q, num_points, 2
            # reference_points 给出每个查询在各特征图上的归一化 2D 坐标 (u,v)
            # 将偏移归一化：除以特征图尺度
            offset_normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], -1).to(query.device)  # (num_levels, 2) -> (W, H)
            # 扩展 reference_points 形状以便与 sampling_offsets 相加: (B, Len_q, 1, num_levels, 1, 2)
            ref_pts = reference_points[:, :, None, None, :, :]
            samp_off = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = ref_pts + samp_off  # (B, Len_q, num_heads, num_levels, num_points, 2)
        elif reference_points.shape[-1] == 4:
            # 若 reference_points 提供了额外维度（例如包含深度或宽高），按照 Deformable DETR 的方式处理
            ref_xy = reference_points[..., :2]   # (B, Len_q, num_levels, 2)
            ref_wh = reference_points[..., 2:]   # (B, Len_q, num_levels, 2)
            ref_xy = ref_xy[:, :, None, :, None, :]  # (B, Len_q, 1, num_levels, 1, 2)
            ref_wh = ref_wh[:, :, None, :, None, :]  # (B, Len_q, 1, num_levels, 1, 2)
            sampling_locations = ref_xy + 0.5 * sampling_offsets * ref_wh  # 利用参考框尺寸调整偏移
        else:
            raise ValueError(f"reference_points 的最后一维必须为2或4，当前为 {reference_points.shape[-1]}")
        # 调用 CUDA 实现的可变形注意力，或使用 PyTorch 实现（较慢）
        value_proj = value_proj.contiguous()
        sampling_locations = sampling_locations.contiguous()
        attention_weights = attention_weights.contiguous()
        # 确保 spatial_shapes 和 level_start_index 为 tensor
        spatial_shapes = spatial_shapes.to(value_proj.device) if not isinstance(spatial_shapes, torch.Tensor) else spatial_shapes
        level_start_index = level_start_index.to(value_proj.device) if not isinstance(level_start_index, torch.Tensor) else level_start_index
        if MultiScaleDeformableAttnFunction_fp32 is not None and value_proj.is_cuda:
            # 使用 CUDA 高效实现（fp32）
            output = MultiScaleDeformableAttnFunction_fp32.apply(
                value_proj, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step
            )
        elif multi_scale_deformable_attn_pytorch is not None:
            # CPU 或无CUDA扩展时，使用 PyTorch 实现
            output = multi_scale_deformable_attn_pytorch(
                value_proj, spatial_shapes, sampling_locations, attention_weights
            )
        else:
            # 若无 mmcv 提供的函数，可在此实现自定义的采样逻辑
            # 这里简单将 output 置零（实际应用中需保证安装 mmcv 或提供实现）
            output = torch.zeros(B, Len_q, self.embed_dims, device=query.device)
        # 输出形状应为 (B, Len_q, embed_dims)
        output = output.view(B, Len_q, self.embed_dims)
        output = self.dropout(output)
        # 如果定义了输出投影层，应用投影
        if self.output_proj is not None:
            output = self.output_proj(output)
        # 若最初 query 不是 batch_first，这里转回原格式
        if not self.batch_first:
            output = output.transpose(0, 1)  # (B, Len_q, C) -> (Len_q, B, C)
        return output

class SpatialCrossAttention(nn.Module):
    """用于 RANGE IMAGE 场景的空间跨模态注意力，将 Range Image 查询与多摄像头图像特征融合。"""
    def __init__(self, embed_dims=256, num_cams=6, dropout=0.1, pc_range=None, num_levels=1, num_points=8):
        super(SpatialCrossAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        # 初始化可变形注意力模块（跨视角跨尺度注意力核心）
        self.deformable_attention = MSDeformableAttention3D(
            embed_dims=embed_dims, num_heads=embed_dims//32,  # 例如将每个头维度设为32（embed_dims=256则num_heads=8）
            num_levels=num_levels, num_points=num_points, batch_first=True
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)  # 输出投影层
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range  # 保存点云范围（可用于位置编码等）
    
    def forward(self, query, key, value, residual=None, query_pos=None,
                reference_points_cam=None, spatial_shapes=None, level_start_index=None, range_image_mask=None):
        """
        query: (B, num_query, C) - Range Image 提取的查询特征。
        key/value: (num_cams, L, B, C) - 6视角图像特征展平序列（L是所有尺度像素数之和）。
        reference_points_cam: (num_cams, B, num_query, D, 2) - 每个查询投影到各摄像头的参考点坐标（归一化），D为每个查询的参考点数。
        range_image_mask: (num_cams, B, num_query, D) - 掩码，指示每个查询在各摄像头下哪些参考点可见（True为可见）。
        """
        B, num_query, _ = query.shape
        inp_residual = query if residual is None else residual
        # 将查询位置编码与查询特征相加
        if query_pos is not None:
            query = query + query_pos
        # 检查必要参数
        assert reference_points_cam is not None and range_image_mask is not None, "需要提供 reference_points_cam 和 range_image_mask"
        num_cams = self.num_cams
        D = reference_points_cam.shape[3]  # 每个查询的参考点数量
        # 计算每个摄像头可见的查询索引列表
        indexes = []
        for cam in range(num_cams):
            # range_image_mask[cam]: (B, num_query, D)
            mask_per_img = range_image_mask[cam]  # 当前摄像头对应的掩码
            # 利用第一个 batch 样本的可见性确定索引（假设各 batch 相似，可减少开销）
            valid_mask = mask_per_img[0].sum(dim=-1) > 0  # (num_query,) 某查询在该相机有至少一个参考点可见
            index_query_per_img = valid_mask.nonzero(as_tuple=False).squeeze(-1)  # 提取索引
            if index_query_per_img.numel() == 0:
                index_query_per_img = torch.zeros(0, dtype=torch.long, device=query.device)
            indexes.append(index_query_per_img)
        max_len = max(idx.size(0) for idx in indexes) if indexes else 0
        # 为每个相机准备 rebatch 后的 query 和 reference_points
        queries_rebatch = query.new_zeros(B, num_cams, max_len, self.embed_dims)            # (B, num_cams, max_len, C)
        reference_points_rebatch = reference_points_cam.new_zeros(B, num_cams, max_len, D, 2)  # (B, num_cams, max_len, D, 2)
        for j in range(B):
            for i, idx in enumerate(indexes):
                Lq = idx.size(0)
                if Lq == 0:
                    continue
                queries_rebatch[j, i, :Lq] = query[j, idx]  # 取出该相机可见的查询特征
                reference_points_rebatch[j, i, :Lq] = reference_points_cam[i, j, idx]  # 对应的参考点坐标
        # 将 key/value 展开，将摄像头维度合并到 batch 维度
        # key, value: 原始形状 (num_cams, L, B, C)
        num_cams_k, L, B_k, C = key.shape
        assert B_k == B, "Key/Value 的 batch 维度应与查询匹配"
        # 调整形状：先 permute 到 (B, num_cams, L, C)，再 reshape 合并摄像头维度
        key_flat = key.permute(2, 0, 1, 3).reshape(B * num_cams, L, C)     # (B*num_cams, L, C)
        value_flat = value.permute(2, 0, 1, 3).reshape(B * num_cams, L, C) # 同上
        query_flat = queries_rebatch.view(B * num_cams, max_len, self.embed_dims)  # (B*num_cams, max_len, C)
        ref_points_flat = reference_points_rebatch.view(B * num_cams, max_len, D, 2)  # (B*num_cams, max_len, D, 2)
        # 通过可变形注意力模块获取融合特征
        # spatial_shapes 和 level_start_index 对每个摄像头都是相同的（假设摄像头特征图尺寸一致）
        out = self.deformable_attention(query_flat, key_flat, value_flat,
                                        reference_points=ref_points_flat,
                                        spatial_shapes=spatial_shapes,
                                        level_start_index=level_start_index)
        # out: (B * num_cams, max_len, C)，调整回原始维度
        out = out.view(B, num_cams, max_len, self.embed_dims)  # (B, num_cams, max_len, C)
        # 将每个摄像头的贡献累加回对应的查询位置
        slots = query.new_zeros(B, num_query, self.embed_dims)  # 初始化输出槽
        for j in range(B):
            for i, idx in enumerate(indexes):
                Lq = idx.size(0)
                if Lq == 0:
                    continue
                slots[j, idx] += out[j, i, :Lq]  # 将第 i 个摄像头对这些查询的注意力输出累加
        # 计算每个查询有多少摄像头提供了有效特征，用于平均
        count = (range_image_mask.sum(dim=-1) > 0).permute(1, 2, 0).sum(dim=-1).clamp(min=1)  # (B, num_query)
        slots = slots / count.unsqueeze(-1)  # 对累加结果取平均
        # 通过输出投影并添加残差连接
        slots = self.output_proj(slots)
        slots = self.dropout(slots)
        return slots + inp_residual  # 返回融合了图像信息的查询特征

class RangeImageEncoder(nn.Module):
    """将 Range Image 特征与多摄像头图像特征融合的编码器。"""
    def __init__(self, embed_dims=128, num_cams=6, num_layers=6, num_levels=1, 
                 num_points=8, pc_range=None):
        super(RangeImageEncoder, self).__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_layers = num_layers
        self.pc_range = pc_range
        # 堆叠多个 SpatialCrossAttention 层
        self.attn_layers = nn.ModuleList([
            SpatialCrossAttention(embed_dims=embed_dims, num_cams=num_cams, dropout=0.1, 
                                   pc_range=pc_range, num_levels=num_levels, num_points=num_points)
            for _ in range(num_layers)
        ])
       
    
    def forward(self, range_features, img_feats, sensor_metas, range_image_h, range_image_w,
                feature_lvl, reference_points_cam, range_image_mask):
        """
        range_features: Range Image 提取的特征 (B, N_query, C) 或 (B, C, H_r, W_r)。
        img_feats: 包含多尺度多摄像头图像特征的列表，每个元素形状为 (B, num_cams, C, H, W)，已投影到 embed_dims 通道。
        reference_points_cam: (num_cams, B, N_query, D, 2) - 每个 Range Image 查询在各相机图像中的归一化参考点。
        range_image_mask: (num_cams, B, N_query, D) - 指示参考点是否在相机视野内的掩码。
        """
        bs = img_feats[0].shape[0]
        range_features = range_features.unsqueeze(0).repeat(bs, 1, 1) # B, N_query, C
        B, N_query, C = range_features.shape

        # 展平每个尺度的图像特征并拼接
        spatial_shapes = []
        flattened_feats = []
        total_length = 0
        for lvl_feat in img_feats:
            # lvl_feat: (B, num_cams, C_feat, H, W)
            B_i, num_cams_i, C_feat, H_feat, W_feat = lvl_feat.shape
            assert B_i == B and num_cams_i == self.num_cams
            spatial_shapes.append((H_feat, W_feat))
            # 展平该尺度特征
            feat_flat = lvl_feat.view(B, num_cams_i, C_feat, H_feat * W_feat)
            feat_flat = feat_flat.permute(1, 3, 0, 2)  # (num_cams, L_lvl, B, C_feat)
            flattened_feats.append(feat_flat)
            total_length += H_feat * W_feat

        # 将各尺度展平特征在长度维度上拼接
        # key/value 形状: (num_cams, total_length, B, C)
        key = torch.cat(flattened_feats, dim=1)
        value = key  # key 和 value 使用同一特征

        # 构建 spatial_shapes 和 level_start_index 张量
        spatial_shapes_tensor = torch.tensor(
            spatial_shapes, dtype=torch.long, device=range_features.device
        )  # (num_levels, 2)
        level_start_index = [0]
        for (h, w) in spatial_shapes:
            level_start_index.append(level_start_index[-1] + h * w)
        level_start_index = torch.tensor(
            level_start_index[:-1], dtype=torch.long, device=range_features.device
        )

        H_r, W_r = range_image_h, range_image_w  # Range Image 的高度和宽度
        # 生成参考点
        reference_points = self.get_reference_points(
            H=H_r,
            W=W_r,
            max_range=self.pc_range,
            num_points_in_ray=8,
            fov_up=3.0,               # 制作 range image 时的参数
            fov_down=-25.0,           # 制作 range image 时的参数
            bs=bs,                    # 批次大小
            device='cuda',
            dtype=torch.float
        )
        _, _, _, h_feat, w_feat = img_feats[feature_lvl].shape
        reference_points_cam, cam_mask = self.point_sampling(
            reference_points=reference_points,
            sensor_metas=sensor_metas,
            orig_img_size=(1600, 900),
            crop_size=(160, 320),
            feat_size=(h_feat, w_feat)
        )

        # 依次通过每一层 SpatialCrossAttention 进行特征融合
        x = range_features  # (B, N_query, C)
        for layer in self.attn_layers:
            x = layer(
                query=x, key=key, value=value, query_pos=None,
                reference_points_cam=reference_points_cam,
                spatial_shapes=spatial_shapes_tensor,
                level_start_index=level_start_index,
                range_image_mask=cam_mask
            )

        x = x.view(B, H_r, W_r, C).permute(0, 3, 1, 2)
        return x

    # 生成参考点，用于生成 range image 到 6环视图的 query，需要进行几何变换
    @staticmethod
    def get_reference_points(H, W, max_range=70.0, num_points_in_ray=8,
                             fov_up=3.0, fov_down=-25.0, bs=1,
                             device='cuda', dtype=torch.float):
        """
        生成用于 SCA/TSA 的参考点，这里参考点基于 range image 投影。
        Args:
            H, W: range image 的高度和宽度。
            max_range: 激光雷达测量的最大范围（单位与点云一致）。
            num_points_in_ray: 每个像素在射线上均匀采样的点数。
            fov_up, fov_down: 垂直视场角（度）。
            bs: 批次大小。
            device, dtype: 设备和数据类型。
        Returns:
            返回形状为 (bs, num_points_in_ray, H*W, 3) 的 3D 参考点张量；

        """

        # 沿深度方向均匀采样候选点 (单位：实际距离)
        rs = torch.linspace(0.5, max_range - 0.5, num_points_in_ray, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(num_points_in_ray, H, W)

        # 对 range image 的每个像素生成横向和纵向坐标
        us = torch.linspace(0.5, W - 0.5, W, dtype=dtype, 
                            device=device).view(1, 1, W).expand(num_points_in_ray, H, W)
        vs = torch.linspace(0.5, H - 0.5, H, dtype=dtype, 
                            device=device).view(1, H, 1).expand(num_points_in_ray, H, W)

        zoom_factor = 1920 / W
        us = us * zoom_factor
        vs = vs * zoom_factor

        # 将 (us, vs, rs) 根据制作 range image 时的参数，投影到3D坐标系
        ref_3d = inverse_range_projection(u=us.cpu().numpy(), v=vs.cpu().numpy(), depth=rs.cpu().numpy())

        # 将 ref_3d 转换为 tensor
        ref_3d = torch.tensor(ref_3d, dtype=dtype, device=device)
        
        # 将 ref_3d 的形状调整为 (bs, 3, num_points_in_ray, H, W)
        ref_3d = ref_3d.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # (bs, 3, num_points_in_ray, H, W)

        # 将 ref_3d 的维度调整为 (bs, num_points_in_ray, H*W, 3)
        ref_3d = ref_3d.permute(0, 2, 3, 4, 1).flatten(2,3)

        return ref_3d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'sensor_metas'))
    def point_sampling(self, reference_points, sensor_metas, orig_img_size, crop_size, feat_size):
        # 关闭 TF32 加速以确保数值精度
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        device = reference_points.device

        # -------------------------------------
        # 1. LiDAR 坐标系 -> Ego vehicle 坐标系
        # -------------------------------------
        lidar_metas = sensor_metas['lidar']

        # LiDAR 的标定信息：从 LiDAR 传感器到 Ego 的变换
        lidar_calib = lidar_metas['calibrated_sensor']
        # 将 translation 和 rotation 从 CPU 转为 GPU 上的 torch.tensor
        lidar_calib_translation = torch.tensor(
            [t[0].item() if t.numel() > 1 else t.item() for t in lidar_calib['translation']],
            dtype=torch.float32, device=device
        )  # shape (3,)
        # 利用 pyquaternion 计算旋转矩阵，再转换为 torch.tensor
        lidar_calib_rotation = Quaternion([t[0].item() if t.numel() > 1 else t.item() for t in lidar_calib['rotation']]).rotation_matrix
        rotation_lidar2ego = torch.tensor(lidar_calib_rotation, dtype=torch.float32, device=device)  # (3,3)

        # --------------------------------------
        # 2. Ego vehicle 坐标系 -> Global 坐标系
        # --------------------------------------
        lidar_pose = lidar_metas['ego_pose']
        lidar_pose_translation = torch.tensor(
            [t[0].item() if t.numel() > 1 else t.item() for t in lidar_pose['translation']],
            dtype=torch.float32, device=device
        )  # shape (3,)
        lidar_pose_rotation = Quaternion([t[0].item() if t.numel() > 1 else t.item() for t in lidar_pose['rotation']]).rotation_matrix
        rotation_ego2global = torch.tensor(lidar_pose_rotation, dtype=torch.float32, device=device)  # (3,3)

        bs, num_points_in_ray, num_query, _ = reference_points.shape
        # 将参考点 reshape 为 (bs, N, 3)，其中 N = num_points_in_ray * num_query
        pc = reference_points.reshape(bs, -1, 3)  # (bs, N, 3)

        # LiDAR 传感器坐标系到 Ego 坐标系
        pc = torch.matmul(pc, rotation_lidar2ego.T) + lidar_calib_translation
        # Ego 坐标系到 Global 坐标系
        pc = torch.matmul(pc, rotation_ego2global.T) + lidar_pose_translation

        # -----------------------------
        # 4. Global 坐标系 -> Camera 坐标系
        # -----------------------------
        camera_metas = sensor_metas['camera']
        # 构建 (num_cam, bs, num_query, num_points_in_ray, 2) 的 tensor: reference_points_cam
        reference_points_cam = torch.zeros(
            (len(camera_metas['calibrated_sensor']), bs, num_query, num_points_in_ray, 2),
            dtype=torch.float32, device=device
        )
        cam_mask = torch.zeros(
            (len(camera_metas['calibrated_sensor']), bs, num_query, num_points_in_ray),
            dtype=torch.bool, device=device
        )

        pc_global = pc.clone()  # 保存 global 坐标系下的点云

        channel_idx = 0
        for channel,cam_calib in camera_metas['calibrated_sensor'].items():
            cam_pose = camera_metas['ego_pose'][channel]
            cam_pose_translation = torch.tensor(
                [t[0].item() if t.numel() > 1 else t.item() for t in cam_pose['translation']],
                dtype=torch.float32, device=device
            )
            cam_pose_rotation = Quaternion([t[0].item() if t.numel() > 1 else t.item() for t in cam_pose['rotation']]).rotation_matrix
            # 注意：转换 global -> ego_cam 时，需要使用逆变换，因此通常使用平移取反，旋转矩阵取转置
            rotation_global2ego_cam = torch.tensor(cam_pose_rotation, dtype=torch.float32, device=device).T
            translation_global2ego_cam = -cam_pose_translation

            # 应用 global -> ego_cam 的变换
            # 先平移，再旋转
            pc = pc_global.clone()  # 恢复 global 坐标系下的点云
            pc = torch.matmul(pc + translation_global2ego_cam, rotation_global2ego_cam.T)
            
            cam_calib_translation = torch.tensor(
                [t[0].item() if t.numel() > 1 else t.item() for t in cam_calib['translation']],
                dtype=torch.float32, device=device
            )
            cam_calib_rotation = Quaternion([t[0].item() if t.numel() > 1 else t.item() for t in cam_calib['rotation']]).rotation_matrix
            # 注意：转换 ego_cam -> cam 时，需要使用逆变换，因此通常使用平移取反，旋转矩阵取转置
            rotation_ego2cam = torch.tensor(cam_calib_rotation, dtype=torch.float32, device=device).T
            translation_ego2cam = -cam_calib_translation

            pc = torch.matmul(pc + translation_ego2cam, rotation_ego2cam.T)

            # -------------------------------------
            # 4. 将点云从相机坐标系投影到图像平面
            # -------------------------------------
            # view_points_gpu 接受 shape 为 (3, n) 的点，因此需要对 batch 内每个样本分别处理
            camera_intrinsic = torch.tensor(
                [[t[0].item() if t.numel() > 1 else t.item() for t in row]
                for row in cam_calib['camera_intrinsic']],
                dtype=torch.float64
            )
            proj_points_list = []
            valid_mask_list = []
            for i in range(bs):
                # 转置为 (3, N)
                points_cam = pc[i].T
                # 使用透视投影：normalize=True 会除以第三行，得到 (3, N) 的齐次投影结果
                proj = view_points_gpu(points_cam, camera_intrinsic, normalize=True)
                # 取前两行，即 (u, v) 坐标；再转置回 (N, 2)
                proj_uv = proj[:2, :].T

                # 上述投影操作是在原图尺寸下进行的
                # 因此需要将投影坐标缩放到模型输入尺寸 - 进一步到特征图尺寸
                resize_origin2input = crop_size[1] / orig_img_size[1]
                proj_uv[:, 0] *= resize_origin2input
                resize_input2feat = feat_size[1] / crop_size[1]
                proj_uv[:, 0] *= resize_input2feat

                # 计算有效点的掩码
                valid_mask = (proj_uv[:, 0] >= 0) & (proj_uv[:, 0] < feat_size[1]) & \
                            (proj_uv[:, 1] >= 0) & (proj_uv[:, 1] < feat_size[0])
                proj_points_list.append(proj_uv)
                valid_mask_list.append(valid_mask)
            # 堆叠得到 (bs, N, 2)
            proj_points = torch.stack(proj_points_list, dim=0)
            valid_mask = torch.stack(valid_mask_list, dim=0)  # (bs, N)
            
            # 根据原始点结构恢复 (bs, num_query, num_points_in_ray, 2)
            proj_points = proj_points.reshape(bs, num_query, num_points_in_ray, 2)
            # 将有效掩码 reshape 为 (bs, num_query, num_points_in_ray)
            valid_mask = valid_mask.reshape(bs, num_query, num_points_in_ray)

            # 使用不同 channel 的 proj_points 为 reference_points_cam 赋值
            reference_points_cam[channel_idx] = proj_points
            cam_mask[channel_idx] = valid_mask
            channel_idx += 1    

        # 恢复 TF32 配置
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, cam_mask


def view_points_gpu(points: torch.Tensor, view: torch.Tensor, normalize: bool) -> torch.Tensor:
    """
    使用 GPU 上的 torch tensor 将 3D 点投影到 2D 平面上。

    Args:
        points: 形状为 (3, n) 的 tensor，每个点为 (x, y, z)
        view: 投影矩阵，形状可以是 (3, 3) 或 (3, 4)
        normalize: 是否归一化第三个坐标（透视投影时设为 True）

    Returns:
        投影后的点，形状为 (3, n)。当 normalize=True 时，第三个坐标被归一化。
    """
    # 确保 view 的维度不超过 4x4，points 为 (3, n)
    assert view.shape[0] <= 4 and view.shape[1] <= 4, "view 矩阵维度应不超过4x4"
    assert points.shape[0] == 3, "points 应为形状 (3, n) 的 tensor"

    # 获取 points 所在设备与数据类型
    device = points.device
    dtype = points.dtype

    # 构造 4x4 的单位矩阵，并将 view 矩阵填充到对应的子矩阵中
    viewpad = torch.eye(4, device=device, dtype=dtype)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # 将 points 扩展为齐次坐标形式，形状 (4, n)
    ones = torch.ones((1, nbr_points), device=device, dtype=dtype)
    points_hom = torch.cat([points, ones], dim=0)

    # 投影：矩阵乘法
    points_proj = torch.matmul(viewpad, points_hom)  # 形状 (4, n)
    points_proj = points_proj[:3, :]  # 取前3行

    if normalize:
        # 防止除以 0，加上一个微小值
        eps = 1e-6
        points_proj = points_proj / (points_proj[2:3, :] + eps).expand_as(points_proj)

    return points_proj
