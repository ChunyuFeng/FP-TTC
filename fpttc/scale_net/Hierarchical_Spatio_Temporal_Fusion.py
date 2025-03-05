import torch
import torch.nn as nn

# 假设我们有 MultiScaleDeformableAttention 模块（例如来自 mmcv 或自定义实现）
# 其构造参数包括嵌入维度、尺度数、注意力头数、每头采样点数等
# forward 接受: query特征、参考点、多尺度键和值特征等，输出融合后的特征。
from mmcv.ops import MultiScaleDeformableAttention  # 如果使用mmcv实现

class Hierarchical_Spatio_Temporal_Fusion(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_scales=2,
                 n_heads=8):
        super(Hierarchical_Spatio_Temporal_Fusion, self).__init__()
        # 多尺度可变形注意力模块：一个用于空间多相机融合，一个用于时间跨帧融合
        self.spatial_attn = MultiScaleDeformableAttention(embed_dims=embed_dim,
                                                          num_levels=num_scales,
                                                          num_heads=n_heads,
                                                          num_points=8,
                                                          batch_first=True)
        self.temporal_attn = MultiScaleDeformableAttention(embed_dims=embed_dim,
                                                           num_levels=num_scales,
                                                           num_heads=n_heads,
                                                           num_points=8,
                                                           batch_first=True)
        # 可选：特征尺度融合层（如卷积）将多尺度特征合并为单一输出
        self.fuse_conv = nn.Conv2d(embed_dim * num_scales, embed_dim, kernel_size=1)  # 将多尺度特征在通道维度融合

    def forward(self, prev_multi_scale_feats, curr_multi_scale_feats):
        """
        prev_feature_list 和 curr_feature_list 是列表，每个包含6个视角的特征，
        每个视角又是一个包含 num_scales 个尺度特征图的列表。
        例如：prev_feature_list[c][s] 表示前一帧第c个相机的第s尺度特征，shape=(B, C, H, W)。
        """

        B, C, _, _ = curr_multi_scale_feats[0].shape
        # num_cams = len(curr_feature_list)          # =6
        # num_scales = len(curr_feature_list[0])     # =2

        ### 1. 空间维度多相机融合 ###
        # 将每帧6个相机的多尺度特征准备为MSDA的输入
        # # 我们先把6个相机的特征在宽度上拼接，得到每个尺度上拼接后的特征图 (B, C, H, 6W)
        # curr_multi_scale_feats = []
        # prev_multi_scale_feats = []
        # for s in range(num_scales):
        #     # 在第s尺度上将6个相机特征在width维度拼接
        #     curr_level_feat = torch.cat([curr_feature_list[cam][s] for cam in range(num_cams)], dim=3)  # shape: (B, C, H, 6W)
        #     prev_level_feat = torch.cat([prev_feature_list[cam][s] for cam in range(num_cams)], dim=3)  # shape: (B, C, H, 6W)
        #     curr_multi_scale_feats.append(curr_level_feat)
        #     prev_multi_scale_feats.append(prev_level_feat)
        # 准备 Multi-Scale Deformable Attention 所需的 reference points 和 shape 信息
        # 假设我们创建每个尺度的空间尺寸列表和每个尺度起始索引，用于MSDA内部计算
        spatial_shapes = []       # 存储每个尺度 (H, width)；例如[(H, 6W), (H, 6W)] 如果两个尺度尺寸相同
        level_start_index = [0]   # 每个尺度flatten起始索引
        total_len = 0
        for lvl_feat in curr_multi_scale_feats:
            _, _, H_lvl, W_lvl = lvl_feat.shape
            spatial_shapes.append((H_lvl, W_lvl))
            total_len += H_lvl * W_lvl
            level_start_index.append(total_len)  # 累计长度作为下一个level的起始
        level_start_index = level_start_index[:-1]  # 最后一项多余

        # 生成参考点坐标 (normalized coordinates)，shape:(B, total_len, num_levels, 2)，用于指示查询的初始采样位置
        # 简化起见，这里假设已经有函数生成参考点网格
        ref_points = self._get_reference_points(spatial_shapes, B, device=curr_multi_scale_feats[0].device)  # 自定义函数，生成每个query位置对应的归一化坐标

        # 将多尺度特征展开为 (B, total_len, C) 供注意力模块使用
        curr_flatten_feats = torch.cat([feat.view(B, C, -1) for feat in curr_multi_scale_feats], dim=2)  # (B, C, total_len)
        curr_flatten_feats = curr_flatten_feats.transpose(1, 2)  # (B, total_len, C)
        prev_flatten_feats = torch.cat([feat.view(B, C, -1) for feat in prev_multi_scale_feats], dim=2)
        prev_flatten_feats = prev_flatten_feats.transpose(1, 2)  # (B, total_len, C)

        # 对当前帧执行多相机融合的自注意力（查询和值都来自当前帧多尺度特征）
        fused_curr_flatten = self.spatial_attn(query=curr_flatten_feats, value=curr_flatten_feats,
                                               key_padding_mask=None, reference_points=ref_points,
                                               spatial_shapes=torch.as_tensor(spatial_shapes, device=curr_flatten_feats.device),
                                               level_start_index=torch.as_tensor(level_start_index, device=curr_flatten_feats.device))
        # 对前一帧执行同样的多相机融合
        fused_prev_flatten = self.spatial_attn(query=prev_flatten_feats, value=prev_flatten_feats,
                                               key_padding_mask=None, reference_points=ref_points,
                                               spatial_shapes=torch.as_tensor(spatial_shapes, device=prev_flatten_feats.device),
                                               level_start_index=torch.as_tensor(level_start_index, device=prev_flatten_feats.device))
        # 将融合后的扁平特征恢复为每个尺度的特征图形状
        fused_curr_levels = []
        fused_prev_levels = []
        for idx, (H_lvl, W_lvl) in enumerate(spatial_shapes):
            start = level_start_index[idx]
            length = H_lvl * W_lvl
            # 提取对应尺度的flatten片段并reshape回原形状
            fused_curr_level = fused_curr_flatten[:, start:start+length, :].transpose(1, 2).reshape(B, C, H_lvl, W_lvl)
            fused_prev_level = fused_prev_flatten[:, start:start+length, :].transpose(1, 2).reshape(B, C, H_lvl, W_lvl)
            fused_curr_levels.append(fused_curr_level)
            fused_prev_levels.append(fused_prev_level)

        ### 2. 时间维度跨帧融合 ###
        # 利用前后帧的融合特征，再次通过MSDA模块进行跨帧的特征交互
        # 将前一帧作为值，当前帧作为查询，捕获跨时间的相关性（然后反过来更新前一帧，使融合对称）
        # 这里我们假设前后帧在空间上对齐（如同一Range Image坐标系），因此可以复用相同的参考点
        # 当前帧查询前一帧（前一帧为KV）
        fused_curr_flatten = torch.cat([feat.view(B, C, -1) for feat in fused_curr_levels], dim=2).transpose(1, 2)  # (B, total_len, C)
        fused_prev_flatten = torch.cat([feat.view(B, C, -1) for feat in fused_prev_levels], dim=2).transpose(1, 2)
        # 当前帧特征查询前帧
        fused_curr_flatten_time = self.temporal_attn(query=fused_curr_flatten, value=fused_prev_flatten,
                                                     key_padding_mask=None, reference_points=ref_points,
                                                     spatial_shapes=torch.as_tensor(spatial_shapes, device=fused_curr_flatten.device),
                                                     level_start_index=torch.as_tensor(level_start_index, device=fused_curr_flatten.device))
        # 前一帧特征查询当前帧（使信息双向流动）
        fused_prev_flatten_time = self.temporal_attn(query=fused_prev_flatten, value=fused_curr_flatten,
                                                     key_padding_mask=None, reference_points=ref_points,
                                                     spatial_shapes=torch.as_tensor(spatial_shapes, device=fused_prev_flatten.device),
                                                     level_start_index=torch.as_tensor(level_start_index, device=fused_prev_flatten.device))
        # 恢复前后帧融合后的多尺度特征图
        fused_curr_levels_time = []
        fused_prev_levels_time = []
        for idx, (H_lvl, W_lvl) in enumerate(spatial_shapes):
            start = level_start_index[idx]
            length = H_lvl * W_lvl
            fused_curr_lvl = fused_curr_flatten_time[:, start:start+length, :].transpose(1, 2).reshape(B, C, H_lvl, W_lvl)
            fused_prev_lvl = fused_prev_flatten_time[:, start:start+length, :].transpose(1, 2).reshape(B, C, H_lvl, W_lvl)
            fused_curr_levels_time.append(fused_curr_lvl)
            fused_prev_levels_time.append(fused_prev_lvl)

        return fused_prev_levels_time, fused_curr_levels_time

        # ### 3. 重组 Range Image 输出 ###
        # # 如有多个尺度输出，我们可以将其融合为单尺度输出；这里通过1x1卷积将两个尺度特征在通道维拼接后压缩
        # # 先将多尺度特征在通道维度拼接: 得到 shape (B, 2*C, H, 6W) 的特征
        # fused_curr_multi_scale = torch.cat([lvl for lvl in fused_curr_levels_time], dim=1)   # (B, 2*C, H, 6W)
        # fused_prev_multi_scale = torch.cat([lvl for lvl in fused_prev_levels_time], dim=1)   # (B, 2*C, H, 6W)
        # # 通过1x1卷积降维到C，融合多尺度信息
        # fused_curr_output = self.fuse_conv(fused_curr_multi_scale)  # (B, C, H, 6W)
        # fused_prev_output = self.fuse_conv(fused_prev_multi_scale)  # (B, C, H, 6W)
        # return fused_prev_output, fused_curr_output

    def _get_reference_points(self, spatial_shapes, batch_size, device):
        """
        根据每个尺度的空间尺寸生成参考点张量 (B, total_len, 2)，
        假设参考点是每个特征图像素归一化到[0,1]的坐标 (y, x)。
        """
        # 计算每个尺度上的网格坐标并归一化
        ref_points_list = []
        for (H, W) in spatial_shapes:
            # grid_y shape: (H, W), range [0, H-1], grid_x shape: (H, W), range [0, W-1]
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H-1, H, device=device),
                                            torch.linspace(0, W-1, W, device=device))
            # 归一化到[0,1]
            ref_y = grid_y / (H - 1) if H > 1 else grid_y  # (H, W)
            ref_x = grid_x / (W - 1) if W > 1 else grid_x
            ref_points_level = torch.stack((ref_x, ref_y), dim=-1)  # (H, W, 2), 格式为(x, y)
            ref_points_list.append(ref_points_level.flatten(0,1))   # 展开为 (H*W, 2)
        ref_points = torch.cat(ref_points_list, dim=0)  # (total_len, 2)
        # 扩展 batch 维度，并增加 levels 维度 (这里每个query在不同level的参考点可设为同一坐标)
        ref_points = ref_points.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, total_len, 2)
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)  # (B, total_len, num_levels, 2)
        return ref_points

class MultiLayerHierarchicalFusion(nn.Module):
    def __init__(self, num_layers, embed_dim, num_scales=2, n_heads=8):
        """
        num_layers: 堆叠的层数
        其他参数与单层 Hierarchical_Spatio_Temporal_Fusion 模块一致
        """
        super(MultiLayerHierarchicalFusion, self).__init__()
        self.layers = nn.ModuleList([
            Hierarchical_Spatio_Temporal_Fusion(embed_dim, num_scales, n_heads)
            for _ in range(num_layers)
        ])

    def forward(self, prev_feature_list, curr_feature_list):
        """
        依次将输入特征传入各层融合模块，每一层输出作为下一层的输入，
        最终返回融合后的前一帧和当前帧特征（结构与输入一致）
        """
        for layer in self.layers:
            prev_feature_list, curr_feature_list = layer(prev_feature_list, curr_feature_list)
        return prev_feature_list, curr_feature_list