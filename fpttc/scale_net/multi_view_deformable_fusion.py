import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import xavier_init, constant_init
from ..modules.position import PositionEmbeddingSine
import math
from .utils.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32

class MultiViewDeformableFusion(nn.Module):
    def __init__(
        self,
        num_layers: int,           # TransformerBlock 层数
        input_dim: int,            # 每尺度原特征通道数（未在本类直接使用，但用于一致性）
        d_model: int,              # Transformer 内部 embedding 维度
        nhead: int,                # 多头注意力头数
        per_level_channels: int,   # 每尺度拼接后通道数 C*num_views，用于 value 投影
        num_scales: int,           # 尺度层数
        num_views: int,            # 视角数
        R: int,                    # 每个 query 在深度方向采样的点数
        range_image_feat_shape: list  # 每尺度的 (H_i, W_i)
    ):
        super().__init__()  # 初始化父类 nn.Module
        # 保存超参数
        self.d_model    = d_model
        self.nhead      = nhead
        self.num_scales = num_scales
        self.num_views  = num_views
        self.R          = R
        self.range_image_feat_shape = range_image_feat_shape

        # 视角嵌入: 为每一路相机和 "invalid"（index==num_views）分配可学习向量
        self.view_embed = nn.Embedding(num_views + 1, d_model)

        # 为每个尺度创建一个可学习的 query feature map，尺寸 (1, d_model, H_i, W_i)
        self.query_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, d_model, H_i, W_i))
            for (H_i, W_i) in range_image_feat_shape
        ])
        # 位置编码，使用正弦位置嵌入，输出 d_model 维度
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        # 将拼接的多视角通道投影到 d_model，用于 deformable attention 的 value
        self.value_proj = nn.Conv1d(per_level_channels, d_model, kernel_size=1)

        # 预留的 R 维度聚合层（本版本未启用）
        self.ref_agg = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=R, bias=False)

        # 构建 num_layers 个 TransformerBlock，每个实现可变形注意力
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_head=nhead,
                num_points=R,
                num_level=num_scales
            ) for _ in range(num_layers)
        ])

        # 对所有参数做 Xavier 初始化
        for p in self.parameters():
            if p.dim() > 1:
                xavier_init(p, distribution='uniform')

    def forward(
        self,
        mv_feats: list,          # List[num_scales] of [B, C*num_views, H_i, W_i]
        uv_map: list,            # List[num_scales] of [B, H0, W0, R, 2]
        cam_idx_map: list,       # List[num_scales] of [B, H0, W0, R]
        valid_mask: list,        # List[num_scales] of [B, H0, W0, R]
        augmentor_affine: torch.Tensor  # [B, 3, 3] 仿射增强矩阵
    ):
        B = mv_feats[0].shape[0]  # batch size

        # 1) 准备 value：拼接所有尺度特征并投影到 d_model
        feat_flats = []
        spatial_shapes = []
        for lvl, feat in enumerate(mv_feats):
            _, C, Hi, Wi = feat.shape
            spatial_shapes.append((Hi, Wi))
            feat_flats.append(feat.flatten(2))  # [B, C, Hi*Wi]
        # 在空间维度上拼接多尺度
        feat_cat = torch.cat(feat_flats, dim=2)  # [B, C_total, S]
        # 投影到 d_model
        value = self.value_proj(feat_cat)        # [B, d_model, S]
        # 调整维度顺序到 [B, S, nhead, d_model]
        value = value.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, self.nhead, 1)

        # 构建 spatial_shapes 张量和级别偏移索引
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=value.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))

        fused_feats = []
        # 对每个尺度做处理
        for lvl in range(self.num_scales):
            H_i, W_i = self.range_image_feat_shape[lvl]
            if lvl == 0:
                # 2) Query 构造 + 位置编码
                query = self.query_embeds[lvl].repeat(B, 1, 1, 1)  # [B, d_model, H0, W0]
                query = query + self.pos_enc(query)
                query_flat = query.flatten(2).permute(0, 2, 1)    # [B, Nq, d_model]

                # 3) 生成参考点、视角索引及有效掩码
                ref, cam_idx_q, valid_q = self.get_reference_points_per_scale(
                    uv_map[lvl], augmentor_affine, mv_feats,
                    lvl, H_i, W_i,
                    cam_idx_map[lvl], valid_mask[lvl]
                )  # ref: [B, Nq, S, R, 2]

                # 4) 将视角嵌入加到 query 上
                view_emb = self.view_embed(cam_idx_q)            # [B, Nq, d_model]
                query_flat = query_flat + view_emb

                # 5) 多层 Deformable Attention
                for layer in self.layers:
                    query_flat = layer(
                        query_flat, value,
                        height=H_i, width=W_i,
                        query_location=ref,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index
                    )
                # 重塑为 [B, d_model, H0, W0]
                fused = query_flat.permute(0, 2, 1).view(B, self.d_model, H_i, W_i)
            else:
                # 对高分辨率尺度，直接对 lvl=0 的结果做上采样
                fused_0 = fused_feats[0]
                fused = F.interpolate(
                    fused_0,
                    size=(H_i, W_i),
                    mode='bilinear',
                    align_corners=True
                )
            fused_feats.append(fused)
        return fused_feats

    def get_reference_points_per_scale(
        self,
        uv: torch.Tensor,            # [B, H0, W0, R, 2] 原始投影坐标
        A: torch.Tensor,             # [B, 3, 3] 仿射增强矩阵
        mv_feats: list,
        lvl: int,
        H_i: int,
        W_i: int,
        cam_idx: torch.Tensor,       # [B, H0, W0, R]
        valid_mask: torch.Tensor     # [B, H0, W0, R]
    ):
        """
        生成参考点坐标，并返回每个 query 对应的 cam_idx 和有效像素掩码。
        返回:
          ref_q: [B, H_i*W_i, num_scales, R, 2]
          cam_idx_q: [B, H_i*W_i]  每个像素点对应的相机索引
          valid_q: [B, H_i*W_i]    像素点是否在任一深度上可见
        """
        B, H0, W0, R, _ = uv.shape
        N0 = H0 * W0 * R

        # (1) 仿射变换
        uv_flat = uv.view(B, N0, 2)
        ones = uv_flat.new_ones((B, N0, 1))
        hom = torch.cat([uv_flat, ones], dim=-1)         # [B, N0, 3]
        hom_t = torch.matmul(hom, A.transpose(1, 2))    # [B, N0, 3]
        uv_t = hom_t[..., :2]                           # [B, N0, 2]

        # (2) 下采样到 feature map 尺度
        uv_lvl = uv_t.clone()
        uv_lvl[..., 0] /= (W0 / W_i)
        uv_lvl[..., 1] /= (H0 / H_i)

        # (3) 归一化到 [-1,1]
        x_norm = (uv_lvl[..., 0] / (W_i - 1)) * 2 - 1
        y_norm = (uv_lvl[..., 1] / (H_i - 1)) * 2 - 1
        ref_flat = torch.stack([x_norm, y_norm], dim=-1)  # [B, N0, 2]

        # (4) 合并 cam_idx 和 valid_mask，并将无效所有深度像素标记
        cam_flat = cam_idx.view(B, H0 * W0, R)           # [B, H0*W0, R]
        valid_flat = valid_mask.view(B, H0 * W0, R)       # [B, H0*W0, R]
        # 对每个像素，统计首个深度可见的相机索引 (或其他策略)
        # 这里选择最小 r 索引处的 cam_idx，如果所有 r 都无效，标记为 invalid_index
        invalid_index = self.num_views
        cam_pixel = torch.full((B, H0 * W0), invalid_index, device=cam_flat.device, dtype=cam_flat.dtype)
        valid_pixel = valid_flat.any(dim=2)               # [B, H0*W0]
        # 找到每个像素第一个 valid 深度
        first_idx = valid_flat.float().argmax(dim=2)      # [B, H0*W0]
        cam_pixel[valid_pixel] = cam_flat[valid_pixel, first_idx[valid_pixel]]
        # mask 横坐标超出
        # 对所有无效像素，把对应的所有 depth 坐标置越界
        ref_flat_pixel = ref_flat.view(B, H0 * W0, R, 2)
        ref_flat_pixel[~valid_flat] = -2.0

        # (5) 取前像素点并 reshape 参考点
        Nq0 = H_i * W_i
        # 取像素维度数据
        ref_pixel = ref_flat_pixel[..., :Nq0, :].view(B, Nq0, R, 2)
        cam_idx_q = cam_pixel[:, :Nq0]                   # [B, Nq0]
        valid_q = valid_pixel[:, :Nq0]                   # [B, Nq0]
        # 扩展尺度维度
        ref_q = ref_pixel.unsqueeze(2).repeat(1, 1, self.num_scales, 1, 1)  # [B, Nq0, S, R, 2]
        return ref_q, cam_idx_q, valid_q
    

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
        level_start_index: [2]
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
       
        # 1) 在第 2 维（head 维）插入一个维度，然后 expand 到 num_head
        reference_points = query_location.unsqueeze(2)   # [B, Q, 1, L, P, 2]
        reference_points = reference_points.expand(
            -1, -1, self.num_head, -1, -1, -1
        )                                                 # [B, Q, H, L, P, 2]

        # 2) 计算最终采样位置
        sampling_locations = reference_points \
            + sampling_offsets / offset_normalizer[None, None, :, None, None, :]
        # sampling_locations = query_location[:, :, None, :, None, :] \
        #     + sampling_offsets \
        #     / offset_normalizer[None, :, None, :]

        output = self.MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,attention_weights)
        #print(self.num_level, self.num_points)
        #print(self.num_head, output.shape)
        
        # output: (bs, num_query, c)
        output = self.output_proj(output)

        return self.dropout(output)
