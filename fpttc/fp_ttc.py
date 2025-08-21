import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from utils.loss import get_loss_scale_map, get_loss_risk_score_map

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet
from .scale_net.utils.spherical_ops import gaussian_splat_to_sphere
from .scale_net.spherical_align import SphericalAlignment

import torch.distributed as dist
import numpy as np
from utils.dist import is_main_process

import matplotlib.pyplot as plt

class CorrEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CorrEncoder, self).__init__()
        self.convc1 = nn.Conv2d(dim_in, 256, 1)
        self.convc2 = nn.Conv2d(256, dim_out, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.convc1(x))
        x = F.relu(self.convc2(x))
        return x

class FpTTC(nn.Module):
    def __init__(self,
                 num_scales             = 2,
                 feature_channels       = 128,
                 upsample_factor        = 4,
                 num_head               = 1,
                 ffn_dim_expansion      = 4,
                 num_transformer_layers = 6,
                 reg_refine             = False
                 ):
        super(FpTTC, self).__init__()
        self.num_scales = num_scales

        self.camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

        self.cnet = CNNEncoder(output_dim        = feature_channels, 
                               num_output_scales = num_scales)
        
        self.featnet = FeatureNet(num_scales             = num_scales,
                                  feature_channels       = feature_channels,
                                  num_head               = num_head, 
                                  ffn_dim_expansion      = ffn_dim_expansion,
                                  num_transformer_layers = num_transformer_layers)   
         
        self.corrnet = FlowNet(num_scales       = num_scales,
                               feature_channels = feature_channels,
                               upsample_factor  = upsample_factor,
                               reg_refine       = reg_refine) 
        
        self.conv_corr = CorrEncoder(dim_in  = 2,
                                     dim_out = feature_channels+1)
        self.scale_net = ScaleNet(num_scales      = num_scales,
                                 feature_channels = feature_channels,
                                 upsample_factor  = upsample_factor,
                                 num_head         = 4,
                                 scale_level      = num_scales, 
                                 reg_refine       = reg_refine, 
                                 head_type        = 'scale')

        # Risk 分支
        self.conv_corr_risk = CorrEncoder(dim_in  = 2,
                                          dim_out = feature_channels+1)
        self.risk_net = ScaleNet(num_scales       = num_scales,
                                 feature_channels = feature_channels,
                                 upsample_factor  = upsample_factor,
                                 num_head         = 4,
                                 scale_level      = num_scales, 
                                 reg_refine       = reg_refine, 
                                 head_type        = 'risk')
        
        # 球面对齐模块
        self.sph_align = SphericalAlignment(in_ch=feature_channels, hidden=128, max_shift=2.0)
        
    def forward(self,
                img_prev, img_curr,
                depth_prev, depth_curr,
                proj_pix_prev, proj_pix_curr,
                proj_xy_prev, proj_xy_curr,
                attn_type,
                attn_splits_list,
                corr_radius_list,
                prop_radius_list,
                num_reg_refine,
                scale_only):

        ### 1）提取输入的多视角图像的底层特征
        img0, img1 = normalize_img(img_prev, img_curr)
        # rgbd0 = torch.cat([img0, depth_prev], dim=2)  # [B, V, 4, H, W]
        # rgbd1 = torch.cat([img1, depth_curr], dim=2)  # [B, V, 4, H, W]
        B, V, C, H_img, W_img = img0.shape
        shared_prev, shared_curr = [], []
        for view in range(img0.size(1)):
            p, c = self.extract_feature(img0[:, view], img1[:, view], branch=None)
            shared_prev.append(p)
            shared_curr.append(c)

        corr_features = []
        multi_level_feats_prev = []
        multi_level_feats_curr = []

        for prev_lvls, curr_lvls in zip(shared_prev, shared_curr):
            corr = None
            lvl_feats_prev = []
            lvl_feats_curr = []
            # 多尺度特征融合与相关性计算
            for lvl in range(self.num_scales):
                prev_feat = prev_lvls[lvl]
                curr_feat = curr_lvls[lvl]
                # 1. 前后帧特征融合
                fused_prev, fused_curr = self.featnet(
                    prev_feat, curr_feat,
                    lvl, attn_type, attn_splits_list,
                    corr
                )
                lvl_feats_prev.append(fused_prev)
                lvl_feats_curr.append(fused_curr)
                # 2. 计算前后帧特征图的相关性
                corr, _ = self.corrnet(
                    fused_prev, fused_curr,
                    lvl, corr_radius_list, prop_radius_list,
                    num_reg_refine, False, corr
                )
                # 3. 如果不是最后一个尺度，则将 correlation map 上采样
                if lvl < self.num_scales - 1:
                    corr = F.interpolate(
                        corr, scale_factor=2,
                        mode='bilinear', align_corners=True
                    ) * 2
            corr_features.append(corr)
            multi_level_feats_prev.append(lvl_feats_prev)
            multi_level_feats_curr.append(lvl_feats_curr)
        
        # === (A) corr 特征 → range（用 curr 的映射） ===
        # 注意：这里需要 Hr/Wr 做到与 proj_xy_* 一致；proj_xy_curr 的 shape 即可拿到 Hr/Wr
        Hr, Wr = proj_xy_curr.shape[1], proj_xy_curr.shape[2]
        corr_range = gaussian_splat_to_sphere(
            corr_features,                     # list(V) of [B,Cc,Hf,Wf]
            proj_pix_curr, proj_xy_curr,       # [B,Hr,Wr,3], [B,Hr,Wr,2]
            H_img=160, W_img=320,              # <- 你原先的标定缩放因子，保持一致
            wrap_horizontal=True
        )   # [B,Cc,Hr,Wr]

        # === (B) multi-level 特征 → range，并做“球面对齐” ===
        multi_level_ranges_prev = []
        multi_level_ranges_curr = []
        for lvl in range(self.num_scales):
            scale = 2 ** (self.num_scales - 1 - lvl)
            if scale > 1:
                proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
                proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
                proj_xy_prev_l = proj_xy_prev[:, ::scale, ::scale, :]
                proj_xy_curr_l = proj_xy_curr[:, ::scale, ::scale, :]
            else:
                proj_prev_lvl = proj_pix_prev
                proj_curr_lvl = proj_pix_curr
                proj_xy_prev_l = proj_xy_prev
                proj_xy_curr_l = proj_xy_curr

            prev_feats = [feats[lvl] for feats in multi_level_feats_prev]  # list(V) of [B,C,Hf,Wf]
            curr_feats = [feats[lvl] for feats in multi_level_feats_curr]

            range_prev, conf_prev, _ = gaussian_splat_to_sphere(
                prev_feats, proj_prev_lvl, proj_xy_prev_l,
                H_img=160, W_img=320, wrap_horizontal=True,
                return_conf=True, conf_norm='max', var_alpha=10.0
            )   # [B,C,Hr_l,Wr_l]
            range_curr, conf_curr, _ = gaussian_splat_to_sphere(
                curr_feats, proj_curr_lvl, proj_xy_curr_l,
                H_img=160, W_img=320, wrap_horizontal=True,
                return_conf=True, conf_norm='max', var_alpha=10.0
            )

            # === 球面对齐：把 curr 对齐到 prev ===
            range_prev_aligned, range_curr_aligned, _ = self.sph_align(range_prev, range_curr)

            multi_level_ranges_prev.append(range_prev_aligned)
            multi_level_ranges_curr.append(range_curr_aligned)

        # # 1) 投影 corr 特征到 range-view（risk）
        # corr_range = self.project_views_to_range(
        #     corr_features, proj_pix_curr,
        #     H_img=160, W_img=320
        # )

        # # 2) multi-lvl、multi-view 特征投影
        # multi_level_ranges_prev = []
        # multi_level_ranges_curr = []
        # for lvl in range(self.num_scales):
        #     scale = 2 ** (self.num_scales - 1 - lvl)
        #     if scale > 1:
        #         proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
        #         proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
        #     else:
        #         proj_prev_lvl = proj_pix_prev
        #         proj_curr_lvl = proj_pix_curr

        #     prev_feats = [feats[lvl] for feats in multi_level_feats_prev]
        #     curr_feats = [feats[lvl] for feats in multi_level_feats_curr]

        #     range_prev = self.project_views_to_range(
        #         prev_feats, proj_prev_lvl,
        #         H_img=160, W_img=320
        #     )
        #     range_curr = self.project_views_to_range(
        #         curr_feats, proj_curr_lvl,
        #         H_img=160, W_img=320
        #     )

        #     multi_level_ranges_prev.append(range_prev)
        #     multi_level_ranges_curr.append(range_curr)

        # 3) 编码并预测风险分支输出
        # scale 分支
        corr_encoded_s = self.conv_corr(corr_range)
        initial_scale = F.softplus(corr_encoded_s[:, :1]) + 1e-3
        corr_encoded_s = corr_encoded_s[:, 1:]
        scales = self.scale_net(
            corr_encoded_s,
            multi_level_ranges_prev,
            multi_level_ranges_curr,
            initial_scale
        )

        if scale_only:
            return scales, None, conf_prev, conf_curr
        
        # risk 分支
        corr_encoded = self.conv_corr_risk(corr_range)
        initial_risk = corr_encoded[:, :1]
        corr_encoded = corr_encoded[:, 1:]
        risk_score = self.risk_net(
            corr_encoded,
            multi_level_ranges_prev,
            multi_level_ranges_curr,
            initial_risk
        )

        return scales, risk_score, conf_prev, conf_curr

    def forward_with_loss( 
            self,
            img_prev,                      # Tensor[B, V, 3, H, W]
            img_curr,                      # Tensor[B, V, 3, H, W]
            depth_prev,                    # Tensor[B, V, 1, H, W]
            depth_curr,                    # Tensor[B, V, 1, H, W]
            proj_pix_prev,                 
            proj_pix_curr,
            proj_xy_prev,
            proj_xy_curr, 
            gt_scale_map_with_mask,        # Tensor[B, 2, H_sph, W_sph]
            gt_risk_score_map_with_mask,   # Tensor[B, 2, H_sph, W_sph]
            attn_type,                     # str
            attn_splits_list,              # List[int]
            corr_radius_list,              # List[int]
            prop_radius_list,              # List[int]
            num_reg_refine,                # int
            scale_only
        ):

        scales, risks, conf_prev, conf_curr = self.forward(
            img_prev         = img_prev,
            img_curr         = img_curr,
            depth_prev       = depth_prev,
            depth_curr       = depth_curr,
            proj_pix_prev    = proj_pix_prev,
            proj_pix_curr    = proj_pix_curr,
            proj_xy_prev     = proj_xy_prev,
            proj_xy_curr     = proj_xy_curr,
            attn_type        = attn_type,
            attn_splits_list = attn_splits_list,
            corr_radius_list = corr_radius_list,
            prop_radius_list = prop_radius_list,
            num_reg_refine   = num_reg_refine,
            scale_only       = scale_only,
        )

        # 将 conf_curr 调整到与 scales/risk_score 相同的分辨率
        if isinstance(scales, (list, tuple)):
            conf_for_loss = [self._resize_conf_to(s, conf_curr) for s in scales]
        else:
            conf_for_loss = self._resize_conf_to(scales, conf_curr)

        if scale_only:
            # 仅计算尺度分支的损失
            loss_s = get_loss_scale_map(scales, 
                                        gt_scale_map_with_mask,
                                        w_conf=conf_for_loss,
                                        gamma=0.9,
                                        loss_kind='charbonnier')
            return scales, None, loss_s, None
        else:
            loss_r = get_loss_risk_score_map(risks, 
                                             gt_risk_score_map_with_mask,
                                             w_conf=conf_for_loss,
                                             gamma=0.9,
                                             loss_kind='charbonnier')
            return None, risks, None, loss_r

    def extract_feature(self, im0, im1, branch):
        x = torch.cat([im0, im1], dim=0)
        feats = self.cnet(x, branch=branch)[::-1]
        p0, p1 = [], []
        for f in feats:
            a, b = torch.chunk(f, 2, dim=0)
            p0.append(a); p1.append(b)
        return p0, p1
    
    def project_views_to_range(
        self,
        features_list,   # list of V tensors, each [B, C, H_feat, W_feat]
        proj_pix,        # LongTensor [B, H_r, W_r, 3] = (cam_idx, u_orig, v_orig)
        H_img=160, W_img=320
    ):
        """
        把 V 路视角、同一尺度的特征图，按 proj_pix 映射到 range-view [B, C, H_r, W_r]。
        """
        B, H_r, W_r, _ = proj_pix.shape
        V = len(features_list)
        C = features_list[0].shape[1]
        H_feat, W_feat = features_list[0].shape[2], features_list[0].shape[3]

        # 1) 下采样比例
        s_u = W_feat / W_img
        s_v = H_feat / H_img

        # 2) 合并成 [B, V, C, Hf, Wf] → [B*V, C, Hf, Wf]
        feats = torch.stack(features_list, dim=1)         # [B,V,C,Hf,Wf]
        feats = feats.view(B*V, C, H_feat, W_feat)
        feats_flat = feats.view(B*V, C, -1)               # [B*V, C, Hf*Wf]

        # 3) 拆出并缩放像素索引
        cam_idx = proj_pix[..., 0].reshape(B, -1)         # [B, N]
        u_orig  = proj_pix[..., 1].float().reshape(B, -1) # [B, N]
        v_orig  = proj_pix[..., 2].float().reshape(B, -1) # [B, N]

        u_feat = (u_orig * s_u).long().clamp(0, W_feat-1) # [B, N]
        v_feat = (v_orig * s_v).long().clamp(0, H_feat-1) # [B, N]

        # 4) 计算扁平化后的批次＋视角索引
        #    batch_idx ∈ [0..B) 重复 N 次，拼接 cam_idx → [B*N]
        batch_idx = torch.arange(B, device=cam_idx.device)\
                        .unsqueeze(1).repeat(1, H_r*W_r)\
                        .reshape(-1)
        view_idx  = batch_idx * V + cam_idx.reshape(-1)   # [B*N]

        # 5) 计算在 Hf*Wf 上的线性化像素
        pix_idx   = (v_feat * W_feat + u_feat).reshape(-1)  # [B*N]

        # 6) 一次性 gather
        #    feats_flat[view_idx, :, pix_idx] → [B*N, C]
        selected = feats_flat[view_idx, :, pix_idx]

        # 7) 重塑回 [B, C, H_r, W_r]
        range_feat = selected.view(B, H_r*W_r, C) \
                            .permute(0,2,1) \
                            .reshape(B, C, H_r, W_r)
        return range_feat
    
    def _resize_conf_to(self, pred, conf):
        # pred: [B,1,H_out,W_out], conf: [B,1,H_in,W_in]
        if pred.shape[-2:] == conf.shape[-2:]:
            return conf
        # 置信度不做平滑，避免“被平均变大/变小”，最近邻更稳
        return F.interpolate(conf, size=pred.shape[-2:], mode='nearest')
