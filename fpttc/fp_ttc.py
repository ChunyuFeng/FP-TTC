import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from utils.loss import get_loss_scale_map, get_loss_risk_score_map

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet
from .scale_net.multi_view_deformable_fusion import MultiViewDeformableFusion

import torch.distributed as dist
import numpy as np
from utils.dist import is_main_process

import matplotlib.pyplot as plt
import time

def _safe_bilinear(x, *, size=None, scale_factor=None, align_corners=True):
    orig_dtype = x.dtype
    if orig_dtype == torch.bfloat16:
        x = x.float()  # or x.half()
    y = F.interpolate(x, size=size, scale_factor=scale_factor,
                      mode='bilinear', align_corners=align_corners)
    return y.to(orig_dtype)
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
        

    def forward(
        self,
        img_prev, img_curr,
        depth_prev, depth_curr,
        proj_pix_prev, proj_pix_curr,
        attn_type,
        attn_splits_list,
        corr_radius_list,
        prop_radius_list,
        num_reg_refine,
        scale_only
    ):
        t0 = time.perf_counter()

        # ----- 预处理 -----
        img0, img1 = normalize_img(img_prev, img_curr)         # [B,V,3,H,W] -> normed
        rgbd0 = torch.cat([img0, depth_prev], dim=2)           # [B,V,4,H,W]
        rgbd1 = torch.cat([img1, depth_curr], dim=2)           # [B,V,4,H,W]
        B, V, C, H_img, W_img = rgbd0.shape

        # 合并视角到 batch 维，一次性提特征
        x0 = rgbd0.view(B*V, C, H_img, W_img)
        x1 = rgbd1.view(B*V, C, H_img, W_img)
        # extract_feature 会在内部 cat([x0, x1], 0) -> cnet -> 多尺度 -> chunk 回来
        prev_lvls_flat, curr_lvls_flat = self.extract_feature(x0, x1, branch=None)  # list[T][B*V,C,Hs,Ws]

        # reshape 回 [B,V,C,Hs,Ws]
        prev_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in prev_lvls_flat]
        curr_lvls = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in curr_lvls_flat]

        t1 = time.perf_counter()
        # print(f"feature extraction time: {t1 - t0:.4f} s")

        # ----- 多尺度 Transformer + 相关性，仍然“只按尺度循环”，但每次处理 B*V -----
        corr = None
        multi_level_feats_prev = []
        multi_level_feats_curr = []

        for lvl in range(self.num_scales):
            # 取出该尺度的 [B,V,C,Hs,Ws]，合并为 [B*V,C,Hs,Ws] 一次性送入
            p = prev_lvls[lvl].reshape(B*V, -1, prev_lvls[lvl].shape[3], prev_lvls[lvl].shape[4])
            c = curr_lvls[lvl].reshape(B*V, -1, curr_lvls[lvl].shape[3], curr_lvls[lvl].shape[4])

            fused_prev, fused_curr = self.featnet(
                p, c, lvl, attn_type, attn_splits_list, corr
            )

            # 保存为 [B,V,C,Hs,Ws] 以便后续投影
            fp = fused_prev.view(B, V, fused_prev.shape[1], fused_prev.shape[2], fused_prev.shape[3])
            fc = fused_curr.view(B, V, fused_curr.shape[1], fused_curr.shape[2], fused_curr.shape[3])
            multi_level_feats_prev.append(fp)
            multi_level_feats_curr.append(fc)

            corr, _ = self.corrnet(
                fused_prev, fused_curr,
                lvl, corr_radius_list, prop_radius_list,
                num_reg_refine, False, corr
            )
            if lvl < self.num_scales - 1:
                corr = _safe_bilinear(corr, scale_factor=2, align_corners=True) * 2


        t2 = time.perf_counter()
        # print(f"correlation computation time: {t2 - t1:.4f} s")

        # corr 此时是 [B*V, 2, Hc, Wc]，变回 [B,V,2,Hc,Wc] 后，为每个视角投影到 range
        Cc, Hc, Wc = corr.shape[1], corr.shape[2], corr.shape[3]
        corr_bv = corr.view(B, V, Cc, Hc, Wc)
        corr_list = [corr_bv[:, v] for v in range(V)]          # list[V] of [B,2,Hc,Wc]

        # 1) 风险/尺度的“相关性特征”投影（当前帧）
        corr_range = self.project_views_to_range(
            corr_list, proj_pix_curr, H_img=H_img, W_img=W_img
        )  # -> [B, Cc, H_r, W_r]

        # 2) 每个尺度的多视角特征投影
        multi_level_ranges_prev, multi_level_ranges_curr = [], []
        for lvl in range(self.num_scales):
            scale = 2 ** (self.num_scales - 1 - lvl)
            if scale > 1:
                proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
                proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
            else:
                proj_prev_lvl = proj_pix_prev
                proj_curr_lvl = proj_pix_curr

            # 组建 list[V] of [B,C,Hs,Ws]
            prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
            curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]

            range_prev = self.project_views_to_range(prev_feats_list, proj_prev_lvl, H_img=H_img, W_img=W_img)
            range_curr = self.project_views_to_range(curr_feats_list, proj_curr_lvl, H_img=H_img, W_img=W_img)

            multi_level_ranges_prev.append(range_prev)  # [B,C,Hr,Wr]
            multi_level_ranges_curr.append(range_curr)

        t3 = time.perf_counter()
        # print(f"projection to range-view time: {t3 - t2:.4f} s")

        # ----- 预测（两分支）-----
        # scale 分支
        corr_encoded_s = self.conv_corr(corr_range)
        initial_scale = F.softplus(corr_encoded_s[:, :1]) + 1e-3
        corr_encoded_s = corr_encoded_s[:, 1:]

        scales = self.scale_net(
            corr_encoded_s, multi_level_ranges_prev, multi_level_ranges_curr, initial_scale
        )
        if scale_only:
            return scales, None

        # risk 分支
        corr_encoded = self.conv_corr_risk(corr_range)
        initial_risk = corr_encoded[:, :1]
        corr_encoded = corr_encoded[:, 1:]
        risk_score = self.risk_net(
            corr_encoded, multi_level_ranges_prev, multi_level_ranges_curr, initial_risk
        )

        t4 = time.perf_counter()
        # print(f"scale & risk prediction time: {t4 - t3:.4f} s")

        return scales, risk_score


    def forward_with_loss( 
            self,
            img_prev,                      # Tensor[B, V, 3, H, W]
            img_curr,                      # Tensor[B, V, 3, H, W]
            depth_prev,                    # Tensor[B, V, 1, H, W]
            depth_curr,                    # Tensor[B, V, 1, H, W]
            proj_pix_prev,                 
            proj_pix_curr,                 
            gt_scale_map_with_mask,        # Tensor[B, 2, H_sph, W_sph]
            gt_risk_score_map_with_mask,   # Tensor[B, 2, H_sph, W_sph]
            attn_type,                     # str
            attn_splits_list,              # List[int]
            corr_radius_list,              # List[int]
            prop_radius_list,              # List[int]
            num_reg_refine,                # int
            scale_only
        ):

        scales, risks = self.forward(
            img_prev         = img_prev,
            img_curr         = img_curr,
            depth_prev       = depth_prev,
            depth_curr       = depth_curr,
            proj_pix_prev    = proj_pix_prev,
            proj_pix_curr    = proj_pix_curr,
            attn_type        = attn_type,
            attn_splits_list = attn_splits_list,
            corr_radius_list = corr_radius_list,
            prop_radius_list = prop_radius_list,
            num_reg_refine   = num_reg_refine,
            scale_only       = scale_only,
        )
        if scale_only:
            # 仅计算尺度分支的损失
            loss_s = get_loss_scale_map(scales, gt_scale_map_with_mask)
            return scales, None, loss_s, None
        else:
            loss_r = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)
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
