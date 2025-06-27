import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from .modules.matching import (global_correlation_softmax, local_correlation_softmax,
                                local_correlation_with_flow, local_scale_correlation)
from utils.loss import get_loss_scale_map, get_loss_risk_score_map

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet
from .scale_net.multi_view_deformable_fusion import MultiViewDeformableFusion
from .scale_net.utils.spherical import build_spherical_voxels, project_spherical_voxels_to_cameras

import torch.distributed as dist
import numpy as np

def is_main_process() -> bool:
    """
    如果没有初始化分布式，或者当前进程 rank == 0，就认为是主卡。
    """
    return (not dist.is_available() or not dist.is_initialized()) or dist.get_rank() == 0


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
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 range_image_feat_shape=[(20, 240), (40, 480)],
                 reg_refine=False,
                 load_cnet=False,
                 pretrained_cnet_path=None,
                 freeze_cnet=False,
                 train=False):
        super(FpTTC, self).__init__()
        self.num_scales = num_scales
        self.is_trainning = train

        self.camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

        # 仅底层共享：CNNEncoder
        self.cnet = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # 加载预训练的 cnet 权重
        if load_cnet:
            self._load_pretrained_cnet(pretrained_cnet_path, freeze_cnet)

        # Scale 分支私有网络
        self.featnet_scale = FeatureNet(num_scales=num_scales, feature_channels=feature_channels,
                                        num_head=num_head, ffn_dim_expansion=ffn_dim_expansion,
                                        num_transformer_layers=num_transformer_layers)
        self.corrnet_scale = FlowNet(num_scales=num_scales, feature_channels=feature_channels,
                                     upsample_factor=upsample_factor, reg_refine=reg_refine)
        self.conv_corr_scale = CorrEncoder(dim_in=2, dim_out=feature_channels+1)
        self.scalenet = ScaleNet(num_scales=num_scales, feature_channels=feature_channels,
                                 upsample_factor=upsample_factor, num_head=4,
                                 scale_level=num_scales, reg_refine=reg_refine, head_type='scale')

        # Risk 分支私有网络
        self.featnet_risk = FeatureNet(num_scales=num_scales, feature_channels=feature_channels,
                                       num_head=num_head, ffn_dim_expansion=ffn_dim_expansion,
                                       num_transformer_layers=num_transformer_layers)
        self.corrnet_risk = FlowNet(num_scales=num_scales, feature_channels=feature_channels,
                                    upsample_factor=upsample_factor, reg_refine=reg_refine)
        self.conv_corr_risk = CorrEncoder(dim_in=2, dim_out=feature_channels+1)
        self.risknet = ScaleNet(num_scales=num_scales, feature_channels=feature_channels,
                                upsample_factor=upsample_factor, num_head=4,
                                scale_level=num_scales, reg_refine=reg_refine, head_type='risk')
        
        self.R = 2
        self.scale_fuser = MultiViewDeformableFusion(
            num_layers=1,
            input_dim=feature_channels,
            d_model=feature_channels,
            nhead=num_head,
            per_level_channels=feature_channels*len(self.camera_channels),
            num_scales=num_scales,
            num_views=len(self.camera_channels),
            R=self.R,
            range_image_feat_shape=range_image_feat_shape
        )

        self.range_image_feat_shape = range_image_feat_shape
        # self.log_sigma_scale = nn.Parameter(torch.zeros(1))
        # self.log_sigma_risk = nn.Parameter(torch.zeros(1))
    
    def _load_pretrained_cnet(self, ckpt_path: str, freeze: bool):
        """
        仅加载预训练 checkpoint 中与 cnet 相关的权重，
        支持 key 前缀 'module.cnet.' 或 'cnet.'，加载后可选冻结。
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # 1) 优先取 'net'，否则尝试 'state_dict'
        if 'net' in ckpt:
            raw_state = ckpt['net']
        elif 'state_dict' in ckpt:
            raw_state = ckpt['state_dict']
        else:
            raw_state = ckpt

        # 2) 筛选出 cnet keys
        enc_state = {}
        for k, v in raw_state.items():
            # 可能的前缀有："module.cnet." 或 "cnet."
            if k.startswith('module.cnet.'):
                new_k = k[len('module.cnet.'):]
            elif k.startswith('cnet.'):
                new_k = k[len('cnet.'):]
            else:
                continue
            enc_state[new_k] = v

        # 3) 加载到 self.cnet（strict=False 允许部分层不匹配）
        missing, unexpected = self.cnet.load_state_dict(enc_state, strict=False)
        if is_main_process():
            if missing:
                print(f"[WARN] cnet missing keys: {missing}")
            if unexpected:
                print(f"[WARN] cnet unexpected keys: {unexpected}")

        # 4) freeze cnet 参数
        if freeze:
            for p in self.cnet.parameters():
                p.requires_grad = False
            if is_main_process():
                print("[INFO] cnet parameters frozen.")

    def forward(self, img0, img1, sensor_metas,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=6,
                testing=False,
                affine=None):
        
        img0, img1 = normalize_img(img0, img1)
        # B, C, H, W = img0.shape
        B, V, C, H, W = img0.shape
        # 提取多视角底层特征
        shared_prev, shared_curr = [], []
        for view in range(img0.size(1)):
            p, c = self.extract_feature(img0[:, view], img1[:, view], branch=None)
            shared_prev.append(p)
            shared_curr.append(c)

        # 假设 range_image_feat_shape = [(20,240),(40,480)]
        uv_map, cam_idx_map, valid_mask = [], [], []
        device = img0.device
        for lvl, (H_sph, W_sph) in enumerate(self.range_image_feat_shape):
            sph, xyz = build_spherical_voxels(H=H_sph, W=W_sph, R=self.R,
                                            r_min=5.0, r_max=30.0,
                                            fov_up_deg=8.0, fov_down_deg=-15.0)
            # 对 batch 中每个样本都 project 一次
            uv_list, camidx_list, valid_list = [], [], []
            for meta in sensor_metas:
                uv_np, cam_np, v_np = project_spherical_voxels_to_cameras(
                    sensor_metas=meta, xyz=xyz,
                    cam_channels=self.camera_channels, min_dist=1.0)
                uv_list.append(uv_np); camidx_list.append(cam_np); valid_list.append(v_np)
            # stack 并转 tensor
            uv_map.append(torch.from_numpy(np.stack(uv_list,axis=0)).to(device))      # [B,H_sph,W_sph,R,2]
            cam_idx_map.append(torch.from_numpy(np.stack(camidx_list,axis=0)).to(device))# [B,H_sph,W_sph,R]
            valid_mask.append(torch.from_numpy(np.stack(valid_list,axis=0)).to(device)) # [B,H_sph,W_sph,R]

        # 同质仿射矩阵
        affine_batch = torch.stack([
            # 如果是 NumPy array
            torch.from_numpy(a) if isinstance(a, np.ndarray)
            # 否则就直接当成 list 转
            else torch.tensor(a, dtype=torch.float32)
            for a in affine
        ], dim=0).to(device)  # [B,3,3]

         # 3) 分支归一化 + 多视角 channel 拼接
        prev_s_feats = [[self.cnet.final_norm_scale(f) for f in p] for p in shared_prev]
        curr_s_feats = [[self.cnet.final_norm_scale(f) for f in c] for c in shared_curr]
        # 每个 scale lvl 上，把 6 路视角的特征在 channel 维度上拼接
        mv_feats_prev = []
        mv_feats_curr = []
        for lvl in range(self.num_scales):
            # p[lvl]: [B, C, H_lvl, W_lvl]
            cat_prev = torch.cat([p[lvl] for p in prev_s_feats], dim=1)  # → [B, C*6, H_lvl, W_lvl]
            cat_curr = torch.cat([c[lvl] for c in curr_s_feats], dim=1)
            mv_feats_prev.append(cat_prev)
            mv_feats_curr.append(cat_curr)

        # 4) 调用可变形注意力 fusion
        feature0_lvls = self.scale_fuser(
            mv_feats=mv_feats_prev,
            uv_map=uv_map, 
            cam_idx_map=cam_idx_map,
            valid_mask=valid_mask,
            augmentor_affine=affine_batch
        )  # [B, C, H_sph, W_sph]

        feature1_lvls = self.scale_fuser(
            mv_feats=mv_feats_curr,
            uv_map=uv_map,
            cam_idx_map=cam_idx_map,
            valid_mask=valid_mask,
            augmentor_affine=affine_batch
        )

        # # —— Scale 分支：先做分支归一化，再组多视角，再预测 Scale Map ——
        # prev_s_feats = [[self.cnet.final_norm_scale(f) for f in p] for p in shared_prev]
        # curr_s_feats = [[self.cnet.final_norm_scale(f) for f in c] for c in shared_curr]
        # # 多视角拼接 - 简单横向拼接
        # feature0_lvls_s, feature1_lvls_s = [], []
        # for lvl in range(self.num_scales):
        #     f0 = torch.cat([f[lvl] for f in prev_s_feats], dim=3)
        #     f1 = torch.cat([f[lvl] for f in curr_s_feats], dim=3)
        #     feature0_lvls_s.append(f0)
        #     feature1_lvls_s.append(f1)

        # === Scale 分支 ===
        corr_s = None
        mlvl_s0, mlvl_s1 = [], []
        for lvl in range(self.num_scales):
            f0, f1 = feature0_lvls[lvl], feature1_lvls[lvl]
            f0_s, f1_s = self.featnet_scale(f0, f1, lvl, attn_type, attn_splits_list, corr_s)
            mlvl_s0.append(f0_s); mlvl_s1.append(f1_s)
            corr_s, _ = self.corrnet_scale(f0_s, f1_s, lvl, corr_radius_list, prop_radius_list, num_reg_refine, False, corr_s)
            if lvl < self.num_scales - 1:
                corr_s = F.interpolate(corr_s, scale_factor=2, mode='bilinear', align_corners=True) * 2
        corr_enc_s = self.conv_corr_scale(corr_s)
        ini_scale = F.softplus(corr_enc_s[:, :1]) + 1e-3
        corr_enc_s = corr_enc_s[:, 1:]
        scales = self.scalenet(corr_enc_s, mlvl_s0, mlvl_s1, ini_scale)

        del prev_s_feats, curr_s_feats
        del corr_s, mlvl_s0, mlvl_s1
        # del feature0_lvls_s, feature1_lvls_s

        # # —— Risk 分支：先做分支归一化，再组多视角，再预测 Risk Map ——
        # prev_r_feats = [[self.cnet.final_norm_risk(f) for f in p] for p in shared_prev]
        # curr_r_feats = [[self.cnet.final_norm_risk(f) for f in c] for c in shared_curr]
        # # 多视角拼接
        # feature0_lvls_r, feature1_lvls_r = [], []
        # for lvl in range(self.num_scales):
        #     f0 = torch.cat([p[lvl] for p in prev_r_feats], dim=3)
        #     f1 = torch.cat([c[lvl] for c in curr_r_feats], dim=3)
        #     feature0_lvls_r.append(f0); feature1_lvls_r.append(f1)

        # === Risk 分支 ===
        corr_r = None
        mlvl_r0, mlvl_r1 = [], []
        for lvl in range(self.num_scales):
            f0, f1 = feature0_lvls[lvl], feature1_lvls[lvl]
            f0_r, f1_r = self.featnet_risk(f0, f1, lvl, attn_type, attn_splits_list, corr_r)
            mlvl_r0.append(f0_r); mlvl_r1.append(f1_r)
            corr_r, _ = self.corrnet_risk(f0_r, f1_r, lvl, corr_radius_list, prop_radius_list, num_reg_refine, False, corr_r)
            if lvl < self.num_scales - 1:
                corr_r = F.interpolate(corr_r, scale_factor=2, mode='bilinear', align_corners=True) * 2
        corr_enc_r = F.relu(self.conv_corr_risk(corr_r))
        ini_risk = corr_enc_r[:, :1]
        corr_enc_r = corr_enc_r[:, 1:]
        risk_score = self.risknet(corr_enc_r, mlvl_r0, mlvl_r1, ini_risk)

        del corr_r, mlvl_r0, mlvl_r1
        del feature0_lvls, feature1_lvls

        del shared_prev, shared_curr

        return scales, risk_score

    def forward_with_loss(self, img0, img1, sensor_meta,
                          gt_scale_map_with_mask, gt_risk_score_map_with_mask, **kwargs):
        scales, risks = self.forward(img0, img1, sensor_meta, **kwargs)
        loss_s = get_loss_scale_map(scales, gt_scale_map_with_mask)
        loss_r = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)
        # loss_s_term = torch.exp(-2 * self.log_sigma_scale) * loss_s + 2 * self.log_sigma_scale
        # loss_r_term = torch.exp(-2 * self.log_sigma_risk)  * loss_r + 2 * self.log_sigma_risk
        # loss = 0.5 * (loss_s_term + loss_r_term)
        return scales, risks, loss_s, loss_r

    def extract_feature(self, im0, im1, branch):
        x = torch.cat([im0, im1], dim=0)
        feats = self.cnet(x, branch=branch)[::-1]
        p0, p1 = [], []
        for f in feats:
            a, b = torch.chunk(f, 2, dim=0)
            p0.append(a); p1.append(b)
        return p0, p1

# class FpTTC(nn.Module):
#     def __init__(self,
#                  num_scales=2,
#                  feature_channels=128,
#                  upsample_factor=8,
#                  num_head=1,
#                  ffn_dim_expansion=4,
#                  num_transformer_layers=6,
#                  range_image_feat_shape=[(20, 240), (40, 480)],
#                  reg_refine=False,
#                  pretrained_cnet_path: str = None,
#                  freeze_cnet: bool = False,
#                  train=False):
#         super(FpTTC, self).__init__()
#         self.num_scales = num_scales
#         self.is_trainning = train

#         # 仅底层共享：CNNEncoder
#         self.cnet = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

#         # 如果提供了预训练 cnet 权重，则只加载该部分
#         if pretrained_cnet_path is not None:
#             self._load_pretrained_cnet(pretrained_cnet_path, freeze_cnet)

#         # Scale 分支私有网络
#         self.featnet_scale = FeatureNet(num_scales=num_scales, feature_channels=feature_channels,
#                                         num_head=num_head, ffn_dim_expansion=ffn_dim_expansion,
#                                         num_transformer_layers=num_transformer_layers)
#         self.corrnet_scale = FlowNet(num_scales=num_scales, feature_channels=feature_channels,
#                                      upsample_factor=upsample_factor, reg_refine=reg_refine)
#         self.conv_corr_scale = CorrEncoder(dim_in=2, dim_out=feature_channels+1)
#         self.scalenet = ScaleNet(num_scales=num_scales, feature_channels=feature_channels,
#                                  upsample_factor=upsample_factor, num_head=4,
#                                  scale_level=num_scales, reg_refine=reg_refine, head_type='scale')
        
#         self.risknet = ScaleNet(num_scales=num_scales, feature_channels=feature_channels,
#                                  upsample_factor=upsample_factor, num_head=4,
#                                  scale_level=num_scales, reg_refine=reg_refine, head_type='risk')
        
#         self.log_sigma_scale = nn.Parameter(torch.zeros(1))
#         self.log_sigma_risk = nn.Parameter(torch.zeros(1))
    
#     def _load_pretrained_cnet(self, ckpt_path: str, freeze: bool):
#         """
#         仅加载预训练 checkpoint 中与 cnet 相关的权重，
#         支持 key 前缀 'module.cnet.' 或 'cnet.'，加载后可选冻结。
#         """
#         ckpt = torch.load(ckpt_path, map_location='cpu')

#         # 1) 优先取 'net'，否则尝试 'state_dict'
#         if 'net' in ckpt:
#             raw_state = ckpt['net']
#         elif 'state_dict' in ckpt:
#             raw_state = ckpt['state_dict']
#         else:
#             raw_state = ckpt

#         # 2) 筛选出 cnet keys
#         enc_state = {}
#         for k, v in raw_state.items():
#             # 可能的前缀有："module.cnet." 或 "cnet."
#             if k.startswith('module.cnet.'):
#                 new_k = k[len('module.cnet.'):]
#             elif k.startswith('cnet.'):
#                 new_k = k[len('cnet.'):]
#             else:
#                 continue
#             enc_state[new_k] = v

#         # 3) 加载到 self.cnet（strict=False 允许部分层不匹配）
#         missing, unexpected = self.cnet.load_state_dict(enc_state, strict=False)
#         if is_main_process():
#             if missing:
#                 print(f"[WARN] cnet missing keys: {missing}")
#             if unexpected:
#                 print(f"[WARN] cnet unexpected keys: {unexpected}")

#         # 4) freeze cnet 参数
#         if freeze:
#             for p in self.cnet.parameters():
#                 p.requires_grad = False
#             if is_main_process():
#                 print("[INFO] cnet parameters frozen.")

#     def forward(self, img0, img1, sensor_metas,
#                 attn_type=None,
#                 attn_splits_list=None,
#                 corr_radius_list=None,
#                 prop_radius_list=None,
#                 num_reg_refine=6,
#                 testing=False):
#         # if self.is_trainning and not testing:
#         #     self.eval()
#         #     torch.set_grad_enabled(False)

#         img0, img1 = normalize_img(img0, img1)
#         # B, C, H, W = img0.shape
#         # 提取多视角底层特征
#         prev_feat_list, curr_feat_list = [], []
#         for view in range(img0.size(1)):
#             p, c = self.extract_feature(img0[:, view], img1[:, view])
#             prev_feat_list.append(p)
#             curr_feat_list.append(c)

#         # 多视角拼接
#         feature0_lvls, feature1_lvls = [], []
#         for lvl in range(len(prev_feat_list[0])):
#             f0 = torch.cat([f[lvl] for f in prev_feat_list], dim=3)
#             f1 = torch.cat([f[lvl] for f in curr_feat_list], dim=3)
#             feature0_lvls.append(f0)
#             feature1_lvls.append(f1)

#         del prev_feat_list, curr_feat_list

#         # === Scale 分支 ===
#         corr_s = None
#         mlvl_s0, mlvl_s1 = [], []
#         for lvl in range(self.num_scales):
#             f0, f1 = feature0_lvls[lvl], feature1_lvls[lvl]
#             f0_s, f1_s = self.featnet_scale(f0, f1, lvl, attn_type, attn_splits_list, corr_s)
#             mlvl_s0.append(f0_s); mlvl_s1.append(f1_s)
#             corr_s, _ = self.corrnet_scale(f0_s, f1_s, lvl, corr_radius_list, prop_radius_list, num_reg_refine, False, corr_s)
#             if lvl < self.num_scales - 1:
#                 corr_s = F.interpolate(corr_s, scale_factor=2, mode='bilinear', align_corners=True) * 2
#         corr_enc_s = F.relu(self.conv_corr_scale(corr_s))
#         ini_scale = corr_enc_s[:, :1]
#         corr_enc_s = corr_enc_s[:, 1:]
#         scales = self.scalenet(corr_enc_s, mlvl_s0, mlvl_s1, ini_scale)

#         corr_enc_r = self.conv_corr_scale(corr_s)
#         ini_risk = corr_enc_r[:, :1]
#         corr_enc_r = corr_enc_r[:, 1:]
#         risk_score = self.risknet(corr_enc_r, mlvl_s0, mlvl_s1, ini_risk)

#         del corr_s, mlvl_s0, mlvl_s1

#         return scales, risk_score

#     def forward_with_loss(self, img0, img1, sensor_meta,
#                           gt_scale_map_with_mask, gt_risk_score_map_with_mask, **kwargs):
#         scales, risk_score = self.forward(img0, img1, sensor_meta, **kwargs)
#         loss_s = get_loss_scale_map(scales, gt_scale_map_with_mask)
#         loss_r = get_loss_risk_score_map(risk_score, gt_risk_score_map_with_mask)
#         loss_s_term = torch.exp(-self.log_sigma_scale) * loss_s + self.log_sigma_scale
#         loss_r_term = torch.exp(-self.log_sigma_risk) * loss_r + self.log_sigma_risk
#         loss = loss_s_term + loss_r_term
#         return scales, risk_score, loss_s, loss_s_term, loss_r, loss_r_term, loss

#     def extract_feature(self, im0, im1):
#         x = torch.cat([im0, im1], dim=0)
#         feats = self.cnet(x)[::-1]
#         p0, p1 = [], []
#         for f in feats:
#             a, b = torch.chunk(f, 2, dim=0)
#             p0.append(a); p1.append(b)
#         return p0, p1