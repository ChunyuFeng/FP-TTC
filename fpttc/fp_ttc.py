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
                 reg_refine             = False,
                 radial_sampling        = 16,
                 load_cnet              = False,
                 pretrained_cnet_path   = None,
                 freeze_cnet            = False,
                 train                  = True):
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
        self.featnet   = FeatureNet(num_scales             = num_scales,
                                    feature_channels       = feature_channels,
                                    num_head               = num_head, 
                                    ffn_dim_expansion      = ffn_dim_expansion,
                                    num_transformer_layers = num_transformer_layers)    
        self.corrnet   = FlowNet(num_scales                = num_scales,
                                feature_channels          = feature_channels,
                                upsample_factor           = upsample_factor,
                                reg_refine                = reg_refine)        
        self.conv_corr = CorrEncoder(dim_in                = 2,
                                           dim_out               = feature_channels+1) 
        self.scalenet_singlebranch = ScaleNet(num_scales               = num_scales,
                                        feature_channels         = feature_channels,
                                        upsample_factor          = upsample_factor,
                                        num_head                 = 4,
                                        scale_level              = num_scales, 
                                        reg_refine               = reg_refine, 
                                        head_type                = 'scale')
        

        # # Scale 分支私有网络
        # self.featnet_scale   = FeatureNet(num_scales             = num_scales,
        #                                   feature_channels       = feature_channels,
        #                                   num_head               = num_head, 
        #                                   ffn_dim_expansion      = ffn_dim_expansion,
        #                                   num_transformer_layers = num_transformer_layers)    
        # self.corrnet_scale   = FlowNet(num_scales                = num_scales,
        #                                feature_channels          = feature_channels,
        #                                upsample_factor           = upsample_factor,
        #                                reg_refine                = reg_refine)        
        # self.conv_corr_scale = CorrEncoder(dim_in                = 2,
        #                                    dim_out               = feature_channels+1) 
        # self.scalenet_singlebranch       = ScaleNet(num_scales               = num_scales,
        #                                 feature_channels         = feature_channels,
        #                                 upsample_factor          = upsample_factor,
        #                                 num_head                 = 4,
        #                                 scale_level              = num_scales, 
        #                                 reg_refine               = reg_refine, 
        #                                 head_type                = 'scale')

        # # Risk 分支私有网络
        # self.featnet_risk    = FeatureNet(num_scales             = num_scales,
        #                                   feature_channels       = feature_channels,
        #                                   num_head               = num_head, 
        #                                   ffn_dim_expansion      = ffn_dim_expansion,
        #                                   num_transformer_layers = num_transformer_layers)
        # self.corrnet_risk    = FlowNet(num_scales                = num_scales,
        #                                feature_channels          = feature_channels,
        #                                upsample_factor           = upsample_factor,
        #                                reg_refine                = reg_refine)
        # self.conv_corr_risk  = CorrEncoder(dim_in                = 2,
        #                                    dim_out               = feature_channels+1)
        # self.risknet         = ScaleNet(num_scales               = num_scales,
        #                                 feature_channels         = feature_channels,
        #                                 upsample_factor          = upsample_factor,
        #                                 num_head                 = 4,
        #                                 scale_level              = num_scales, 
        #                                 reg_refine               = reg_refine, 
        #                                 head_type                = 'risk')
        
        # # 特征融合和投影到透视图
        # self.inner_fusion    = InnerFeatureFusion(channels       = feature_channels)
        # self.voxel_to_pv     = VoxelToPV(in_channels             = feature_channels,
        #                                  R                       = radial_sampling)
    
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

    def forward(self,
                img_prev,
                img_curr,
                depth_prev,
                depth_curr,
                proj_pix_prev,
                proj_pix_curr,
                affine_matrix,
                # idx_uv_prev,
                # idx_uv_curr,
                attn_type,
                attn_splits_list,
                corr_radius_list,
                prop_radius_list,
                num_reg_refine,
                testing):

        # ### 1）提取输入的多视角图像的底层特征
        # img0, img1 = normalize_img(img_prev, img_curr)
        # B, V, C, H_img, W_img = img0.shape
        # shared_prev, shared_curr = [], []
        # for view in range(img0.size(1)):
        #     p, c = self.extract_feature(img0[:, view], img1[:, view], branch=None)
        #     shared_prev.append(p)
        #     shared_curr.append(c)

        ### 1）提取输入的多视角图像的底层特征
        img0, img1 = normalize_img(img_prev, img_curr)
        rgbd0 = torch.cat([img0, depth_prev], dim=2)  # [B, V, 4, H, W]
        rgbd1 = torch.cat([img1, depth_curr], dim=2)  # [B, V, 4, H, W]
        B, V, C, H_img, W_img = img0.shape
        shared_prev, shared_curr = [], []
        for view in range(rgbd0.size(1)):
            p, c = self.extract_feature(rgbd0[:, view], rgbd1[:, view], branch=None)
            shared_prev.append(p)
            shared_curr.append(c)

        # ### 2）融合多视角特征 - 基于 bilinear sampling，参考 SimpleBEV 

        # # 用于 voxel bilinear sampling 的 base 2D feature
        # # shared_prev 和 shared_curr 是 list，包含了每个视角的特征
        # base_feat_prev = [ self.inner_fusion(p[0], p[1]) for p in shared_prev ]
        # base_feat_curr = [ self.inner_fusion(c[0], c[1]) for c in shared_curr ]

        # # voxel - 2D raw image(900*1600) 的位置对应关系为 idx_uv_prev 和 idx_uv_curr
        # # affine_matrix 表示模型输入图像(160*320)相对于数据集原始图像(900*1600)的变换矩阵，需要应用到 idx_uv_prev 和 idx_uv_curr
        # # 从输入图像(160*320)到特征图(40*80)，还存在一个缩放比例，通过 norm_factor 来归一化 (u,v) 坐标
        # norm_factor = W_img / base_feat_prev[0].shape[3]  # 计算缩放因子
        # idx_uv_prev_transformed = self.apply_affine(idx_uv_prev, affine_matrix,
        #                                             normalize=True, norm_factor=norm_factor)
        # idx_uv_curr_transformed = self.apply_affine(idx_uv_curr, affine_matrix,
        #                                             normalize=True, norm_factor=norm_factor)

        # fused_prev = self.voxel_to_pv(
        #     surround_view_feats = base_feat_prev,
        #     idx_uv              = idx_uv_prev_transformed
        # )
        # fused_curr = self.voxel_to_pv(
        #     surround_view_feats = base_feat_curr,
        #     idx_uv              = idx_uv_curr_transformed
        # )

        # # 将 fused_feat_prev 和 fused_feat_curr 下采样，重新构建成多尺度特征
        # # 将下采样后的较小的特征图放在列表前面，原尺寸的放在后面
        # feature0_lvls = []
        # feature1_lvls = []
        # prev_feat = fused_prev
        # curr_feat = fused_curr
        # for _ in range(self.num_scales):
        #     feature0_lvls.insert(0, prev_feat)
        #     feature1_lvls.insert(0, curr_feat)
        #     prev_feat = F.max_pool2d(prev_feat, kernel_size=2, stride=2)
        #     curr_feat = F.max_pool2d(curr_feat, kernel_size=2, stride=2)

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


        corr_s_ = []
        mlvl_s0_ = []
        mlvl_s1_ = []
        # === Scale 分支 ===
        for feat_prev, feat_curr in zip(shared_prev, shared_curr):
            feature0_lvls = feat_prev
            feature1_lvls = feat_curr
            corr_s = None
            mlvl_s0, mlvl_s1 = [], []
            for lvl in range(self.num_scales):
                f0, f1     = feature0_lvls[lvl], feature1_lvls[lvl]
                f0_s, f1_s = self.featnet(f0, f1, lvl, attn_type, attn_splits_list, corr_s)
                mlvl_s0.append(f0_s)
                mlvl_s1.append(f1_s)
                corr_s, _  = self.corrnet(f0_s, f1_s, lvl, corr_radius_list, prop_radius_list, num_reg_refine, False, corr_s)
                if lvl < self.num_scales - 1:
                    corr_s = F.interpolate(corr_s, scale_factor=2, mode='bilinear', align_corners=True) * 2
            corr_s_.append(corr_s)
            mlvl_s0_.append(mlvl_s0)
            mlvl_s1_.append(mlvl_s1)
        
        corr_s = self.project_views_to_range(
            corr_s_,
            proj_pix_curr,
            H_img=160, W_img=320
        )

        # visualize_corr_s(corr_s, title="Projected corr_s")
        s0_lvl0 = [view_feats[0] for view_feats in mlvl_s0_]
        s0_lvl1 = [view_feats[1] for view_feats in mlvl_s0_]
        range_s0_lvl0 = self.project_views_to_range(
            s0_lvl0,
            proj_pix_prev,
            H_img=160, W_img=320
        )
        range_s0_lvl1 = self.project_views_to_range(
            s0_lvl1,
            proj_pix_prev,
            H_img=160, W_img=320
        )
        
        s1_lvl0 = [view_feats[0] for view_feats in mlvl_s1_]
        s1_lvl1 = [view_feats[1] for view_feats in mlvl_s1_]
        range_s1_lvl0 = self.project_views_to_range(
            s1_lvl0,
            proj_pix_curr,
            H_img=160, W_img=320
        )
        range_s1_lvl1 = self.project_views_to_range(
            s1_lvl1,
            proj_pix_curr,
            H_img=160, W_img=320
        )
        mlvl_s0 = [range_s0_lvl0, range_s0_lvl1]
        mlvl_s1 = [range_s1_lvl0, range_s1_lvl1]        

        corr_enc_s = self.conv_corr(corr_s)
        ini_scale  = F.softplus(corr_enc_s[:, :1]) + 1e-3
        corr_enc_s = corr_enc_s[:, 1:]
        scales     = self.scalenet_singlebranch(corr_enc_s, mlvl_s0, mlvl_s1, ini_scale)

        del corr_s, mlvl_s0, mlvl_s1

        return scales

        # # === Risk 分支 ===
        # corr_r = None
        # mlvl_r0, mlvl_r1 = [], []
        # for lvl in range(self.num_scales):
        #     f0, f1     = feature0_lvls[lvl], feature1_lvls[lvl]
        #     f0_r, f1_r = self.featnet_risk(f0, f1, lvl, attn_type, attn_splits_list, corr_r)
        #     mlvl_r0.append(f0_r)
        #     mlvl_r1.append(f1_r)
        #     corr_r, _  = self.corrnet_risk(f0_r, f1_r, lvl, corr_radius_list, prop_radius_list, num_reg_refine, False, corr_r)
        #     if lvl < self.num_scales - 1:
        #         corr_r = F.interpolate(corr_r, scale_factor=2, mode='bilinear', align_corners=True) * 2
        # corr_enc_r = F.relu(self.conv_corr_risk(corr_r))
        # ini_risk   = corr_enc_r[:, :1]
        # corr_enc_r = corr_enc_r[:, 1:]
        # risk_score = self.risknet(corr_enc_r, mlvl_r0, mlvl_r1, ini_risk)

        # del corr_r, mlvl_r0, mlvl_r1
        # del feature0_lvls, feature1_lvls

        # return scales, risk_score

    # def forward_with_loss( 
    #         self,
    #         img_prev,                      # Tensor[B, V, 3, H, W]
    #         img_curr,                      # Tensor[B, V, 3, H, W]
    #         gt_scale_map_with_mask,        # Tensor[B, 2, H_sph, W_sph]
    #         gt_risk_score_map_with_mask,   # Tensor[B, 2, H_sph, W_sph]
    #         affine_matrix,                 # Tensor[B, 3, 3]
    #         idx_uv_prev,                   # Tensor[B, H_sph, W_sph, R, 3]
    #         idx_uv_curr,                   # Tensor[B, H_sph, W_sph, R, 3]
    #         attn_type,                     # str
    #         attn_splits_list,              # List[int]
    #         corr_radius_list,              # List[int]
    #         prop_radius_list,              # List[int]
    #         num_reg_refine,                # int
    #         testing                        # bool
    #     ):

    #     scales, risks = self.forward(
    #         img_prev         = img_prev,
    #         img_curr         = img_curr,
    #         affine_matrix    = affine_matrix,
    #         idx_uv_prev      = idx_uv_prev,
    #         idx_uv_curr      = idx_uv_curr,
    #         attn_type        = attn_type,
    #         attn_splits_list = attn_splits_list,
    #         corr_radius_list = corr_radius_list,
    #         prop_radius_list = prop_radius_list,
    #         num_reg_refine   = num_reg_refine,
    #         testing          = testing
    #     )
    #     loss_s = get_loss_scale_map(scales, gt_scale_map_with_mask)
    #     loss_r = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)
    #     # loss_s_term = torch.exp(-2 * self.log_sigma_scale) * loss_s + 2 * self.log_sigma_scale
    #     # loss_r_term = torch.exp(-2 * self.log_sigma_risk)  * loss_r + 2 * self.log_sigma_risk
    #     # loss = 0.5 * (loss_s_term + loss_r_term)
    #     return scales, risks, loss_s, loss_r

    def forward_with_loss( 
            self,
            img_prev,                      # Tensor[B, V, 3, H, W]
            img_curr,                      # Tensor[B, V, 3, H, W]
            depth_prev,                    # Tensor[B, V, 1, H, W]
            depth_curr,                    # Tensor[B, V, 1, H, W]
            proj_pix_prev,
            proj_pix_curr,
            gt_scale_map_with_mask,        # Tensor[B, 2, H_sph, W_sph]
            affine_matrix,                 # Tensor[B, 3, 3]
            # gt_risk_score_map_with_mask,   # Tensor[B, 2, H_sph, W_sph]
            # affine_matrix,                 # Tensor[B, 3, 3]
            # idx_uv_prev,                   # Tensor[B, H_sph, W_sph, R, 3]
            # idx_uv_curr,                   # Tensor[B, H_sph, W_sph, R, 3]
            attn_type,                     # str
            attn_splits_list,              # List[int]
            corr_radius_list,              # List[int]
            prop_radius_list,              # List[int]
            num_reg_refine,                # int
            testing                        # bool
        ):

        scales = self.forward(
            img_prev         = img_prev,
            img_curr         = img_curr,
            depth_prev       = depth_prev,
            depth_curr       = depth_curr,
            proj_pix_prev    = proj_pix_prev,
            proj_pix_curr    = proj_pix_curr,
            affine_matrix    = affine_matrix,
            # idx_uv_prev      = idx_uv_prev,
            # idx_uv_curr      = idx_uv_curr,
            attn_type        = attn_type,
            attn_splits_list = attn_splits_list,
            corr_radius_list = corr_radius_list,
            prop_radius_list = prop_radius_list,
            num_reg_refine   = num_reg_refine,
            testing          = testing
        )
        loss_s = get_loss_scale_map(scales, gt_scale_map_with_mask)
        return scales, loss_s

    def extract_feature(self, im0, im1, branch):
        x = torch.cat([im0, im1], dim=0)
        feats = self.cnet(x, branch=branch)[::-1]
        p0, p1 = [], []
        for f in feats:
            a, b = torch.chunk(f, 2, dim=0)
            p0.append(a); p1.append(b)
        return p0, p1

    def apply_affine(self, idx_uv, affine_matrix, 
                     normalize=True, norm_factor=1.0):
        """
        将仿射矩阵 affine 应用到 idx_uv 中的 (u, v) 坐标上。
        idx_uv[...,0] = cam_idx 保持不变；idx_uv[...,1:3] = (u,v) 会被映射。

        Args:
            idx_uv       :  float Tensor, shape [B, Hs, Ws, R, 3]
            affine_matrix:  float Tensor, shape [B, 3, 3]
            normalize    :  bool, 是否将 (u,v) 坐标归一化到指定尺寸范围
            norm_factor  :  float, 归一化因子，默认为 1.0

        Returns:
            idx_uv_transformed: Tensor with same shape, uv 部分已被变换
        """
        B, Hs, Ws, R, _ = idx_uv.shape

        # 1) 拆出 uv 并添加齐次分量
        uv = idx_uv[..., 1:3]                            # [B,Hs,Ws,R,2]
        ones = uv.new_ones(B, Hs, Ws, R, 1)              # [B,Hs,Ws,R,1]
        homo = torch.cat([uv, ones], dim=-1)             # [B,Hs,Ws,R,3]

        # 2) 扁平化便于矩阵乘
        N = Hs * Ws * R
        homo_flat = homo.view(B, N, 3)                   # [B,N,3]
        
        # 3) 批量仿射
        A_T = affine_matrix.transpose(1, 2)              # [B,3,3]
        trans_flat = torch.matmul(homo_flat, A_T)        # [B,N,3]

        # 4) reshape 回去
        trans = trans_flat.view(B, Hs, Ws, R, 3)         # [B,Hs,Ws,R,3]

        # 5) 用变换后的 u',v' 更新原 idx_uv
        idx_uv_transformed = idx_uv.clone()
        idx_uv_transformed[..., 1:3] = trans[..., 0:2]

        # 6）
        if normalize:
            idx_uv_transformed[..., 1] /= norm_factor
            idx_uv_transformed[..., 2] /= norm_factor

        return idx_uv_transformed
    

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
    
def visualize_corr_s(corr_s, title="corr_s projection"):
    """
    corr_s: Tensor of shape [B, C, H_r, W_r]
    """
    # 1) 选第 0 个 batch
    x = corr_s[0]  # [C, H_r, W_r]

    # 2) 聚合通道：你可以选单个通道，也可以做平均/最大
    #    这里我们取通道维度的平均，得到 [H_r, W_r]
    img = x.mean(dim=0).cpu().detach().numpy()  

    # 3) 画图
    plt.figure(figsize=(8, 2))
    plt.title(title + " (mean over channels)")
    plt.imshow(img, cmap='jet', aspect='auto')  
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("range width")
    plt.ylabel("range height")
    plt.tight_layout()
    plt.show()

class InnerFeatureFusion(nn.Module):
    """
    将低分辨率特征上采样并与高分辨率特征拼接，
    再经过两层 Conv+InstanceNorm+ReLU 融合，输出通道不变。
    输入：
        feat_h4: Tensor, shape (B, C, H/4, W/4)
        feat_h8: Tensor, shape (B, C, H/8, W/8)
    输出：
        Tensor, shape (B, C, H/4, W/4)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.in1   = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2   = nn.InstanceNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, feat_h8: torch.Tensor, feat_h4: torch.Tensor) -> torch.Tensor:
        up = self.upsample(feat_h8)                      # -> (B, C, H/4, W/4)
        cat = torch.cat([feat_h4, up], dim=1)            # -> (B, 2C, H/4, W/4)
        x = self.relu(self.in1(self.conv1(cat)))         # -> (B, C, H/4, W/4)
        x = self.relu(self.in2(self.conv2(x)))           # -> (B, C, H/4, W/4)
        return x

class VoxelToPV(nn.Module):
    def __init__(self, in_channels: int, R: int):
        """
        Args:
          in_channels: 每个相机特征图的通道数 C
          R: 每条透视射线上采样的径向点数
        """
        super().__init__()
        self.R = R
        # 将 C*R 通道压回 C
        self.conv_reduce = nn.Conv2d(in_channels * R, in_channels,
                                     kernel_size=3, padding=1, bias=True)

    def forward(self,
                surround_view_feats,
                idx_uv):
        """
        Args:
          surround_view_feats: List of length 6, 每个 [B, C, Hf, Wf]
          idx_uv:              [B, H_sph, W_sph, R, 3]，最后一维 (cam_idx, u, v)

        Returns:
          pv_feat: [B, C, H_sph, W_sph]
        """
        B, Hs, Ws, R, _ = idx_uv.shape
        C = surround_view_feats[0].shape[1]
        device = surround_view_feats[0].device

        # 拆分 cam_idx_map 与 uv 坐标
        cam_idx_map = idx_uv[..., 0].long()        # -> [B, Hs, Ws, R]
        uv = idx_uv[..., 1:].float()               # -> [B, Hs, Ws, R, 2]

        # 归一化坐标到 [-1,1]，供 grid_sample 使用
        # 假设所有视角特征图 Hf, Wf 相同：
        Hf, Wf = surround_view_feats[0].shape[2:]
        u = uv[..., 0]  # x 方向像素
        v = uv[..., 1]  # y 方向像素
        x_norm = 2.0 * u / (Wf - 1) - 1.0
        y_norm = 2.0 * v / (Hf - 1) - 1.0
        # 组装成 grid， grid 的最后一维顺序是 (x, y)
        grid = torch.stack([x_norm, y_norm], dim=-1)  # [B, Hs, Ws, R, 2]
        # 为了一次性采样，把 (Hs, Ws, R) 合并：
        grid_flat = grid.view(B, Hs, Ws * R, 2)        # [B, Hs, Ws*R, 2]

        # 在所有相机视角上做采样并累加
        feat_sum   = torch.zeros(B, C, Hs, Ws, R, device=device)
        count_mask = torch.zeros(B, 1, Hs, Ws, R, device=device)  # 用于计数每个位置的采样次数

        for cam_idx, feat in enumerate(surround_view_feats):
            # feat: [B, C, Hf, Wf]
            # grid_sample 输出 [B, C, Hs, Ws*R]
            sampled = F.grid_sample(
                feat, grid_flat,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            # reshape 回 [B, C, Hs, Ws, R]
            sampled = sampled.view(B, C, Hs, Ws, R)
            # 当前相机有效投影mask
            mask_i = (cam_idx_map == cam_idx).unsqueeze(1).float()  # [B,1,Hs,Ws,R]

            feat_sum = feat_sum + sampled * mask_i
            count_mask = count_mask + mask_i

        # 计算每个位置的平均值
        valid_mask = (count_mask > 0.0).float()        
        count_mask = count_mask.clamp(min=1.0)
        fused = feat_sum / count_mask
        fused = fused * valid_mask

        # sampled: [B, C, Hs, Ws, R]
        # 把 C 和 R 合并到一起
        # 先 permute 到 [B, Hs, Ws, R, C]
        x = fused.permute(0, 2, 3, 4, 1)
        # reshape 到 [B, Hs, Ws, R*C]
        x = x.reshape(B, Hs, Ws, R * C)
        # 再 permute 到 [B, R*C, Hs, Ws]
        x = x.permute(0, 3, 1, 2)

        # 最后 3×3 卷积降回 C
        pv_feat = self.conv_reduce(x)  # [B, C, Hs, Ws]

        valid_mask_2d = (valid_mask.sum(dim=-1) > 0).float()  # [B, 1, Hs, Ws]

        # 统计当前 batch 上正常位置特征的最大绝对值
        max_val = (pv_feat * valid_mask_2d).abs().max().item()
        sentinel = - (max_val * 3)
        pv_feat = pv_feat * valid_mask_2d + (1 - valid_mask_2d) * sentinel  # 保持无效位置为 -1.0
        return pv_feat