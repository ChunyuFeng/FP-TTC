import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from utils.loss import get_loss_scale_map, get_loss_risk_score_map

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet

import torch.distributed as dist
import numpy as np
from utils.dist import is_main_process

from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from .modules.geomwarp import (warp_image, warp_depth, build_proj_pix_from_depth_torch)
import os

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
                 # NEW:
                 use_da_head            = False,
                 da_encoder             = 'vitl',
                 da_max_depth           = 80.0,
                 online_proj            = False,
                 detach_depth_for_proj  = True,
                 ddcl_w                 = 0.10,
                 mvrcl_w                = 0.20,
                 smooth_scale_w         = 0.01,
                 smooth_risk_w          = 0.01,
                 rv_size                = (160,1920),
                 rv_fov_up              = 10.0,
                 rv_fov_down            = -20.0,
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
        
        # --- consistency config
        self.use_da_head = use_da_head
        self.online_proj = online_proj
        self.detach_depth_for_proj = detach_depth_for_proj
        self.ddcl_w = ddcl_w
        self.mvrcl_w = mvrcl_w
        self.smooth_scale_w = smooth_scale_w
        self.smooth_risk_w = smooth_risk_w
        self.rv_H, self.rv_W = rv_size
        self.rv_fov_up = rv_fov_up
        self.rv_fov_down = rv_fov_down

        # --- DepthAnything head (finetune head only)
        self.da_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        if self.use_da_head:
            self.da = DepthAnythingV2(**{**self.da_model_configs[da_encoder], 'max_depth': da_max_depth})
            # freeze backbone
            for p in self.da.pretrained.parameters():
                p.requires_grad = False
            # only head trainable
            for p in self.da.depth_head.parameters():
                p.requires_grad = True
        else:
            self.da = None

        # 相机邻接（环形相邻）
        self.camera_channels = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT']
        self.neighbours = [-1, 1]  # 每个视角与左右相邻
        self.ddcl_viz = True
        

    def forward(self,
                img_prev, img_curr,
                depth_prev, depth_curr,
                proj_pix_prev, proj_pix_curr,
                attn_type,
                attn_splits_list,
                corr_radius_list,
                prop_radius_list,
                num_reg_refine,
                scale_only,
                affine_matrix,
                K_curr,
                T_E_from_C_curr):

    
        # 保留原图 (0..1) 给 photometric/warping
        rgb_prev_raw = img_prev.clamp(0,255) / 255.0   # [B,V,3,H,W]
        rgb_curr_raw = img_curr.clamp(0,255) / 255.0

        # 1) 决定本轮用的 depth（可选 DA head finetune）
        if self.da is not None and self.use_da_head:
            # 可反传版本：训练阶段用 forward()，推理可以换成 no_grad 的 infer
            depth_prev_pred = self._da_forward_batch(img_prev)  # [B,V,1,H,W]
            depth_curr_pred = self._da_forward_batch(img_curr)
        else:
            depth_prev_pred = depth_prev        # 使用数据集提供的 DepthAnything 结果
            depth_curr_pred = depth_curr

        # 2) 在线刷新 proj_pix（如果启用）
        if self.online_proj:
            depth_for_proj_prev = depth_prev_pred.detach() if self.detach_depth_for_proj else depth_prev_pred
            depth_for_proj_curr = depth_curr_pred.detach() if self.detach_depth_for_proj else depth_curr_pred

            # 需要 K_curr / T_E_from_C_curr，从 forward 的参数携带
            # 为了兼容旧接口，我们把它们缓存进 self.tmp_calib（在 forward_with_loss 里赋值）
            K = K_curr        # [B,V,3,3]
            T = T_E_from_C_curr    # [B,V,4,4]

            proj_pix_prev, _, _ = build_proj_pix_from_depth_torch(
                depth_for_proj_prev, K, T, self.rv_H, self.rv_W, self.rv_fov_up, self.rv_fov_down, affine_matrix
            )
            proj_pix_curr, _, _ = build_proj_pix_from_depth_torch(
                depth_for_proj_curr, K, T, self.rv_H, self.rv_W, self.rv_fov_up, self.rv_fov_down, affine_matrix
            )

        ### 1）提取输入的多视角图像的底层特征
        img0, img1 = normalize_img(img_prev, img_curr)
        rgbd0 = torch.cat([img0, depth_prev_pred], dim=2)  # [B, V, 4, H, W]
        rgbd1 = torch.cat([img1, depth_curr_pred], dim=2)  # [B, V, 4, H, W]
        B, V, C, H_img, W_img = img0.shape
        shared_prev, shared_curr = [], []
        for view in range(rgbd0.size(1)):
            p, c = self.extract_feature(rgbd0[:, view], rgbd1[:, view], branch=None)
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

        # 1) 投影 corr 特征到 range-view（risk）
        corr_range = self.project_views_to_range(
            corr_features, proj_pix_curr,
            H_img=160, W_img=320
        )

        # 2) multi-lvl、multi-view 特征投影
        multi_level_ranges_prev = []
        multi_level_ranges_curr = []
        for lvl in range(self.num_scales):
            scale = 2 ** (self.num_scales - 1 - lvl)
            if scale > 1:
                proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
                proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
            else:
                proj_prev_lvl = proj_pix_prev
                proj_curr_lvl = proj_pix_curr

            prev_feats = [feats[lvl] for feats in multi_level_feats_prev]
            curr_feats = [feats[lvl] for feats in multi_level_feats_curr]

            range_prev = self.project_views_to_range(
                prev_feats, proj_prev_lvl,
                H_img=160, W_img=320
            )
            range_curr = self.project_views_to_range(
                curr_feats, proj_curr_lvl,
                H_img=160, W_img=320
            )

            multi_level_ranges_prev.append(range_prev)
            multi_level_ranges_curr.append(range_curr)

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
            return scales, None
        
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
            gt_depth_map,
            attn_type,                     # str
            attn_splits_list,              # List[int]
            corr_radius_list,              # List[int]
            prop_radius_list,              # List[int]
            num_reg_refine,                # int
            scale_only,
            # NEW:
            K_curr,                # [B,V,3,3]
            affine_matrix,
            T_E_from_C_curr,       # [B,V,4,4]
            T_Ecurr_from_Eprev,    # [B,V,4,4]
        ):

        self.tmp_K_curr = K_curr
        self.tmp_T_E_from_C = T_E_from_C_curr
        self.tmp_T_Ecurr_from_Eprev = T_Ecurr_from_Eprev

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
            affine_matrix    = affine_matrix,
            K_curr           = K_curr,
            T_E_from_C_curr  = T_E_from_C_curr,
        )
        
        if not hasattr(self, "mag_aligner"):
            from utils.mag_aligner import MagAligner
            self.mag_aligner = MagAligner(
                beta=0.98, warmup_steps=1500,
                target_ratio_ddcl=1.0,     # 让 ddcl 贡献≈loss_s
                target_ratio_smooth=0.2,   # 让 smooth 贡献≈20% loss_s
                clamp_ddcl=(0.005, 0.05),
                clamp_smooth=(0.5, 5.0),
            )

        # supervised（你现有的）
        if scale_only:
            loss_s = get_loss_scale_map(scales, gt_scale_map_with_mask)
            total_loss = loss_s
            # consistency losses + smooth on SCALE
            if self.ddcl_w > 0 or self.mvrcl_w > 0 or self.smooth_scale_w > 0:
                # 取当前帧的深度（来自 DA 或数据集）
                depth_curr_used = depth_curr if (self.da is None or not self.use_da_head) else self._da_forward_batch(img_curr)
                # 几何标定
                K = self.tmp_K_curr; K = affine_matrix @ K  # [B,V,3,3]
                T = self.tmp_T_E_from_C; Trel = self.tmp_T_Ecurr_from_Eprev
                # DDCL
                if self.ddcl_w > 0:
                    ddcl = self._ddcl_loss(img_curr.clamp(0,255)/255.0, depth_curr_used, K, T)
                    total_loss = total_loss + self.ddcl_w * ddcl
                # MVRCL
                if self.mvrcl_w > 0:
                    depth_prev_used = depth_prev if (self.da is None or not self.use_da_head) else self._da_forward_batch(img_prev)
                    mvrcl = self._mvrcl_loss(img_prev.clamp(0,255)/255.0, img_curr.clamp(0,255)/255.0,
                                              depth_prev_used, K, T, Trel)
                    total_loss = total_loss + self.mvrcl_w * mvrcl
                # Smooth on scale map（range-view 的引导用 GT range depth）
                if self.smooth_scale_w > 0:
                    # gt_depth = gt_scale_map_with_mask.new_zeros(gt_scale_map_with_mask.shape[0],1,self.rv_H,self.rv_W)
                    gt_depth = gt_depth_map
                    smooth = self._edge_aware_smooth(scales[-1] if isinstance(scales,list) else scales, gt_depth)
                    total_loss = total_loss + self.smooth_scale_w * smooth

            # 量级对齐
            loss_scale = float(loss_s.detach().mean().item())
            loss_ddcl = float(ddcl.detach().mean().item()) if self.ddcl_w > 0 else None
            loss_smooth = float(smooth.detach().mean().item()) if self.smooth_scale_w > 0 else None
            ddcl_w, smooth_w = self.mag_aligner.update_and_get_weights(loss_scale, loss_ddcl, loss_smooth)
            total_loss = loss_s
            if ddcl   is not None: total_loss = total_loss + ddcl_w   * ddcl
            if smooth is not None: total_loss = total_loss + smooth_w * smooth

            return scales, None, total_loss, None

        else:
            loss_r = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)
            total_loss = loss_r
            # 在 risk 阶段，也让 DA head 继续通过 DDCL/MVRCL 学一致性
            if self.ddcl_w > 0 or self.mvrcl_w > 0 or self.smooth_risk_w > 0:
                depth_curr_used = depth_curr if (self.da is None or not self.use_da_head) else self._da_forward_batch(img_curr)
                K = self.tmp_K_curr; T = self.tmp_T_E_from_C; Trel = self.tmp_T_Ecurr_from_Eprev
                if self.ddcl_w > 0:
                    ddcl = self._ddcl_loss(img_curr.clamp(0,255)/255.0, depth_curr_used, K, T)
                    total_loss = total_loss + self.ddcl_w * ddcl
                if self.mvrcl_w > 0:
                    depth_prev_used = depth_prev if (self.da is None or not self.use_da_head) else self._da_forward_batch(img_prev)
                    mvrcl = self._mvrcl_loss(img_prev.clamp(0,255)/255.0, img_curr.clamp(0,255)/255.0,
                                              depth_prev_used, K, T, Trel)
                    total_loss = total_loss + self.mvrcl_w * mvrcl
                if self.smooth_risk_w > 0:
                    gt_depth = gt_risk_score_map_with_mask.new_zeros(gt_risk_score_map_with_mask.shape[0],1,self.rv_H,self.rv_W)
                    smooth = self._edge_aware_smooth(risks, gt_depth)
                    total_loss = total_loss + self.smooth_risk_w * smooth

            return None, risks, None, total_loss


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
        features_list,        # list 长度 = V，每个 [B, C, Hf, Wf]
        proj_pix,             # [B, H_r, W_r, 3]  (cam_idx, u_img, v_img)，允许 -1 表示无效
        H_img, W_img          # 相机图像分辨率（增强后）
    ):
        """
        从多视角特征提取 range-view 特征，安全处理无效映射（cam=-1 或 u/v=-1）。
        """
        device = proj_pix.device
        B, H_r, W_r, _ = proj_pix.shape
        V = len(features_list)

        # [B,V,C,Hf,Wf] -> 合并视角到 batch: [B*V, C, Hf*Wf]
        feats = torch.stack(features_list, dim=1)     # [B,V,C,Hf,Wf]
        B_, V_, C, Hf, Wf = feats.shape
        assert B_ == B and V_ == V
        feats_flat = feats.view(B * V, C, Hf * Wf).contiguous()

        # 像素 -> 特征图的缩放
        s_u = Wf / float(W_img)
        s_v = Hf / float(H_img)

        # 取出 (cam,u,v)
        cam = proj_pix[..., 0].long()                 # [B,Hr,Wr]
        u   = proj_pix[..., 1].float()
        v   = proj_pix[..., 2].float()

        # 有效掩码（-1 无效）
        valid = (cam >= 0) & (u >= 0) & (v >= 0)      # [B,Hr,Wr]

        # 安全索引（无效的先占位 0，最后用 valid 清零）
        cam_safe = torch.where(valid, cam, torch.zeros_like(cam))
        u_safe   = torch.where(valid, u,   torch.zeros_like(u))
        v_safe   = torch.where(valid, v,   torch.zeros_like(v))

        # 映射到特征分辨率（显式 floor 再 clamp）
        uf = torch.floor(u_safe * s_u).long().clamp(0, Wf - 1)   # [B,Hr,Wr]
        vf = torch.floor(v_safe * s_v).long().clamp(0, Hf - 1)   # [B,Hr,Wr]

        # 展平（**长度从数据推导**，不要手写 H_r*W_r）
        cam_flat   = cam_safe.view(B, -1)            # [B, Npix]
        uf_flat    = uf.view(B, -1)                  # [B, Npix]
        vf_flat    = vf.view(B, -1)                  # [B, Npix]
        valid_flat = valid.view(B, -1)               # [B, Npix]
        Npix       = cam_flat.size(1)                # = H_r * W_r

        # 源特征的线性像素索引
        pix_idx = (vf_flat * Wf + uf_flat).view(-1)  # [B*Npix]

        # 视角索引：把视角合并到 batch 维 => view_idx ∈ [0, B*V-1]
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, Npix)   # [B,Npix]
        view_idx  = (batch_idx * V + cam_flat).view(-1)                         # [B*Npix]

        # 只对有效位置取值
        sel = valid_flat.view(-1)                                               # [B*Npix] bool

        # 目标画布
        out = torch.zeros(B * Npix, C, device=device, dtype=feats.dtype)        # [B*Npix, C]
        if sel.any():
            out[sel] = feats_flat[view_idx[sel], :, pix_idx[sel]]               # gather 有效位置

        # 复原成 [B, C, H_r, W_r]
        out = out.view(B, Npix, C).permute(0, 2, 1).contiguous().view(B, C, H_r, W_r)
        return out

    # def project_views_to_range(
    #     self,
    #     features_list,   # list长度 = V，每个 [B, C, Hf, Wf]
    #     proj_pix,        # [B, H_r, W_r, 3]  (cam_idx, u_img, v_img), 允许 -1 表示无效
    #     H_img, W_img               # 相机图像分辨率（增强后）
    # ):
    #     """
    #     从多视角特征提取 range-view 特征，安全处理无效映射（cam=-1 或 u/v=-1）。
    #     等价于你 numpy 版思路，但不会在 GPU 上越界。
    #     """
    #     B, H_r, W_r, _ = proj_pix.shape
    #     V = len(features_list)
    #     device = proj_pix.device
    #     dtype  = proj_pix.dtype

    #     # [B,V,C,Hf,Wf] -> [B*V, C, Hf*Wf]
    #     feats = torch.stack(features_list, dim=1)                # [B,V,C,Hf,Wf]
    #     B_, V_, C, Hf, Wf = feats.shape
    #     assert B_ == B and V_ == V
    #     feats_flat = feats.view(B*V, C, Hf*Wf).contiguous()

    #     # 下采样比例（从图像像素到特征像素）
    #     s_u = Wf / float(W_img)
    #     s_v = Hf / float(H_img)

    #     cam = proj_pix[..., 0].long()                            # [B,Hr,Wr]
    #     u   = proj_pix[..., 1].float()
    #     v   = proj_pix[..., 2].float()

    #     # 有效位置：cam>=0 且 u/v>=0
    #     valid = (cam >= 0) & (u >= 0) & (v >= 0)                # [B,Hr,Wr]

    #     # 安全索引（无效位置先用 0 占位，后续再用 valid 清零）
    #     cam_safe = torch.where(valid, cam, torch.zeros_like(cam))
    #     u_safe   = torch.where(valid, u,   torch.zeros_like(u))
    #     v_safe   = torch.where(valid, v,   torch.zeros_like(v))

    #     # 映射到特征分辨率并裁剪
    #     uf = (u_safe * s_u).long().clamp(0, Wf-1)               # [B,Hr,Wr]
    #     vf = (v_safe * s_v).long().clamp(0, Hf-1)

    #     # 线性化 index
    #     pix_idx = (vf * Wf + uf).view(B, -1)                    # [B, Hr*Wr]
    #     # batch 维
    #     batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, H_r*W_r).view(-1)  # [B*Hr*Wr]
    #     # 视角合并：view_idx = batch*V + cam
    #     view_idx = (batch_idx * V + cam_safe.view(B, -1)).view(-1)                           # [B*Hr*Wr]
    #     pix_idx  = pix_idx.view(-1)

    #     # gather
    #     selected = feats_flat[view_idx, :, pix_idx]             # [B*Hr*Wr, C]
    #     selected = selected * valid.view(-1, 1).float()         # 无效像素清零

    #     # reshape 回 [B,C,Hr,Wr]
    #     range_feat = selected.view(B, H_r*W_r, C).permute(0, 2, 1).contiguous().view(B, C, H_r, W_r)
    #     return range_feat

    
    # def project_views_to_range(
    #     self,
    #     features_list,   # list of V tensors, each [B, C, H_feat, W_feat]
    #     proj_pix,        # LongTensor [B, H_r, W_r, 3] = (cam_idx, u_orig, v_orig)
    #     H_img=160, W_img=320
    # ):
    #     """
    #     把 V 路视角、同一尺度的特征图，按 proj_pix 映射到 range-view [B, C, H_r, W_r]。
    #     """
    #     B, H_r, W_r, _ = proj_pix.shape
    #     V = len(features_list)
    #     C = features_list[0].shape[1]
    #     H_feat, W_feat = features_list[0].shape[2], features_list[0].shape[3]

    #     # 1) 下采样比例
    #     s_u = W_feat / W_img
    #     s_v = H_feat / H_img

    #     # 2) 合并成 [B, V, C, Hf, Wf] → [B*V, C, Hf, Wf]
    #     feats = torch.stack(features_list, dim=1)         # [B,V,C,Hf,Wf]
    #     feats = feats.view(B*V, C, H_feat, W_feat)
    #     feats_flat = feats.view(B*V, C, -1)               # [B*V, C, Hf*Wf]

    #     # 3) 拆出并缩放像素索引
    #     cam_idx = proj_pix[..., 0].reshape(B, -1)         # [B, N]
    #     u_orig  = proj_pix[..., 1].float().reshape(B, -1) # [B, N]
    #     v_orig  = proj_pix[..., 2].float().reshape(B, -1) # [B, N]

    #     u_feat = (u_orig * s_u).long().clamp(0, W_feat-1) # [B, N]
    #     v_feat = (v_orig * s_v).long().clamp(0, H_feat-1) # [B, N]

    #     # 4) 计算扁平化后的批次＋视角索引
    #     #    batch_idx ∈ [0..B) 重复 N 次，拼接 cam_idx → [B*N]
    #     batch_idx = torch.arange(B, device=cam_idx.device)\
    #                     .unsqueeze(1).repeat(1, H_r*W_r)\
    #                     .reshape(-1)
    #     view_idx  = batch_idx * V + cam_idx.reshape(-1)   # [B*N]

    #     # 5) 计算在 Hf*Wf 上的线性化像素
    #     pix_idx   = (v_feat * W_feat + u_feat).reshape(-1)  # [B*N]

    #     # 6) 一次性 gather
    #     #    feats_flat[view_idx, :, pix_idx] → [B*N, C]
    #     selected = feats_flat[view_idx, :, pix_idx]

    #     # 7) 重塑回 [B, C, H_r, W_r]
    #     range_feat = selected.view(B, H_r*W_r, C) \
    #                         .permute(0,2,1) \
    #                         .reshape(B, C, H_r, W_r)
    #     return range_feat
    
    def load_depthanything_ckpt(self, ckpt_path: str, strict: bool = False):
        """
        将 ckpt_path 中的权重加载到 self.da (DepthAnythingV2)。
        兼容几种常见结构：{'model':...}, {'state_dict':...}, 纯 state_dict，DDP 的 'module.' 前缀等。
        只吸收以 'pretrained.' (DINOv2 编码器) 与 'depth_head.' 开头的 key。
        """
        if ckpt_path is None or not os.path.isfile(ckpt_path):
            print(f"[DA] checkpoint not found: {ckpt_path}")
            return

        sd_in = torch.load(ckpt_path, map_location='cpu')

        # 取出真正的 state_dict
        if isinstance(sd_in, dict) and 'state_dict' in sd_in:
            sd = sd_in['state_dict']
        elif isinstance(sd_in, dict) and 'model' in sd_in:
            sd = sd_in['model']
        elif isinstance(sd_in, dict):
            sd = sd_in
        else:
            print("[DA] Unsupported checkpoint format; expect dict or dict with 'state_dict'/'model'.")
            return

        # 规范 key：去掉 'module.' / 'da.' 前缀，只保留 da 内的两部分
        filtered = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[len('module.'):]
            if nk.startswith('da.'):
                nk = nk[len('da.'):]
            # 只接收 DepthAnythingV2 内部结构的 key
            if nk.startswith('pretrained.') or nk.startswith('depth_head.'):
                filtered[nk] = v

        if not hasattr(self, "da"):
            print("[DA] FpTTC has no attribute `da` (DepthAnythingV2). Cannot load.")
            return

        # 加载
        load_info = self.da.load_state_dict(filtered, strict=False if not strict else True)

        # 只在主进程打印
        try:
            from utils.dist import is_main_process
            mainp = is_main_process()
        except Exception:
            mainp = True

        if mainp:
            loaded_keys = set(filtered.keys()) - set(getattr(load_info, 'unexpected_keys', []))
            print(f"[DA] Loaded {len(loaded_keys)} keys into DepthAnythingV2.")
            if getattr(load_info, 'missing_keys', []):
                print(f"[DA] Missing ({len(load_info.missing_keys)}) keys (ok if heads mismatch):")
                for k in load_info.missing_keys[:20]:
                    print("    ", k)
                if len(load_info.missing_keys) > 20:
                    print("    ...")
            if getattr(load_info, 'unexpected_keys', []):
                print(f"[DA] Unexpected ({len(load_info.unexpected_keys)}) keys (ignored):")
                for k in load_info.unexpected_keys[:20]:
                    print("    ", k)
                if len(load_info.unexpected_keys) > 20:
                    print("    ...")

        # 不改变现有推理/训练逻辑（是否 finetune 仍由你现有逻辑控制）
        return
    
    def _da_forward_batch(self, imgs):  # imgs: [B,V,3,H,W]
        """
        训练用 DepthAnythingV2 前向：可反传。
        1) ImageNet 归一化；2) 尺寸对齐到 14 的倍数；3) 深度预测；4) 还原回原尺寸。
        返回: depth [B,V,1,H,W]
        """
        B, V, C, H, W = imgs.shape
        device = imgs.device
        dtype  = imgs.dtype

        # 正确的广播形状：用于 [B,3,H,W]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)

        outs = []
        for v in range(V):
            x = imgs[:, v]  # [B,3,H,W]
            # 像素值归一化到 0..1（若已是 0..1 则不会影响数值稳定性）
            if x.max() > 1.5:
                x = x / 255.0
            # ImageNet 标准化（与 DA 的 transforms 保持一致）
            x = (x - mean) / std

            # 对齐到 14 的倍数
            Hn = int((H + 13) // 14 * 14)
            Wn = int((W + 13) // 14 * 14)
            if (Hn != H) or (Wn != W):
                x = F.interpolate(x, size=(Hn, Wn), mode="bilinear", align_corners=True)

            # DepthAnythingV2 正向（可反传）
            depth_v = self.da.forward(x)  # [B, Hn, Wn]

            # 还原到原始分辨率
            if (Hn != H) or (Wn != W):
                depth_v = F.interpolate(depth_v[:, None], size=(H, W), mode="bilinear", align_corners=True)[:, 0]
            else:
                depth_v = depth_v  # [B,H,W]

            outs.append(depth_v.unsqueeze(1))  # [B,1,H,W]

        return torch.stack(outs, dim=1)  # [B,V,1,H,W]
    
    @torch.no_grad()
    def _da_infer_batch(self, imgs):  # imgs: [B,V,3,H,W] (0..255)
        """慢一点但简单稳妥：逐(B,V)送入 DA 的 image2tensor -> forward"""
        B,V,_,H,W = imgs.shape
        depths = []
        for b in range(B):
            row = []
            for v in range(V):
                # image2tensor 内部会做 resize/normalize
                d = self.da.infer_image(imgs[b,v].permute(1,2,0).cpu().numpy(), input_size=518)  # [H,W] numpy
                row.append(torch.from_numpy(d).to(imgs.device).float().unsqueeze(0))             # [1,H,W]
            depths.append(torch.stack(row, dim=0))  # [V,1,H,W]
        depths = torch.stack(depths, dim=0)         # [B,V,1,H,W]
        return depths
    
    # def _edge_aware_smooth(self, pred_map, guide_depth):
    #     """
    #     pred_map   : [B,1,H_r,W_r]   (scale 或 risk)
    #     guide_depth: [B,1,H_r,W_r]   (range-view 深度图：可用 GT range depth)
    #     """
    #     # 梯度
    #     dx = torch.abs(pred_map[:, :, :, 1:] - pred_map[:, :, :, :-1])
    #     dy = torch.abs(pred_map[:, :, 1:, :] - pred_map[:, :, :-1, :])
    #     # 引导边缘
    #     gx = torch.mean(torch.abs(guide_depth[:, :, :, 1:] - guide_depth[:, :, :, :-1]), dim=1, keepdim=True)
    #     gy = torch.mean(torch.abs(guide_depth[:, :, 1:, :] - guide_depth[:, :, :-1, :]), dim=1, keepdim=True)
    #     wx = torch.exp(-gx)
    #     wy = torch.exp(-gy)
    #     loss = (dx * wx).mean() + (dy * wy).mean()
    #     return loss

    def _edge_aware_smooth(self, pred_map, guide_depth, eps=1e-6):
        """
        pred_map   : [B,1,H,W]  (scale 或 risk 的最终输出分辨率)
        guide_depth: [B,1,H',W'] (range-view 引导深度，建议为 GT range depth；可与 pred_map 不同分辨率)
        说明：自动将 guide_depth 双线性插值到 pred_map 的分辨率，再做边缘引导的有限差分平滑。
        """
        # ---- 规范形状 [B,1,H,W] ----
        if guide_depth.dim() == 3:
            guide_depth = guide_depth.unsqueeze(1)         # [B,1,H',W']
        if pred_map.dim() == 3:
            pred_map = pred_map.unsqueeze(1)               # [B,1,H,W]
        if pred_map.size(1) != 1:
            pred_map = pred_map[:, :1]                     # 只取第一通道作为标量图
        if guide_depth.size(1) != 1:
            guide_depth = guide_depth[:, :1]               # 只取第一通道

        # ---- 分辨率对齐：把 guide_depth 拉到与 pred_map 一致 ----
        H, W = pred_map.shape[-2], pred_map.shape[-1]
        if guide_depth.shape[-2:] != (H, W):
            guide_depth = F.interpolate(
                guide_depth, size=(H, W),
                mode='bilinear', align_corners=False
            )

        # ---- 有限差分梯度 ----
        dx = (pred_map[:, :, :, 1:] - pred_map[:, :, :, :-1]).abs()  # [B,1,H,W-1]
        dy = (pred_map[:, :, 1:, :] - pred_map[:, :, :-1, :]).abs()  # [B,1,H-1,W]

        gx = (guide_depth[:, :, :, 1:] - guide_depth[:, :, :, :-1]).abs()  # [B,1,H,W-1]
        gy = (guide_depth[:, :, 1:, :] - guide_depth[:, :, :-1, :]).abs()  # [B,1,H-1,W]

        # ---- 边缘权重：exp(-|∇guide|) ----
        wx = torch.exp(-gx)
        wy = torch.exp(-gy)

        # ---- 可选：无效深度（<=0）不参与
        valid = (guide_depth > eps).float()
        valid_x = valid[:, :, :, 1:] * valid[:, :, :, :-1]
        valid_y = valid[:, :, 1:, :] * valid[:, :, :-1, :]

        loss_x = ((dx * wx) * valid_x).sum() / (valid_x.sum() + eps)
        loss_y = ((dy * wy) * valid_y).sum() / (valid_y.sum() + eps)
        return loss_x + loss_y

    # def _ddcl_loss(self, rgb_curr, depth_curr, K, T_E_from_C):
    #     """
    #     跨视角深度一致：把相机 i 的 3D 点投到相机 j，比较 j 相机的(投影)深度与 j 的深度图采样值
    #     rgb_curr   : [B,V,3,H,W]  (仅用于生成有效 mask，可不严格用)
    #     depth_curr : [B,V,1,H,W]
    #     K          : [B,V,3,3]
    #     T_E_from_C : [B,V,4,4]
    #     """
    #     B,V,_,H,W = rgb_curr.shape
    #     device = rgb_curr.device
    #     total = 0.0
    #     count = 0
    #     for dv in self.neighbours:  # (-1,+1)
    #         j_ofs = dv
    #         for vi in range(V):
    #             vj = (vi + j_ofs) % V
    #             # i->j
    #             z_pred_j, uv_j, valid = warp_depth(
    #                 depth_curr[:,vi:vi+1],      # [B,1,1,H,W]
    #                 K[:,vi:vi+1],
    #                 T_E_from_C[:,vi:vi+1],
    #                 K[:,vj:vj+1],
    #                 T_E_from_C[:,vj:vj+1],
    #                 H, W
    #             )  # z_pred_j:[B,1,1,H,W], uv_j:[B,1,2,H,W], valid:[B,1,1,H,W]

    #             # 采样 j 的深度为观测
    #             grid_u = 2.0*(uv_j[:,0,0]/(W-1.0)) - 1.0
    #             grid_v = 2.0*(uv_j[:,0,1]/(H-1.0)) - 1.0
    #             grid = torch.stack([grid_u, grid_v], dim=-1)  # [B,H,W,2]
    #             depth_j = depth_curr[:,vj]                    # [B,1,H,W]
    #             depth_j_flat = F.grid_sample(depth_j, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    #             # L1(log 深度差)
    #             mask = valid[:,0] * (depth_j_flat>0).float()
    #             if mask.sum() < 1: 
    #                 continue
    #             loss_ij = (torch.log(z_pred_j[:,0].clamp_min(1e-3)) - torch.log(depth_j_flat.clamp_min(1e-3))).abs()
    #             total += (loss_ij * mask).sum() / (mask.sum()+1e-6)
    #             count += 1
    #     return total / max(1, count)


    # def _ddcl_loss(self, rgb_curr, depth_curr, K, T_E_from_C,
    #             viz_prob: float = 1, viz_dir: str = "./log/ddcl_debug"):
    #     """
    #     跨视角深度一致（DDCL）：
    #     将相机 i 的深度投影到相机 j，得到 z_pred_j 与 uv_j，
    #     用 grid_sample 从 j 的深度图采样得到 depth_j_flat，
    #     最后对 log 深度差做 L1（仅在有效区域）。
    #     可视化：随机抽样将 (i->j) 的投影/误差等保存到 viz_dir。
    #     Args:
    #     rgb_curr    : [B,V,3,H,W]   （仅用于可视化）
    #     depth_curr  : [B,V,1,H,W]
    #     K           : [B,V,3,3]
    #     T_E_from_C  : [B,V,4,4]
    #     viz_prob    : 每个 (i->j) 对被可视化的概率
    #     viz_dir     : 可视化输出目录
    #     Returns:
    #     loss (Tensor scalar)
    #     """
    #     B, V, _, H, W = rgb_curr.shape
    #     device = rgb_curr.device

    #     # 结果与计数
    #     total = torch.zeros((), device=device, dtype=depth_curr.dtype)
    #     count = 0

    #     # 规范化 self.neighbours，避免 (tuple-of-tuple) 之类带来 int+tuple 报错
    #     neigh = getattr(self, "neighbours", [-1, +1])
    #     if isinstance(neigh, (int, float)):
    #         neigh = [int(neigh)]
    #     elif isinstance(neigh, tuple):
    #         neigh = [int(x[0] if isinstance(x, (list, tuple)) else x) for x in neigh]
    #     elif isinstance(neigh, list):
    #         neigh = [int(x[0] if isinstance(x, (list, tuple)) else x) for x in neigh]
    #     else:
    #         neigh = [-1, +1]

    #     for j_ofs in neigh:                 # 相机环的偏移（例如 -1 或 +1）
    #         for vi in range(V):             # 源相机 i
    #             vj = (vi + j_ofs) % V       # 目标相机 j（环绕相邻）

    #             # i -> j：投影得到 j 像素上的 uv 与 j 坐标系下的 z 预测
    #             z_pred_j, uv_j, valid = warp_depth(
    #                 depth_curr[:, vi:vi+1],      # [B,1,1,H,W]
    #                 K[:, vi:vi+1],               # [B,1,3,3]
    #                 T_E_from_C[:, vi:vi+1],      # [B,1,4,4]
    #                 K[:, vj:vj+1],               # [B,1,3,3]
    #                 T_E_from_C[:, vj:vj+1],      # [B,1,4,4]
    #                 H, W
    #             )  # z_pred_j: [B,1,1,H,W]  uv_j: [B,1,2,H,W]  valid: [B,1,1,H,W]

    #             # 将 uv_j 像素坐标归一化到 [-1,1] 以便 grid_sample
    #             grid_u = 2.0 * (uv_j[:, 0, 0] / (W - 1.0)) - 1.0   # [B,H,W]
    #             grid_v = 2.0 * (uv_j[:, 0, 1] / (H - 1.0)) - 1.0   # [B,H,W]
    #             grid = torch.stack([grid_u, grid_v], dim=-1)       # [B,H,W,2]
    #             # 处理 NaN / Inf
    #             grid = torch.where(torch.isfinite(grid), grid, torch.zeros_like(grid))

    #             # 从 j 的深度图采样
    #             depth_j = depth_curr[:, vj]  # [B,1,H,W]
    #             depth_j_flat = F.grid_sample(
    #                 depth_j, grid, mode='bilinear',
    #                 padding_mode='zeros', align_corners=True
    #             )  # [B,1,H,W]

    #             # 有效掩码：warp 有效 且 被采样到的深度 > 0
    #             mask = valid[:, 0] * (depth_j_flat > 0).float()  # [B,1,H,W]
    #             if mask.sum() < 1:
    #                 # 可选：少量可视化帮助定位“完全无覆盖”的对
    #                 if is_main_process() and hasattr(self, "ddcl_viz") and self.ddcl_viz:
    #                     if torch.rand(1, device=device).item() < viz_prob:
    #                         try:
    #                             viz_ddcl_pair(
    #                                 out_dir=viz_dir,
    #                                 rgb_i=rgb_curr[:, vi],          # [B,3,H,W]
    #                                 rgb_j=rgb_curr[:, vj],          # [B,3,H,W]
    #                                 depth_j=depth_curr[:, vj],      # [B,1,H,W]
    #                                 uv_j=uv_j,                      # [B,1,2,H,W]
    #                                 z_pred_j=z_pred_j,              # [B,1,1,H,W]
    #                                 valid=valid,                    # [B,1,1,H,W]
    #                                 prefix=f'i{vi}_to_j{vj}_nocover'
    #                             )
    #                         except Exception as e:
    #                             if is_main_process():
    #                                 print(f"[DDCL viz] skip (no cover) due to {e}")
    #                 continue

    #             # log-L1 深度一致
    #             loss_ij = (torch.log(z_pred_j[:, 0].clamp_min(1e-3))  # [B,1,H,W]
    #                     - torch.log(depth_j_flat.clamp_min(1e-3))).abs()
    #             total = total + (loss_ij * mask).sum() / (mask.sum() + 1e-6)
    #             count += 1

    #             # 随机可视化一部分 (i->j) 对，避免生成过多图像
    #             if is_main_process() and hasattr(self, "ddcl_viz") and self.ddcl_viz:
    #                 if torch.rand(1, device=device).item() < viz_prob:
    #                     try:
    #                         viz_ddcl_pair(
    #                             out_dir=viz_dir,
    #                             rgb_i=rgb_curr[:, vi],          # [B,3,H,W]
    #                             rgb_j=rgb_curr[:, vj],          # [B,3,H,W]
    #                             depth_j=depth_curr[:, vj],      # [B,1,H,W]
    #                             uv_j=uv_j,                      # [B,1,2,H,W]
    #                             z_pred_j=z_pred_j,              # [B,1,1,H,W]
    #                             valid=valid,                    # [B,1,1,H,W]
    #                             prefix=f'i{vi}_to_j{vj}'
    #                         )
    #                     except Exception as e:
    #                         # 可视化失败不影响训练
    #                         if is_main_process():
    #                             print(f"[DDCL viz] skip due to {e}")

    #     # 返回平均损失
    #     return total / max(1, count)

    def _ddcl_loss(self, rgb_curr, depth_curr, K, T_E_from_C,
                viz_prob: float = 0.1, viz_dir: str = "./log/ddcl_debug"):
        """
        跨视角深度一致（DDCL）：
        1) 将相机 i 的深度 warp 到相机 j，得到 z_pred_j 与 uv_j(像素坐标)
        2) 用 grid_sample 从 j 的深度图采样得到 depth_j_flat
        3) 在有效区域上做 log-L1 损失

        Args:
        rgb_curr    : [B,V,3,H,W]  (仅可视化用，数值范围任意)
        depth_curr  : [B,V,1,H,W]
        K           : [B,V,3,3]
        T_E_from_C  : [B,V,4,4]
        viz_prob    : 每个(i->j)对被可视化的概率
        viz_dir     : 可视化输出目录

        Returns:
        loss (scalar tensor)
        """
        B, V, _, H, W = rgb_curr.shape
        device = rgb_curr.device

        # 结果与计数
        total = torch.zeros((), device=device, dtype=depth_curr.dtype)
        count = 0

        # 规范化邻接关系
        neigh = getattr(self, "neighbours", [-1, +1])
        if isinstance(neigh, (int, float)):
            neigh = [int(neigh)]
        elif isinstance(neigh, tuple):
            neigh = [int(x[0] if isinstance(x, (list, tuple)) else x) for x in neigh]
        elif isinstance(neigh, list):
            neigh = [int(x[0] if isinstance(x, (list, tuple)) else x) for x in neigh]
        else:
            neigh = [-1, +1]

        os.makedirs(viz_dir, exist_ok=True)

        for j_ofs in neigh:              # 例如 -1 / +1
            for vi in range(V):          # 源相机 i
                vj = (vi + j_ofs) % V    # 目标相机 j

                # i -> j：投影得到 j 像素上的 uv 与 j 坐标系下的 z 预测
                z_pred_j, uv_j, valid = warp_depth(
                    depth_curr[:, vi:vi+1],      # [B,1,1,H,W]
                    K[:, vi:vi+1],               # [B,1,3,3]
                    T_E_from_C[:, vi:vi+1],      # [B,1,4,4]
                    K[:, vj:vj+1],               # [B,1,3,3]
                    T_E_from_C[:, vj:vj+1],      # [B,1,4,4]
                    H, W
                )  # z_pred_j:[B,1,1,H,W], uv_j:[B,1,2,H,W], valid:[B,1,1,H,W]

                # uv_j(像素) -> grid([-1,1])
                grid_u = 2.0 * (uv_j[:, 0, 0] / (W - 1.0)) - 1.0   # [B,H,W]
                grid_v = 2.0 * (uv_j[:, 0, 1] / (H - 1.0)) - 1.0   # [B,H,W]
                grid = torch.stack([grid_u, grid_v], dim=-1)       # [B,H,W,2]
                grid = torch.where(torch.isfinite(grid), grid, torch.zeros_like(grid))

                # 从 j 的深度图采样
                depth_j = depth_curr[:, vj]  # [B,1,H,W]
                depth_j_flat = F.grid_sample(
                    depth_j, grid, mode='bilinear',
                    padding_mode='zeros', align_corners=True
                )  # [B,1,H,W]

                valid_for_viz = valid
                if valid_for_viz.ndim == 4:              # [B,1,H,W] -> [B,1,1,H,W]
                    valid_for_viz = valid_for_viz.unsqueeze(1)
                elif valid_for_viz.ndim == 2:            # [H,W] -> [B,1,1,H,W]（仅当你真的传了2D时）
                    B, _, H, W = depth_curr.shape[0], depth_curr.shape[1], depth_curr.shape[-2], depth_curr.shape[-1]
                    valid_for_viz = valid_for_viz[None, None, None, :, :].expand(B, 1, 1, H, W)
                elif valid_for_viz.ndim != 5:            # 其它形状一律报错，避免静默错误
                    raise ValueError(f"valid has unexpected shape {valid_for_viz.shape}, expected [B,1,1,H,W]")

                # 有效区域：warp 有效 且 被采样到的深度 > 0
                mask = valid[:, 0] * (depth_j_flat > 0).float()  # [B,1,H,W]
                if mask.sum() < 1:
                    # 少量可视化帮助定位“完全无覆盖”的对
                    if getattr(self, "ddcl_viz", False) and torch.rand(1, device=device).item() < viz_prob:
                        try:
                            save_path = os.path.join(viz_dir, f"i{vi}_to_j{vj}_nocover.png")
                            viz_ddcl_pair(
                                out_dir=viz_dir,
                                rgb_j=rgb_curr[:, vj],          # [B,3,H,W]
                                depth_j=depth_curr[:, vj],      # [B,1,H,W]
                                uv_j=uv_j,                      # [B,1,2,H,W]
                                z_pred_j=z_pred_j,              # [B,1,1,H,W]
                                valid=valid_for_viz,                    # [B,1,1,H,W]
                                prefix=f"i{vi}_to_j{vj}"
                            )
                        except Exception as e:
                            if is_main_process():
                                print(f"[DDCL viz] skip (no cover) due to {e}")
                    continue

                # log-L1 深度一致
                loss_ij = (torch.log(z_pred_j[:, 0].clamp_min(1e-3))   # [B,1,H,W]
                        - torch.log(depth_j_flat.clamp_min(1e-3))).abs()
                total = total + (loss_ij * mask).sum() / (mask.sum() + 1e-6)
                count += 1

                # 随机可视化一部分(i->j)
                if getattr(self, "ddcl_viz", False) and torch.rand(1, device=device).item() < viz_prob:
                    try:
                        save_path = os.path.join(viz_dir, f"i{vi}_to_j{vj}.png")
                        viz_ddcl_pair(
                            out_dir=viz_dir,
                            rgb_j=rgb_curr[:, vj],          # [B,3,H,W]
                            depth_j=depth_curr[:, vj],      # [B,1,H,W]
                            uv_j=uv_j,                      # [B,1,2,H,W]
                            z_pred_j=z_pred_j,              # [B,1,1,H,W]
                            valid=valid_for_viz,                    # [B,1,1,H,W]
                            prefix=f"i{vi}_to_j{vj}"
                        )
                    except Exception as e:
                        if is_main_process():
                            print(f"[DDCL viz] skip due to {e}")

        return total / max(1, count)


    # def _mvrcl_loss(self, rgb_prev, rgb_curr, depth_prev, K, T_E_from_C, T_Ecurr_from_Eprev):
    #     """
    #     同一相机的时序重建：用 prev 的深度把 prev warp 到 curr，做 photometric。
    #     """
    #     B,V,_,H,W = rgb_curr.shape
    #     photometric = 0.0
    #     count = 0
    #     for v in range(V):
    #         # prev camera -> ego(prev) -> ego(curr) -> curr camera
    #         # 等价于把 T_E_from_C_prev 和 T_E_from_C_curr 结合 T_Ecurr_from_Eprev[v]
    #         # 这里我们把 src 的自车系点右乘 T_Ecurr_from_Eprev
    #         # 做法：先用 src(T_E_from_C_prev) 得到 Ego(prev)，再乘 rel，最后投到 curr 相机
    #         T_src = T_E_from_C[:, v:v+1]                   # [B,1,4,4]
    #         T_tgt = T_E_from_C[:, v:v+1]
    #         # 临时把相对位姿乘到目标侧（简化 warp_image 的接口：把 src->tgt 的变换藏在T_tgt里）
    #         T_rel = T_Ecurr_from_Eprev[:, v:v+1]           # [B,1,4,4]
    #         # 组合：Xc_src -> Ego(prev) -> Ego(curr) = T_rel @ (T_src @ Xc_src)
    #         # 在 warp_image 里我们是用 "Xc_tgt = inv(T_tgt) @ (T_src @ Xc_src)"，
    #         # 因此这里把 T_tgt 改成  inv(T_rel) @ T_tgt  等价于在函数内部右乘 T_rel
    #         T_tgt_eff = torch.inverse(T_rel) @ T_tgt

    #         recon, valid = warp_image(
    #             rgb_prev[:,v:v+1], depth_prev[:,v:v+1],
    #             K[:,v:v+1], T_src,
    #             K[:,v:v+1], T_tgt_eff,
    #             H, W
    #         )
    #         # photometric (SSIM+L1)
    #         curr = rgb_curr[:,v:v+1]
    #         abs_diff = torch.abs(curr - recon).mean(2, True)  # [B,1,H,W]
    #         # 简化的 ssim（可以替换为你贴过的 compute_photometric_loss）
    #         loss = abs_diff
    #         mask = valid
    #         if mask.sum() < 1:
    #             continue
    #         photometric += (loss * mask).sum() / (mask.sum()+1e-6)
    #         count += 1
    #     return photometric / max(1, count)
    
    def _mvrcl_loss(
        self,
        rgb_prev, rgb_curr, depth_prev,
        K, T_E_from_C, T_Ecurr_from_Eprev,
        viz_prob: float = 0.0,
        viz_dir: str = "./log/mvrcl_debug"
    ):
        """
        prev->curr：用 prev 深度重建 curr 并做 photometric（这里用 L1）。
        另：在 src 坐标系上可视化 src→curr 的位移，证明 src 确实被几何 warp 过。
        """
        B, V, _, H, W = rgb_curr.shape
        photometric, count = 0.0, 0
        device = rgb_curr.device

        for v in range(V):
            T_src = T_E_from_C[:, v:v+1]           # prev 相机 -> Ego(prev)
            T_tgt = T_E_from_C[:, v:v+1]           # curr 相机 -> Ego(curr)
            T_rel = T_Ecurr_from_Eprev[:, v:v+1]   # Ego(prev) -> Ego(curr)
            T_tgt_eff = torch.inverse(T_rel) @ T_tgt

            # 用 prev 深度得到 src->tgt 的 uv
            z_pred_tgt, uv_tgt, valid = warp_depth(
                depth_prev[:, v:v+1],     # [B,1,1,H,W]
                K[:, v:v+1],              # [B,1,3,3]
                T_src,                    # [B,1,4,4]
                K[:, v:v+1],              # [B,1,3,3]
                T_tgt_eff,                # [B,1,4,4]
                H, W
            )  # uv_tgt:[B,1,2,H,W], valid:[B,1,1,H,W]

            # 归一化 uv → grid
            grid_u = 2.0 * (uv_tgt[:, 0, 0] / (W - 1.0)) - 1.0   # [B,H,W]
            grid_v = 2.0 * (uv_tgt[:, 0, 1] / (H - 1.0)) - 1.0   # [B,H,W]
            grid   = torch.stack([grid_u, grid_v], dim=-1)       # [B,H,W,2]
            grid   = torch.where(torch.isfinite(grid), grid, torch.zeros_like(grid))

            # === 关键修正：给 grid_sample 的输入要是 4D [B,3,H,W]，不要 [B,1,3,H,W] ===
            src_rgb = rgb_prev[:, v]          # [B,3,H,W]  (之前是 rgb_prev[:, v:v+1] → [B,1,3,H,W])
            tgt_rgb = rgb_curr[:, v]          # [B,3,H,W]

            recon_on_curr = F.grid_sample(
                src_rgb,                      # 4D
                grid, mode='bilinear',
                padding_mode='zeros', align_corners=True
            )  # [B,3,H,W]

            # Photometric：对通道取平均，mask 也要挤掉多余维度
            abs_diff = torch.abs(tgt_rgb - recon_on_curr).mean(1, keepdim=True)  # [B,1,H,W]
            mask     = valid[:, 0]                                               # [B,1,H,W]
            if mask.sum() > 0:
                photometric += (abs_diff * mask).sum() / (mask.sum() + 1e-6)
                count += 1

            # 可视化（按概率）
            if getattr(self, "mvrcl_viz", True) and is_main_process():
                if torch.rand(1, device=device).item() < viz_prob:
                    try:
                        viz_mvrcl_pair_with_src_overlay(
                            out_dir=viz_dir,
                            rgb_curr=tgt_rgb.unsqueeze(1),         # 变回 [B,1,3,H,W] 以兼容可视化函数
                            recon_on_curr=recon_on_curr.unsqueeze(1),
                            valid_on_curr=valid,                    # [B,1,1,H,W]
                            rgb_src=src_rgb.unsqueeze(1),
                            uv_src2tgt=uv_tgt,                      # [B,1,2,H,W]
                            prefix=f"v{v}",
                            sample_id=0,
                            alpha=0.5
                        )
                    except Exception as e:
                        print(f"[MVRCL viz] skip due to {e}")

        return photometric / max(1, count)



import os
import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def viz_ddcl_pair(
    out_dir: str,
    rgb_j: torch.Tensor,      # [B,3,H,W], 0..1 或 0..255
    depth_j: torch.Tensor,    # [B,1,H,W], 米/任意单位，>0 有效
    uv_j: torch.Tensor,       # [B,1,2,H,W], 以像素坐标 (u,v)
    z_pred_j: torch.Tensor,   # [B,1,1,H,W], 将 i 深度投到 j 后的 Z
    valid: torch.Tensor,      # [B,1,1,H,W]，warp 有效掩码
    prefix: str = "i2j",
    sample_id: int = 0,
    alpha: float = 0.5,       # 覆盖到 RGB 的透明度
    cmap: str = "magma"       # 统一用一个 colormap
):
    """
    生成三张图：
      1) j 视角深度 (depth_j)
      2) i->j 投影深度在 j 像素平面上的栅格化 (z_pred_on_j)
      3) j 视角 RGB 叠加 2)
    注意：2) 与 1) 使用**相同的 vmin/vmax**，colorbar 一致，便于对比。
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- 取 batch 中的一条 ----
    rgb = rgb_j[sample_id]          # [3,H,W]
    dep = depth_j[sample_id, 0]     # [H,W]
    H, W = dep.shape
    uv  = uv_j[sample_id, 0]        # [2,H,W]
    zp  = z_pred_j[sample_id, 0, 0] # [H,W]
    vm  = valid[sample_id, 0, 0]    # [H,W]  (True/False)

    # ---- 准备 j 的 RGB ----
    # 允许输入 0..1 或 0..255，这里统一到 uint8
    rgb_np = rgb.detach().cpu().float().clamp(0, 255).numpy()
    if rgb_np.max() <= 1.0:
        rgb_np = (rgb_np * 255.0).round()
    rgb_np = rgb_np.astype(np.uint8).transpose(1, 2, 0)  # [H,W,3]

    # ---- j 的深度（用于 colorbar 标定）----
    dep_np = dep.detach().cpu().numpy()
    dep_mask = (dep_np > 0) & np.isfinite(dep_np)
    if dep_mask.sum() == 0:
        # 没有有效深度，直接退出
        return

    # 用 j 深度的分位数做稳健可视化范围（也可以改成 min/max）
    vals = dep_np[dep_mask]
    vmin = float(np.percentile(vals, 2.0))
    vmax = float(np.percentile(vals, 98.0))
    if vmin >= vmax:   # 极端情况回退
        vmin, vmax = float(vals.min()), float(vals.max())

    # ---- 把 i->j 的 z_pred 按 uv 栅格化到 j 平面（Z-buffer 取近处）----
    # 像素化
    u = uv[0].detach().cpu().round().long()
    v = uv[1].detach().cpu().round().long()
    z = zp.detach().cpu().float()
    m = vm.detach().cpu().bool() & torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(z) \
        & (u>=0) & (u<W) & (v>=0) & (v<H) & (z>0)

    if m.any():
        u = u[m]; v = v[m]; z = z[m]
        lin = (v * W + u).view(-1)                    # [N]
        # 用 scatter_reduce 做“每个像素取最小深度”（更靠近相机）
        # 需要 torch>=1.12; 若你的版本不支持，可换成手写 z-buffer 循环（见注释）
        canvas = torch.full((H*W,), float("inf"))
        canvas = canvas.scatter_reduce(0, lin, z.view(-1), reduce="amin", include_self=True)
        zpred_on_j = canvas.view(H, W).numpy()
        zpred_valid = np.isfinite(zpred_on_j)
        zpred_on_j[~zpred_valid] = np.nan
    else:
        zpred_on_j = np.full((H, W), np.nan, dtype=np.float32)
        zpred_valid = np.zeros((H, W), dtype=bool)

    # ================= 保存 3 张图 =================
    # 1) j 的深度（带 colorbar）
    plt.figure(figsize=(8, 4))
    im = plt.imshow(dep_np, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("depth_j")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_depth_j.png"), dpi=200)
    plt.close()

    # 2) i->j 栅格化后的深度（用与 1 相同的 vmin/vmax 与 colorbar）
    plt.figure(figsize=(8, 4))
    im = plt.imshow(zpred_on_j, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("z_pred_on_j (rasterized)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_zpred_on_j.png"), dpi=200)
    plt.close()

    # 3) RGB 叠加 z_pred_on_j（同一 colorbar）
    plt.figure(figsize=(8, 4))
    plt.imshow(rgb_np)  # 底图
    z_masked = np.ma.masked_invalid(zpred_on_j)  # nan 处透明
    plt.imshow(z_masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    # 单独再画一个 colorbar，保持与深度相同的范围
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_clim(vmin, vmax)
    plt.colorbar(mappable, fraction=0.046, pad=0.04)
    plt.title("rgb_j + z_pred_on_j overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_rgb_j_overlay_zpred.png"), dpi=200)
    plt.close()


import os
import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def viz_mvrcl_pair_with_src_overlay(
    out_dir: str,
    rgb_curr: torch.Tensor,          # [B,1,3,H,W] 或 [B,3,H,W]
    recon_on_curr: torch.Tensor,     # [B,1,3,H,W] 或 [B,3,H,W]
    valid_on_curr: torch.Tensor,     # [B,1,1,H,W] / [B,1,H,W] / [H,W]
    *,
    rgb_src: torch.Tensor,           # [B,1,3,H,W] 或 [B,3,H,W]
    uv_src2tgt: torch.Tensor,        # [B,1,2,H,W]  —— 这是 src→curr 的 uv
    prefix: str = "mvrcl",
    sample_id: int = 0,
    alpha: float = 0.5,
    flow_stride: int = 16,           # 画稀疏箭头的步长
    flow_max: float = 40.0           # 用于颜色归一化的最大位移（像素）
):
    os.makedirs(out_dir, exist_ok=True)

    def _img(x):  # -> [3,H,W]
        if x.dim() == 5: x = x[sample_id, 0]
        elif x.dim() == 4: x = x[sample_id]
        else: raise ValueError(f"unexpected dim {tuple(x.shape)}")
        return x

    def _to_u8(x3):  # [3,H,W] -> [H,W,3] uint8
        x = x3.detach().cpu().float().clamp(0, 255).numpy()
        if x.max() <= 1.0: x *= 255.0
        return x.astype(np.uint8).transpose(1, 2, 0)

    def _valid(m):  # -> [H,W] bool
        if m.dim() == 5: m = m[sample_id, 0, 0]
        elif m.dim() == 4: m = m[sample_id, 0]
        return (m > 0)

    curr  = _to_u8(_img(rgb_curr))
    recon = _to_u8(_img(recon_on_curr))
    vm_c  = _valid(valid_on_curr).cpu().numpy()  # [H,W]
    H, W  = vm_c.shape

    # ===== (1) curr 侧三张 =====
    # curr
    plt.figure(figsize=(8,4)); plt.imshow(curr); plt.title("curr RGB"); plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_curr_rgb.png"), dpi=200); plt.close()
    # recon_on_curr
    plt.figure(figsize=(8,4)); plt.imshow(recon); plt.title("recon (prev→curr)"); plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_recon_on_curr.png"), dpi=200); plt.close()
    # overlay on curr（只在 valid 区域叠加）
    mask3 = np.broadcast_to((~vm_c)[...,None], (H,W,3))
    plt.figure(figsize=(8,4))
    plt.imshow(curr)
    plt.imshow(np.ma.array(recon, mask=mask3), alpha=alpha)
    plt.title("overlay on curr (valid-only)")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_overlay_on_curr.png"), dpi=200)
    plt.close()

    # ===== (2) src 侧：在 src 上画 “src→curr 位移” =====
    src = _to_u8(_img(rgb_src))
    # 取 uv 并计算位移（像素）
    uv = uv_src2tgt[sample_id, 0]        # [2,H,W]
    u2 = uv[0].detach().cpu().numpy()    # tgt u
    v2 = uv[1].detach().cpu().numpy()    # tgt v
    u1, v1 = np.meshgrid(np.arange(W), np.arange(H))  # src 坐标
    du, dv = u2 - u1, v2 - v1
    mag = np.sqrt(du**2 + dv**2)

    # (2a) 位移热力图覆盖 src（证明 warp 确实发生）
    mag_norm = np.clip(mag / flow_max, 0, 1)
    plt.figure(figsize=(8,4))
    plt.imshow(src)
    plt.imshow(mag_norm, cmap="magma", alpha=0.6)
    plt.title("src + displacement magnitude (src→curr)")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_src_overlay_disp.png"), dpi=200)
    plt.close()

    # (2b) 稀疏箭头（更直观的方向/位移）
    step = max(1, flow_stride)
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    YY, XX = np.meshgrid(ys, xs, indexing='ij')
    DU = du[YY, XX]; DV = dv[YY, XX]

    plt.figure(figsize=(8,4))
    plt.imshow(src)
    # 只画有效区域的箭头
    valid_sparse = vm_c[YY, XX]
    plt.quiver(
        XX[valid_sparse], YY[valid_sparse],
        DU[valid_sparse], DV[valid_sparse],
        angles='xy', scale_units='xy', scale=1.0, width=0.002, headwidth=3
    )
    plt.title("src + sparse flow vectors (src→curr)")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_src_overlay_arrows.png"), dpi=200)
    plt.close()

def _sanity_check_proj_pix(proj_pix, H_img, W_img, V, tag="curr"):
    # proj_pix: [B, H_r, W_r, 3]  (cam_idx, u, v)
    cam = proj_pix[..., 0].clone()
    u   = proj_pix[..., 1].clone()
    v   = proj_pix[..., 2].clone()

    # 统计
    total = cam.numel()
    holes = (cam < 0).sum().item()
    bad_cam = ((cam >= V) | (cam < -1)).sum().item()
    bad_u   = ((u < -1) | (u >= W_img)).sum().item()
    bad_v   = ((v < -1) | (v >= H_img)).sum().item()

    print(f"[check {tag}] total={total}, holes={holes}, bad_cam={bad_cam}, bad_u={bad_u}, bad_v={bad_v}")

    # 触发错误位置
    assert bad_cam == 0, "cam_idx out of range"
    # 注意：允许 -1（表示洞），但不允许 < -1 或 >= W/H
    # 真正 gather 前最好把 -1 的位置用 mask 屏蔽掉（见第二步）

