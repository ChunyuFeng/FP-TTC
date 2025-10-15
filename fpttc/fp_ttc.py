import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from utils.loss import get_loss_scale_map, get_loss_risk_score_map

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet
from fpttc.rvt.range_view_transformer import RangeViewTransformer

import torch.distributed as dist
import numpy as np
from utils.dist import is_main_process

import matplotlib.pyplot as plt
import time


def _safe_bilinear(x, *, size=None, scale_factor=None, align_corners=True):
    orig_dtype = x.dtype
    if orig_dtype == torch.bfloat16:
        x = x.float()
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
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=4,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,
                 # ---- 蒸馏相关超参 ----
                 lambda_feat=1.0,       # 多尺度 Range 特征蒸馏系数
                 lambda_corr=1.0,       # corr Range 蒸馏系数
                 lambda_cos=0.1,        # 方向余弦损失权重
                 use_kd=True            # 是否启用蒸馏
                 ):
        super(FpTTC, self).__init__()
        self.num_scales = num_scales
        self.use_kd = use_kd
        self.lambda_feat = float(lambda_feat)
        self.lambda_corr = float(lambda_corr)
        self.lambda_cos = float(lambda_cos)

        self.camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

        self.cnet = CNNEncoder(output_dim=feature_channels,
                               num_output_scales=num_scales)

        self.featnet = FeatureNet(num_scales=num_scales,
                                  feature_channels=feature_channels,
                                  num_head=num_head,
                                  ffn_dim_expansion=ffn_dim_expansion,
                                  num_transformer_layers=num_transformer_layers)

        self.corrnet = FlowNet(num_scales=num_scales,
                               feature_channels=feature_channels,
                               upsample_factor=upsample_factor,
                               reg_refine=reg_refine)

        self.conv_corr = CorrEncoder(dim_in=feature_channels,
                                     dim_out=feature_channels+1)

        self.scale_net = ScaleNet(num_scales=num_scales,
                                  feature_channels=feature_channels,
                                  upsample_factor=upsample_factor,
                                  num_head=4,
                                  scale_level=num_scales,
                                  reg_refine=reg_refine,
                                  head_type='scale')

        # Risk 分支
        self.conv_corr_risk = CorrEncoder(dim_in=2,
                                          dim_out=feature_channels+1)
        self.risk_net = ScaleNet(num_scales=num_scales,
                                 feature_channels=feature_channels,
                                 upsample_factor=upsample_factor,
                                 num_head=4,
                                 scale_level=num_scales,
                                 reg_refine=reg_refine,
                                 head_type='risk')

        K_bins = 32
        num_views = len(self.camera_channels)
        self.register_buffer(
            "depth_bins",
            torch.logspace(math.log10(1.0), math.log10(40.0), K_bins)
        )
        self.conv_corr_ = CorrEncoder(dim_in=2,
                                      dim_out=feature_channels)

        # 特征分支：按尺度各一个
        self.rvt_feat = nn.ModuleList([
            RangeViewTransformer(
                num_layers=2,
                input_dim=feature_channels,
                d_model=feature_channels,
                nhead=4,
                num_level=num_views,
                num_points=K_bins,
                fov_up=8.0,
                fov_down=-15.0
            )
            for _ in range(num_scales)
        ])

        # corr 分支：输入是 conv_corr_(corr) 的输出通道 = feature_channels
        self.rvt_corr = RangeViewTransformer(
            num_layers=2,
            input_dim=feature_channels,
            d_model=feature_channels,
            nhead=4,
            num_level=num_views,
            num_points=K_bins,
            fov_up=8.0,
            fov_down=-15.0
        )

    # ------------------------- 前向 -------------------------
    def forward(
        self,
        img_prev, img_curr,
        depth_prev, depth_curr,
        proj_pix_prev,
        proj_pix_curr,
        sensor_metas,
        attn_type,
        attn_splits_list,
        corr_radius_list,
        prop_radius_list,
        num_reg_refine,
        scale_only
    ):
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

        # ----- 多尺度 Transformer + 相关性 -----
        corr = None
        multi_level_feats_prev = []
        multi_level_feats_curr = []

        for lvl in range(self.num_scales):
            # [B,V,C,Hs,Ws] -> [B*V,C,Hs,Ws]
            p = prev_lvls[lvl].reshape(B*V, -1, prev_lvls[lvl].shape[3], prev_lvls[lvl].shape[4])
            c = curr_lvls[lvl].reshape(B*V, -1, curr_lvls[lvl].shape[3], curr_lvls[lvl].shape[4])

            fused_prev, fused_curr = self.featnet(
                p, c, lvl, attn_type, attn_splits_list, corr
            )

            # 保存成 [B,V,C,Hs,Ws]
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

        # corr: [B*V, 2, Hc, Wc] -> [B,V,2,Hc,Wc]
        Cc, Hc, Wc = corr.shape[1], corr.shape[2], corr.shape[3]
        corr_bv = corr.view(B, V, Cc, Hc, Wc)
        corr_list = [corr_bv[:, v] for v in range(V)]  # list[V] of [B,2,Hc,Wc]

        # ====== RVT 聚合到 Range View（Student，用于任务损失）======
        # corr -> range（当前帧）
        corr_encoded_list = [self.conv_corr_(c) for c in corr_list]  # list[V] of [B, Ccorr, Hc, Wc]
        H_r = corr_encoded_list[0].shape[2]
        W_r = corr_encoded_list[0].shape[3] * V
        ini_corr_query = torch.cat(corr_encoded_list, dim=3)  # [B, Ccorr, Hc, Wc*V]
        corr_range = self.rvt_corr(
            feats_by_cam=corr_encoded_list,
            cam_K=[sensor_metas['curr'][cam]['K'] for cam in self.camera_channels],
            cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
            cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
            affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
            Hr=H_r, Wr=W_r,
            depth_bins=self.depth_bins,
            ini_query=ini_corr_query
        )

        # 多尺度特征 -> range（前/当前帧）
        multi_level_ranges_prev, multi_level_ranges_curr = [], []
        for lvl in range(self.num_scales):
            prev_feats_list = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
            curr_feats_list = [multi_level_feats_curr[lvl][:, v] for v in range(V)]
            H_r_l = prev_feats_list[0].shape[2]
            W_r_l = prev_feats_list[0].shape[3] * V

            range_prev = self.rvt_feat[lvl](
                feats_by_cam=prev_feats_list,
                cam_K=[sensor_metas['prev'][cam]['K'] for cam in self.camera_channels],
                cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r_l, Wr=W_r_l,
                depth_bins=self.depth_bins
            )
            range_curr = self.rvt_feat[lvl](
                feats_by_cam=curr_feats_list,
                cam_K=[sensor_metas['curr'][cam]['K'] for cam in self.camera_channels],
                cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r_l, Wr=W_r_l,
                depth_bins=self.depth_bins
            )
            multi_level_ranges_prev.append(range_prev)  # [B,C,Hr,Wr]
            multi_level_ranges_curr.append(range_curr)

        # ====== Teacher：DepthAnything 风格“直接投影到 Range”（不经 RVT）======
        with torch.no_grad():
            # corr teacher：先把 2ch corr 编码到 feature_channels，再做投影
            corr_encoded_list_t = [self.conv_corr_(c) for c in corr_list]
            corr_range_t = self.project_views_to_range(
                corr_encoded_list_t, proj_pix_curr, H_img=H_img, W_img=W_img
            )  # [B, Ccorr, H_r, W_r]（这里 H_r/W_r 来自 proj_pix_curr）

            # 多尺度 teacher
            multi_level_ranges_prev_t, multi_level_ranges_curr_t = [], []
            for lvl in range(self.num_scales):
                scale = 2 ** (self.num_scales - 1 - lvl)
                if scale > 1:
                    proj_prev_lvl = proj_pix_prev[:, ::scale, ::scale, :]
                    proj_curr_lvl = proj_pix_curr[:, ::scale, ::scale, :]
                else:
                    proj_prev_lvl = proj_pix_prev
                    proj_curr_lvl = proj_pix_curr

                prev_feats_list_t = [multi_level_feats_prev[lvl][:, v] for v in range(V)]
                curr_feats_list_t = [multi_level_feats_curr[lvl][:, v] for v in range(V)]

                range_prev_t = self.project_views_to_range(prev_feats_list_t, proj_prev_lvl, H_img=H_img, W_img=W_img)
                range_curr_t = self.project_views_to_range(curr_feats_list_t, proj_curr_lvl, H_img=H_img, W_img=W_img)

                multi_level_ranges_prev_t.append(range_prev_t)
                multi_level_ranges_curr_t.append(range_curr_t)

        # ----- 预测（两分支）-----
        # scale 分支
        corr_encoded_s = self.conv_corr(corr_range)
        initial_scale = F.softplus(corr_encoded_s[:, :1]) + 1e-3
        corr_encoded_s = corr_encoded_s[:, 1:]

        scales = self.scale_net(
            corr_encoded_s, multi_level_ranges_prev, multi_level_ranges_curr, initial_scale
        )
        if scale_only:
            kd_cache = {
                'corr_encoded_list': corr_encoded_list,
                'ini_corr_query': ini_corr_query,
                'multi_level_feats_prev': multi_level_feats_prev,
                'multi_level_feats_curr': multi_level_feats_curr,
                'teacher': {
                    'corr_range_t': corr_range_t,
                    'multi_level_ranges_prev_t': multi_level_ranges_prev_t,
                    'multi_level_ranges_curr_t': multi_level_ranges_curr_t
                }
            }
            return scales, None, kd_cache

        # risk 分支
        corr_encoded = self.conv_corr_risk(corr_range)
        initial_risk = corr_encoded[:, :1]
        corr_encoded = corr_encoded[:, 1:]
        risk_score = self.risk_net(
            corr_encoded, multi_level_ranges_prev, multi_level_ranges_curr, initial_risk
        )

        kd_cache = {
            'corr_encoded_list': corr_encoded_list,
            'ini_corr_query': ini_corr_query,
            'multi_level_feats_prev': multi_level_feats_prev,
            'multi_level_feats_curr': multi_level_feats_curr,
            'teacher': {
                'corr_range_t': corr_range_t,
                'multi_level_ranges_prev_t': multi_level_ranges_prev_t,
                'multi_level_ranges_curr_t': multi_level_ranges_curr_t
            }
        }
        return scales, risk_score, kd_cache

    # ------------------------- 训练前向（含损失） -------------------------
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
            sensor_metas,
            attn_type,                     # str
            attn_splits_list,              # List[int]
            corr_radius_list,              # List[int]
            prop_radius_list,              # List[int]
            num_reg_refine,                # int
            scale_only
        ):
        # 主干前向（返回 kd_cache）
        scales, risks, kd_cache = self.forward(
            img_prev=img_prev,
            img_curr=img_curr,
            depth_prev=depth_prev,
            depth_curr=depth_curr,
            proj_pix_prev=proj_pix_prev,
            proj_pix_curr=proj_pix_curr,
            sensor_metas=sensor_metas,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
            scale_only=scale_only,
        )

        # ----- 任务损失 -----
        if scale_only:
            loss_task = get_loss_scale_map(scales, gt_scale_map_with_mask)
        else:
            loss_task = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)

        # ----- 蒸馏损失（仅反传到 RVT） -----
        loss_kd = scales.new_tensor(0.0)

        if self.use_kd:
            # Teacher（已经 no_grad 生成）
            T_corr = kd_cache['teacher']['corr_range_t']                             # [B,C,Hr,Wr]
            T_prevL = kd_cache['teacher']['multi_level_ranges_prev_t']               # list[L][B,C,Hr,Wr]
            T_currL = kd_cache['teacher']['multi_level_ranges_curr_t']               # list[L][B,C,Hr,Wr]

            # Student for KD：再跑一遍 RVT，但把输入 detach()，使梯度只更新 RVT
            corr_encoded_list_det = [t.detach() for t in kd_cache['corr_encoded_list']]
            ini_corr_query_det = kd_cache['ini_corr_query'].detach()

            # 需要 V（根据 list 长度）
            V = len(corr_encoded_list_det)
            B = T_corr.shape[0]

            # 维度（以 corr 编码的单视角尺寸为准）
            H_r = corr_encoded_list_det[0].shape[2]
            W_r = corr_encoded_list_det[0].shape[3] * V

            # Corr KD
            S_corr_kd = self.rvt_corr(
                feats_by_cam=corr_encoded_list_det,
                cam_K=[sensor_metas['curr'][cam]['K'] for cam in self.camera_channels],
                cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                Hr=H_r, Wr=W_r,
                depth_bins=self.depth_bins,
                ini_query=ini_corr_query_det
            )  # [B,C,Hr,Wr]

            # mask（Teacher 非零）
            mask_corr = (T_corr.abs().sum(dim=1, keepdim=True) > 0).float()

            L_corr = self._charbonnier(S_corr_kd - T_corr, mask=mask_corr)
            L_corr += self.lambda_cos * self._cosine_loss(S_corr_kd, T_corr, mask=mask_corr)

            # 多尺度 KD
            S_prevL_kd, S_currL_kd = [], []
            for lvl in range(self.num_scales):
                prev_feats_list_det = [kd_cache['multi_level_feats_prev'][lvl][:, v].detach()
                                       for v in range(V)]
                curr_feats_list_det = [kd_cache['multi_level_feats_curr'][lvl][:, v].detach()
                                       for v in range(V)]

                H_r_l = prev_feats_list_det[0].shape[2]
                W_r_l = prev_feats_list_det[0].shape[3] * V

                S_prev_kd = self.rvt_feat[lvl](
                    feats_by_cam=prev_feats_list_det,
                    cam_K=[sensor_metas['prev'][cam]['K'] for cam in self.camera_channels],
                    cam_R=[sensor_metas['prev'][cam]['R_l2c'] for cam in self.camera_channels],
                    cam_t=[sensor_metas['prev'][cam]['t_l2c'] for cam in self.camera_channels],
                    affine_M=[sensor_metas['prev'][cam]['affine'] for cam in self.camera_channels],
                    Hr=H_r_l, Wr=W_r_l,
                    depth_bins=self.depth_bins
                )
                S_curr_kd = self.rvt_feat[lvl](
                    feats_by_cam=curr_feats_list_det,
                    cam_K=[sensor_metas['curr'][cam]['K'] for cam in self.camera_channels],
                    cam_R=[sensor_metas['curr'][cam]['R_l2c'] for cam in self.camera_channels],
                    cam_t=[sensor_metas['curr'][cam]['t_l2c'] for cam in self.camera_channels],
                    affine_M=[sensor_metas['curr'][cam]['affine'] for cam in self.camera_channels],
                    Hr=H_r_l, Wr=W_r_l,
                    depth_bins=self.depth_bins
                )
                S_prevL_kd.append(S_prev_kd)
                S_currL_kd.append(S_curr_kd)

            L_feat = S_corr_kd.new_tensor(0.0)
            for lvl in range(self.num_scales):
                Tp, Tc = T_prevL[lvl], T_currL[lvl]
                Sp, Sc = S_prevL_kd[lvl], S_currL_kd[lvl]
                mask_p = (Tp.abs().sum(dim=1, keepdim=True) > 0).float()
                mask_c = (Tc.abs().sum(dim=1, keepdim=True) > 0).float()

                Lp = self._charbonnier(Sp - Tp, mask=mask_p) + self.lambda_cos * self._cosine_loss(Sp, Tp, mask=mask_p)
                Lc = self._charbonnier(Sc - Tc, mask=mask_c) + self.lambda_cos * self._cosine_loss(Sc, Tc, mask=mask_c)
                L_feat = L_feat + (Lp + Lc)

            L_feat = L_feat / (2 * self.num_scales)

            loss_kd = self.lambda_corr * L_corr + self.lambda_feat * L_feat

        # ----- 汇总 -----
        loss_total = loss_task + loss_kd

        if scale_only:
            return scales, None, loss_total, None
        else:
            return None, risks, None, loss_total

    # ------------------------- 工具函数 -------------------------
    def extract_feature(self, im0, im1, branch):
        x = torch.cat([im0, im1], dim=0)
        feats = self.cnet(x, branch=branch)[::-1]
        p0, p1 = [], []
        for f in feats:
            a, b = torch.chunk(f, 2, dim=0)
            p0.append(a)
            p1.append(b)
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
        batch_idx = torch.arange(B, device=cam_idx.device)\
                        .unsqueeze(1).repeat(1, H_r*W_r)\
                        .reshape(-1)
        view_idx  = batch_idx * V + cam_idx.reshape(-1)   # [B*N]

        # 5) 计算在 Hf*Wf 上的线性化像素
        pix_idx   = (v_feat * W_feat + u_feat).reshape(-1)  # [B*N]

        # 6) 一次性 gather
        selected = feats_flat[view_idx, :, pix_idx]         # [B*N, C]

        # 7) 重塑回 [B, C, H_r, W_r]
        range_feat = selected.view(B, H_r*W_r, C) \
                            .permute(0, 2, 1) \
                            .reshape(B, C, H_r, W_r)
        return range_feat

    @staticmethod
    def _reduce_mean(x, mask=None, eps=1e-6):
        if mask is None:
            return x.mean()
        if mask.shape[1] < x.shape[1]:
            mask = mask.expand_as(x)  # 关键：沿通道扩展
        denom = mask.sum().clamp_min(1.0)
        return (x * mask).sum() / (denom + eps)

    def _charbonnier(self, x, mask=None, eps=1e-6, alpha=0.5):
        loss = torch.pow(x, 2) + eps * eps
        loss = torch.pow(loss, alpha)
        if mask is not None:
            if mask.dim() < x.dim():
                mask = mask.expand_as(x)  # 同上
            return self._reduce_mean(loss, mask=mask)
        return loss.mean()


    def _cosine_loss(self, p, q, mask=None, eps=1e-6):
        # 1 - cos(p, q) over channel dim
        # p,q: [B,C,H,W]
        p_flat = p
        q_flat = q
        num = (p_flat * q_flat).sum(dim=1, keepdim=True)       # [B,1,H,W]
        pd = p_flat.norm(dim=1, keepdim=True).clamp_min(eps)
        qd = q_flat.norm(dim=1, keepdim=True).clamp_min(eps)
        cos = num / (pd * qd)
        loss = 1.0 - cos
        if mask is not None:
            return self._reduce_mean(loss, mask=mask)
        return loss.mean()
