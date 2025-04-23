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

from .range_image_encoder_new import RangeImageEncoder

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
                 train=False):
        super(FpTTC, self).__init__()
        self.num_scales = num_scales
        self.is_trainning = train

        # 仅底层共享：CNNEncoder
        self.cnet = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

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
                                upsample_factor=upsample_factor, num_head=8,
                                scale_level=num_scales, reg_refine=reg_refine, head_type='risk')

    def forward(self, img0, img1, sensor_metas,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=6,
                testing=False):
        # if self.is_trainning and not testing:
        #     self.eval()
        #     torch.set_grad_enabled(False)

        img0, img1 = normalize_img(img0, img1)
        # B, C, H, W = img0.shape
        # 提取多视角底层特征
        prev_feat_list, curr_feat_list = [], []
        for view in range(img0.size(1)):
            p, c = self.extract_feature(img0[:, view], img1[:, view])
            prev_feat_list.append(p)
            curr_feat_list.append(c)

        # 多视角拼接
        feature0_lvls, feature1_lvls = [], []
        for lvl in range(len(prev_feat_list[0])):
            f0 = torch.cat([f[lvl] for f in prev_feat_list], dim=3)
            f1 = torch.cat([f[lvl] for f in curr_feat_list], dim=3)
            feature0_lvls.append(f0)
            feature1_lvls.append(f1)

        del prev_feat_list, curr_feat_list

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
        corr_enc_s = F.relu(self.conv_corr_scale(corr_s))
        ini_scale = corr_enc_s[:, :1]
        corr_enc_s = corr_enc_s[:, 1:]
        scales = self.scalenet(corr_enc_s, mlvl_s0, mlvl_s1, ini_scale)

        del corr_s, mlvl_s0, mlvl_s1

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

        return scales, risk_score

    def forward_with_loss(self, img0, img1, sensor_meta,
                          gt_scale_map_with_mask, gt_risk_score_map_with_mask, **kwargs):
        scales, risks = self.forward(img0, img1, sensor_meta, **kwargs)
        loss_s = get_loss_scale_map(scales, gt_scale_map_with_mask)
        loss_r = get_loss_risk_score_map(risks, gt_risk_score_map_with_mask)
        loss = loss_s + loss_r
        return scales, risks, loss_s, loss_r, loss

    def extract_feature(self, im0, im1):
        x = torch.cat([im0, im1], dim=0)
        feats = self.cnet(x)[::-1]
        p0, p1 = [], []
        for f in feats:
            a, b = torch.chunk(f, 2, dim=0)
            p0.append(a); p1.append(b)
        return p0, p1
