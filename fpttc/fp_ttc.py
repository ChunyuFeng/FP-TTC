import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from .modules.matching import (global_correlation_softmax, local_correlation_softmax, \
                                local_correlation_with_flow, local_scale_correlation)
from utils.loss import get_loss_scale_map, get_loss_risk_score_map

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet

from .range_image_encoder_new import RangeImageEncoder
# from .range_image_encoder import RangeImageEncoder
from .scale_net.Hierarchical_Spatio_Temporal_Fusion import Hierarchical_Spatio_Temporal_Fusion, MultiLayerHierarchicalFusion


class FpTTC(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 range_image_feat_shape=[(20, 240), (40, 480)],
                 reg_refine=False,  # optional local regression refinement
                 train=False,
                #  local_radius=3,
                 ):
        super(FpTTC, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine
        self.is_trainning = train
        # self.local_radius = local_radius

        self.range_image_feat_shape = range_image_feat_shape

        # CNN
        #norm_layer=nn.BatchNorm2d
        #self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales, norm_layer=nn.BatchNorm2d)
        self.cnet = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        self.featnet = FeatureNet(num_scales=num_scales, feature_channels=feature_channels, 
                                  num_head=num_head, ffn_dim_expansion=ffn_dim_expansion,
                                  num_transformer_layers=num_transformer_layers)
        self.corrnet = FlowNet(num_scales=num_scales, feature_channels=feature_channels,
                               upsample_factor=upsample_factor, reg_refine=reg_refine)
        self.conv_corr = CorrEncoder(dim_in=2, dim_out=feature_channels+1)
        self.scalenet = ScaleNet(num_scales=num_scales, feature_channels=feature_channels,
                                 upsample_factor=upsample_factor, num_head=8,
                                 scale_level=num_scales, reg_refine=reg_refine, head_type='scale')
        self.risknet = ScaleNet(num_scales=num_scales, feature_channels=feature_channels,
                                 upsample_factor=upsample_factor, num_head=8,
                                 scale_level=num_scales, reg_refine=reg_refine, head_type='risk')
        

        # Transformer
        # self.rangeimageencoder = RangeImageEncoder(num_layers=num_transformer_layers,
        #                             d_model=feature_channels,
        #                             nhead=num_head,
        #                             num_feature_levels=num_scales,
        #                             num_level=num_scales)

        self.RI_Encoder = RangeImageEncoder(embed_dims=feature_channels,
                                            num_cams=6,
                                            num_layers=num_transformer_layers,
                                            num_levels=num_scales,
                                            num_points=8, # SpatialCrossAttention 模块使用的参数
                                            pc_range=70.0
                                            )
        # 分别为 prev range image/curr range image 的 low level/high level 创建可学习的embedding
        self.range_image_embeds = nn.ModuleDict({
            'prev_llvl': nn.Embedding(range_image_feat_shape[0][0] * range_image_feat_shape[0][1], feature_channels),
            'curr_llvl': nn.Embedding(range_image_feat_shape[0][0] * range_image_feat_shape[0][1], feature_channels),
            'prev_hlvl': nn.Embedding(range_image_feat_shape[1][0] * range_image_feat_shape[1][1], feature_channels),
            'curr_hlvl': nn.Embedding(range_image_feat_shape[1][0] * range_image_feat_shape[1][1], feature_channels),
        })

        # self.MultiLayerHierarchicalFusion = MultiLayerHierarchicalFusion(num_layers=num_transformer_layers,
        #                                                                  embed_dim=feature_channels,
        #                                                                  num_scales=num_scales,
        #                                                                  n_heads=8)

        self.log_sigma_scale = nn.Parameter(torch.zeros(1))
        self.log_sigma_risk = nn.Parameter(torch.zeros(1))

    # 0 ~ 5 6个视角 同时处理
    def forward(self, img0, img1, sensor_metas,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=6,
                pred_bidir_flow=False,
                testing=False,
                ):

        if self.is_trainning and not testing:
            self.eval()
            torch.set_grad_enabled(False)

        scale, corr = None, None

        img0, img1 = normalize_img(img0, img1)

        camera_channel_num = 6
        prev_feature_list = []
        curr_feature_list = []

        for view in range(camera_channel_num):
            prev_feature, curr_feature = self.extract_feature(img0[:, view, :, :, :], img1[:, view, :, :, :])
            prev_feature_list.append(prev_feature)
            curr_feature_list.append(curr_feature)


        # # 使用Spatial Cross Attention模块进行特征融合
        # prev_feat_stacked = []
        # curr_feat_stacked = []
        # # 合并多个视角的特征图： B, num_cams, C, H, W
        # for scale in range(self.num_scales):
        #     scale_feats = [prev_feature_list[cam][scale] for cam in range(camera_channel_num)]
        #     scale_feats_stacked = torch.stack(scale_feats, dim=1)
        #     prev_feat_stacked.append(scale_feats_stacked)

        #     scale_feats = [curr_feature_list[cam][scale] for cam in range(camera_channel_num)]
        #     scale_feats_stacked = torch.stack(scale_feats, dim=1)
        #     curr_feat_stacked.append(scale_feats_stacked)
        
        # # range image 特征图作为 query
        # dtype = prev_feat_stacked[0].dtype
        # prev_ri_query_llvl, prev_ri_query_hlvl, curr_ri_query_llvl, curr_ri_query_hlvl = [
        #     self.range_image_embeds[key].weight.to(dtype) 
        #     for key in ['prev_llvl', 'prev_hlvl', 'curr_llvl', 'curr_hlvl']
        # ]

        # prev_feature_llvl, curr_feature_llvl = [
        #     self.RI_Encoder(
        #     range_features=query,
        #     img_feats=prev_feat_stacked,
        #     sensor_metas=sensor_metas[0] if query is prev_ri_query_llvl else sensor_metas[1],
        #     range_image_h=self.range_image_feat_shape[0][0],
        #     range_image_w=self.range_image_feat_shape[0][1],
        #     feature_lvl=0,
        #     reference_points_cam=None,
        #     range_image_mask=None
        #     )
        #     for query in (prev_ri_query_llvl, curr_ri_query_llvl)
        # ]
        # prev_feature_hlvl, curr_feature_hlvl = [
        #     self.RI_Encoder(
        #     range_features=query,
        #     img_feats=curr_feat_stacked,
        #     sensor_metas=sensor_metas[0] if query is prev_ri_query_hlvl else sensor_metas[1],
        #     range_image_h=self.range_image_feat_shape[1][0],
        #     range_image_w=self.range_image_feat_shape[1][1],
        #     feature_lvl=1,
        #     reference_points_cam=None,
        #     range_image_mask=None
        #     )
        #     for query in (prev_ri_query_hlvl, curr_ri_query_hlvl)
        # ]

        # feature0_listc = [prev_feature_llvl, prev_feature_hlvl]
        # feature1_listc = [curr_feature_llvl, curr_feature_hlvl]

        feature0_listc = []
        feature1_listc = []
        num_feat_levels = len(prev_feature_list[0])
        # 将不同视角的图像按照channel维度拼接
        for i in range(num_feat_levels):
            tensor_to_concat = [entry[i] for entry in prev_feature_list]
            concat_tensor = torch.cat(tensor_to_concat, dim=3)
            feature0_listc.append(concat_tensor)
        
            tensor_to_concat = [entry[i] for entry in curr_feature_list]
            concat_tensor = torch.cat(tensor_to_concat, dim=3)
            feature1_listc.append(concat_tensor)
        
        # for i in range(len(feature0_listc)):
        #     feature0_listc[i] = self.rangeimageencoder(feature0_listc, feature1_listc, query_lvl=i, ini_query=None)
        #     feature1_listc[i] = self.rangeimageencoder(feature0_listc, feature1_listc, query_lvl=i, ini_query=None)


        mlvl_feats0, mlvl_feats1 = [], []
        if self.is_trainning and not testing:
            torch.set_grad_enabled(True)
            self.train()

        for scale_idx in range(self.num_scales):
            if scale_idx < 1:
                feature0, feature1 = feature0_listc[scale_idx], feature1_listc[scale_idx]
                feature0, feature1 = self.featnet(feature0, feature1, scale_idx, attn_type, attn_splits_list,
                                                  corr)
                mlvl_feats0.append(feature0)
                mlvl_feats1.append(feature1)
                corr, final = self.corrnet(feature0, feature1, scale_idx, corr_radius_list,
                                                             prop_radius_list, num_reg_refine, False, corr)
                corr = F.interpolate(corr, scale_factor=2, mode='bilinear', align_corners=True) * 2
            else:
                feature0, feature1 = feature0_listc[scale_idx], feature1_listc[scale_idx]
                feature0_f, feature1_f = self.featnet(feature0, feature1, scale_idx, attn_type, attn_splits_list,
                                                      corr)
                mlvl_feats0.append(feature0_f)
                mlvl_feats1.append(feature1_f)
                corr, final = self.corrnet(feature0_f, feature1_f, scale_idx, corr_radius_list,
                                                             prop_radius_list, num_reg_refine, False, corr)

        corr = self.conv_corr(corr)
        ini_scale, corr = corr[:, 0:1, ...], corr[:, 1:, ...]
        scales = self.scalenet(corr, mlvl_feats0, mlvl_feats1, ini_scale)
        risk_score = self.risknet(corr, mlvl_feats0, mlvl_feats1, ini_scale)

        return scales, risk_score

    def forward_with_loss(self, img0, img1, sensor_meta, 
                          gt_scale_map_with_mask, gt_risk_score_map_with_mask, **kwargs):
        """
        完整前向流程：先计算预测，再计算各任务损失，最后用不确定性加权得到总损失。
        """
        scale, risk_score = self.forward(img0, img1, sensor_meta, **kwargs)
        loss_scale = get_loss_scale_map(scale, gt_scale_map_with_mask)
        loss_risk  = get_loss_risk_score_map(risk_score, gt_risk_score_map_with_mask)
        # 使用不确定性加权损失公式：loss = exp(-log_sigma) * L + log_sigma
        # loss = (torch.exp(-self.log_sigma_scale) * loss_scale + self.log_sigma_scale +
        #         torch.exp(-self.log_sigma_risk)  * loss_risk  + self.log_sigma_risk)
        loss = loss_scale + loss_risk
        return scale, risk_score, loss_scale, loss_risk, loss
    
    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.cnet(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def scale_loss(self, flow_f, scale_src):

        d_kernel = torch.tensor([[[[-1,0,0],[0,1,0],[0,0,0]],
                                [[0,-1,0],[0,1,0],[0,0,0]],
                                [[0,0,-1],[0,1,0],[0,0,0]],
                                [[0,0,0],[-1,1,0],[0,0,0]],
                                [[0,0,0],[0,1,-1],[0,0,0]],
                                [[0,0,0],[0,1,0],[-1,0,0]],
                                [[0,0,0],[0,1,0],[0,-1,0]],
                                [[0,0,0],[0,1,0],[0,0,-1]]]]).permute(1,0,2,3).type(torch.float32).cuda() # (8,1,3,3)  

        b, _, h, w = scale_src.size()
        grid_w = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(b, 1, h, w).cuda()
        grid_h = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(b, 1, h, w).cuda()
        flow_u = flow_f[:,0:1,...] + grid_w
        flow_v = flow_f[:,1:,...] + grid_h

        pad = (1,1,1,1)
        grid_w = F.pad(grid_w, pad, mode='replicate')
        grid_h = F.pad(grid_h, pad, mode='replicate')
        flow_u = F.pad(flow_u, pad, mode='replicate')
        flow_v = F.pad(flow_v, pad, mode='replicate')

        #in:(b,1,h,w) out:(b,8,h,w)
        d_scale = torch.abs(F.conv2d(scale_src.log(), d_kernel, padding=(1,1))).type(torch.float32)
        d_u = F.conv2d(flow_u, d_kernel)
        d_v = F.conv2d(flow_v, d_kernel)
        d_w = F.conv2d(grid_w, d_kernel)
        d_h = F.conv2d(grid_h, d_kernel)

        index = torch.argmin(d_scale,dim=1).unsqueeze(0)

        d_scale = torch.gather(d_scale, 1, index)  #(b,1,h,w)
        d_u = torch.gather(d_u, 1, index)
        # print(d_u, torch.max(d_u), torch.min(d_u))
        d_v = torch.gather(d_v, 1, index)
        d_w = torch.gather(d_w, 1, index)
        d_h = torch.gather(d_h, 1, index)
        # print(d_w, torch.max(d_w), torch.min(d_w))

        scale_change = (scale_src-1)*(d_w**2+d_h**2)
        flow_change = d_u**2 + d_v**2

        a = torch.ones_like(scale_src)
        a[scale_src<1] = -1

        b = flow_change / (d_w**2+d_h**2)
        return 1/b
    

class CorrEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CorrEncoder, self).__init__()
        self.convc1 = nn.Conv2d(dim_in, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, dim_out, 3, padding=1)

    def forward(self, corr):
        # cor = F.relu(self.convc1(corr))
        # cor = F.relu(self.convc2(cor))
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        return cor