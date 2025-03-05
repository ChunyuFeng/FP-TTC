import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from .modules.matching import (global_correlation_softmax, local_correlation_softmax, \
                                local_correlation_with_flow, local_scale_correlation)

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet

from .range_image_encoder import RangeImageEncoder
from .scale_net.Hierarchical_Spatio_Temporal_Fusion import Hierarchical_Spatio_Temporal_Fusion, MultiLayerHierarchicalFusion


class FpTTC(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 train=False,
                 local_radius=3
                 ):
        super(FpTTC, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine
        self.is_trainning = train
        self.local_radius = local_radius

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
                                 upsample_factor=upsample_factor, num_head=num_head,
                                 scale_level=num_scales, reg_refine=reg_refine)

        # Transformer
        self.rangeimageencoder = RangeImageEncoder(num_layers=num_transformer_layers,
                                    d_model=feature_channels,
                                    nhead=num_head,
                                    num_feature_levels=num_scales,
                                    num_level=num_scales)

        self.MultiLayerHierarchicalFusion = MultiLayerHierarchicalFusion(num_layers=num_transformer_layers,
                                                                         embed_dim=feature_channels,
                                                                         num_scales=num_scales,
                                                                         n_heads=num_head)

    '''
    # 3 views & 3 views
    def forward(self, img0, img1,
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

        scale, corr, final = None, None, None
        mlvl_feats0, mlvl_feats1, mlvl_flows, mlvl_flows_back = [], [], [], []

        start = time.time()

        # 这一步现在还有用吗？
        img0, img1 = normalize_img(img0, img1)

        camera_channel_num = 6
        prev_feature_list = []
        curr_feature_list = []

        for view in range(camera_channel_num):
            prev_feature, curr_feature = self.extract_feature(img0[:, view, :, :, :], img1[:, view, :, :, :])
            prev_feature_list.append(prev_feature)
            curr_feature_list.append(curr_feature)

        num_feat_levels = len(prev_feature_list[0])
        feature0_listc = []
        feature1_listc = []
        for i in range(num_feat_levels):
            tensor_to_concat = [entry[i] for entry in prev_feature_list]
            concat_tensor = torch.cat(tensor_to_concat, dim=3)
            feature0_listc.append(concat_tensor)

            tensor_to_concat = [entry[i] for entry in curr_feature_list]
            concat_tensor = torch.cat(tensor_to_concat, dim=3)
            feature1_listc.append(concat_tensor)

        del concat_tensor, img0, img1, prev_feature, curr_feature, prev_feature_list, curr_feature_list
        torch.cuda.empty_cache()

        for i in range(len(feature0_listc)):
            feature0_listc[i] = self.rangeimageencoder(feature0_listc, feature1_listc, query_lvl=i, ini_query=None)
            feature1_listc[i] = self.rangeimageencoder(feature0_listc, feature1_listc, query_lvl=i, ini_query=None)


        mlvl_feats0_view0123, mlvl_feats1_view0123 = [], []
        mlvl_feats0_view3450, mlvl_feats1_view3450 = [], []
        mlvl_feats0, mlvl_feats1 = [], []
        if self.is_trainning and not testing:
            torch.set_grad_enabled(True)
            self.train()

        corr_view0123 = None
        corr_view3450 = None
        # 视角 0 1 2 3

        for scale_idx in range(self.num_scales):
            if scale_idx < 1:
                feature0, feature1 = feature0_listc[scale_idx][:, :, :, 0:192], feature1_listc[scale_idx][:, :, :,
                                                                                0:192]
                feature0, feature1 = self.featnet(feature0, feature1, scale_idx, attn_type, attn_splits_list,
                                                    corr_view0123)
                mlvl_feats0_view0123.append(feature0)
                mlvl_feats1_view0123.append(feature1)
                corr_view0123, final_view0123 = self.corrnet(feature0, feature1, scale_idx, corr_radius_list,
                                                                prop_radius_list, num_reg_refine, False, corr_view0123)
                corr_view0123 = F.interpolate(corr_view0123, scale_factor=2, mode='bilinear',
                                                align_corners=True) * 2
            else:
                feature0, feature1 = feature0_listc[scale_idx][:, :, :, 0:384], feature1_listc[scale_idx][:, :, :,
                                                                                0:384]
                feature0_f, feature1_f = self.featnet(feature0, feature1, scale_idx, attn_type, attn_splits_list,
                                                        corr_view0123)
                mlvl_feats0_view0123.append(feature0_f)
                mlvl_feats1_view0123.append(feature1_f)
                corr_view0123, final_view0123 = self.corrnet(feature0_f, feature1_f, scale_idx, corr_radius_list,
                                                                prop_radius_list, num_reg_refine, False, corr_view0123)

        # 清理视角0-3的计算结果，释放显存
        del feature0, feature1, feature0_f, feature1_f, final_view0123
        torch.cuda.empty_cache()

        # 视角 3 4 5 0
        for scale_idx in range(self.num_scales):
            if scale_idx < 1:
                feature0 = feature0_listc[scale_idx][:, :, :, 192:384]
                feature1 = feature1_listc[scale_idx][:, :, :, 192:384]
                feature0, feature1 = self.featnet(feature0, feature1, scale_idx, attn_type, attn_splits_list,
                                                    corr_view3450)
                mlvl_feats0_view3450.append(feature0)
                mlvl_feats1_view3450.append(feature1)
                corr_view3450, final_view3450 = self.corrnet(feature0, feature1, scale_idx, corr_radius_list,
                                                                prop_radius_list, num_reg_refine, False, corr_view3450)
                corr_view3450 = F.interpolate(corr_view3450, scale_factor=2, mode='bilinear',
                                                align_corners=True) * 2
            else:
                feature0 = feature0_listc[scale_idx][:, :, :, 384:768]
                feature1 = feature1_listc[scale_idx][:, :, :, 384:768]

                del feature0_listc, feature1_listc
                torch.cuda.empty_cache()

                feature0_f, feature1_f = self.featnet(feature0, feature1, scale_idx, attn_type, attn_splits_list,
                                                        corr_view3450)
                mlvl_feats0_view3450.append(feature0_f)
                mlvl_feats1_view3450.append(feature1_f)
                corr_view3450, final_view3450 = self.corrnet(feature0_f, feature1_f, scale_idx, corr_radius_list,
                                                                prop_radius_list, num_reg_refine, False, corr_view3450)

        # 清理视角3-5的计算结果，释放显存
        del feature0, feature1, feature0_f, feature1_f, final_view3450
        torch.cuda.empty_cache()

        # 特征融合
        corr = torch.cat((corr_view0123, corr_view3450), dim=3)
        mlvl_feats0 = [torch.cat((mlvl_feats0_view0123[i], mlvl_feats0_view3450[i]), dim=3) for i in
                       range(self.num_scales)]
        mlvl_feats1 = [torch.cat((mlvl_feats1_view0123[i], mlvl_feats1_view3450[i]), dim=3) for i in
                       range(self.num_scales)]

        corr = self.conv_corr(corr)
        ini_scale, corr = corr[:, 0:1, ...], corr[:, 1:, ...]
        scales = self.scalenet(corr, mlvl_feats0, mlvl_feats1, ini_scale)

        return scales, None, None
        '''


    # 0 ~ 5 6个视角 同时处理
    def forward(self, img0, img1,
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
        mlvl_feats0, mlvl_feats1, mlvl_flows, mlvl_flows_back = [], [], [], []

        start = time.time()

        img0, img1 = normalize_img(img0, img1)

        camera_channel_num = 6
        prev_feature_list = []
        curr_feature_list = []

        for view in range(camera_channel_num):
            prev_feature, curr_feature = self.extract_feature(img0[:, view, :, :, :], img1[:, view, :, :, :])
            prev_feature_list.append(prev_feature)
            curr_feature_list.append(curr_feature)

        num_feat_levels = len(prev_feature_list[0])
        feature0_listc = []
        feature1_listc = []
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

        feature0_listc, feature1_listc = self.MultiLayerHierarchicalFusion(feature0_listc, feature1_listc)

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

        return scales, final, None

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