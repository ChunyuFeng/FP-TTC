import torch
import torch.nn as nn
import torch.nn.functional as F

from .scale_encoder import ScaleEncoder
from .decoder import SharedDecoder
from ..modules.attention import SelfAttnPropagation
from ..modules.geometry import flow_wrap

from .feature_net.feature_net import GmaAtten

class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super(FeatureFusion, self).__init__()
        self.convc1 = nn.Conv2d(dim, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(dim, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, dim, 3, padding=1)
    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return out


class MultiTaskScaleNet(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=4,
                 num_head=1,
                 num_transformer_layers=6,
                 scale_level=1,
                 reg_refine=False,
                 query_lvl=-1,
                 num_blocks=2):
        super(MultiTaskScaleNet, self).__init__()
        # encoder + attention
        self.encoder = ScaleEncoder(
            num_layers=num_transformer_layers,
            input_dim=feature_channels,
            d_model=feature_channels,
            nhead=num_head,
            num_feature_levels=num_scales,
            num_level=scale_level)
        self.gma = GmaAtten(
            num_scales=1,
            feature_channels=feature_channels,
            num_head=num_head,
            ffn_dim_expansion=2,
            num_transformer_layers=1)
        # shared decoder
        self.decoder = SharedDecoder(
            input_dim=feature_channels,
            hidden_dim=feature_channels//1,
            upsample_factor=upsample_factor,
            num_blocks=num_blocks)

    def forward(self, corr, feature0_listc, feature1_listc, ini_scale):
        # encode
        scale_feat = self.encoder(feature0_listc,
                                  feature1_listc,
                                  query_lvl=-1,
                                  ini_query=corr)
        # aggregate correlation
        agg_corr = self.gma(scale_feat, corr,
                            attn_type='swin',
                            attn_splits_list=[2,8])
        # decode multi-task
        scale, risk = self.decoder(scale_feat,
                                   feature0_listc[-1],
                                   agg_corr,
                                   ini_scale)
        return scale, risk



