import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.utils import upsample_scale_with_mask
from ..modules.attention import SelfAttnPropagation

# ------------------------- Heads -------------------------
class ScaleHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(ScaleHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return F.relu(self.conv2(self.relu(self.conv1(x))))

class RiskHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(RiskHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Only first conv uses ReLU; final output can be negative
        return self.conv2(self.relu(self.conv1(x)))


class BasicMotionEncoder(nn.Module):
    def __init__(self, dim):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(dim, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-1, 3, padding=1)
    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

# --------------------- Shared Update Block ---------------------
class SharedBasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        super(SharedBasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(dim=input_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim,
                              input_dim=hidden_dim + input_dim)
        self.scale_head = ScaleHead(input_dim=hidden_dim, hidden_dim=256)
        self.risk_head = RiskHead(input_dim=hidden_dim, hidden_dim=256)

    def forward(self, net, inp, agg_corr, scale, risk, first_block=False):
        # motion encoding
        motion_features = self.encoder(scale, agg_corr)
        inp2 = torch.cat([inp, motion_features], dim=1)

        # update hidden state
        net = self.gru(net, inp2)

        # per-task head outputs
        scale_out = self.scale_head(net)
        risk_out = self.risk_head(net)

        # combine with previous prediction
        if first_block:
            new_scale = scale_out
            new_risk = risk_out
        else:
            new_scale = scale * scale_out
            new_risk = risk * risk_out
        return net, new_scale, new_risk


# ---------------------- Shared Decoder ----------------------
class SharedDecoder(nn.Module):
    def __init__(self,
                 input_dim=128,
                 hidden_dim=96,
                 upsample_factor=4,
                 num_blocks=2):
        super(SharedDecoder, self).__init__()
        self.upsample_factor = upsample_factor
        self.refine_proj = nn.Conv2d(input_dim * 2, 256, 1)

        # iterative update blocks
        self.update_blocks = nn.ModuleList([
            SharedBasicUpdateBlock(hidden_dim, input_dim)
            for _ in range(num_blocks)
        ])

        # upsampler unchanged
        self.upsampler = nn.Sequential(
            nn.Conv2d(1 + input_dim, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, upsample_factor ** 2 * 9, 1)
        )

    def forward(self, scale_feature, cfeat0, agg_corr, ini_scale):
        # combine feature for update
        proj = self.refine_proj(torch.cat([scale_feature, cfeat0], dim=1))
        net, inp = torch.chunk(proj, chunks=2, dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # initial predictions
        scale = ini_scale
        risk = ini_scale
        # iterative updates
        for i, block in enumerate(self.update_blocks):
            net, scale, risk = block(net, inp, agg_corr, scale, risk, first_block=(i==0))

        # upsample both task outputs
        scale_f = upsample_scale_with_mask(scale,
            self.upsampler(torch.cat([scale, cfeat0], dim=1)),
            upsample_factor=self.upsample_factor)
        risk_f  = upsample_scale_with_mask(risk,
            self.upsampler(torch.cat([risk,  cfeat0], dim=1)),
            upsample_factor=self.upsample_factor)
        return scale_f, risk_f

    def upsample_flow_with_mask(self, flow, up_mask, upsample_factor):
        # convex upsampling following raft
        mask = up_mask
        b, flow_channel, h, w = flow.shape
        mask = mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

        up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
        up_flow = up_flow.reshape(b, flow_channel, upsample_factor * h,
                                upsample_factor * w)  # [B, 2, K*H, K*W]
        return up_flow

    def upsample_scale(self, scale, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):

        concat = torch.cat((scale, feature), dim=1)
        mask = self.upsampler(concat)
        up_scale = upsample_scale_with_mask(scale, mask, upsample_factor=self.upsample_factor,
                                            is_depth=is_depth)
        return up_scale



