import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.utils import normalize_img
from utils.loss import get_loss_scale_map, get_loss_risk_score_map

from .scale_net.backbone import CNNEncoder
from .scale_net.feature_net.feature_net import FeatureNet
from .scale_net.flow_net import FlowNet
from .scale_net.scale_net import ScaleNet
import math
from utils.dist import is_main_process

from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from depthanything.metric_depth.depth_anything_v2.dinov2 import DINOv2

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
        
        # 1) DINOv2 Transformer: 提取多尺度特征
        self.dinov2 = DINOv2(model_name='vitb')

        # 2) Bridge: 将 DPTHead 输出的多尺度通道映射到统一 feature_channels
        dpt_out_channels = [96, 384]
        self.bridge_convs = nn.ModuleList([
            nn.Conv2d(dpt_out_channels[i], feature_channels, kernel_size=1)
            for i in range(len(dpt_out_channels))
        ])

        self.resize_layers = nn.ModuleList([
            # upsample 2×
            nn.ConvTranspose2d(
            in_channels=dpt_out_channels[0],
            out_channels=dpt_out_channels[0],
            kernel_size=2,
            stride=2,
            padding=0
            ),
            # upsample 1×
            nn.Identity(),
        ])

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.dinov2.embed_dim,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in dpt_out_channels
        ])

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
        

    def forward(self,
                img_prev, img_curr,
                depth_prev, depth_curr,
                proj_pix_prev, proj_pix_curr,
                attn_type,
                attn_splits_list,
                corr_radius_list,
                prop_radius_list,
                num_reg_refine,
                scale_only):
        
        B, V, C, H, W = img_prev.shape
        shared_prev, shared_curr = [], []

        # 1) 多视角特征提取：从 DINOv2 Transformer 取 num_scales 层 token，再用 DPTHead 的 projects+resize_layers 得到多尺度特征
        patch_size = self.dinov2.patch_size
        patch_h, patch_w = H // patch_size, W // patch_size
        # idxs = self.depth_model.intermediate_layer_idx[self.depth_model.encoder] # [2, 5, 8, 11] for vitb，取 2 8 两层
        idxs = [2, 8]
        for v in range(V):
            # im0 = image_prev_resized[:, v]
            # im1 = image_curr_resized[:, v]
            im0 = img_prev[:, v]
            im1 = img_curr[:, v]
            with torch.no_grad():
                tokens0 = self.dinov2.get_intermediate_layers(im0, idxs, return_class_token=False)
                tokens1 = self.dinov2.get_intermediate_layers(im1, idxs, return_class_token=False)
            p0, p1 = [], []
            for i, (t0, t1) in enumerate(zip(tokens0, tokens1)):
                # [B, N, 768] -> [B, 768, patch_h, patch_w]
                Ce = t0.shape[-1]
                fmap0 = t0.permute(0,2,1).reshape(B, Ce, patch_h, patch_w)
                fmap1 = t1.permute(0,2,1).reshape(B, Ce, patch_h, patch_w)
                # DPTHead projects: 768->out_channels[i]
                proj = self.projects[i] # 1
                fmap0 = proj(fmap0)
                fmap1 = proj(fmap1)
                # DPTHead resize_layers: 恢复到相同空间尺度
                resize = self.resize_layers[i]
                fmap0 = resize(fmap0)
                fmap1 = resize(fmap1)
                # Bridge conv: out_channels[i] -> feature_channels
                fmap0 = self.bridge_convs[i](fmap0)
                fmap1 = self.bridge_convs[i](fmap1)
                p0.append(fmap0)
                p1.append(fmap1)
            p0.reverse()
            p1.reverse()
            shared_prev.append(p0)
            shared_curr.append(p1)

        # ### 1）提取输入的多视角图像的底层特征
        # img0, img1 = normalize_img(img_prev, img_curr)
        # rgbd0 = torch.cat([img0, depth_prev], dim=2)  # [B, V, 4, H, W]
        # rgbd1 = torch.cat([img1, depth_curr], dim=2)  # [B, V, 4, H, W]
        # B, V, C, H_img, W_img = img0.shape
        # shared_prev, shared_curr = [], []
        # for view in range(rgbd0.size(1)):
        #     p, c = self.extract_feature(rgbd0[:, view], rgbd1[:, view], branch=None)
        #     shared_prev.append(p)
        #     shared_curr.append(c)

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

    def batch_resize_images_tensor(
        self,
        images: torch.Tensor,
        input_size: int,
        patch_multiple: int = 14
    ):
        """
        Resize a batch of images (B, V, C, H, W) so that:
        1) aspect ratio is preserved,
        2) both height and width >= input_size,
        3) final H and W are multiples of `patch_multiple`.

        Args:
            images: torch.Tensor of shape (B, V, C, H, W)
            input_size: minimum target size for both H and W
            patch_multiple: align factor (e.g. 14)

        Returns:
            resized: torch.Tensor of shape (B, V, C, H_new, W_new)
            original_sizes: List[List[(H, W)]] of shape (B, V)
        """
        B, V, C, H, W = images.shape
        # Record original sizes per view
        original_sizes = [[(H, W) for _ in range(V)] for _ in range(B)]

        # Flatten batch and view dims to N
        x = images.view(B * V, C, H, W)

        # 1. Compute scale to meet input_size
        scale = max(input_size / H, input_size / W)
        H_scaled = math.ceil(H * scale)
        W_scaled = math.ceil(W * scale)

        # 2. Align to patch_multiple
        H_new = math.ceil(H_scaled / patch_multiple) * patch_multiple
        W_new = math.ceil(W_scaled / patch_multiple) * patch_multiple

        # 3. Batch interpolate
        resized_flat = F.interpolate(
            x,
            size=(H_new, W_new),
            mode='bilinear',
            align_corners=True
        )

        # Reshape back to (B, V, C, H_new, W_new)
        resized = resized_flat.view(B, V, C, H_new, W_new)

        return resized, original_sizes