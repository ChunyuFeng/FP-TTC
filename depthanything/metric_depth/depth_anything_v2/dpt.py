import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, out_features, patch_h, patch_w, return_intermediates=False):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        if return_intermediates:
            return out, {
                'path_1': path_1,
                'path_2': path_2,
                'path_3': path_3,
                'path_4': path_4,
            }

        return out


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        max_depth=20.0
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.max_depth = max_depth
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x, return_intermediates=False):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth_out = self.depth_head(features, patch_h, patch_w, return_intermediates=return_intermediates)

        if return_intermediates:
            depth, intermediates = depth_out
            return depth.squeeze(1) * self.max_depth, intermediates

        depth = depth_out * self.max_depth

        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    @torch.no_grad()
    def infer_images(self, raw_images, input_size=518, pad_value=0.0, return_torch=False, device=None):
        """
        批量版推理：raw_images 为 list[np.ndarray(BGR,H,W,3)] 或 list[RGB 要先转 BGR 再传]
        - 预处理时保持纵横比、边长对齐到 14 的倍数（和 infer_image 一致）
        - 对每张图得到 (3, Hi, Wi)，对齐到 (Hmax, Wmax) 右下角补零后 stack 成 [N,3,Hmax,Wmax]
        - 前向得到 [N, Hmax, Wmax]，再按各自 (Hi,Wi) 裁剪并 resize 回原始 (h,w)
        """
        if device is None:
            device = next(self.parameters()).device

        # 逐张预处理到 (3, Hi, Wi)，记录原始尺寸与网络输入尺寸
        tensors = []
        orig_sizes = []
        proc_sizes = []
        for img in raw_images:
            t, (h, w), (Hi, Wi) = self.image2tensor_nobatch(img, input_size)
            tensors.append(t)           # (3, Hi, Wi) on CPU
            orig_sizes.append((h, w))   # 原始分辨率
            proc_sizes.append((Hi, Wi)) # 预处理后的分辨率(14的倍数)

        # 右下角补零到同一尺寸
        Hmax = max(h for h, _ in proc_sizes)
        Wmax = max(w for _, w in proc_sizes)
        padded = []
        for t, (Hi, Wi) in zip(tensors, proc_sizes):
            pad_h, pad_w = Hmax - Hi, Wmax - Wi
            t = F.pad(t, (0, pad_w, 0, pad_h), value=pad_value)  # (left,right,top,bottom)
            padded.append(t)
        batch = torch.stack(padded, dim=0).to(device, non_blocking=True)  # [N,3,Hmax,Wmax]

        # 前向
        depth = self.forward(batch)  # [N, Hmax, Wmax]  （DPTHead 已经上采样回输入尺寸）

        # 逐张裁剪+还原
        outs = []
        for i, ((ho, wo), (Hi, Wi)) in enumerate(zip(orig_sizes, proc_sizes)):
            d = depth[i:i+1, :Hi, :Wi]                        # (1, Hi, Wi)
            d = F.interpolate(d.unsqueeze(1), (ho, wo), mode="bilinear", align_corners=True)[0, 0]
            outs.append(d if return_torch else d.detach().cpu().numpy())
        return outs

    def image2tensor_nobatch(self, raw_image, input_size=518):
        """
        和 image2tensor 基本一致，但不加 batch 维、不搬到设备，用于在 Python 侧先做 pad/stack。
        返回：tensor(3,Hi,Wi)[CPU]，(h,w 原图尺寸)，(Hi,Wi 预处理尺寸)
        """
        transform = Compose([
            Resize(
                width=input_size, height=input_size,
                resize_target=False, keep_aspect_ratio=True,
                ensure_multiple_of=14, resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        h, w = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0  # 注意：期望传入 BGR
        image = transform({'image': image})['image']                 # numpy (3, Hi, Wi)
        tensor = torch.from_numpy(image).float()                     # torch (3, Hi, Wi)
        Hi, Wi = tensor.shape[-2], tensor.shape[-1]
        return tensor, (h, w), (Hi, Wi)

    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
