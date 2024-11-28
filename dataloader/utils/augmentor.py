import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow,dc):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            dc = cv2.resize(dc, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                dc = dc[:, ::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                dc = dc[::-1, :]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        dc   = dc[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow,dc

    def __call__(self, img1, img2, flow,dc):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow,dc = self.spatial_transform(img1, img2, flow,dc)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        dc = np.ascontiguousarray(dc)
        return img1, img2, flow,dc

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_exp_map(self, flow,exp,d1,d2, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        exp = exp.reshape(-1).astype(np.float32)
        d1 = d1.reshape(-1).astype(np.float32)
        d2 = d2.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]
        exp0 = exp[valid >= 1]
        d10 = d1[valid >= 1]
        d20 = d2[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]
        exp1 = exp0[v]
        d1_1 = d10[v]
        d2_1 = d20[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)
        exp_img = np.zeros([ht1, wd1], dtype=np.float32)
        d1_img = np.zeros([ht1, wd1], dtype=np.float32)
        d2_img = np.zeros([ht1, wd1], dtype=np.float32)

        exp_img[yy,xx] = exp1
        d1_img[yy, xx] = d1_1
        d2_img[yy, xx] = d2_1
        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img, exp_img,d1_img,d2_img

    def spatial_transform(self, img1, img2,d1,d2,d1pre,d2pre,bkmask,ints, flow, valid,dc):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:#如果使用随机缩放

            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            d1pre = cv2.resize(d1pre, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
            d2pre = cv2.resize(d2pre, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
            bkmask = cv2.resize(bkmask, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
            ints[0] = ints[0] * scale_x
            ints[1] = ints[1] * scale_y
            flow, valid, exp,d1,d2 = self.resize_sparse_flow_exp_map(flow,dc[:,:,0],d1,d2, valid, fx=scale_x, fy=scale_y)
            dc_out = np.ones_like(flow)
            dc_out[:,:,0] = exp
            dc_out[:,:,1] = valid
        else:
            dc_out = np.ones_like(flow)
            dc_out[:, :, 0] = dc[:,:,0]
            dc_out[:, :, 1] = valid
        if self.do_flip:
            if np.random.rand() < 0.5: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]
                dc_out = dc_out[:, ::-1]
                d1 = d1[:, ::-1]
                d2 = d2[:, ::-1]
                d1pre = d1pre[:, ::-1]
                d2pre = d2pre[:, ::-1]
                bkmask = bkmask[:,::-1]
        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        d1_out = d1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        d2_out = d2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        d1_outp = d1pre[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        d2_outp = d2pre[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        dc_out   = dc_out[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        bkmask = bkmask[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        ints[0,2] = ints[0,2]-x0
        ints[1, 2] = ints[1,2]-y0
        return img1, img2, flow, dc_out,d1_out,d2_out,d1_outp,d2_outp,bkmask,ints, valid


    def __call__(self, img1, img2,d1,d2,d1pre,d2pre,bkmask,ints, flow,dc, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow,dc,d1,d2,d1pre,d2pre,bkmask,ints,valid = self.spatial_transform(img1, img2,d1,d2,d1pre,d2pre,bkmask,ints, flow, valid,dc)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        d1 = np.ascontiguousarray(d1)
        d2 = np.ascontiguousarray(d2)
        d1pre = np.ascontiguousarray(d1pre)
        d2pre = np.ascontiguousarray(d2pre)
        valid = np.ascontiguousarray(valid)
        bkmask = np.ascontiguousarray(bkmask)
        dc = np.ascontiguousarray(dc)
        return img1, img2, flow,dc,d1,d2,d1pre,d2pre,bkmask,ints, valid

class SparseFlowAugmentorm:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3  ), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_exp_map(self, flow, exp, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        exp = exp.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]
        exp0 = exp[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]
        exp1 = exp0[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)
        exp_img = np.ones([ht1, wd1], dtype=np.float32)

        exp_img[yy, xx] = exp1
        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img, exp_img

    def spatial_transform(self, img1, img2, flow, valid, dc):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:

            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # dc   = cv2.resize(dc  , None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
            flow, valid, exp = self.resize_sparse_flow_exp_map(flow, dc[:, :, 0], valid, fx=scale_x, fy=scale_y)
            dc_out = np.ones_like(flow)
            dc_out[:, :, 0] = exp
            dc_out[:, :, 1] = valid
        else:
            dc_out = np.ones_like(flow)
            dc_out[:, :, 0] = dc[:, :, 0]
            dc_out[:, :, 1] = valid
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]
                dc_out = dc_out[:, ::-1]
            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                valid = valid[::-1, :]
                dc_out = dc_out[::-1, :]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        dc_out = dc_out[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow, dc_out, valid

    def __call__(self, img1, img2, flow, dc, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, dc, valid = self.spatial_transform(img1, img2, flow, valid, dc)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)
        dc = np.ascontiguousarray(dc)
        return img1, img2, flow, dc, valid


class NuscAugmentor:
    def __init__(self, crop_size, do_flip=False):
        # 增强参数
        self.crop_size = crop_size  # 裁剪后图像的大小
        self.do_flip = do_flip  # 是否进行翻转
        # self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)

    # def color_transform(self, img):
    #     """颜色增强（对称）"""
    #     img = np.array(self.photo_aug(Image.fromarray(img)), dtype=np.uint8)
    #     return img

    def resize_and_crop(self, img):
        """将图像缩放并裁剪到指定大小"""
        w, h = img.size
        crop_h, crop_w = self.crop_size
        resize = max(crop_h / h, crop_w / w)

        resize_h, resize_w = int(h * resize), int(w * resize)
        crop_h_start = (resize_h - crop_h) // 2
        crop_w_start = (resize_w - crop_w) // 2
        crop = (crop_w_start, crop_h_start, crop_w_start + crop_w, crop_h_start + crop_h)

        if img.mode == 'RGB':
            img = img.resize((resize_w, resize_h), Image.BILINEAR)
        elif img.mode == 'F':
            img = self.resize_with_explicit_mapping(img, (resize_h, resize_w))
        else:
            raise ValueError(f'Unsupported image mode: {img.mode}')

        img = img.crop(crop)

        return img

    def resize_with_explicit_mapping(self, img, resize_dims, fill_value=np.nan):
        # 将图像转换为 NumPy 数组
        img_np = np.array(img)
        # 提取有效数据的位置和值（非 NaN 的位置）
        valid_coords = np.argwhere(~np.isnan(img_np))  # 获取有效数据的坐标 (y, x)
        valid_values = img_np[~np.isnan(img_np)]  # 获取有效数据的值
        # 计算缩放因子
        resize_coef = max(resize_dims[0] / img_np.shape[0], resize_dims[1] / img_np.shape[1])
        # 对有效坐标进行缩放
        scaled_coords = (valid_coords * resize_coef).astype(int)
        # 创建填充了 fill_value 的新图像
        resized_img_np = np.full((resize_dims[0], resize_dims[1]), fill_value, dtype=np.float32)
        # 将有效值填充到新的位置
        for (y, x), value in zip(scaled_coords, valid_values):
            if 0 <= y < resize_dims[0] and 0 <= x < resize_dims[1]:  # 检查边界
                resized_img_np[y, x] = value
        # 转换为 PIL 图像返回
        return Image.fromarray(resized_img_np, mode='F')


    def center_crop(self, img):
        """从图像中心裁剪到指定大小"""
        ht, wd = img.shape[:2]
        crop_h, crop_w = self.crop_size

        # 计算中心裁剪的起始坐标
        center_y, center_x = ht // 2, wd // 2
        start_y = max(0, center_y - crop_h // 2)
        start_x = max(0, center_x - crop_w // 2)

        # 裁剪图像
        img_cropped = img[start_y:start_y + crop_h, start_x:start_x + crop_w]

        # 如果裁剪区域小于目标大小，用填充补齐
        if img_cropped.shape[0] < crop_h or img_cropped.shape[1] < crop_w:
            pad_h = crop_h - img_cropped.shape[0]
            pad_w = crop_w - img_cropped.shape[1]
            img_cropped = cv2.copyMakeBorder(img_cropped, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        return img_cropped

    def flip_transform(self, img):
        """图像翻转"""
        if self.do_flip:
            if np.random.rand() < 0.5:  # 水平翻转
                img = img[:, ::-1]
            if np.random.rand() < 0.1:  # 垂直翻转
                img = img[::-1, :]
        return img

    def __call__(self, img):

        img = self.resize_and_crop(img)

        # 将 PIL 图像转换为 NumPy 数组
        if isinstance(img, Image.Image):
            img = np.array(img)

        return np.ascontiguousarray(img)