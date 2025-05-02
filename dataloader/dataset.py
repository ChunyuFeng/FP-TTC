import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import os
import pickle
import math
import random
from glob import glob
import os.path as osp

from .utils.rectangle_noise import retangle
from .utils import frame_utils
import  cv2
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentorm, NuscAugmentor, NuscRangeImageAugmentor


def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def readPFM(file):
    import re
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:
        return readPFM(path)[0]
def get_grid_np(B,H,W):
    meshgrid_base = np.meshgrid(range(0, W), range(0, H))[::-1]
    basey = np.reshape(meshgrid_base[0], [1, 1, 1, H, W])
    basex = np.reshape(meshgrid_base[1], [1, 1, 1, H, W])
    grid = torch.tensor(np.concatenate((basex.reshape((-1, H, W, 1)), basey.reshape((-1, H, W, 1))), -1)).float()
    return grid.view( H, W, 2)

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, vkitti2=False, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorm(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.driving = False
        self.is_test = False
        self.init_seed = False
        self.test_scene = False
        self.stereo = False
        self.flow_list = []
        self.dispnet =[]
        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.mask_list = []
        self.occ_list = []
        self.rect = retangle()
        self.kit = 0
        self.k = 1
        self.kr = 0
        self.get_depth = 0
        self.kitti_test = 0
        self.sintel_test = 0

        self.vkitti2 = vkitti2

        self.last_image = np.random.randn(320,960,3)
    def __getitem__(self, index):
        self.kit = self.kit +1
        if self.test_scene:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            dispnet = np.abs(disparity_loader(self.dispnet[index]))
            return img1, img2, self.extra_info[index],dispnet
        if self.is_test and not self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        if self.get_depth:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            mask[dc_change > 1.5] = 0
            mask[dc_change < 0.5] = 0
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            dc_change = np.concatenate((dc_change[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
            #读取光流结果
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            for i in range(int(self.kr)):
                imgb1, imgb2, ansb, flag = self.rect.get_mask(img1)
                if flag > 1:
                    img1[imgb1 > 0] = imgb1[imgb1 > 0]
                    img2[imgb2 > 0] = imgb2[imgb2 > 0]
                    flow[imgb1[:, :, 0] > 0, :] = ansb[imgb1[:, :, 0] > 0, :2]
                    dc_change[imgb1[:, :, 0] > 0, 0:1] = ansb[imgb1[:, :, 0] > 0, 2:]
                    d1[imgb1[:, :, 0] > 0] = 10
                    d2[imgb1[:, :, 0] > 0] = dc_change[imgb1[:, :, 0] > 0,0]*10
                    li = ansb[:, :, 2] > 0
                    dc_change[li, 1] = 1
                    mask[imgb1[:, :, 0] > 0]=2

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0

            return img1,img2,flow,dc_change,d1,d2,disp1,disp2,mask,self.extra_info[index]#这个mask是是否有噪音块的掩膜
        if self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            #mask = frame_utils.read_gen(self.mask_list[index])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            # 读取光流结果
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            mask = np.array(mask).astype(np.uint8)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0
            return img1, img2, flow, dc_change, d1, d2, disp1, disp2, mask,valid, self.extra_info[index]  # 这个mask是是否有噪音块的掩膜
        if self.sintel_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            #mask = frame_utils.read_gen(self.mask_list[index])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            # 读取光流结果
            flow = frame_utils.read_gen(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            mask = np.array(mask).astype(np.uint8)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0
            return img1, img2, flow, dc_change, d1, d2, disp1, disp2, mask,0, self.extra_info[index]
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        d1,d2,mask = self.get_dc(index)

        dc_change = d2/d1
        mask[dc_change>1.5] = 0
        mask[dc_change <0.5] = 0
        dc_change[mask==0]=0
        if self.occlusion:
            dcc = dc_change
            dcc = abs(cv2.filter2D(dcc,-1,kernel=self.kernel2))
            maskd = torch.from_numpy(dcc>1).bool()
            dc_change[maskd!=0] = 0
            masku = dc_change>0
            #再加一个遮挡
            dc_change = np.concatenate((dc_change[:,:,np.newaxis],masku[:,:,np.newaxis]),axis =2 )
        else:
            dc_change = np.concatenate((dc_change[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
        if self.sparse:
            if self.driving:
                flow, valid = frame_utils.readFlowdriving(self.flow_list[index])
            elif self.stereo:
                flowx = disparity_loader(self.depth_list[index][0])
                flow = np.concatenate((flowx[:, :, np.newaxis], flowx[:, :, np.newaxis]), axis=2)
                valid = flowx>0
                flow[:,:,1]=0
            elif self.vkitti2:
                flow, valid = frame_utils.read_vkitti2_flow(self.flow_list[index])
                mask = np.logical_and(mask, valid)
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])

        else:
            flow = frame_utils.read_gen(self.flow_list[index])


        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        dc_change = np.array(dc_change).astype(np.float32)


        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]


        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow,dc_change, valid = self.augmentor(img1, img2, flow,dc_change, valid)
            else:
                img1, img2, flow, dc_change = self.augmentor(img1, img2, flow,dc_change)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change   = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        # return img1, img2, flow, dc_change, valid.float()
        return img1, img2, dc_change

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.depth_list = v * self.depth_list
        self.occ_list = v * self.occ_list
        return self

    def __len__(self):
        return len(self.image_list)

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/mnt/pool/Datasets/OpticalFlow/FlyingThings3D/', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)
        exclude = np.loadtxt('/mnt/pool/Datasets/OpticalFlow/exclude.txt', delimiter=' ', dtype=np.unicode_)
        exclude = set(exclude)
        self.occlusion = False
        self.driving = True
        for cam in ['left','right']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                d0_dirs = sorted(glob(osp.join(root, 'disparity/TRAIN/*/*')))
                d0_dirs = sorted([osp.join(f, cam) for f in d0_dirs])

                dc_dirs = sorted(glob(osp.join(root, 'disparity_change/TRAIN/*/*')))
                dc_dirs = sorted([osp.join(f, direction, cam) for f in dc_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir,d0dir,dcdir in zip(image_dirs, flow_dirs,d0_dirs,dc_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    d0s = sorted(glob(osp.join(d0dir, '*.pfm')))
                    dcs = sorted(glob(osp.join(dcdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        tag = '/'.join(images[i].split('/')[-5:])
                        if tag in exclude:
                            print("Excluding %s" % tag)
                            continue
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                            self.depth_list += [[d0s[i], dcs[i]]]
                            frame_id = images[i].split('/')[-1]
                            self.extra_info += [[frame_id]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]
                            self.depth_list += [[d0s[i+1], dcs[i+1]]]
                            frame_id = images[i+1].split('/')[-1]
                            self.extra_info += [[frame_id]]
    def triangulation(self, disp, bl=1):#kitti flow 2015

        fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1])+d1)
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='./Datasets/kitti/data_scene_flow',get_depth=0):
        super(KITTI, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        if split == 'testing':
            self.is_test = True
        if split == 'submit':
            self.is_test = True
        if split == 'submitother':
            self.is_test = True
        if split =='test':
            self.test_scene = True
        self.occlusion = False
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]

        root_img = './Datasets/kitti/data_scene_flow_multi'
        if split == 'training':
            root = osp.join(root, split)
            root_img = osp.join(root_img, split)

            images1o = sorted(glob(osp.join(root_img, 'image_2/*_09.png')))
            images2o = sorted(glob(osp.join(root_img, 'image_2/*_10.png')))
            disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

            for j in range(images2o.__len__()):
                if j%5!=0:
                    print(images1o[j])
                    images1.append(images1o[j])
                    images2.append(images2o[j])
                    disp1.append(disp1o[j])
                    disp2.append(disp2o[j])


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]
        if split == 'training':
            flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            for j in range(flowo.__len__()):
                if j%5!=0:
                    flow.append(flowo[j])
        elif split == 'testing':
            flow = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        self.flow_list = flow


    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask

class Driving(FlowDataset):
    def __init__(self, aug_params=None,  split='training',root='/mnt/pool/Datasets/OpticalFlow/Driving'):
        super(Driving, self).__init__(aug_params, sparse=True)
        self.calib = []
        self.occlusion = False
        self.driving = True
        level_stars = '/*' * 6
        candidate_pool = glob('%s/optical_flow%s' % (root, level_stars))
        for flow_path in sorted(candidate_pool):
            idd = flow_path.split('/')[-1].split('_')[-2]
            if 'into_future' in flow_path:
                idd_p1 = '%04d' % (int(idd) - 1)
            else:
                idd_p1 = '%04d' % (int(idd) + 1)
            if os.path.exists(flow_path.replace(idd, idd_p1)):
                d0_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','disparity')
                d0_path = '%s/%s.pfm' % (d0_path.rsplit('/', 1)[0], idd)
                dc_path = flow_path.replace('optical_flow', 'disparity_change')
                dc_path = '%s/%s.pfm' % (dc_path.rsplit('/', 1)[0], idd)
                im_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','frames_cleanpass')
                im0_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd_p1)
                im1_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd)
                frame_id = im1_path.split('/')[-1]
                self.extra_info += [[frame_id]]
                #calib.append('%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0]))
                self.flow_list += [flow_path]
                self.image_list += [[im0_path,im1_path]]
                self.depth_list += [[d0_path,dc_path]]
                self.calib +=['%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0])]
    def triangulation(self, disp,index, bl=1):#kitti flow 2015
        if '15mm_' in self.calib[index]:
            fl = 450  # 450
        else:
            fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1])+d1)
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1,index),self.triangulation(d2,index),mask


class nuScenes(data.Dataset):
    def __init__(self, aug_params=None, split='training', train_info_file='train_infos_500_ground_31mix_.pkl',
                 root='/home/chunyu/WorkSpace/BugStudio/FP-TTC/Datasets/nuscenes'):
        self.aug_params = aug_params
        self.split = split
        self.root = root
        self.train_info_file = train_info_file
        self.data = None  # 用于存储从 pkl 文件中加载的数据

        # 根据 split 加载对应的 pkl 文件
        pkl_file_path = osp.join(root, self.train_info_file)

        # 检查文件是否存在
        if osp.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded data from {pkl_file_path}")
        else:
            raise FileNotFoundError(f"No such file: {pkl_file_path}")

        # 数据增强设置
        self.augmentor = None
        if self.aug_params is not None:
            self.augmentor = NuscAugmentor(**self.aug_params)

        # 将 pkl 文件中的数据转换为模型所需的格式
        self.image_list = []
        self.timestamp_list = []
        self.depth_list = []

        infos = self.data['infos']
        for i in range(len(infos) - 1):
            current_info = infos[i]
            next_info = infos[i + 1]

            # 25 ms < timestamp_diff < 125 ms
            if abs(next_info['timestamp'] - current_info['timestamp'])/1e3 < 25 or abs(next_info['timestamp'] - current_info['timestamp'])/1e3 > 125:
                continue
            else:
                self.image_list.append([current_info['imgs_path']['CAM_FRONT'], next_info['imgs_path']['CAM_FRONT']])
                self.timestamp_list.append([current_info['timestamp'], next_info['timestamp']])
                self.depth_list.append([current_info['gt_path']['CAM_FRONT'], next_info['gt_path']['CAM_FRONT']])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # 获取图像对、时间戳和深度文件路径
        img1_path, img2_path = self.image_list[index]
        timestamp1, timestamp2 = self.timestamp_list[index]
        depth1_path, depth2_path = self.depth_list[index]

        image_path_prefix = '/home/chunyu/WorkSpace/BugStudio/FP-TTC/Datasets/nuscenes/'
        img1_path = image_path_prefix + img1_path
        img2_path = image_path_prefix + img2_path

        # 读取图像和深度数据
        img1 = frame_utils.read_nusc_image(img1_path)
        img2 = frame_utils.read_nusc_image(img2_path)
        gt_scale= frame_utils.read_nusc_scale(depth1_path)

        # 数据增强
        if self.augmentor is not None:
            random_state = random.getstate()
            img1 = self.augmentor(img1)
            random.setstate(random_state)
            img2 = self.augmentor(img2)
            random.setstate(random_state)
            gt_scale = self.augmentor(gt_scale)

            # img1, img2, gt_scale= [self.augmentor(x) for x in [img1, img2, gt_scale]]

        mask = gt_scale > 0

        # 转换为 Tensor
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float()
        gt_scale = torch.from_numpy(gt_scale).float()
        mask = torch.from_numpy(mask)

        # 拼接gt_scale和mask
        gt_scale_with_mask = torch.cat((gt_scale.unsqueeze(0), mask.unsqueeze(0).float()), dim=0)

        # #################### 统计gt中的有效值 ####################
        # # 点云数只占图像像素数的 1%
        # non_nan_values = gt_scale[~torch.isnan(gt_scale)]
        # non_nan_count = non_nan_values.numel()


        # 返回图像、时间戳、深度
        # return img1_tensor, img2_tensor, timestamp1, timestamp2, gt_scale, mask
        return img1_tensor, img2_tensor, gt_scale_with_mask

    def __rmul__(self, v):
        self.timestamp_list = v * self.timestamp_list
        self.image_list = v * self.image_list
        self.depth_list = v * self.depth_list
        # self.occ_list = v * self.occ_list
        return self

class nuScenes_range_image(data.Dataset):
    def __init__(self,
                 aug_params=None,
                 split='training',
                 train_info_path='./Datasets/nuscenes/2_trainval_test_infos',
                 train_info_file='nusc_trainval_infos_160_1920.pkl'
                 ):
        self.aug_params = aug_params
        self.split = split
        # self.root = root
        self.train_info_path = train_info_path
        self.train_info_file = train_info_file
        self.data = None  # 用于存储从 pkl 文件中加载的数据
        # self.train_location = train_location

        # 根据 split 加载对应的 pkl 文件
        pkl_file_path = osp.join(self.train_info_path, self.train_info_file)

        # 检查文件是否存在
        if osp.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded data from {pkl_file_path}")
        else:
            raise FileNotFoundError(f"No such file: {pkl_file_path}")

        # # 使用 scene_indice 为 4 的数据，做小数据集快速迭代测试
        # self.data = [d for d in self.data if d.get('scene_indice') == '4']

        # 数据增强设置
        self.augmentor = None
        if self.aug_params is not None:
            self.augmentor = NuscRangeImageAugmentor(**self.aug_params)

        # pkl 文件中包含：
        # - surround view images pairs (nusc sample data format)
        # - lidar data pairs (nusc sample data format)
        # - sensor metas (including calibrated infos and ego pose infos of LiDAR and cameras)
        # - gt range images path (including scale map, depth map and risk score map)
        # - nuscenes scene flow pointcloud path
        self.image_list = [] # input images
        self.sensor_meta_list = []
        self.scale_map_list = [] # ground truth
        self.risk_score_map_list = []
        # TODO: 目前 depth map 是将碰撞点反投影回图像平面时使用的，仅在可视化时使用
        self.depth_map_list = []

        for i in range(len(self.data)):
            
            range_image_path = os.path.join(self.data[i]['gt_map_path'], 'range_image.npy')
            if not osp.exists(range_image_path):
                print(f"Range image file {range_image_path} does not exist.")
                continue
            # 读取 range image
            range_image = np.load(range_image_path, allow_pickle=True).item()

            self.image_list.append([self.data[i]['prev_camera_data'],
                                    self.data[i]['curr_camera_data']])
            self.sensor_meta_list.append(self.data[i]['sensor_metas'])
            self.scale_map_list.append(range_image['scale'])
            self.risk_score_map_list.append(range_image['risk_score'])
            self.depth_map_list.append(range_image['depth'])


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        prev_surr_view_imgs = {}
        curr_surr_view_imgs = {}

        # camera_channels = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
        #                    'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT']
        # 后续会按照这个顺序拼接，和 Range Image 对应 
        camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                           'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        
        path_prefix = './Datasets/nuscenes/'
        for channel in camera_channels:
            prev_surr_view_imgs_path = os.path.join(path_prefix,
                                                    self.image_list[index][0][channel]['filename'])
            prev_surr_view_imgs[channel] = Image.open(prev_surr_view_imgs_path)

            curr_surr_view_imgs_path = os.path.join(path_prefix,
                                                    self.image_list[index][1][channel]['filename'])
            curr_surr_view_imgs[channel] = Image.open(curr_surr_view_imgs_path)
        
        gt_scale_map = self.scale_map_list[index]
        gt_risk_score_map = self.risk_score_map_list[index]
        gt_depth_map = self.depth_map_list[index]

        sensor_meta = self.sensor_meta_list[index]

        # 数据增强
        if self.augmentor is not None:
            prev_surr_view_imgs = self.augmentor(prev_surr_view_imgs)
            curr_surr_view_imgs = self.augmentor(curr_surr_view_imgs)
        
        # 转换为 Tensor
        for channel in camera_channels:
            prev_surr_view_imgs[channel] = torch.from_numpy(prev_surr_view_imgs[channel]).permute(2, 0, 1).float()
            curr_surr_view_imgs[channel] = torch.from_numpy(curr_surr_view_imgs[channel]).permute(2, 0, 1).float()
        
        prev_surr_view_imgs_tensor = torch.stack([prev_surr_view_imgs[channel] for channel in camera_channels], dim=0)
        curr_surr_view_imgs_tensor = torch.stack([curr_surr_view_imgs[channel] for channel in camera_channels], dim=0)

        gt_scale_map = torch.from_numpy(gt_scale_map).float()
        gt_risk_score_map = torch.from_numpy(gt_risk_score_map).float()
        gt_depth_map = torch.from_numpy(gt_depth_map).float()

        # scale 取 (0.3, 3.0) 之间的值
        mask_scale = (gt_scale_map > 0.3) & (gt_scale_map < 3.0)
        # risk score & depth 取相同的 mask
        mask_risk_score = mask_scale
        mask_depth = mask_scale
        # 拼接gt_scale和mask
        gt_scale_map_with_mask = torch.cat((gt_scale_map.unsqueeze(0), mask_scale.unsqueeze(0).float()), dim=0)
        gt_risk_score_map_with_mask = torch.cat((gt_risk_score_map.unsqueeze(0), mask_risk_score.unsqueeze(0).float()), dim=0)
        gt_depth_map_with_mask = torch.cat((gt_depth_map.unsqueeze(0), mask_depth.unsqueeze(0).float()), dim=0)

        return (prev_surr_view_imgs_tensor,
                curr_surr_view_imgs_tensor,
                gt_scale_map_with_mask, # model 的输入以及真值
                gt_risk_score_map_with_mask, # model 的输入以及真值
                gt_depth_map_with_mask, # model 的输入以及真值
                sensor_meta)

    def __rmul__(self, v):
        self.image_list = v * self.image_list
        self.sensor_meta_list = v * self.sensor_meta_list
        self.scale_map_list = v * self.scale_map_list
        self.risk_score_map_list = v * self.risk_score_map_list
        self.depth_map_list = v * self.depth_map_list
        return self

def fetch_dataloader(args, TRAIN_DS='C+T+K/S'):
    """ Create the data loader for the corresponding trainign set """
    train_dataset = None

    if args.stage == 'driving':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        train_dataset = driving

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        driving = Driving(aug_params, split='training')
        train_dataset = clean_dataset+driving

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        kitti = KITTI(aug_params, split='training')
        train_dataset = 100*kitti

    elif args.stage == 'nuscenes':
        aug_params = {'crop_size': args.image_size, 'do_flip': False, 'rotate': True, 'rotate_prob': 0.1, 'rotate_angle': 90}
        train_info_file = 'train_infos_500_ground.pkl'
        nuscenes = nuScenes(aug_params, train_info_file=train_info_file, split='training')
        train_dataset = 100*nuscenes

    elif args.stage == 'nuscenes_range_image':
        aug_params = {'crop_size': args.image_size, 'do_flip': False, 'rotate': False, 'rotate_prob': 0.1, 'rotate_angle': 90}
        train_info_file = 'nusc_trainval_infos_160_1920.pkl'
        train_info_path = './Datasets/nuscenes/2_trainval_test_infos'

        nuscenes = nuScenes_range_image(aug_params,
                                        train_info_file=train_info_file,
                                        train_info_path=train_info_path,
                                        split='training')

        train_dataset = 1*nuscenes

    elif args.stage == 'mix':
        nusc_aug_params = {'crop_size': args.image_size, 'do_flip': False, 'rotate': True, 'rotate_prob': 0.2, 'rotate_angle': 90}
        kitti_aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        train_info_file = 'train_infos_500_ground_31mix_.pkl'
        nuscenes = nuScenes(nusc_aug_params, train_info_file=train_info_file, split='training')
        kitti = KITTI(kitti_aug_params, split='training')
        train_dataset = torch.utils.data.ConcatDataset([50*nuscenes, 50*kitti])

    # print('Training with %d image pairs' % len(train_dataset.image_list))
    return train_dataset
