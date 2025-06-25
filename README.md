# FP-TTC: Fast Prediction of Time-to-Collision using Monocular Images

Official implementation of "FP-TTC: Fast Prediction of Time-to-Collision using Monocular Images". Submitted to T-CSVT on May **, 2024.

![pipeline](./images/pipeline_pano.png)



## Abstract

**Time-to-Collision (TTC)** is a measure of the time until an object collides with the observation plane which is a critical input indicator for obstacle avoidance and other downstream modules. Previous works have utilized deep neural networks to estimate TTC with monocular cameras in an end-toend manner, which obtain the state-of-the-art (SOTA) accuracy performance. However, these models usually have deep layers and numerous parameters, resulting in long inference time and high computational overhead. Moreover, existing methods use two frames which are the current and future moments as input to calculate the TTC resulting in a delay during the calculation process. To solve these issues, we propose a novel fast TTC prediction model: FP-TTC. We first use an attention-based scale encoder to model the scale-matching process between images, which significantly reduces the computational overhead as well as improves the model’s accuracy. Meanwhile, a simple but powerful trick is introduced to the model, where we built a time-series decoder and predict the current TTC from RGB images in the past, avoiding the computational delay caused by the system time step interval, and further improved the TTC prediction speed. Compared to the previous SOTA work, our model achieves a parameter reduction of 89.1%, a 6-fold increase in inference speed, a 19.3% improvement in accuracy.

## Setup

### Environment

Our experiments are conducted in Ubuntu20.04 with Anaconda3, Pytorch 1.12.0, CUDA 11.3, 3090 GPU.

1. create conda environment:

```shell
conda create -n fpttc python=3.8 -y
conda activate fpttc
```

2. install dependencies:

```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt 
```

3. clone our code:

```
git clone https://github.com/LChanglin/FP-TTC.git
```

4. download our pretrained weights from [link](https://drive.google.com/drive/folders/1WL2cuKDt2YPERB8WaAScX9qbO4x4p4hI?usp=sharing).



### Datasets

Download Driving and KITTI for training. 

```bash
Datasets
|-- Driving
|   |-- camera_data
|   |-- disparity
|   |-- disparity_change
|   |-- frames_cleanpass
|   `-- optical_flow
`-- kitti
    |-- data_scene_flow
    |   |-- testing
    |   `-- training
    |-- data_scene_flow_calib
    |   |-- testing
    |   `-- training
    `-- data_scene_flow_multi
    |-- testing
    `--training
```



## Usage

We use 3090 GPUs for training and testing.

### training

```bash
# train with our settings
# --resume: load with pretrained weights, used for finetuning (default:./pretrained/fpttc_mix.pth.tar)
# --epoch: training epoches
# --lr: learning rate: set as mentioned in out paper
# --image_size: resolution
sh train.sh
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--epoch 2000 \
--lr 4e-5 \
--batch_size 6 \
--stage 'nuscenes_range_image' \
--image_size 160 320 \
--parallel \
--load_cnet \
--load_cnet_path './pretrained/fpttc_mix.pth.tar' \
--freeze_cnet
```



### inference with your own data

```bash
# test with our settings
# --resume: load with pretrained weights (default:./pretrained/fpttc_mix.pth.tar)
# --inference_dir: tested images
sh train.sh
```



### evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py \
--resume ./pretrained/fpttc_mix.pth.tar \
--inference_dir [PATH TO KITTI]/testing/image_2/
```

The evaluation results will be saved as .npy.

| Pretrained Weights       | Mid Error |
| ------------------------ | --------- |
| fintuned on kitti        | **59.35** |
| trained on mixed dataset | 62.30     |



## Visualization

KITTI:

![viz](./images/viz.png)

## [TODO]
- [x] 维护一张修改过的模型图
- [x] 单任务分别训练，记录其 loss 变化情况 - llw可视化工具

- [x] 将每个场景的数据分别制作真值，记录场景idx，方便后续训练使用连续场景
- [x] ~~通过限制深度的离群值、或者限制scale的比值，来限制一下真值的最大、最小值；否则可视化图像中会出现颜色深浅不一，制作成视频后会闪烁~~
- [x] 视频闪烁原因：真值中每组图像之间的时间间隔不同（100ms和50ms两种时间间隔交替出现），导致其Scale不均匀（速度均匀变化的前提下），导致可视化结果闪烁。
- [x] 图像Input选择合适的帧率，且尽量保持一致。（目前 1/3 image pairs 的时间间隔为 50ms，2/3 image pairs 的时间间隔为 100ms）。现在统一成了100ms。
  
- [x] 由于真值中需要保存路径信息，做如下统一：
  - [x] pkl文件中只保留相对路径，文件保存到 /mnt/data 中，创建符号链接指向 ./Datasets
  - [x] 将 scale map、depth map、risk score map 放到同一个 npy 文件中；

- [x] 调整 bounding box 的尺寸以及位置，使其尽可能不包含除目标 object 以外的点云。

- [x] 预训练：能否找现有的在nuscenes上的模型，用于深度估计、目标检测等任务的模型，作为预训练模型使用？
  - [x] 使用FPTTC与训练的模型，加载cnet的参数，因为cnet结构没有变化。
  
- [ ] Range Image 点云和图像进行区域对齐；目前只是将全部点云做成了Range Image，将图像裁剪到相应区域，并没有进行对齐。
  - [ ] 可能可以采取的方法：将点云投影到图像上（已实现）；然后将投影到图像上的点云制作成 Range Image 格式。
  - [ ] ...其他方法

- [x]  ~~Range Image 如何变稠密？目前已有的大模型可以估计图像的深度，但是没法直接给 Range Image 赋值，是否可以通过 PV Image Depth 2 Point Cloud 2 Range Image Value~~
- [ ]  scale和risk score多任务学习，目前不用考虑显存占用，如何区分两个任务共用的模块？
- [ ]  test.py 中，图像预处理的函数直接调用模型中图像预处理的函数，不要二次实现，否则难以保持统一。
- [ ]  模型训练可视化、测试可视化的代码也只保留一份。
- [ ]  check一下loss 函数，原本scale收敛到0.01的loss时，已经有模糊的轮廓，为什么现在没有了？

- [ ]  将risk改为成“符号分类 + 幅度回归”？
- [ ]  是否可以复用更多的模块？或者调整risk loss，现在risk分支难以收敛。（risk可以收敛，就是效果不好。）

- [x]  统一模型中所有的mask，包括两个任务的mask，以及loss函数所使用的mask
  - [x]  mask 统一为了 $gt_scale\in(0.3, 3.0)$ 的区域
- [x]  ~~能否统一两个任务的值域？$scale\in(0,+\infty)$，基本分布在1附近；$risk\in(-\infty,+\infty)$,能否也映射到$(0,+\infty)$，且以1为分界线？~~
  - [x]  ~~对risk做归一化，更容易收敛一些(tanh)~~
    - [x]  ~~不用做归一化，分析了 risk 的值，值域为 $(-2.5, 2.12)$~~
- [x]  不再freeze，对cnet参数进行微调。
- [ ]  scale的无效值设置为1，risk score的无效值设置为0，计算loss时不加mask，是否可以将空白区域也学习出来？
- [ ]  数据预处理中，增加数据的随机旋转等图像增强；
- [x]  只在主卡打印关键日志；
- [x]  只在主卡输出neptune信息
- [x]  使用小数据集进行测试
  - [x]  目前选取了scene_indice=4的120条数据进行训练
- [ ]  ~~目前用1059组数据，4卡，bs=6，lr=3e-4会收敛很快，但是触底震荡；- 测试一下完整的训练过程；~~
- [ ]  ~~现在采用120组数据，4卡，bs=6，lr=1e-4实验；~~
- [ ]  参考minkocc使用伪标签的方式。
- [ ]  

