# 1) 创建 conda 环境（或用 environment.yml）
conda create -n fpttc_py39 python=3.9 -y
conda activate fpttc_py39

# 2) 安装 torch/cu117（conda）
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# 3) 安装 numpy/scipy/matplotlib（conda-forge）
conda install -y -c conda-forge numpy=1.24.3 scipy=1.10.1 matplotlib=3.5.3

# 4) 安装 mmcv-full（按 cu117/torch1.13 的 wheel）
python -m pip install -U pip
python -m pip uninstall -y mmcv mmcv-full || true
python -m pip install "mmcv-full==1.7.1" -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# 5) 安装 pip 依赖（关键：--no-deps）
python -m pip install -r requirements-pip.txt --no-deps

# 6) 清掉可能的 opencv headless（若被装入）
python -m pip uninstall -y opencv-python-headless || true

# 7) 一键验收
python -c "import torch, mmcv, numpy as np; import cv2; import matplotlib.pyplot as plt; \
print('torch', torch.__version__, torch.version.cuda); \
print('mmcv', mmcv.__version__); \
print('numpy', np.__version__); \
print('cv2', cv2.__version__); \
print('mpl ok')"