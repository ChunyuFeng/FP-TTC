import os
import pickle
import shutil  # 用于文件拷贝

# 输入路径
camera_lut_path = './Datasets/nuscenes/camera_lut_sorted_ground.pkl'

# 输出目录
destination_dir = './test/imgs'

# 确保目标目录存在
os.makedirs(destination_dir, exist_ok=True)

# 读取 camera_lut.pkl
with open(camera_lut_path, 'rb') as file:
    camera_lut = pickle.load(file)

# 遍历并拷贝文件
for i in range(0, 420):
    filename = camera_lut[i]['CAM_FRONT']['filename']
    filepath = os.path.join('./Datasets/nuscenes/', filename)
    if os.path.isfile(filepath):  # 检查文件是否存在
        # 构造目标文件路径
        destination_path = os.path.join(destination_dir, os.path.basename(filename))
        # 拷贝文件
        shutil.copy(filepath, destination_path)
        print(f"Copied: {filepath} -> {destination_path}")
    else:
        print(f"File not found: {filepath}")

print('Files have been copied successfully.')
