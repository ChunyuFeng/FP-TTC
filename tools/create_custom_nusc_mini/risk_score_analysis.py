import pickle
import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.nusc_paths import infer_nusc_dataset_root, resolve_nusc_path

def load_pkl_file(filepath):
    """
    从指定路径读取 pkl 文件，并返回内容.
    """
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def main(args):
    """
    主函数，解析命令行参数并调用加载函数.
    """
    data = load_pkl_file(args.pkl_file_path)
    max_val = []
    min_val = []
    if data is not None:
        dataset_root = infer_nusc_dataset_root(args.pkl_file_path)
        data_length = len(data)
        for i in tqdm(range(data_length), desc="Processing data"):
            gt_map_path = resolve_nusc_path(data[i]['gt_map_path'], dataset_root) / 'range_image.npy'
            gt_map = np.load(gt_map_path, allow_pickle=True).item()
            scale_map = gt_map['scale']
            mask = (scale_map>0.3)&(scale_map<3.0)
            risk_score_map = gt_map['risk_score']
            risk_score_map_valid = risk_score_map[mask]
            max_ = np.max(risk_score_map_valid) if risk_score_map_valid.size > 0 else None
            min_ = np.min(risk_score_map_valid) if risk_score_map_valid.size > 0 else None
            max_val.append(max_)
            min_val.append(min_)
        
        # 过滤掉 None 值后计算最大和最小值
        filtered_max_val = [v for v in max_val if v is not None]
        filtered_min_val = [v for v in min_val if v is not None]

        if filtered_max_val:
            max_in_max_val = max(filtered_max_val)
            min_in_max_val = min(filtered_max_val)
        else:
            max_in_max_val = None
            min_in_max_val = None

        if filtered_min_val:
            max_in_min_val = max(filtered_min_val)
            min_in_min_val = min(filtered_min_val)
        else:
            max_in_min_val = None
            min_in_min_val = None

        print("max_val 中的最大值:", max_in_max_val)
        print("max_val 中的最小值:", min_in_max_val)
        print("min_val 中的最大值:", max_in_min_val)
        print("min_val 中的最小值:", min_in_min_val)
        print("finish")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pkl file and print its content.")
    parser.add_argument('--pkl_file_path', type=str, required=True, help="Path to the pkl file.")
    args = parser.parse_args()
    main(args=args)
