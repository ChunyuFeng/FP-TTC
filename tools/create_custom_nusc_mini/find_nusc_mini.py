import pickle
import argparse
import os
import shutil
from tqdm import tqdm

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

    if data is not None:
        data_length = len(data)
        for i in tqdm(range(data_length), desc="Processing data"):
            prev_camera_data = data[i]['prev_camera_data']
            curr_camera_data = data[i]['curr_camera_data']

            for channel, camera_sample_data in prev_camera_data.items():
                src_file_path = os.path.join(args.nusc_root, camera_sample_data['filename'])
                dst_file_path = os.path.join(args.output_dir, camera_sample_data['filename'])
                if os.path.exists(src_file_path):
                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                    shutil.copy(src_file_path, dst_file_path)
            
            for channel, camera_sample_data in curr_camera_data.items():
                src_file_path = os.path.join(args.nusc_root, camera_sample_data['filename'])
                dst_file_path = os.path.join(args.output_dir, camera_sample_data['filename'])
                if os.path.exists(src_file_path):
                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                    shutil.copy(src_file_path, dst_file_path)

        print(f"成功处理 {data_length} 条数据.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pkl file and print its content.")
    parser.add_argument('--pkl_file_path', type=str, required=True, help="Path to the pkl file.")
    parser.add_argument('--nusc_root', type=str, default='/mnt/data/nuScenes', help="Root directory of the nuScenes dataset.")
    parser.add_argument('--output_dir', type=str, default='/mnt/data/custom_nusc', help="Output directory for copied files.")
    parser.add_argument('--surr_view_imgs_vis', action='store_true', help="view the raw surrounding images.")
    args = parser.parse_args()
    main(args=args)