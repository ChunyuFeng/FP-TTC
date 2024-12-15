import os
import shutil


def copy_cam_front_images(src_dir, dst_dir):
    """
    从源目录中拷贝指定通道(CAM_FRONT)的图像到目标目录。

    :param src_dir: 源目录路径
    :param dst_dir: 目标目录路径
    """
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 获取源目录下的所有文件夹
    folder_list = [folder for folder in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, folder))]

    # 根据文件夹名称中的时间戳排序
    folder_list.sort(key=lambda x: int(x.split("__")[-1]))

    # 遍历每个文件夹
    for folder in folder_list:
        folder_path = os.path.join(src_dir, folder)
        # 获取该文件夹下的所有文件
        files = os.listdir(folder_path)

        # 筛选出CAM_FRONT通道的文件
        cam_front_files = [f for f in files if "CAM_FRONT" in f and f.endswith(".jpg")]

        for file_name in cam_front_files:
            src_file = os.path.join(folder_path, file_name)
            dst_file = os.path.join(dst_dir, file_name)

            # 拷贝文件到目标目录
            shutil.copy(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")


# 示例调用
source_directory = "/home/chunyu/WorkSpace/BugStudio/AEB/visualization/scale_gt_with_ori_img31_"  # 源目录路径
destination_directory = "./test/gt_with_img31_"  # 目标目录路径

copy_cam_front_images(source_directory, destination_directory)
