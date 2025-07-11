import os
import pickle

def merge_dict_lists_pkl(pkl1_path: str, pkl2_path: str, output_path: str):
    """
    从两个 pkl 文件中读取 list（每个元素都是 dict），
    并把第二个 list 的元素拼接到第一个 list 末尾，然后保存。

    参数:
      pkl1_path   第一个 pkl 路径（包含 list<dict>）
      pkl2_path   第二个 pkl 路径（包含 list<dict>）
      output_path 合并后 list 的输出文件路径

    返回:
      merged_list 合并后的 list<dict>
    """
    # 1. 读取两个文件
    with open(pkl1_path, 'rb') as f1:
        list1 = pickle.load(f1)
    with open(pkl2_path, 'rb') as f2:
        list2 = pickle.load(f2)

    # 2. 类型检查
    if not isinstance(list1, list) or not isinstance(list2, list):
        raise TypeError(f"Expected both pickles to contain lists, got {type(list1)} and {type(list2)}")

    # 3. 合并
    merged_list = list1 + list2
    #    或者原地合并： list1.extend(list2); merged_list = list1

    # 4. 保存到新的 pkl
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as fout:
        pickle.dump(merged_list, fout)

    return merged_list

# 示例调用（请替换成你自己的路径）：
if __name__ == "__main__":
    pkl_dir    = "Datasets/nuscenes/2_trainval_test_infos"
    file_a     = "nusc_trainval_infos_160_1920_fov_8_15.pkl"
    file_b     = "nusc_trainval_infos_160_1920_fov_8_15_.pkl"
    out_file   = os.path.join(pkl_dir, "merged.pkl")

    merged = merge_dict_lists_pkl(
        os.path.join(pkl_dir, file_a),
        os.path.join(pkl_dir, file_b),
        out_file
    )
    print(f"合并完成，最终列表长度：{len(merged)}")  