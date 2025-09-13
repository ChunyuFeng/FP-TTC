#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import pickle
import shutil
from typing import List, Set

def load_used_scene_flow_dirs_from_pkls(pkl_paths: List[str]) -> Set[str]:
    """
    从多个 infos.pkl 中收集被用到的 scene_flow 文件夹的“绝对路径”集合。
    期望条目结构里包含形如 info['scene_flow_path'] 的键。
    """
    used = set()
    for pkl in pkl_paths:
        with open(pkl, 'rb') as f:
            infos = pickle.load(f)
        for it in infos:
            # 兼容：有的工程字段名可能不同，优先使用示例中的 'scene_flow_path'
            sf_path = it.get('scene_flow_path', None)
            if sf_path is None:
                # 兜底：有些工程只存了 folder_name 或 gt_map_path 里带了 folder_name
                folder_name = None
                if 'gt_map_path' in it:
                    folder_name = os.path.basename(it['gt_map_path'].rstrip('/'))
                if folder_name is None and 'prev_lidar_data' in it:
                    # 若你想更强的兜底可从别的字段拼回 folder_name
                    pass
                if folder_name is not None:
                    # 需要调用者传 root，用相对名拼出绝对路径，下面会做二次规范化匹配
                    # 这里先跳过，在主流程里做由 folder_name -> abs path 的匹配
                    used.add(folder_name)
                continue

            # 规范化为绝对路径
            used.add(os.path.abspath(sf_path))
    return used

def list_scene_flow_dirs(root: str) -> List[str]:
    """
    列出 scene_flow 根目录下一层的所有子目录（每个子目录就是一个样本，如 scene_XX_...）
    """
    out = []
    for name in os.listdir(root):
        d = os.path.join(root, name)
        if os.path.isdir(d):
            out.append(os.path.abspath(d))
    return out

def dir_size_bytes(path: str) -> int:
    total = 0
    for dp, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total

def human_bytes(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def main():
    parser = argparse.ArgumentParser(description="Prune unused scene_flow folders based on infos.pkl.")
    parser.add_argument("--scene_flow_root", required=True,
                        help="例如 ./Datasets/nuscenes/0_scene_flow/scene_flow_all_frames")
    parser.add_argument("--pkl", nargs="+", required=True,
                        help="一个或多个 infos.pkl 路径（支持 train/val/test 多份）")
    parser.add_argument("--delete", action="store_true",
                        help="真的删除（默认只预览统计）")
    parser.add_argument("--safe_only_contains", nargs="+", default=["pc_prev.npy", "pc_curr.npy"],
                        help="为安全起见，仅删除包含这些文件名的目录（默认：pc_prev.npy 与 pc_curr.npy 都存在才删）")
    parser.add_argument("--skip_size", action="store_true",
                        help="跳过大小统计（更快）")
    parser.add_argument("--log_file", default="pruned_scene_flow_list.txt",
                        help="把删除/候删清单记录到该文本文件")
    args = parser.parse_args()

    root = os.path.abspath(args.scene_flow_root)
    all_dirs = list_scene_flow_dirs(root)

    # 1) 从 pkl 里收集被用到的“绝对路径”或“文件夹名”
    used_from_pkls = load_used_scene_flow_dirs_from_pkls(args.pkl)

    # 2) 兼容：如果 used 集合里有纯文件夹名（非绝对路径），映射到 root 下真实目录
    #    形如 'scene_20_n015-...__LIDAR_TOP__1531884294899270'
    used_abs = set()
    names_to_abs = {os.path.basename(d): d for d in all_dirs}
    for item in used_from_pkls:
        if os.path.isabs(item):
            used_abs.add(os.path.abspath(item))
        else:
            # 当作目录名匹配
            if item in names_to_abs:
                used_abs.add(names_to_abs[item])
            else:
                # 尝试更宽松匹配：完全包含关系
                hits = [d for d in all_dirs if os.path.basename(d) == item or item in os.path.basename(d)]
                used_abs.update(hits)

    # 3) 计算未使用目录集合
    all_abs = set(all_dirs)
    unused_abs = sorted(all_abs - used_abs)

    # 4) 安全过滤：只删包含关键文件的目录（避免误删误伤）
    def contains_required_files(d):
        return all(os.path.exists(os.path.join(d, fn)) for fn in args.safe_only_contains)

    candidates = [d for d in unused_abs if contains_required_files(d)]

    # 打印统计
    print(f"[统计] scene_flow 根目录：{root}")
    print(f"[统计] 全部目录数：{len(all_dirs)}")
    print(f"[统计] 被使用目录数：{len(used_abs)}")
    print(f"[统计] 计划删除（未使用且包含关键文件）目录数：{len(candidates)}")

    if not args.skip_size:
        total_bytes = sum(dir_size_bytes(d) for d in candidates)
        print(f"[统计] 预计可释放空间：{human_bytes(total_bytes)}")

    # 写日志
    with open(args.log_file, "w") as f:
        f.write("# Unused scene_flow folders (candidates for deletion)\n")
        for d in candidates:
            f.write(d + "\n")
    print(f"[输出] 候删清单已写入: {args.log_file}")

    # 5) 真删除
    if args.delete:
        print("[执行] 开始删除……")
        for d in candidates:
            try:
                shutil.rmtree(d)
                print(f"  已删除: {d}")
            except Exception as e:
                print(f"  删除失败: {d} -> {e}")
        print("[完成] 删除流程结束。")
    else:
        print("[提示] 目前是 dry-run（只统计不删除）。确认无误后可加 --delete 真删。")

if __name__ == "__main__":
    main()
