import argparse
import json
import math
import os
import textwrap
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

from select_nusc_scene_splits import KEYWORDS


SPLITS = ("train", "val", "test")
SCENE_LIST_FILENAMES = {
    "train": "scene_150_train.txt",
    "val": "scene_150_val.txt",
    "test": "scene_150_test.txt",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="./Datasets/nuscenes/", type=str)
    parser.add_argument("--version", default="v1.0-trainval", type=str)
    parser.add_argument(
        "--scene_list_dir",
        default="./tools/generate_scene_flow_nuscenes/scene_lists",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="./tools/generate_scene_flow_nuscenes/scene_lists/visualization",
        type=str,
    )
    parser.add_argument("--examples_per_tag", default=6, type=int)
    return parser.parse_args()


def read_scene_lists(scene_list_dir):
    split_scene_names = {}
    all_scene_names = set()
    for split_name, filename in SCENE_LIST_FILENAMES.items():
        path = os.path.join(scene_list_dir, filename)
        with open(path, "r") as file:
            scene_names = [line.strip() for line in file if line.strip()]
        split_scene_names[split_name] = scene_names
        all_scene_names.update(scene_names)
    return split_scene_names, all_scene_names


def get_middle_sample(nusc, scene):
    sample_token = scene["first_sample_token"]
    num_steps = scene["nbr_samples"] // 2
    sample = nusc.get("sample", sample_token)
    for _ in range(num_steps):
        if not sample["next"]:
            break
        sample = nusc.get("sample", sample["next"])
    return sample


def build_scene_records(nusc, scene_names):
    scene_records = {}
    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue

        log = nusc.get("log", scene["log_token"])
        sample = get_middle_sample(nusc, scene)
        cam_front = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        image_path = os.path.join(nusc.dataroot, cam_front["filename"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        description_lower = scene["description"].lower()
        matched_tags = [keyword for keyword in KEYWORDS if keyword in description_lower]
        scene_records[scene["name"]] = {
            "scene_name": scene["name"],
            "scene_token": scene["token"],
            "log_token": scene["log_token"],
            "location": log["location"],
            "description": scene["description"],
            "num_samples": scene["nbr_samples"],
            "sample_token": sample["token"],
            "image_path": image_path,
            "image_relpath": cam_front["filename"],
            "matched_tags": matched_tags,
            "tag_flags": {keyword: keyword in description_lower for keyword in KEYWORDS},
        }
    return scene_records


def compute_summary(split_scene_names, scene_records):
    summary = {}
    for split_name, scene_names in split_scene_names.items():
        records = [scene_records[scene_name] for scene_name in scene_names]
        keyword_counts = Counter()
        location_counts = Counter()
        for record in records:
            location_counts[record["location"]] += 1
            for keyword in KEYWORDS:
                keyword_counts[keyword] += int(record["tag_flags"][keyword])

        summary[split_name] = {
            "num_scenes": len(records),
            "num_logs": len({record["log_token"] for record in records}),
            "location_counts": dict(sorted(location_counts.items())),
            "keyword_counts": dict(sorted(keyword_counts.items())),
        }
    return summary


def choose_examples(records, tag, examples_per_tag):
    tag_records = [record for record in records if record["tag_flags"][tag]]
    tag_records.sort(key=lambda x: x["scene_name"])

    picked_records = []
    used_logs = set()
    remaining_records = []
    for record in tag_records:
        if record["log_token"] in used_logs:
            remaining_records.append(record)
            continue
        picked_records.append(record)
        used_logs.add(record["log_token"])
        if len(picked_records) == examples_per_tag:
            return picked_records

    for record in remaining_records:
        if len(picked_records) == examples_per_tag:
            break
        picked_records.append(record)

    return picked_records


def format_title(record, tag):
    matched_tags = ",".join(record["matched_tags"]) if record["matched_tags"] else "none"
    description = "\n".join(textwrap.wrap(record["description"], width=42, break_long_words=False))
    return (
        f"{record['scene_name']} | {record['location']}\n"
        f"focus={tag} | tags={matched_tags}\n"
        f"{description}"
    )


def save_contact_sheet(split_name, tag, records, output_path):
    if not records:
        return []

    num_examples = len(records)
    num_cols = min(3, num_examples)
    num_rows = math.ceil(num_examples / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5.5 * num_rows))
    if hasattr(axes, "flatten"):
        axes = list(axes.flatten())
    else:
        axes = [axes]

    manifest_entries = []
    for ax, record in zip(axes, records):
        image = mpimg.imread(record["image_path"])
        ax.imshow(image)
        ax.set_title(format_title(record, tag), fontsize=9)
        ax.axis("off")
        manifest_entries.append(
            {
                "scene_name": record["scene_name"],
                "location": record["location"],
                "log_token": record["log_token"],
                "description": record["description"],
                "matched_tags": record["matched_tags"],
                "focus_tag": tag,
                "image_path": record["image_path"],
                "image_relpath": record["image_relpath"],
                "sample_token": record["sample_token"],
            }
        )

    for ax in axes[num_examples:]:
        ax.axis("off")

    fig.suptitle(f"{split_name} | {tag} | {num_examples} examples", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return manifest_entries


def save_summary_plot(summary, output_path):
    fig, axes = plt.subplots(1, len(KEYWORDS), figsize=(4.8 * len(KEYWORDS), 5.5), sharey=False)
    split_order = list(SPLITS)
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]

    for ax, keyword in zip(axes, KEYWORDS):
        counts = [summary[split_name]["keyword_counts"][keyword] for split_name in split_order]
        bars = ax.bar(split_order, counts, color=colors)
        ax.set_title(keyword)
        ax.set_ylabel("num scenes")
        ax.set_ylim(0, max(counts) + 2)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2.0, count + 0.1, str(count), ha="center", va="bottom")

    fig.suptitle("Scene Tag Counts by Split", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ensure_scene_records_complete(split_scene_names, scene_records):
    missing_scene_names = []
    for scene_names in split_scene_names.values():
        for scene_name in scene_names:
            if scene_name not in scene_records:
                missing_scene_names.append(scene_name)
    if missing_scene_names:
        raise RuntimeError(f"Failed to build records for scenes: {sorted(missing_scene_names)}")


def main():
    args = parse_args()
    if args.examples_per_tag <= 0:
        raise ValueError("--examples_per_tag must be a positive integer.")

    split_scene_names, all_scene_names = read_scene_lists(args.scene_list_dir)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scene_records = build_scene_records(nusc, all_scene_names)
    ensure_scene_records_complete(split_scene_names, scene_records)

    summary = compute_summary(split_scene_names, scene_records)

    summary_dir = os.path.join(args.output_dir, "summary")
    examples_dir = os.path.join(args.output_dir, "examples")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)

    summary_json_path = os.path.join(summary_dir, "scene_tag_counts.json")
    with open(summary_json_path, "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    summary_plot_path = os.path.join(summary_dir, "scene_tag_counts.png")
    save_summary_plot(summary, summary_plot_path)

    manifest = {}
    for split_name in SPLITS:
        split_records = [scene_records[scene_name] for scene_name in split_scene_names[split_name]]
        for tag in KEYWORDS:
            picked_records = choose_examples(split_records, tag, args.examples_per_tag)
            output_path = os.path.join(examples_dir, f"{split_name}_{tag}.png")
            manifest_key = f"{split_name}_{tag}"
            manifest[manifest_key] = {
                "split": split_name,
                "tag": tag,
                "num_examples": len(picked_records),
                "examples": save_contact_sheet(split_name, tag, picked_records, output_path),
                "output_path": output_path if picked_records else None,
            }

    manifest_path = os.path.join(examples_dir, "examples_manifest.json")
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)

    print(f"Saved summary JSON to {summary_json_path}")
    print(f"Saved summary plot to {summary_plot_path}")
    print(f"Saved example manifest to {manifest_path}")


if __name__ == "__main__":
    main()
