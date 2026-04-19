#!/usr/bin/env python3
import argparse
import os
import pickle
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tools.cyberrock.sjtu_pipeline_utils import CAMERA_CHANNELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-sample SAM 2 automatic mask caches for SJTU curr camera images"
    )
    parser.add_argument("--test_info_path", type=str, required=True, help="SJTU test info pkl path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save per-sample npz caches")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM 2.1 checkpoint")
    parser.add_argument("--sam2_model_cfg", type=str, required=True, help="SAM 2 model config path")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample caches")
    parser.add_argument("--vis_dir", type=str, default="", help="Optional directory to save per-sample mask visualizations")
    parser.add_argument("--start_idx", type=int, default=0, help="Start sample index (inclusive)")
    parser.add_argument("--end_idx", type=int, default=-1, help="End sample index (exclusive), -1 means all")
    parser.add_argument("--sample_idx", type=int, default=-1, help="Optional single sample index to process")
    parser.add_argument(
        "--camera_channels",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of camera channels to process, e.g. CAM_FRONT",
    )
    return parser.parse_args()


def load_test_infos(test_info_path):
    with open(test_info_path, "rb") as f:
        return pickle.load(f)


def import_sam2():
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise ImportError(
            "SAM 2 is not available in the current environment. "
            "Create a dedicated Python>=3.10 env with torch>=2.5.1 and install the official "
            "facebookresearch/sam2 package, then rerun this script."
        ) from exc
    return torch, SAM2AutomaticMaskGenerator, build_sam2


def build_generator(args):
    torch, SAM2AutomaticMaskGenerator, build_sam2 = import_sam2()
    model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=args.device)
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        min_mask_region_area=80,
        output_mode="binary_mask",
        use_m2m=False,
    )
    return torch, generator


def load_rgb_image(image_path):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def convert_bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    x1 = int(round(float(x)))
    y1 = int(round(float(y)))
    x2 = int(round(float(x + w - 1)))
    y2 = int(round(float(y + h - 1)))
    return [x1, y1, x2, y2]


def filter_and_deduplicate_masks(mask_dicts, image_shape):
    image_area = int(image_shape[0] * image_shape[1])
    filtered = []
    for mask_dict in mask_dicts:
        area = int(mask_dict["area"])
        predicted_iou = float(mask_dict["predicted_iou"])
        stability_score = float(mask_dict["stability_score"])
        if predicted_iou < 0.88:
            continue
        if stability_score < 0.92:
            continue
        if area < 80:
            continue
        if area > int(round(0.35 * image_area)):
            continue
        filtered.append(mask_dict)

    filtered.sort(key=lambda m: (-float(m["predicted_iou"]), int(m["area"])))
    kept = []
    for mask_dict in filtered:
        seg = np.asarray(mask_dict["segmentation"], dtype=bool)
        duplicate = False
        for kept_dict in kept:
            kept_seg = np.asarray(kept_dict["segmentation"], dtype=bool)
            if mask_iou(seg, kept_seg) >= 0.85:
                duplicate = True
                break
        if not duplicate:
            kept.append(mask_dict)
    return kept


def encode_camera_masks(mask_dicts, image_shape):
    h, w = image_shape[:2]
    if not mask_dicts:
        return {
            "mask_count": np.array([0], dtype=np.int32),
            "mask_shape": np.array([h, w], dtype=np.int32),
            "mask_bits": np.zeros((0, (h * w + 7) // 8), dtype=np.uint8),
            "bbox_xyxy": np.zeros((0, 4), dtype=np.int32),
            "area": np.zeros((0,), dtype=np.int32),
            "predicted_iou": np.zeros((0,), dtype=np.float32),
            "stability_score": np.zeros((0,), dtype=np.float32),
        }

    masks = np.stack([np.asarray(m["segmentation"], dtype=np.uint8).reshape(-1) for m in mask_dicts], axis=0)
    mask_bits = np.packbits(masks, axis=1)
    bbox_xyxy = np.asarray([convert_bbox_xywh_to_xyxy(m["bbox"]) for m in mask_dicts], dtype=np.int32)
    area = np.asarray([int(m["area"]) for m in mask_dicts], dtype=np.int32)
    predicted_iou = np.asarray([float(m["predicted_iou"]) for m in mask_dicts], dtype=np.float32)
    stability_score = np.asarray([float(m["stability_score"]) for m in mask_dicts], dtype=np.float32)
    return {
        "mask_count": np.array([len(mask_dicts)], dtype=np.int32),
        "mask_shape": np.array([h, w], dtype=np.int32),
        "mask_bits": mask_bits.astype(np.uint8),
        "bbox_xyxy": bbox_xyxy,
        "area": area,
        "predicted_iou": predicted_iou,
        "stability_score": stability_score,
    }


def color_for_index(index):
    palette = [
        (231, 76, 60),
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
        (26, 188, 156),
        (149, 165, 166),
        (243, 156, 18),
        (52, 73, 94),
    ]
    return palette[index % len(palette)]


def overlay_camera_masks(rgb, mask_dicts, camera_channel):
    vis = rgb.copy()
    overlay = rgb.copy()
    for mask_idx, mask_dict in enumerate(mask_dicts):
        color = np.asarray(color_for_index(mask_idx), dtype=np.uint8)
        seg = np.asarray(mask_dict["segmentation"], dtype=bool)
        overlay[seg] = (0.65 * overlay[seg] + 0.35 * color).astype(np.uint8)
        x1, y1, x2, y2 = convert_bbox_xywh_to_xyxy(mask_dict["bbox"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)
        label = f"m{mask_idx} iou={float(mask_dict['predicted_iou']):.2f} s={float(mask_dict['stability_score']):.2f}"
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color.tolist(),
            2,
            cv2.LINE_AA,
        )
    vis = overlay
    header = f"{camera_channel} masks={len(mask_dicts)}"
    cv2.putText(
        vis,
        header,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def build_visualization_mosaic(camera_visuals, tile_hw=(216, 384)):
    tiles = []
    th, tw = tile_hw
    for ch in CAMERA_CHANNELS:
        rgb = camera_visuals[ch]
        tile = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    row1 = np.concatenate(tiles[:3], axis=1)
    row2 = np.concatenate(tiles[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def save_sample_visualization(vis_path, sample_idx, sample_info, camera_visuals):
    if len(camera_visuals) == 1:
        camera_channel, rgb = next(iter(camera_visuals.items()))
        canvas_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        title = (
            f"sample={sample_idx}  scene={sample_info.get('scene_indice', '')}  "
            f"ros_msg_seq_curr={sample_info.get('ros_msg_seq_curr', '')}  "
            f"camera={camera_channel}"
        )
        cv2.putText(
            canvas_bgr,
            title,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(vis_path), canvas_bgr)
        return

    mosaic_rgb = build_visualization_mosaic(camera_visuals)
    mosaic_bgr = cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR)
    title = (
        f"sample={sample_idx}  scene={sample_info.get('scene_indice', '')}  "
        f"ros_msg_seq_curr={sample_info.get('ros_msg_seq_curr', '')}"
    )
    cv2.putText(
        mosaic_bgr,
        title,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(vis_path), mosaic_bgr)


def save_sample_cache(output_path, sample_idx, sample_info, camera_payloads):
    payload = {
        "sample_idx": np.array([sample_idx], dtype=np.int32),
        "scene_indice": np.array([sample_info.get("scene_indice", "")]),
        "ros_msg_seq_curr": np.array([sample_info.get("ros_msg_seq_curr", "")]),
    }
    for ch, camera_payload in camera_payloads.items():
        prefix = ch
        for key, value in camera_payload.items():
            payload[f"{prefix}__{key}"] = value
    np.savez_compressed(output_path, **payload)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
    test_infos = load_test_infos(args.test_info_path)
    torch, generator = build_generator(args)
    if args.sample_idx >= 0:
        args.start_idx = args.sample_idx
        args.end_idx = args.sample_idx + 1
    end_idx = len(test_infos) if args.end_idx < 0 else min(args.end_idx, len(test_infos))
    camera_channels = list(args.camera_channels) if args.camera_channels else list(CAMERA_CHANNELS)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device.startswith("cuda")
        else nullcontext()
    )

    with torch.inference_mode():
        for sample_idx in tqdm(range(args.start_idx, end_idx)):
            sample_info = test_infos[sample_idx]
            out_path = Path(args.output_dir) / f"pred_{sample_idx}.npz"
            if out_path.exists() and not args.overwrite:
                if not args.vis_dir:
                    continue

            camera_payloads = {}
            camera_visuals = {}
            for ch in camera_channels:
                image_path = sample_info["curr_camera_data"][ch]["filename"]
                rgb = load_rgb_image(image_path)
                with autocast_ctx:
                    mask_dicts = generator.generate(rgb)
                filtered = filter_and_deduplicate_masks(mask_dicts, rgb.shape[:2])
                camera_payloads[ch] = encode_camera_masks(filtered, rgb.shape[:2])
                if args.vis_dir:
                    camera_visuals[ch] = overlay_camera_masks(rgb, filtered, ch)

            if not out_path.exists() or args.overwrite:
                save_sample_cache(str(out_path), sample_idx, sample_info, camera_payloads)
            if args.vis_dir:
                if len(camera_visuals) == 1:
                    only_camera = next(iter(camera_visuals.keys()))
                    vis_path = Path(args.vis_dir) / f"sam2_masks_{sample_idx}_{only_camera}.jpg"
                else:
                    vis_path = Path(args.vis_dir) / f"sam2_masks_{sample_idx}.jpg"
                save_sample_visualization(vis_path, sample_idx, sample_info, camera_visuals)


if __name__ == "__main__":
    main()
