#!/usr/bin/env python3
"""Render side-by-side GT/Pred comparison panels for each frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cyberrock.real_vehicle_annotation_common import ensure_dir, load_csv_rows
from utils.draw import make_risk_analysis_rgb, make_scale_analysis_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GT/Pred comparison images.")
    parser.add_argument("--comparison-csv", required=True, type=str)
    parser.add_argument("--warning-png", default="tools/collision_prediction/warning.png", type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--scene", nargs="*", default=None)
    parser.add_argument("--sequence-dir", nargs="*", default=None)
    parser.add_argument("--frame-case", nargs="*", default=None)
    parser.add_argument("--only-with-point", action="store_true")
    parser.add_argument("--max-frames", default=0, type=int)
    parser.add_argument("--column-width", default=1400, type=int)
    parser.add_argument("--rgb-icon-size", default=38, type=int)
    parser.add_argument("--rv-icon-size", default=38, type=int)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def rgb_from_float(image: np.ndarray) -> Image.Image:
    array = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def resize_keep_aspect(image: Image.Image, target_width: int) -> Image.Image:
    scale = float(target_width) / float(image.width)
    target_height = max(1, int(round(image.height * scale)))
    return image.resize((target_width, target_height), Image.Resampling.BILINEAR)


def paste_warning_markers(
    base: Image.Image,
    points: list[dict[str, object]],
    x_key: str,
    y_key: str,
    warning_icon: Image.Image,
) -> Image.Image:
    canvas = base.convert("RGBA")
    for point in points:
        x_val = point.get(x_key)
        y_val = point.get(y_key)
        if x_val is None or y_val is None:
            continue
        x = int(round(float(x_val)))
        y = int(round(float(y_val)))
        canvas.alpha_composite(warning_icon, (x - warning_icon.width // 2, y - warning_icon.height // 2))
    return canvas.convert("RGB")


def parse_points(cell: str) -> list[dict[str, object]]:
    if not cell:
        return []
    return list(json.loads(cell))


def build_proj_to_stitched_lookup(pred_payload: dict[str, object], stitched_size: tuple[int, int]) -> dict[str, np.ndarray]:
    proj = np.asarray(pred_payload["proj_pix_curr_fullres"], dtype=np.float32)
    cam_idx = proj[:, :, 0].astype(np.int32)
    u_proc = proj[:, :, 1].astype(np.float32)
    v_proc = proj[:, :, 2].astype(np.float32)
    valid = (cam_idx >= 0) & (cam_idx < 6)

    stitched_w, stitched_h = stitched_size
    cam_w = float(stitched_w) / 6.0
    proc_w = max(int(np.nanmax(u_proc)) + 1, 1)
    proc_h = max(int(np.nanmax(v_proc)) + 1, 1)

    x_orig = (u_proc + 0.5) * cam_w / float(proc_w) - 0.5
    y_orig = (v_proc + 0.5) * float(stitched_h) / float(proc_h) - 0.5
    x_stitched = cam_idx.astype(np.float32) * cam_w + x_orig
    return {
        "valid": valid,
        "x_stitched": x_stitched,
        "y_stitched": y_orig,
    }


def fallback_rv_point_from_stitched(
    point: dict[str, object],
    proj_lookup: dict[str, np.ndarray],
) -> tuple[float | None, float | None]:
    px = point.get("x_stitched_orig")
    py = point.get("y_stitched_orig")
    if px is None or py is None:
        return None, None
    valid = proj_lookup["valid"]
    if not np.any(valid):
        return None, None
    ys, xs = np.where(valid)
    d2 = (proj_lookup["x_stitched"][ys, xs] - float(px)) ** 2 + (
        proj_lookup["y_stitched"][ys, xs] - float(py)
    ) ** 2
    best = int(np.argmin(d2))
    return float(xs[best]), float(ys[best])


def build_panel_triplet(
    stitched_image_path: Path,
    pred_npy_path: Path,
    points: list[dict[str, object]],
    column_width: int,
    warning_rgb: Image.Image,
    warning_rv: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    stitched = Image.open(stitched_image_path).convert("RGB")
    pred_payload = np.load(pred_npy_path, allow_pickle=True).item()
    proj_lookup = build_proj_to_stitched_lookup(pred_payload, stitched.size)

    scale_panel = rgb_from_float(make_scale_analysis_rgb(pred_payload["scale_pred"]))
    orientation_panel = rgb_from_float(make_risk_analysis_rgb(pred_payload["risk_pred"]))

    stitched_disp = resize_keep_aspect(stitched, column_width)
    scale_disp = resize_keep_aspect(scale_panel, column_width)
    orientation_disp = resize_keep_aspect(orientation_panel, column_width)

    rgb_points = []
    rv_points = []
    stitched_x_scale = float(stitched_disp.width) / float(stitched.width)
    stitched_y_scale = float(stitched_disp.height) / float(stitched.height)
    rv_w = int(pred_payload["scale_pred"].shape[1])
    rv_h = int(pred_payload["scale_pred"].shape[0])
    rv_x_scale = float(scale_disp.width) / float(rv_w)
    rv_y_scale = float(scale_disp.height) / float(rv_h)
    for point in points:
        px = point.get("x_stitched_orig")
        py = point.get("y_stitched_orig")
        if px is not None and py is not None:
            rgb_points.append(
                {
                    "x": float(px) * stitched_x_scale,
                    "y": float(py) * stitched_y_scale,
                }
            )
        rx = point.get("x_rv")
        ry = point.get("y_rv")
        if (rx is None or ry is None) and px is not None and py is not None:
            rx, ry = fallback_rv_point_from_stitched(point, proj_lookup)
        if rx is not None and ry is not None:
            rv_points.append(
                {
                    "x": float(rx) * rv_x_scale,
                    "y": float(ry) * rv_y_scale,
                }
            )

    stitched_disp = paste_warning_markers(stitched_disp, rgb_points, "x", "y", warning_rgb)
    scale_disp = paste_warning_markers(scale_disp, rv_points, "x", "y", warning_rv)
    orientation_disp = paste_warning_markers(orientation_disp, rv_points, "x", "y", warning_rv)
    return stitched_disp, scale_disp, orientation_disp


def add_caption(image: Image.Image, text: str, font: ImageFont.ImageFont, pad_top: int = 12, pad_bottom: int = 10) -> Image.Image:
    dummy = Image.new("RGB", (image.width, image.height), "white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_h = bbox[3] - bbox[1]
    out = Image.new("RGB", (image.width, image.height + pad_top + text_h + pad_bottom), "white")
    out.paste(image, (0, pad_top + text_h + pad_bottom))
    draw = ImageDraw.Draw(out)
    draw.text((14, pad_top), text, fill=(20, 20, 20), font=font)
    return out


def stack_vertical(images: list[Image.Image], gap: int, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), bg)
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    return canvas


def join_columns(left: Image.Image, right: Image.Image, title: str, subtitle: str) -> Image.Image:
    title_font = load_font(34)
    subtitle_font = load_font(22)
    gap = 36
    header_h = 110
    canvas_w = left.width + right.width + gap * 3
    canvas_h = max(left.height, right.height) + header_h + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 18), title, fill=(15, 15, 15), font=title_font)
    draw.text((gap, 62), subtitle, fill=(60, 60, 60), font=subtitle_font)
    canvas.paste(left, (gap, header_h))
    canvas.paste(right, (gap * 2 + left.width, header_h))
    return canvas


def main() -> None:
    args = parse_args()
    comparison_csv = Path(args.comparison_csv).resolve()
    out_dir = ensure_dir(Path(args.out_dir).resolve())
    warning_png = Path(args.warning_png).resolve()

    rows = load_csv_rows(comparison_csv)
    selected_scenes = set(args.scene or [])
    selected_sequences = set(args.sequence_dir or [])
    selected_cases = set(args.frame_case or [])

    filtered = []
    for row in rows:
        if selected_scenes and row["scene"] not in selected_scenes:
            continue
        if selected_sequences and row["sequence_dir_name"] not in selected_sequences:
            continue
        if selected_cases and row["frame_case"] not in selected_cases:
            continue
        if args.only_with_point and not (row["pred_has_point"] == "1" or row["gt_has_point"] == "1"):
            continue
        filtered.append(row)

    if args.max_frames > 0:
        filtered = filtered[: args.max_frames]

    warning_icon = Image.open(warning_png).convert("RGBA")
    warning_rgb = warning_icon.resize((args.rgb_icon_size, args.rgb_icon_size), Image.Resampling.LANCZOS)
    warning_rv = warning_icon.resize((args.rv_icon_size, args.rv_icon_size), Image.Resampling.LANCZOS)
    label_font = load_font(24)

    for row in filtered:
        pred_points = parse_points(row["pred_points_json"])
        gt_points = parse_points(row["gt_points_json"])
        stitched_path = Path(row["stitched_image_path"])
        pred_npy_path = Path(row["pred_npy_path"])

        pred_rgb, pred_scale, pred_orientation = build_panel_triplet(
            stitched_path,
            pred_npy_path,
            pred_points,
            args.column_width,
            warning_rgb,
            warning_rv,
        )
        gt_rgb, gt_scale, gt_orientation = build_panel_triplet(
            stitched_path,
            pred_npy_path,
            gt_points,
            args.column_width,
            warning_rgb,
            warning_rv,
        )

        left = stack_vertical(
            [
                add_caption(pred_rgb, "Prediction / Stitched RGB", label_font),
                add_caption(pred_scale, "Prediction / Scale", label_font),
                add_caption(pred_orientation, "Prediction / Orientation", label_font),
            ],
            gap=18,
        )
        right = stack_vertical(
            [
                add_caption(gt_rgb, "Ground Truth / Stitched RGB", label_font),
                add_caption(gt_scale, "Ground Truth / Scale", label_font),
                add_caption(gt_orientation, "Ground Truth / Orientation", label_font),
            ],
            gap=18,
        )

        title = f"{row['scene']} | {row['sequence_dir_name']} | group {row['group_id']} | {row['frame_case']}"
        subtitle = (
            f"pred_points={row['pred_point_count']} | gt_points={row['gt_point_count']} | "
            f"ros_seq={row['ros_msg_seq_curr']}"
        )
        canvas = join_columns(left, right, title, subtitle)
        out_name = f"{row['scene']}__{row['sequence_dir_name']}__g{int(row['group_id']):04d}.jpg"
        canvas.save(out_dir / out_name, quality=95)

    print(f"[ok] rendered {len(filtered)} frames to {out_dir}")


if __name__ == "__main__":
    main()
