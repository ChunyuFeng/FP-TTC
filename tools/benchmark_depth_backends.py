import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import argparse
import datetime
import importlib.util
import json
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.utils.augmentor import NuscRangeImageAugmentor
from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from utils.nusc_paths import resolve_nusc_path


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


CAMERA_CHANNELS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


def xformers_available() -> bool:
    return importlib.util.find_spec("xformers") is not None


def xformers_disabled_by_env() -> bool:
    value = os.environ.get("DEPTHANYTHING_DISABLE_XFORMERS", "")
    return value.lower() in {"1", "true", "yes", "on"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_info_path",
        default="./Datasets/nuscenes/2_trainval_test_infos/test/nusc_test_infos_key_frames_160_1920_fov_8_15.pkl",
        type=str,
    )
    parser.add_argument("--sample_index", default=0, type=int)
    parser.add_argument("--image_size", default=[160, 320], type=int, nargs=2)
    parser.add_argument("--depth_input_size", default=160, type=int)
    parser.add_argument("--depthanything_ckpt_dir", default="", type=str)
    parser.add_argument("--benchmark_backend", default="pytorch", choices=["pytorch", "compile", "onnxruntime"])
    parser.add_argument("--compile_mode", default="reduce-overhead", choices=["reduce-overhead", "max-autotune"])
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--onnx_path", default="", type=str)
    parser.add_argument("--num_warmup", default=20, type=int)
    parser.add_argument("--num_iters", default=100, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--amp_dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--return_torch", action="store_true")
    parser.add_argument("--output_dir", default="test", type=str)
    return parser.parse_args()


def parse_amp_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {name}")


def autocast_context(enabled: bool, dtype_name: str):
    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=parse_amp_dtype(dtype_name))
    return torch.cuda.amp.autocast(dtype=parse_amp_dtype(dtype_name))


def resolve_depthanything_ckpt_dir(requested_dir: str) -> Path:
    candidates = []
    if requested_dir:
        candidates.append(Path(requested_dir))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "pretrained" / "depthanything")
    candidates.append(repo_root.parent / "FP-TTC-hardproj-150scene-v1" / "pretrained" / "depthanything")

    for candidate in candidates:
        ckpt_path = candidate / "depth_anything_v2_metric_vkitti_vits.pth"
        if ckpt_path.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find depth_anything_v2_metric_vkitti_vits.pth. "
        "Pass --depthanything_ckpt_dir or provide the pretrained weights."
    )


def load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def build_fixed_nusc_fast_preprocess(
    crop_size: Tuple[int, int],
    example_image_size: Tuple[int, int],
) -> Dict[str, Tuple[int, int]]:
    augmentor = NuscRangeImageAugmentor(crop_size=crop_size)
    params = augmentor.sample_params(example_image_size)
    if params["flip_h"] or params["flip_v"] or params["rotate"]:
        raise ValueError("Expected flip and rotate to be disabled for fixed preprocess.")
    resize_w, resize_h = params["resize"]
    crop_x, crop_y = params["crop"]
    return {
        "resize_hw": (resize_h, resize_w),
        "crop_yx": (crop_y, crop_x),
        "crop_hw": tuple(crop_size),
    }


def fixed_resize_crop_rgb_image(
    image_rgb: np.ndarray,
    resize_hw: Tuple[int, int],
    crop_yx: Tuple[int, int],
    crop_hw: Tuple[int, int],
) -> np.ndarray:
    resize_h, resize_w = resize_hw
    resized = cv2.resize(image_rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    crop_y, crop_x = crop_yx
    crop_h, crop_w = crop_hw
    return resized[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy()


def load_nusc_rgb_image(entry: dict, frame_type: str, camera_channel: str) -> np.ndarray:
    image_rel = entry[f"{frame_type}_camera_data"][camera_channel]["filename"]
    image_path = resolve_nusc_path(image_rel)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_rgb_pair_tensor(entry: dict, image_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    first_path = resolve_nusc_path(entry["prev_camera_data"][CAMERA_CHANNELS[0]]["filename"])
    with Image.open(first_path) as sample_image:
        preprocess = build_fixed_nusc_fast_preprocess(image_size, sample_image.size)

    ordered_images = []
    for frame_type in ("prev", "curr"):
        for camera_channel in CAMERA_CHANNELS:
            image_rgb = load_nusc_rgb_image(entry, frame_type, camera_channel)
            image_rgb = fixed_resize_crop_rgb_image(
                image_rgb,
                preprocess["resize_hw"],
                preprocess["crop_yx"],
                preprocess["crop_hw"],
            )
            ordered_images.append(image_rgb)

    batch_np = np.stack(ordered_images, axis=0)
    batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().float()
    return batch.to(device, non_blocking=torch.cuda.is_available())


def build_depth_model(device: torch.device, ckpt_dir: Path) -> DepthAnythingV2:
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    }
    depth_model = DepthAnythingV2(**{**model_configs["vits"], "max_depth": 80})
    checkpoint = torch.load(ckpt_dir / "depth_anything_v2_metric_vkitti_vits.pth", map_location="cpu")
    depth_model.load_state_dict(checkpoint)
    depth_model.to(device).eval()
    return depth_model


def maybe_compile_depth_model(model: DepthAnythingV2, enabled: bool, mode: str):
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this environment.")
    return torch.compile(model, mode=mode)


def build_onnx_path(args, batch_shape: Tuple[int, int, int, int]) -> Path:
    if args.onnx_path:
        return Path(args.onnx_path)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_tag = "x".join(str(v) for v in batch_shape)
    return output_root / f"depthanything_vits_{batch_tag}.onnx"


def export_depthanything_onnx(model: DepthAnythingV2, onnx_path: Path, batch_shape: Tuple[int, int, int, int]):
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(batch_shape, device=next(model.parameters()).device, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["images"],
            output_names=["depth"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )


class ORTDepthRunner:
    def __init__(self, onnx_path: Path, batch_shape: Tuple[int, int, int, int], device: torch.device):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.batch_shape = tuple(batch_shape)
        self.device = device
        self.providers = list(self.session.get_providers())
        self.use_cuda_iobinding = (
            device.type == "cuda"
            and "CUDAExecutionProvider" in self.session.get_providers()
        )

    def infer(self, processed_batch: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_iobinding:
            output = torch.empty(
                (self.batch_shape[0], self.batch_shape[2], self.batch_shape[3]),
                device=processed_batch.device,
                dtype=torch.float32,
            )
            io_binding = self.session.io_binding()
            io_binding.bind_input(
                name=self.input_name,
                device_type="cuda",
                device_id=processed_batch.device.index or 0,
                element_type=np.float32,
                shape=tuple(processed_batch.shape),
                buffer_ptr=processed_batch.data_ptr(),
            )
            io_binding.bind_output(
                name=self.output_name,
                device_type="cuda",
                device_id=processed_batch.device.index or 0,
                element_type=np.float32,
                shape=tuple(output.shape),
                buffer_ptr=output.data_ptr(),
            )
            self.session.run_with_iobinding(io_binding)
            return output

        outputs = self.session.run([self.output_name], {self.input_name: processed_batch.detach().cpu().numpy()})[0]
        return torch.from_numpy(outputs).to(self.device, non_blocking=False)


def run_depth_stage_pytorch(
    depth_model: DepthAnythingV2,
    rgb_batch: torch.Tensor,
    input_size: int,
    amp_dtype_name: str,
    enable_amp: bool,
    return_torch: bool,
) -> List[torch.Tensor]:
    with torch.no_grad():
        with autocast_context(rgb_batch.is_cuda and enable_amp, amp_dtype_name):
            return depth_model.infer_tensor_batch_rgb(
                rgb_batch,
                input_size=input_size,
                return_torch=return_torch,
                raw_is_rgb=True,
            )


def run_depth_stage_onnxruntime(
    depth_model: DepthAnythingV2,
    ort_runner: ORTDepthRunner,
    rgb_batch: torch.Tensor,
    input_size: int,
    return_torch: bool,
) -> List[torch.Tensor]:
    processed_batch, orig_size, proc_size = depth_model.preprocess_tensor_batch_rgb(
        rgb_batch,
        input_size=input_size,
        raw_is_rgb=True,
    )
    with torch.no_grad():
        depth = ort_runner.infer(processed_batch)
    return depth_model.postprocess_depth_batch(depth, orig_size, proc_size, return_torch=return_torch)


def benchmark_backend(
    backend: str,
    depth_model: DepthAnythingV2,
    rgb_batch: torch.Tensor,
    input_size: int,
    amp_dtype_name: str,
    enable_amp: bool,
    return_torch: bool,
    num_warmup: int,
    num_iters: int,
    ort_runner: ORTDepthRunner = None,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    if backend == "onnxruntime" and ort_runner is None:
        raise ValueError("ORT runner is required for onnxruntime benchmark.")

    if backend == "onnxruntime":
        runner = lambda: run_depth_stage_onnxruntime(depth_model, ort_runner, rgb_batch, input_size, return_torch)
    else:
        runner = lambda: run_depth_stage_pytorch(
            depth_model,
            rgb_batch,
            input_size,
            amp_dtype_name,
            enable_amp,
            return_torch,
        )

    for _ in range(num_warmup):
        runner()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    records_ms = []
    output = None
    for _ in range(num_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = runner()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        records_ms.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(records_ms, dtype=np.float64)
    metrics = {
        "avg_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "num_iters": int(num_iters),
        "num_warmup": int(num_warmup),
    }
    return output, metrics


def compare_outputs(reference: List[torch.Tensor], candidate: List[torch.Tensor]) -> Dict[str, float]:
    if len(reference) != len(candidate):
        raise ValueError("Output lengths differ.")
    max_abs = 0.0
    mean_abs_values = []
    for ref_item, cand_item in zip(reference, candidate):
        ref_tensor = ref_item if torch.is_tensor(ref_item) else torch.from_numpy(ref_item)
        cand_tensor = cand_item if torch.is_tensor(cand_item) else torch.from_numpy(cand_item)
        ref_tensor = ref_tensor.detach().float().cpu()
        cand_tensor = cand_tensor.detach().float().cpu()
        diff = (ref_tensor - cand_tensor).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        mean_abs_values.append(float(diff.mean().item()))
    return {
        "max_abs_diff": float(max_abs),
        "mean_abs_diff": float(np.mean(mean_abs_values)),
    }


def build_output_dir(output_root: str, backend: str) -> Path:
    timestamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S_%f")
    path = Path(output_root) / f"{timestamp}_depth_{backend}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_summary(output_dir: Path, payload: Dict):
    json_path = output_dir / "depth_backend_benchmark.json"
    md_path = output_dir / "depth_backend_benchmark.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    lines = [
        "# Depth Backend Benchmark",
        "",
        f"- Backend: {payload['backend']}",
        f"- Device: {payload['device']}",
        f"- xformers available: {payload['xformers_available']}",
        f"- xformers enabled: {payload['xformers_enabled']}",
        f"- AMP enabled: {payload['amp_enabled']} ({payload['amp_dtype']})",
        f"- Sample index: {payload['sample_index']}",
        f"- RGB batch shape: `{payload['rgb_batch_shape']}`",
        f"- Processed batch shape: `{payload['processed_batch_shape']}`",
        f"- ONNX path: `{payload.get('onnx_path', '')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| avg_ms | {payload['timing']['avg_ms']:.2f} |",
        f"| p50_ms | {payload['timing']['p50_ms']:.2f} |",
        f"| p95_ms | {payload['timing']['p95_ms']:.2f} |",
        f"| min_ms | {payload['timing']['min_ms']:.2f} |",
        f"| max_ms | {payload['timing']['max_ms']:.2f} |",
    ]
    if payload.get("output_diff"):
        lines.extend([
            "",
            "| Diff vs eager | Value |",
            "| --- | ---: |",
            f"| max_abs_diff | {payload['output_diff']['max_abs_diff']:.6f} |",
            f"| mean_abs_diff | {payload['output_diff']['mean_abs_diff']:.6f} |",
        ])
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt_dir = resolve_depthanything_ckpt_dir(args.depthanything_ckpt_dir)
    test_entries = load_pickle(args.test_info_path)
    entry = test_entries[args.sample_index]
    image_size = tuple(args.image_size)

    rgb_batch = load_rgb_pair_tensor(entry, image_size=image_size, device=device)
    depth_model = build_depth_model(device, ckpt_dir)

    processed_batch, _, _ = depth_model.preprocess_tensor_batch_rgb(
        rgb_batch,
        input_size=args.depth_input_size,
        raw_is_rgb=True,
    )

    eager_output, _ = benchmark_backend(
        backend="pytorch",
        depth_model=depth_model,
        rgb_batch=rgb_batch,
        input_size=args.depth_input_size,
        amp_dtype_name=args.amp_dtype,
        enable_amp=not args.disable_amp,
        return_torch=True,
        num_warmup=max(2, min(5, args.num_warmup)),
        num_iters=3,
    )

    ort_runner = None
    benchmark_model = depth_model
    onnx_path = None
    if args.benchmark_backend == "compile":
        benchmark_model = maybe_compile_depth_model(depth_model, enabled=True, mode=args.compile_mode)
    elif args.benchmark_backend == "onnxruntime":
        onnx_path = build_onnx_path(args, tuple(processed_batch.shape))
        if args.export_onnx or not onnx_path.exists():
            export_depthanything_onnx(depth_model, onnx_path, tuple(processed_batch.shape))
        ort_runner = ORTDepthRunner(onnx_path, tuple(processed_batch.shape), device)

    output, timing = benchmark_backend(
        backend=args.benchmark_backend,
        depth_model=benchmark_model,
        rgb_batch=rgb_batch,
        input_size=args.depth_input_size,
        amp_dtype_name=args.amp_dtype,
        enable_amp=not args.disable_amp,
        return_torch=args.return_torch or args.benchmark_backend != "onnxruntime",
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        ort_runner=ort_runner,
    )

    payload = {
        "backend": args.benchmark_backend,
        "device": str(device),
        "xformers_available": xformers_available(),
        "xformers_enabled": xformers_available() and not xformers_disabled_by_env(),
        "amp_enabled": not args.disable_amp,
        "amp_dtype": args.amp_dtype,
        "sample_index": args.sample_index,
        "rgb_batch_shape": list(rgb_batch.shape),
        "processed_batch_shape": list(processed_batch.shape),
        "timing": timing,
        "output_diff": compare_outputs(eager_output, output),
    }
    if onnx_path is not None:
        payload["onnx_path"] = str(onnx_path)
    if ort_runner is not None:
        payload["ort_providers"] = ort_runner.providers

    output_dir = build_output_dir(args.output_dir, args.benchmark_backend)
    write_summary(output_dir, payload)
    print(json.dumps(payload, indent=2))
    print(f"[depth-bench] output_dir={output_dir}")


if __name__ == "__main__":
    main()
