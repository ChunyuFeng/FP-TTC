import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import sys
import cv2
cv2.setNumThreads(1)

import json
import inspect
import time
import pickle
import argparse
import datetime
import shutil
import subprocess
from copy import deepcopy
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from fpttc.fp_ttc import FpTTC
from utils.draw import (
    make_risk_analysis_rgb,
    make_scale_analysis_rgb,
)
from dataloader.utils.augmentor import NuscRangeImageAugmentor
from dataloader.dataset import (
    build_camera_mapping_fast,
    build_frame_sensor_metas,
    finalize_frame_mapping_fast,
    tensorize_sensor_metas,
)
from utils.dtype_audit import (
    dump_dtype_audit,
    log_dtype_event,
    reset_dtype_audit,
    set_dtype_audit_enabled,
)
from utils.nusc_paths import resolve_nusc_path

try:
    from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
except ModuleNotFoundError:
    sibling_repo = Path(__file__).resolve().parents[1] / "FP-TTC-hardproj-150scene-v1"
    if sibling_repo.exists():
        for module_name in list(sys.modules.keys()):
            if module_name == "depthanything" or module_name.startswith("depthanything."):
                del sys.modules[module_name]
        sys.path.insert(0, str(sibling_repo))
        from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    else:
        raise


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def parse_amp_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def autocast_context(enabled: bool, dtype_name: str):
    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.amp.autocast(dtype=parse_amp_dtype(dtype_name))


parser = argparse.ArgumentParser()

parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--strict_resume", action="store_true")
parser.add_argument("--image_size", default=[160, 320], type=int, nargs="+")
parser.add_argument("--padding_factor", default=32, type=int)
parser.add_argument("--num_scales", default=2, type=int)
parser.add_argument("--feature_channels", default=128, type=int)
parser.add_argument("--upsample_factor", default=4, type=int)
parser.add_argument("--num_head", default=1, type=int)
parser.add_argument("--ffn_dim_expansion", default=4, type=int)
parser.add_argument("--num_transformer_layers", default=6, type=int)
parser.add_argument("--reg_refine", action="store_true")
parser.add_argument("--rvt_depth_guided_sampling", action="store_true")
parser.add_argument("--attn_type", default="swin", type=str)
parser.add_argument("--attn_splits_list", default=[2, 8], type=int, nargs="+")
parser.add_argument("--corr_radius_list", default=[-1, 4], type=int, nargs="+")
parser.add_argument("--prop_radius_list", default=[-1, 1], type=int, nargs="+")
parser.add_argument("--num_reg_refine", default=1, type=int)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--count_time", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--sjtu_test", action="store_true")

parser.add_argument("--scale_only", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--test_profile", default="custom", choices=["custom", "fastest", "accurate"])
parser.add_argument("--save_visualizations", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--compute_scale_metrics", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument(
    "--compute_scale_orientation_metrics",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
)
parser.add_argument("--compute_table3_metrics", dest="compute_scale_orientation_metrics", type=str2bool, nargs="?", const=True, help=argparse.SUPPRESS)
parser.add_argument("--compute_scene_mid_err", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--max_test_samples", default=-1, type=int)
parser.add_argument("--output_dir", default=None, type=str)

parser.add_argument("--save_pred_npy", action="store_true")
parser.add_argument("--pred_npy_dir", default="./Datasets/nuscenes/3_visualization/collision_pred", type=str)
parser.add_argument(
    "--test_info_path",
    default="./Datasets/nuscenes/2_trainval_test_infos/test/nusc_test_infos_key_frames_160_1920_fov_8_15.pkl",
    type=str,
)
parser.add_argument(
    "--depthanything_ckpt_dir",
    default="",
    type=str,
    help="Directory containing depth_anything_v2_metric_vkitti_vits.pth",
)

parser.add_argument("--num_io_workers", default=12, type=int)
parser.add_argument("--enable_nusc_io_prefetch", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--enable_fast_nusc_preprocess", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--depth_input_size", default=320, type=int)
parser.add_argument("--num_depth_preprocess_workers", default=1, type=int)
parser.add_argument("--depth_backend", default="pytorch", choices=["pytorch", "onnxruntime", "tensorrt"])
parser.add_argument("--depth_trt_engine", default="", type=str)
parser.add_argument("--depth_onnx_path", default="", type=str)
parser.add_argument("--compile_depth", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--compile_model_submodules", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--compile_mode", default="reduce-overhead", choices=["reduce-overhead", "max-autotune"])
parser.add_argument("--enable_cuda_graphs", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--cuda_graph_warmup_iters", default=3, type=int)
parser.add_argument("--cache_geom_constants", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--dump_dtype_audit", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--range_size", default=["40", "480"], type=str, nargs="+")
parser.add_argument("--mapping_pixel_stride", default=1, type=int)
parser.add_argument("--mapping_range_backend", default="auto", choices=["auto", "gpu", "cpu"])
parser.add_argument("--enable_depth_amp", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--depth_amp_dtype", default="fp16", choices=["fp16", "bf16"])
parser.add_argument("--enable_model_amp", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--model_amp_dtype", default="bf16", choices=["fp16", "bf16"])
parser.add_argument("--empty_cache_before_model", type=str2bool, nargs="?", const=True, default=False)

args = parser.parse_args()


TEST_PROFILE_OVERRIDES = {
    "fastest": {
        "depth_input_size": 160,
        "enable_nusc_io_prefetch": True,
        "enable_fast_nusc_preprocess": True,
        "mapping_range_backend": "auto",
        "enable_depth_amp": True,
        "depth_amp_dtype": "fp16",
        "enable_model_amp": True,
        "model_amp_dtype": "bf16",
        "num_depth_preprocess_workers": 12,
    },
    "accurate": {
        "depth_input_size": 320,
        "enable_nusc_io_prefetch": True,
        "enable_fast_nusc_preprocess": False,
        "mapping_range_backend": "cpu",
        "enable_depth_amp": False,
        "enable_model_amp": False,
        "num_depth_preprocess_workers": 1,
        "empty_cache_before_model": True,
    },
}


TIMING_KEYS = (
    "io_ms",
    "augment_ms",
    "depth_ms",
    "mapping_ms",
    "model_ms",
    "vis_ms",
    "metrics_ms",
)

MAPPING_DETAIL_KEYS = (
    "transform_setup_ms",
    "camera_backproject_ms",
    "range_project_ms",
    "hole_fill_ms",
)


@dataclass
class TestRuntime:
    output_dir: str
    split_name: str
    camera_channels: List[str]
    range_h: int
    range_w: int
    mapping_range_backend: str
    fixed_nusc_fast: Optional[Dict[str, object]] = None
    fixed_depth_proc_hw: Optional[Tuple[int, int]] = None
    depth_backend_summary: str = "pytorch"
    xformers_available: bool = False
    model_cuda_graph_runner: Any = None


@dataclass
class NuscInputBundle:
    raw_prev: Dict[str, object]
    raw_curr: Dict[str, object]
    proc_prev: Dict[str, np.ndarray]
    proc_curr: Dict[str, np.ndarray]
    prev_batch: torch.Tensor
    curr_batch: torch.Tensor
    affine_matrix: np.ndarray
    rgb_12: Optional[torch.Tensor] = None


def torch_compile_available() -> bool:
    return hasattr(torch, "compile")


def xformers_available() -> bool:
    try:
        import xformers  # noqa: F401
    except Exception:
        return False
    return True


def maybe_compile_module(module: torch.nn.Module, enabled: bool, mode: str, label: str) -> torch.nn.Module:
    if not enabled:
        return module
    if not torch_compile_available():
        print(f"[Accel] torch.compile is unavailable in torch {torch.__version__}; skipping {label}.")
        return module
    try:
        compiled = torch.compile(module, mode=mode)
        print(f"[Accel] Enabled torch.compile for {label} with mode={mode}.")
        return compiled
    except Exception as exc:
        print(f"[Accel] Failed to compile {label}: {type(exc).__name__}: {exc}")
        return module


def maybe_compile_model_submodules(model: torch.nn.Module, enabled: bool, mode: str):
    if not enabled:
        return
    compile_targets = [
        "cnet",
        "conv_corr",
        "conv_corr_risk",
        "corr_residual_decoder",
    ]
    for attr_name in compile_targets:
        module = getattr(model, attr_name, None)
        if isinstance(module, torch.nn.Module):
            setattr(model, attr_name, maybe_compile_module(module, True, mode, f"model.{attr_name}"))


def tensor_tree_empty_like(obj):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return torch.empty_like(obj)
    if isinstance(obj, dict):
        return {key: tensor_tree_empty_like(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [tensor_tree_empty_like(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(tensor_tree_empty_like(value) for value in obj)
    return obj


def tensor_tree_copy_(dst, src):
    if dst is None or src is None:
        return
    if torch.is_tensor(dst):
        dst.copy_(src)
        return
    if isinstance(dst, dict):
        for key in dst:
            tensor_tree_copy_(dst[key], src[key])
        return
    if isinstance(dst, list):
        for dst_item, src_item in zip(dst, src):
            tensor_tree_copy_(dst_item, src_item)
        return
    if isinstance(dst, tuple):
        for dst_item, src_item in zip(dst, src):
            tensor_tree_copy_(dst_item, src_item)
        return
    return


def infer_shape_signature(*items):
    signature = []
    for item in items:
        signature.append(_shape_signature_for_item(item))
    return tuple(signature)


def _shape_signature_for_item(item):
    if item is None:
        return None
    if torch.is_tensor(item):
        return (tuple(item.shape), str(item.dtype), str(item.device))
    if isinstance(item, dict):
        return tuple((key, _shape_signature_for_item(value)) for key, value in sorted(item.items()))
    if isinstance(item, list):
        return tuple(_shape_signature_for_item(value) for value in item)
    if isinstance(item, tuple):
        return tuple(_shape_signature_for_item(value) for value in item)
    return type(item).__name__


class DepthCudaGraphRunner:
    def __init__(self, model: DepthAnythingV2, amp_enabled: bool, amp_dtype_name: str, warmup_iters: int):
        self.model = model
        self.amp_enabled = amp_enabled
        self.amp_dtype_name = amp_dtype_name
        self.warmup_iters = max(1, int(warmup_iters))
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.shape_signature = None
        self.disabled = False

    def run(self, processed_batch: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            with torch.inference_mode():
                with autocast_context(self.amp_enabled, self.amp_dtype_name):
                    return self.model(processed_batch)
        signature = infer_shape_signature(processed_batch)
        if self.graph is None or self.shape_signature != signature:
            self._capture(processed_batch)
        self.static_input.copy_(processed_batch)
        self.graph.replay()
        return self.static_output

    def _capture(self, processed_batch: torch.Tensor):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require CUDA.")
        try:
            self.shape_signature = infer_shape_signature(processed_batch)
            self.static_input = torch.empty_like(processed_batch)
            self.static_output = None

            with torch.inference_mode():
                for _ in range(self.warmup_iters):
                    with autocast_context(self.amp_enabled, self.amp_dtype_name):
                        _ = self.model(self.static_input.copy_(processed_batch))
                sync_cuda_if_available()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    with autocast_context(self.amp_enabled, self.amp_dtype_name):
                        self.static_output = self.model(self.static_input)
        except Exception as exc:
            self.disabled = True
            self.graph = None
            self.static_input = None
            self.static_output = None
            print(f"[Accel] Depth CUDA graph capture failed; falling back to eager: {type(exc).__name__}: {exc}")


class ModelCudaGraphRunner:
    def __init__(self, model: torch.nn.Module, amp_enabled: bool, amp_dtype_name: str, warmup_iters: int):
        self.model = model
        self.amp_enabled = amp_enabled
        self.amp_dtype_name = amp_dtype_name
        self.warmup_iters = max(1, int(warmup_iters))
        self.graph = None
        self.static_kwargs = None
        self.static_outputs = None
        self.shape_signature = None
        self.disabled = False

    def run(self, **kwargs):
        if self.disabled:
            with torch.inference_mode():
                with autocast_context(self.amp_enabled, self.amp_dtype_name):
                    return self.model.forward(**kwargs)
        signature = infer_shape_signature(kwargs)
        if self.graph is None or self.shape_signature != signature:
            self._capture(kwargs)
        if self.disabled:
            with torch.inference_mode():
                with autocast_context(self.amp_enabled, self.amp_dtype_name):
                    return self.model.forward(**kwargs)
        tensor_tree_copy_(self.static_kwargs, kwargs)
        self.graph.replay()
        return self.static_outputs

    def _capture(self, kwargs: Dict[str, Any]):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require CUDA.")
        try:
            self.shape_signature = infer_shape_signature(kwargs)
            self.static_kwargs = tensor_tree_empty_like(kwargs)
            tensor_tree_copy_(self.static_kwargs, kwargs)

            with torch.inference_mode():
                for _ in range(self.warmup_iters):
                    with autocast_context(self.amp_enabled, self.amp_dtype_name):
                        _ = self.model.forward(**self.static_kwargs)
                sync_cuda_if_available()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    with autocast_context(self.amp_enabled, self.amp_dtype_name):
                        self.static_outputs = self.model.forward(**self.static_kwargs)
        except Exception as exc:
            self.disabled = True
            self.graph = None
            self.static_kwargs = None
            self.static_outputs = None
            print(f"[Accel] Model CUDA graph capture failed; falling back to eager: {type(exc).__name__}: {exc}")


class DepthPyTorchBackend:
    def __init__(
        self,
        model: DepthAnythingV2,
        *,
        input_size: int,
        enable_amp: bool,
        amp_dtype_name: str,
        compile_depth: bool,
        compile_mode: str,
        enable_cuda_graphs: bool,
        cuda_graph_warmup_iters: int,
    ):
        self.model = maybe_compile_module(model, compile_depth, compile_mode, "depth_model")
        self.input_size = input_size
        self.enable_amp = enable_amp
        self.amp_dtype_name = amp_dtype_name
        self.enable_cuda_graphs = enable_cuda_graphs
        self.cuda_graph_warmup_iters = cuda_graph_warmup_iters
        self.cuda_graph_runner = None
        if self.enable_cuda_graphs and torch.cuda.is_available():
            self.cuda_graph_runner = DepthCudaGraphRunner(
                self.model,
                amp_enabled=enable_amp,
                amp_dtype_name=amp_dtype_name,
                warmup_iters=cuda_graph_warmup_iters,
            )

    @property
    def summary(self) -> str:
        xformers_tag = "xformers" if xformers_available() else "no-xformers"
        compile_tag = f"compile={args.compile_mode}" if args.compile_depth and torch_compile_available() else "compile=off"
        graph_tag = "cudagraph=on" if self.cuda_graph_runner is not None else "cudagraph=off"
        return f"pytorch/{xformers_tag}/{compile_tag}/{graph_tag}"

    def infer_depth_pair(
        self,
        inputs: NuscInputBundle,
        camera_channels: List[str],
        num_preprocess_workers: int,
    ):
        if inputs.rgb_12 is not None:
            log_dtype_event(
                "depth_input.rgb_12",
                src_dtype=inputs.rgb_12.dtype,
                dst_dtype=inputs.rgb_12.dtype,
                shape=inputs.rgb_12.shape,
                device=inputs.rgb_12.device,
                copied=False,
            )
            processed_batch, orig_size, proc_size = self.model.preprocess_tensor_batch_rgb(
                inputs.rgb_12,
                input_size=self.input_size,
                raw_is_rgb=True,
            )
            log_dtype_event(
                "depth_preprocess.tensor_batch",
                src_dtype=inputs.rgb_12.dtype,
                dst_dtype=processed_batch.dtype,
                shape=processed_batch.shape,
                device=processed_batch.device,
                copied=True,
                note=f"orig={orig_size} proc={proc_size}",
            )
            with torch.inference_mode():
                if self.cuda_graph_runner is not None:
                    depth_batch = self.cuda_graph_runner.run(processed_batch)
                else:
                    with autocast_context(self.enable_amp, self.amp_dtype_name):
                        depth_batch = self.model(processed_batch)
            depth_list_12 = self.model.postprocess_depth_batch(
                depth_batch,
                orig_size=orig_size,
                proc_size=proc_size,
                return_torch=False,
            )
            log_dtype_event(
                "depth_output.tensor_batch",
                src_dtype=depth_batch.dtype,
                dst_dtype=depth_batch.dtype,
                shape=depth_batch.shape,
                device=depth_batch.device,
                copied=False,
            )
            return depth_list_12

        imgs_12 = [inputs.proc_prev[ch] for ch in camera_channels] + [inputs.proc_curr[ch] for ch in camera_channels]
        infer_images_kwargs = {"input_size": self.input_size}
        if "num_preprocess_workers" in inspect.signature(self.model.infer_images).parameters:
            infer_images_kwargs["num_preprocess_workers"] = num_preprocess_workers
        with torch.inference_mode():
            with autocast_context(self.enable_amp, self.amp_dtype_name):
                return self.model.infer_images(imgs_12, **infer_images_kwargs)


def export_depthanything_onnx(depth_model: DepthAnythingV2, onnx_path: str, batch_size: int, input_hw: Tuple[int, int]):
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(batch_size, 3, input_hw[0], input_hw[1], device=device, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            depth_model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["depth"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    return onnx_path


def build_trt_engine_from_onnx(onnx_path: str, engine_path: str, input_hw: Tuple[int, int], batch_size: int):
    trtexec_path = shutil.which("trtexec")
    if trtexec_path is None:
        raise FileNotFoundError("trtexec was not found on PATH; cannot build a TensorRT engine automatically.")
    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--skipInference",
        f"--minShapes=input:{batch_size}x3x{input_hw[0]}x{input_hw[1]}",
        f"--optShapes=input:{batch_size}x3x{input_hw[0]}x{input_hw[1]}",
        f"--maxShapes=input:{batch_size}x3x{input_hw[0]}x{input_hw[1]}",
    ]
    subprocess.run(command, check=True)


class DepthONNXRuntimeBackend:
    def __init__(
        self,
        *,
        depth_model: DepthAnythingV2,
        onnx_path: str,
        input_size: int,
        fixed_batch_size: int,
        fixed_input_hw: Tuple[int, int],
    ):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("Depth ONNX Runtime backend requires the onnxruntime package.") from exc

        if not onnx_path:
            raise ValueError("--depth_onnx_path must be provided or resolved before --depth_backend=onnxruntime.")

        self.ort = ort
        self.depth_model = depth_model
        self.onnx_path = Path(onnx_path)
        self.input_size = input_size
        self.fixed_batch_size = fixed_batch_size
        self.fixed_input_hw = fixed_input_hw

        if not self.onnx_path.exists():
            export_depthanything_onnx(
                depth_model=self.depth_model,
                onnx_path=str(self.onnx_path),
                batch_size=self.fixed_batch_size,
                input_hw=self.fixed_input_hw,
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = list(self.session.get_providers())
        self.use_cuda_iobinding = torch.cuda.is_available() and "CUDAExecutionProvider" in self.providers

    @property
    def summary(self) -> str:
        provider_tag = self.providers[0] if self.providers else "unknown"
        return f"onnxruntime/{provider_tag}/model={self.onnx_path.name}"

    def infer_depth_pair(
        self,
        inputs: NuscInputBundle,
        camera_channels: List[str],
        num_preprocess_workers: int,
    ):
        if inputs.rgb_12 is None:
            raise RuntimeError("ONNX Runtime depth backend requires the fixed tensor RGB path; enable fast nuScenes preprocess.")

        processed_batch, orig_size, proc_size = self.depth_model.preprocess_tensor_batch_rgb(
            inputs.rgb_12,
            input_size=self.input_size,
            raw_is_rgb=True,
        )
        expected_shape = (self.fixed_batch_size, 3, self.fixed_input_hw[0], self.fixed_input_hw[1])
        if tuple(processed_batch.shape) != expected_shape:
            raise ValueError(
                "ONNX Runtime depth backend only supports the fixed benchmark shape "
                f"{expected_shape}; got {tuple(processed_batch.shape)}."
            )

        if processed_batch.dtype != torch.float32:
            processed_batch = processed_batch.float()
        processed_batch = processed_batch.contiguous()

        if self.use_cuda_iobinding:
            depth_batch = torch.empty(
                (self.fixed_batch_size, self.fixed_input_hw[0], self.fixed_input_hw[1]),
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
                buffer_ptr=int(processed_batch.data_ptr()),
            )
            io_binding.bind_output(
                name=self.output_name,
                device_type="cuda",
                device_id=processed_batch.device.index or 0,
                element_type=np.float32,
                shape=tuple(depth_batch.shape),
                buffer_ptr=int(depth_batch.data_ptr()),
            )
            self.session.run_with_iobinding(io_binding)
        else:
            depth_np = self.session.run([self.output_name], {self.input_name: processed_batch.detach().cpu().numpy()})[0]
            depth_batch = torch.from_numpy(depth_np).to(device, non_blocking=False)

        return self.depth_model.postprocess_depth_batch(
            depth_batch,
            orig_size=orig_size,
            proc_size=proc_size,
            return_torch=False,
        )


class DepthTensorRTBackend:
    def __init__(
        self,
        *,
        depth_model: DepthAnythingV2,
        engine_path: str,
        onnx_path: str,
        input_size: int,
        fixed_batch_size: int,
        fixed_input_hw: Tuple[int, int],
    ):
        try:
            import tensorrt as trt
        except Exception as exc:
            raise RuntimeError("Depth TensorRT backend requires the tensorrt Python package.") from exc

        self.trt = trt
        self.depth_model = depth_model
        self.input_size = input_size
        self.engine_path = Path(engine_path) if engine_path else None
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.fixed_batch_size = fixed_batch_size
        self.fixed_input_hw = fixed_input_hw
        self.logger = trt.Logger(trt.Logger.WARNING)

        if self.engine_path is None:
            raise ValueError("--depth_trt_engine must be provided when --depth_backend=tensorrt.")
        if not self.engine_path.exists():
            if self.onnx_path is None:
                raise FileNotFoundError(
                    f"TensorRT engine not found at {self.engine_path}. "
                    "Pass --depth_onnx_path to allow ONNX export + engine build."
                )
            export_depthanything_onnx(
                depth_model=self.depth_model,
                onnx_path=str(self.onnx_path),
                batch_size=self.fixed_batch_size,
                input_hw=self.fixed_input_hw,
            )
            build_trt_engine_from_onnx(
                onnx_path=str(self.onnx_path),
                engine_path=str(self.engine_path),
                input_hw=self.fixed_input_hw,
                batch_size=self.fixed_batch_size,
            )

        with open(self.engine_path, "rb") as handle:
            engine_bytes = handle.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

    @property
    def summary(self) -> str:
        return f"tensorrt/engine={self.engine_path.name}"

    def infer_depth_pair(
        self,
        inputs: NuscInputBundle,
        camera_channels: List[str],
        num_preprocess_workers: int,
    ):
        if inputs.rgb_12 is None:
            raise RuntimeError("TensorRT depth backend requires the fixed tensor RGB path; enable fast nuScenes preprocess.")
        processed_batch, orig_size, proc_size = self.depth_model.preprocess_tensor_batch_rgb(
            inputs.rgb_12,
            input_size=self.input_size,
            raw_is_rgb=True,
        )
        if tuple(processed_batch.shape) != (self.fixed_batch_size, 3, self.fixed_input_hw[0], self.fixed_input_hw[1]):
            raise ValueError(
                "TensorRT depth backend only supports the fixed benchmark shape "
                f"{(self.fixed_batch_size, 3, self.fixed_input_hw[0], self.fixed_input_hw[1])}; "
                f"got {tuple(processed_batch.shape)}."
            )

        if processed_batch.dtype != torch.float32:
            processed_batch = processed_batch.float()
        processed_batch = processed_batch.contiguous()

        self.context.set_input_shape(self.input_name, tuple(processed_batch.shape))
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        depth_batch = torch.empty(output_shape, device=processed_batch.device, dtype=torch.float32)
        self.context.set_tensor_address(self.input_name, int(processed_batch.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(depth_batch.data_ptr()))
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.depth_model.postprocess_depth_batch(
            depth_batch,
            orig_size=orig_size,
            proc_size=proc_size,
            return_torch=False,
        )

class ScaleMetricsAccumulator:
    def __init__(self):
        self.err_thresholds = (1.0, 2.0, 5.0)
        self.sample_count = 0
        self.gt_sample_count = 0
        self.valid_pixel_count = 0
        self.sum_mid_err = 0.0
        self.err_mismatch_counts = {threshold: 0 for threshold in self.err_thresholds}

    def update(
        self,
        scale_pred: np.ndarray,
        scale_gt: np.ndarray,
        valid_mask: np.ndarray,
        delta_t: float,
    ):
        valid_mask = np.asarray(valid_mask, dtype=bool)
        valid_pixels = int(valid_mask.sum())
        if valid_pixels == 0:
            return

        pred_scale_valid = np.asarray(scale_pred, dtype=np.float64)[valid_mask]
        gt_scale_valid = np.asarray(scale_gt, dtype=np.float64)[valid_mask]
        pred_eta = np.clip(pred_scale_valid, 1e-6, None)
        gt_eta = np.clip(gt_scale_valid, 1e-6, None)

        self.valid_pixel_count += valid_pixels
        self.sum_mid_err += np.abs(np.log(gt_eta) - np.log(pred_eta)).sum() * 1e4

        pred_ttc = compute_ttc_from_scale(pred_scale_valid, delta_t)
        gt_ttc = compute_ttc_from_scale(gt_scale_valid, delta_t)
        for threshold in self.err_thresholds:
            pred_label = (pred_ttc > 0.0) & (pred_ttc < threshold)
            gt_label = (gt_ttc > 0.0) & (gt_ttc < threshold)
            self.err_mismatch_counts[threshold] += int((pred_label != gt_label).sum())

    def finalize(self, checkpoint_path: str, split_name: str) -> Dict[str, object]:
        valid = float(self.valid_pixel_count)
        return {
            "checkpoint_path": checkpoint_path,
            "split_name": split_name,
            "sample_count": int(self.sample_count),
            "gt_sample_count": int(self.gt_sample_count),
            "valid_pixel_count": int(self.valid_pixel_count),
            "mid_err": float(self.sum_mid_err / valid) if valid > 0 else 0.0,
            "err_1": float(self.err_mismatch_counts[1.0] / valid) if valid > 0 else 0.0,
            "err_2": float(self.err_mismatch_counts[2.0] / valid) if valid > 0 else 0.0,
            "err_5": float(self.err_mismatch_counts[5.0] / valid) if valid > 0 else 0.0,
        }


class Table3MetricsAccumulator:
    def __init__(self):
        self.tau_theta = float(np.pi / 12.0)
        self.tau_t = 2.0
        self.beta = 2.0
        self.err_thresholds = (1.0, 2.0, 5.0)

        self.sample_count = 0
        self.gt_sample_count = 0
        self.valid_pixel_count = 0

        self.sum_mid_err = 0.0
        self.sum_mae_theta = 0.0
        self.sum_acc_hits = 0
        self.err_mismatch_counts = {threshold: 0 for threshold in self.err_thresholds}

        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        scale_pred: np.ndarray,
        scale_gt: np.ndarray,
        orientation_pred: np.ndarray,
        orientation_gt: np.ndarray,
        valid_mask: np.ndarray,
        delta_t: float,
    ):
        valid_mask = np.asarray(valid_mask, dtype=bool)
        valid_pixels = int(valid_mask.sum())
        if valid_pixels == 0:
            return

        pred_scale_valid = np.asarray(scale_pred, dtype=np.float64)[valid_mask]
        gt_scale_valid = np.asarray(scale_gt, dtype=np.float64)[valid_mask]
        pred_eta = np.clip(pred_scale_valid, 1e-6, None)
        gt_eta = np.clip(gt_scale_valid, 1e-6, None)

        pred_orientation = np.clip(
            np.asarray(orientation_pred, dtype=np.float64)[valid_mask], 0.0, np.pi
        )
        gt_orientation = np.clip(
            np.asarray(orientation_gt, dtype=np.float64)[valid_mask], 0.0, np.pi
        )

        self.valid_pixel_count += valid_pixels
        self.sum_mid_err += np.abs(np.log(gt_eta) - np.log(pred_eta)).sum() * 1e4

        theta_diff = np.abs(pred_orientation - gt_orientation)
        self.sum_mae_theta += theta_diff.sum()
        self.sum_acc_hits += int((theta_diff <= self.tau_theta).sum())

        pred_ttc = compute_ttc_from_scale(pred_scale_valid, delta_t)
        gt_ttc = compute_ttc_from_scale(gt_scale_valid, delta_t)
        for threshold in self.err_thresholds:
            pred_label = (pred_ttc > 0.0) & (pred_ttc < threshold)
            gt_label = (gt_ttc > 0.0) & (gt_ttc < threshold)
            self.err_mismatch_counts[threshold] += int((pred_label != gt_label).sum())

        gt_high_risk = (
            (gt_ttc > 0.0)
            & (gt_ttc < self.tau_t)
            & (gt_orientation <= self.tau_theta)
        )
        pred_high_risk = (
            (pred_ttc > 0.0)
            & (pred_ttc < self.tau_t)
            & (pred_orientation <= self.tau_theta)
        )

        self.tp += int((pred_high_risk & gt_high_risk).sum())
        self.fp += int((pred_high_risk & ~gt_high_risk).sum())
        self.fn += int((~pred_high_risk & gt_high_risk).sum())

    def finalize(self, checkpoint_path: str, split_name: str) -> Dict[str, object]:
        valid = float(self.valid_pixel_count)
        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)
        beta_sq = self.beta ** 2
        if precision == 0.0 and recall == 0.0:
            hr_f2 = 0.0
        else:
            hr_f2 = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)

        return {
            "checkpoint_path": checkpoint_path,
            "split_name": split_name,
            "sample_count": int(self.sample_count),
            "gt_sample_count": int(self.gt_sample_count),
            "valid_pixel_count": int(self.valid_pixel_count),
            "mid_err": float(self.sum_mid_err / valid) if valid > 0 else 0.0,
            "err_1": float(self.err_mismatch_counts[1.0] / valid) if valid > 0 else 0.0,
            "err_2": float(self.err_mismatch_counts[2.0] / valid) if valid > 0 else 0.0,
            "err_5": float(self.err_mismatch_counts[5.0] / valid) if valid > 0 else 0.0,
            "mae_theta": float(self.sum_mae_theta / valid) if valid > 0 else 0.0,
            "acc_tau_theta": float(self.sum_acc_hits / valid) if valid > 0 else 0.0,
            "hr_precision": float(precision),
            "hr_recall": float(recall),
            "hr_f2": float(hr_f2),
        }


class SceneMidErrAccumulator:
    def __init__(self):
        self.scene_state = {}

    def update(
        self,
        scene_id: str,
        scale_pred: np.ndarray,
        scale_gt: np.ndarray,
        valid_mask: np.ndarray,
    ):
        valid_mask = np.asarray(valid_mask, dtype=bool)
        valid_pixels = int(valid_mask.sum())
        if valid_pixels == 0:
            return

        pred_scale_valid = np.asarray(scale_pred, dtype=np.float64)[valid_mask]
        gt_scale_valid = np.asarray(scale_gt, dtype=np.float64)[valid_mask]
        pred_eta = np.clip(pred_scale_valid, 1e-6, None)
        gt_eta = np.clip(gt_scale_valid, 1e-6, None)

        state = self.scene_state.setdefault(
            str(scene_id),
            {"valid_pixel_count": 0, "mid_error_sum": 0.0},
        )
        state["valid_pixel_count"] += valid_pixels
        state["mid_error_sum"] += np.abs(np.log(gt_eta) - np.log(pred_eta)).sum() * 1e4

    def finalize(self, checkpoint_path: str, split_name: str) -> Dict[str, object]:
        rows = []
        for scene_id, state in self.scene_state.items():
            valid = int(state["valid_pixel_count"])
            mid_err = float(state["mid_error_sum"] / valid) if valid > 0 else 0.0
            rows.append(
                {
                    "scene_id": scene_id,
                    "valid_pixel_count": valid,
                    "mid_err": mid_err,
                }
            )
        rows.sort(key=lambda row: row["mid_err"])
        return {
            "checkpoint_path": checkpoint_path,
            "split_name": split_name,
            "scene_count": len(rows),
            "rows": rows,
        }


def parse_range_size(range_size_args) -> Tuple[int, int]:
    if isinstance(range_size_args, str):
        raw_tokens = [range_size_args]
    else:
        raw_tokens = list(range_size_args)

    tokens: List[str] = []
    for token in raw_tokens:
        for sep in ("x", "X", ","):
            token = token.replace(sep, " ")
        tokens.extend(part for part in token.split() if part)

    if len(tokens) != 2:
        raise ValueError(f"Expected range size to resolve to 2 integers, got: {range_size_args}")

    range_h, range_w = int(tokens[0]), int(tokens[1])
    if range_h <= 0 or range_w <= 0:
        raise ValueError(f"Range size must be positive, got: {(range_h, range_w)}")
    return range_h, range_w


args.range_size = parse_range_size(args.range_size)


def resolve_mapping_range_backend(requested: str) -> str:
    if requested == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if requested == "gpu" and not torch.cuda.is_available():
        return "cpu"
    return requested


def resolve_test_profile(args) -> Dict[str, object]:
    profile_name = str(args.test_profile).strip().lower()
    args.test_profile = profile_name
    if profile_name == "custom":
        return {}

    overrides = TEST_PROFILE_OVERRIDES[profile_name]
    for key, value in overrides.items():
        setattr(args, key, value)
    return dict(overrides)


def format_effective_test_profile(args, applied_profile_overrides: Dict[str, object]) -> str:
    header = f"[Profile] test_profile={args.test_profile}"
    if args.test_profile == "custom":
        return f"{header} (no preset overrides)"
    return f"{header}\n{pformat(applied_profile_overrides, sort_dicts=False)}"


def sanitize_accel_flags(args):
    metrics_enabled = args.compute_scale_metrics if args.scale_only else (
        args.compute_scale_orientation_metrics or args.compute_scene_mid_err
    )
    fixed_depth_backends = {"onnxruntime", "tensorrt"}
    if args.enable_cuda_graphs and (
        args.save_visualizations
        or metrics_enabled
    ):
        print("[Accel] Disabling CUDA graphs because vis/metrics are enabled.")
        args.enable_cuda_graphs = False

    if args.compile_depth and args.depth_backend != "pytorch":
        print(f"[Accel] Ignoring --compile_depth because depth backend is {args.depth_backend}.")
        args.compile_depth = False

    if args.enable_cuda_graphs and not args.enable_fast_nusc_preprocess and args.depth_backend in fixed_depth_backends:
        print(f"[Accel] {args.depth_backend} depth backend requires fast nuScenes preprocess; disabling CUDA graphs.")
        args.enable_cuda_graphs = False

    if args.depth_backend in fixed_depth_backends and not args.enable_fast_nusc_preprocess:
        raise ValueError(f"--depth_backend={args.depth_backend} currently requires --enable_fast_nusc_preprocess true.")

    if args.depth_backend in fixed_depth_backends and args.sjtu_test:
        raise ValueError(f"--depth_backend={args.depth_backend} is currently only wired for the fixed nuScenes test path.")


if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize_height, resize_width = args.image_size
augmentor = NuscRangeImageAugmentor(
    crop_size=(resize_height, resize_width),
    do_flip=False,
    rotate=False,
)


def maybe_pin_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.pin_memory() if torch.cuda.is_available() else tensor


def sync_cuda_if_available():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_rgb_image(path: str, rgb_image: np.ndarray):
    rgb_u8 = np.clip(rgb_image * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(rgb_u8).save(path)


def stack_imgs_to_tensor(
    imgs: Dict[str, np.ndarray],
    camera_channels: List[str],
    device: torch.device,
) -> torch.Tensor:
    tensors = [torch.from_numpy(imgs[ch]).permute(2, 0, 1).float() for ch in camera_channels]
    batch = torch.stack(tensors, dim=0).unsqueeze(0)
    batch = maybe_pin_cpu_tensor(batch)
    batch_gpu = batch.to(device, non_blocking=torch.cuda.is_available())
    log_dtype_event(
        "input_stack.rgb_batch",
        src_dtype=batch.dtype,
        dst_dtype=batch_gpu.dtype,
        shape=batch_gpu.shape,
        device=batch_gpu.device,
        copied=True,
    )
    return batch_gpu


def stack_rgb_pair_dicts_to_device_batches(
    prev_imgs: Dict[str, np.ndarray],
    curr_imgs: Dict[str, np.ndarray],
    camera_channels: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_np = np.stack(
        [prev_imgs[ch] for ch in camera_channels] + [curr_imgs[ch] for ch in camera_channels],
        axis=0,
    )
    tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous()
    tensor = maybe_pin_cpu_tensor(tensor)
    tensor_gpu = tensor.to(device, non_blocking=torch.cuda.is_available()).float()
    log_dtype_event(
        "input_stack.rgb_pair_batch",
        src_dtype=tensor.dtype,
        dst_dtype=tensor_gpu.dtype,
        shape=tensor_gpu.shape,
        device=tensor_gpu.device,
        copied=True,
    )
    num_cams = len(camera_channels)
    return tensor_gpu[:num_cams].unsqueeze(0), tensor_gpu[num_cams:].unsqueeze(0), tensor_gpu


def to_device_batch(array: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    tensor = torch.from_numpy(array if dtype is None else array.astype(dtype))
    tensor = maybe_pin_cpu_tensor(tensor)
    return tensor.unsqueeze(0).to(device, non_blocking=torch.cuda.is_available())


def stack_depth_maps_to_device_batch(
    depth_maps: Dict[str, np.ndarray],
    camera_channels: List[str],
    device: torch.device,
) -> torch.Tensor:
    depth_np = np.stack([depth_maps[ch] for ch in camera_channels], axis=0).astype(np.float32, copy=False)
    tensor = torch.from_numpy(depth_np).unsqueeze(1).contiguous()
    tensor = maybe_pin_cpu_tensor(tensor)
    tensor_gpu = tensor.unsqueeze(0).to(device, non_blocking=torch.cuda.is_available())
    log_dtype_event(
        "depth_stack.depth_batch",
        src_dtype=tensor.dtype,
        dst_dtype=tensor_gpu.dtype,
        shape=tensor_gpu.shape,
        device=tensor_gpu.device,
        copied=True,
    )
    return tensor_gpu


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


def preprocess_nusc_rgb_dict(
    imgs: Dict[str, np.ndarray],
    camera_channels: List[str],
    resize_hw: Tuple[int, int],
    crop_yx: Tuple[int, int],
    crop_hw: Tuple[int, int],
    pool: ThreadPoolExecutor,
) -> Dict[str, np.ndarray]:
    futures = [
        pool.submit(fixed_resize_crop_rgb_image, imgs[ch], resize_hw, crop_yx, crop_hw)
        for ch in camera_channels
    ]
    return {ch: future.result() for ch, future in zip(camera_channels, futures)}


def load_nusc_image(ch: str, frame_type: str, idx: int, test_entries: list):
    image_path = resolve_nusc_path(test_entries[idx][f"{frame_type}_camera_data"][ch]["filename"])
    with Image.open(image_path) as image:
        return ch, image.convert("RGB")


def load_nusc_image_np(ch: str, frame_type: str, idx: int, test_entries: list):
    image_path = resolve_nusc_path(test_entries[idx][f"{frame_type}_camera_data"][ch]["filename"])
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return ch, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def submit_nusc_image_prefetch(
    prefetch_pool: ThreadPoolExecutor,
    image_loader,
    idx: int,
    test_entries: list,
    camera_channels: List[str],
):
    prev_futs = [prefetch_pool.submit(image_loader, ch, "prev", idx, test_entries) for ch in camera_channels]
    curr_futs = [prefetch_pool.submit(image_loader, ch, "curr", idx, test_entries) for ch in camera_channels]
    return prev_futs, curr_futs


def consume_nusc_image_prefetch(prefetch_handles):
    prev_futs, curr_futs = prefetch_handles
    prev_images = {ch: img for ch, img in (f.result() for f in prev_futs)}
    curr_images = {ch: img for ch, img in (f.result() for f in curr_futs)}
    return prev_images, curr_images


def build_fixed_nusc_fast_preprocess(
    augmentor: NuscRangeImageAugmentor,
    example_image_size: Tuple[int, int],
):
    params = augmentor.sample_params(example_image_size)
    if params["flip_h"] or params["flip_v"] or params["rotate"]:
        raise ValueError("Fast nuScenes preprocess expects flip/rotate to be disabled.")
    resize_w, resize_h = params["resize"]
    crop_x, crop_y = params["crop"]
    return {
        "resize_hw": (resize_h, resize_w),
        "crop_yx": (crop_y, crop_x),
        "crop_hw": tuple(augmentor.crop_size),
        "affine_matrix": augmentor.get_affine_matrix(params),
    }


def init_timing_totals() -> Dict[str, float]:
    return {key: 0.0 for key in TIMING_KEYS}


def init_timing_records() -> Dict[str, List[float]]:
    records = {key: [] for key in TIMING_KEYS}
    records["total_ms"] = []
    return records


def init_mapping_detail_totals() -> Dict[str, float]:
    return {key: 0.0 for key in MAPPING_DETAIL_KEYS}


def init_mapping_detail_records() -> Dict[str, List[float]]:
    return {key: [] for key in MAPPING_DETAIL_KEYS}


def accumulate_timing(totals: Dict[str, float], sample_timing: Dict[str, float]):
    for key in TIMING_KEYS:
        totals[key] += sample_timing.get(key, 0.0)


def record_timing_sample(records: Dict[str, List[float]], sample_timing: Dict[str, float]):
    total_ms = 0.0
    for key in TIMING_KEYS:
        value = float(sample_timing.get(key, 0.0))
        records[key].append(value)
        total_ms += value
    records["total_ms"].append(total_ms)


def accumulate_mapping_detail(totals: Dict[str, float], detail: Dict[str, float]):
    for key in MAPPING_DETAIL_KEYS:
        totals[key] += float(detail.get(key, 0.0))


def record_mapping_detail_sample(records: Dict[str, List[float]], detail: Dict[str, float]):
    for key in MAPPING_DETAIL_KEYS:
        records[key].append(float(detail.get(key, 0.0)))


def get_timing_status(stage_key: str) -> str:
    if stage_key == "vis_ms":
        return "active" if args.save_visualizations else "disabled"
    if stage_key == "metrics_ms":
        if args.scale_only:
            return "active" if args.compute_scale_metrics else "disabled"
        return "active" if args.compute_scale_orientation_metrics else "disabled"
    return "active"


def build_timing_summary_markdown(
    label: str,
    totals: Dict[str, float],
    records: Dict[str, List[float]],
    sample_count: int,
    mapping_detail_totals: Dict[str, float],
    mapping_detail_records: Dict[str, List[float]],
    mapping_range_backend: str,
) -> str:
    if sample_count <= 0:
        return ""

    total_ms = sum(totals[key] for key in TIMING_KEYS)
    avg_total_ms = total_ms / sample_count
    lines = [
        f"# Timing Summary ({label})",
        "",
        f"- Samples: {sample_count}",
        f"- Avg total: {avg_total_ms:.2f} ms",
        f"- Mapping range backend: {mapping_range_backend}",
        "",
        "| Stage | Avg ms | Share % | p50 ms | p95 ms | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for key in TIMING_KEYS:
        avg_ms = totals[key] / sample_count
        share = (totals[key] / total_ms * 100.0) if total_ms > 0 else 0.0
        values = np.asarray(records.get(key, []), dtype=np.float64)
        p50 = float(np.percentile(values, 50)) if values.size else 0.0
        p95 = float(np.percentile(values, 95)) if values.size else 0.0
        lines.append(
            f"| {key.replace('_ms', '')} | {avg_ms:.2f} | {share:.1f} | {p50:.2f} | {p95:.2f} | {get_timing_status(key)} |"
        )

    total_values = np.asarray(records.get("total_ms", []), dtype=np.float64)
    total_p50 = float(np.percentile(total_values, 50)) if total_values.size else 0.0
    total_p95 = float(np.percentile(total_values, 95)) if total_values.size else 0.0
    lines.append(f"| total | {avg_total_ms:.2f} | 100.0 | {total_p50:.2f} | {total_p95:.2f} | active |")
    lines.extend([
        "",
        "## Mapping Breakdown",
        "",
        "| Stage | Avg ms | Share % of mapping | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])

    mapping_total = sum(mapping_detail_totals[key] for key in MAPPING_DETAIL_KEYS)
    for key in MAPPING_DETAIL_KEYS:
        avg_ms = mapping_detail_totals[key] / sample_count
        share = (mapping_detail_totals[key] / mapping_total * 100.0) if mapping_total > 0 else 0.0
        values = np.asarray(mapping_detail_records.get(key, []), dtype=np.float64)
        p50 = float(np.percentile(values, 50)) if values.size else 0.0
        p95 = float(np.percentile(values, 95)) if values.size else 0.0
        lines.append(
            f"| {key.replace('_ms', '')} | {avg_ms:.2f} | {share:.1f} | {p50:.2f} | {p95:.2f} |"
        )
    return "\n".join(lines)


def print_timing_summary(
    label: str,
    totals: Dict[str, float],
    records: Dict[str, List[float]],
    sample_count: int,
    output_dir: str,
    mapping_detail_totals: Dict[str, float],
    mapping_detail_records: Dict[str, List[float]],
    mapping_range_backend: str,
):
    if sample_count <= 0:
        return
    markdown = build_timing_summary_markdown(
        label,
        totals,
        records,
        sample_count,
        mapping_detail_totals,
        mapping_detail_records,
        mapping_range_backend,
    )
    print(f"\n{markdown}")
    timing_path = os.path.join(output_dir, "timing_breakdown.md")
    with open(timing_path, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")
    print(f"[INFO] Saved timing breakdown to {timing_path}")


def resize_map_to_shape(map_array: np.ndarray, target_shape: Tuple[int, int], interpolation) -> np.ndarray:
    map_array = np.asarray(map_array)
    if map_array.shape == tuple(target_shape):
        return map_array
    target_h, target_w = target_shape
    resized = cv2.resize(map_array.astype(np.float32), (target_w, target_h), interpolation=interpolation)
    return resized.astype(np.float32)


def align_prediction_to_shape(pred_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    return resize_map_to_shape(pred_map, target_shape, interpolation=cv2.INTER_LINEAR)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def compute_ttc_from_scale(scale_map: np.ndarray, delta_t: float) -> np.ndarray:
    scale_map = np.asarray(scale_map, dtype=np.float64)
    denom = 1.0 - scale_map
    ttc = np.full(scale_map.shape, np.inf, dtype=np.float64)
    valid = np.abs(denom) > 1e-12
    np.divide(float(delta_t), denom, out=ttc, where=valid)
    return ttc


def print_scale_metrics_summary(metrics_payload: Dict[str, object]):
    print()
    print("Scale metrics")
    print(f"  MiD Err: {metrics_payload['mid_err']:.6f}")
    print(f"  Err-1:   {metrics_payload['err_1']:.6f}")
    print(f"  Err-2:   {metrics_payload['err_2']:.6f}")
    print(f"  Err-5:   {metrics_payload['err_5']:.6f}")
    print(f"  Valid pixels: {metrics_payload['valid_pixel_count']}")


def print_scale_orientation_metrics_summary(metrics_payload: Dict[str, object]):
    print()
    print("Scale & orientation metrics")
    print(f"  MiD Err:   {metrics_payload['mid_err']:.6f}")
    print(f"  Err-1:     {metrics_payload['err_1']:.6f}")
    print(f"  Err-2:     {metrics_payload['err_2']:.6f}")
    print(f"  Err-5:     {metrics_payload['err_5']:.6f}")
    print(f"  MAEθ:      {metrics_payload['mae_theta']:.6f}")
    print(f"  Acc@τθ:    {metrics_payload['acc_tau_theta']:.6f}")
    print(f"  Precision: {metrics_payload['hr_precision']:.6f}")
    print(f"  Recall:    {metrics_payload['hr_recall']:.6f}")
    print(f"  HR-F2:     {metrics_payload['hr_f2']:.6f}")
    print(f"  Valid pixels: {metrics_payload['valid_pixel_count']}")


def build_scene_mid_err_markdown(scene_payload: Dict[str, object]) -> str:
    lines = [
        "# Scene-wise MiD Err",
        "",
        f"- Checkpoint: {scene_payload['checkpoint_path']}",
        f"- Split: {scene_payload['split_name']}",
        f"- Scene count: {scene_payload['scene_count']}",
        "",
        "| Scene | Valid pixels | MiD Err |",
        "| --- | ---: | ---: |",
    ]
    for row in scene_payload["rows"]:
        lines.append(
            f"| {row['scene_id']} | {row['valid_pixel_count']} | {row['mid_err']:.6f} |"
        )
    return "\n".join(lines)


def print_scene_mid_err_summary(scene_payload: Dict[str, object]):
    print()
    print(build_scene_mid_err_markdown(scene_payload))


def save_analysis_pair(output_dir: str, idx: int, prefix: str, scale_rgb: np.ndarray, risk_rgb: np.ndarray):
    save_rgb_image(os.path.join(output_dir, f"{prefix}_scale_analysis_{idx}.png"), scale_rgb)
    save_rgb_image(os.path.join(output_dir, f"{prefix}_risk_analysis_{idx}.png"), risk_rgb)


def save_pred_visualizations(
    output_dir: str,
    idx: int,
    scale_map: np.ndarray,
    risk_map: np.ndarray,
):
    scale_analysis_rgb = make_scale_analysis_rgb(scale_map)
    risk_analysis_rgb = make_risk_analysis_rgb(risk_map)
    save_analysis_pair(output_dir, idx, "pred", scale_analysis_rgb, risk_analysis_rgb)


def save_gt_analysis_visualizations(
    output_dir: str,
    idx: int,
    gt_scale_map: np.ndarray,
    gt_risk_map: np.ndarray,
    gt_valid_mask: np.ndarray,
):
    gt_scale_rgb = make_scale_analysis_rgb(gt_scale_map, valid_mask=gt_valid_mask)
    gt_risk_rgb = make_risk_analysis_rgb(gt_risk_map, valid_mask=gt_valid_mask)
    save_analysis_pair(output_dir, idx, "gt", gt_scale_rgb, gt_risk_rgb)


def create_output_dir(output_dir_arg: Optional[str]) -> str:
    if output_dir_arg:
        os.makedirs(output_dir_arg, exist_ok=True)
        return output_dir_arg
    time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S_%f")
    suffix = "scale_test" if args.scale_only else "surround_ttc"
    output_dir = f"./test/{time_stamp}_{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_test_entries(info_path: str, max_test_samples: int) -> Tuple[list, str]:
    with open(info_path, "rb") as f:
        test_entries = pickle.load(f)
    if max_test_samples > 0:
        test_entries = test_entries[:max_test_samples]
    split_name = os.path.splitext(os.path.basename(info_path))[0]
    return test_entries, split_name


def _batchify_sensor_metas(sensor_metas):
    return {
        frame_key: {
            channel: {
                key: value.unsqueeze(0)
                for key, value in channel_metas.items()
            }
            for channel, channel_metas in frame_sensor_metas.items()
        }
        for frame_key, frame_sensor_metas in sensor_metas.items()
    }


def _move_sensor_metas_to_device(sensor_metas, device):
    for frame_key in sensor_metas:
        for channel in sensor_metas[frame_key]:
            for key in sensor_metas[frame_key][channel]:
                sensor_metas[frame_key][channel][key] = sensor_metas[frame_key][channel][key].to(device)
    return sensor_metas


def resolve_repo_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def undistort_sjtu_bgr_image(image_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray):
    h, w = image_bgr.shape[:2]
    K = np.asarray(K, dtype=np.float32)
    dist = np.asarray(dist, dtype=np.float32).reshape(-1)
    if dist.size == 0:
        return image_bgr.copy(), K.copy(), (0, 0, w, h)

    K_undist, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    image_undist = cv2.undistort(image_bgr, K, dist, None, K_undist)
    return image_undist, K_undist.astype(np.float32), roi


def compose_sjtu_500ms_entries(raw_entries: list, max_test_samples: int) -> list:
    composed = []
    front_channel = "CAM_FRONT"

    for start in range(0, max(0, len(raw_entries) - 4), 5):
        end = start + 4
        if end >= len(raw_entries):
            break

        prev_entry = raw_entries[start]
        curr_entry = raw_entries[end]
        prev_ts = int(prev_entry["prev_camera_data"][front_channel]["timestamp"])
        curr_ts = int(curr_entry["curr_camera_data"][front_channel]["timestamp"])
        composed.append(
            {
                "prev_camera_data": deepcopy(prev_entry["prev_camera_data"]),
                "curr_camera_data": deepcopy(curr_entry["curr_camera_data"]),
                "prev_lidar_data": prev_entry.get("prev_lidar_data"),
                "curr_lidar_data": curr_entry.get("curr_lidar_data"),
                "sensor_metas_prev": deepcopy(prev_entry["sensor_metas_prev"]),
                "sensor_metas_curr": deepcopy(curr_entry["sensor_metas_curr"]),
                "gt_map_path": None,
                "scene_flow_path": None,
                "scene_indice": prev_entry.get("scene_indice", "18"),
                "ros_msg_seq_prev": prev_entry.get("ros_msg_seq"),
                "ros_msg_seq_curr": curr_entry.get("ros_msg_seq"),
                "ros_msg_seq": curr_entry.get("ros_msg_seq"),
                "time_diff_cam_us": curr_ts - prev_ts,
            }
        )

    print(
        f"[SJTU] raw_100ms_samples={len(raw_entries)} "
        f"regrouped_500ms_samples={len(composed)} "
        f"sjtu_frame_gap_ms=500"
    )
    for pair_idx, sample in enumerate(composed[:3]):
        print(
            f"[SJTU] pair[{pair_idx}] seq_prev={sample.get('ros_msg_seq_prev')} "
            f"seq_curr={sample.get('ros_msg_seq_curr')} "
            f"delta_us={sample['time_diff_cam_us']}"
        )

    if max_test_samples > 0:
        composed = composed[:max_test_samples]
    return composed


def _load_checkpoint_flexibly(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "net" in checkpoint:
        raw_state_dict = checkpoint["net"]
    elif "state_dict" in checkpoint:
        raw_state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        raw_state_dict = checkpoint["model"]
    else:
        raw_state_dict = checkpoint

    processed_state_dict = {}
    for key, value in raw_state_dict.items():
        processed_state_dict[key[len("module."):] if key.startswith("module.") else key] = value

    model_state_dict = model.state_dict()
    loadable = {}
    for key, value in processed_state_dict.items():
        if key in model_state_dict and value.shape == model_state_dict[key].shape:
            loadable[key] = value

    load_info = model.load_state_dict(loadable, strict=False)
    print(f"[TEST] loaded {len(loadable)} keys from {checkpoint_path}")
    if load_info.missing_keys:
        print(f"[TEST] missing {len(load_info.missing_keys)} keys")
    if load_info.unexpected_keys:
        print(f"[TEST] unexpected {len(load_info.unexpected_keys)} keys")


def resolve_depthanything_ckpt_dir(requested_dir: str) -> Path:
    candidates = []
    if requested_dir:
        candidates.append(Path(requested_dir))
    repo_root = Path(__file__).resolve().parent
    candidates.append(repo_root / "pretrained" / "depthanything")
    candidates.append(repo_root.parent / "FP-TTC-hardproj-150scene-v1" / "pretrained" / "depthanything")

    for candidate in candidates:
        ckpt_path = candidate / "depth_anything_v2_metric_vkitti_vits.pth"
        if ckpt_path.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find depth_anything_v2_metric_vkitti_vits.pth. "
        "Pass --depthanything_ckpt_dir or provide the pretrained depthanything weights."
    )


def build_models_and_runtime(args, test_entries: list, split_name: str) -> Tuple[torch.nn.Module, Any, TestRuntime]:
    model_kwargs = dict(
        num_scales=args.num_scales,
        feature_channels=args.feature_channels,
        upsample_factor=args.upsample_factor,
        num_head=args.num_head,
        ffn_dim_expansion=args.ffn_dim_expansion,
        num_transformer_layers=args.num_transformer_layers,
        reg_refine=args.reg_refine,
        cache_geom_constants=args.cache_geom_constants,
    )
    if "rvt_depth_guided_sampling" in inspect.signature(FpTTC).parameters:
        model_kwargs["rvt_depth_guided_sampling"] = args.rvt_depth_guided_sampling
    model = FpTTC(**model_kwargs).to(device).eval()
    maybe_compile_model_submodules(model, args.compile_model_submodules, args.compile_mode)
    if args.resume:
        _load_checkpoint_flexibly(model, args.resume, device)

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    depth_ckpt_dir = resolve_depthanything_ckpt_dir(args.depthanything_ckpt_dir)
    depth_model = DepthAnythingV2(**{**model_configs["vits"], "max_depth": 80})
    depth_model.load_state_dict(
        torch.load(depth_ckpt_dir / "depth_anything_v2_metric_vkitti_vits.pth", map_location="cpu")
    )
    depth_model.to(device).eval()

    camera_channels = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
    ]
    fixed_nusc_fast = None
    fixed_depth_proc_hw = None
    if args.enable_fast_nusc_preprocess and test_entries and not args.sjtu_test:
        sample_path = resolve_nusc_path(test_entries[0]["prev_camera_data"][camera_channels[0]]["filename"])
        with Image.open(sample_path) as sample_image:
            fixed_nusc_fast = build_fixed_nusc_fast_preprocess(augmentor, sample_image.size)
        crop_h, crop_w = fixed_nusc_fast["crop_hw"]
        fixed_depth_proc_hw = DepthAnythingV2.get_preprocess_size(crop_h, crop_w, input_size=args.depth_input_size)

    runtime = TestRuntime(
        output_dir=create_output_dir(args.output_dir),
        split_name=split_name,
        camera_channels=camera_channels,
        range_h=args.range_size[0],
        range_w=args.range_size[1],
        mapping_range_backend=resolve_mapping_range_backend(args.mapping_range_backend),
        fixed_nusc_fast=fixed_nusc_fast,
        fixed_depth_proc_hw=fixed_depth_proc_hw,
        xformers_available=xformers_available(),
    )
    fixed_batch_size = len(camera_channels) * 2
    default_depth_onnx_path = None
    if runtime.fixed_depth_proc_hw is not None:
        default_depth_onnx_path = (
            Path(runtime.output_dir)
            / f"depthanything_vits_{fixed_batch_size}x3x{runtime.fixed_depth_proc_hw[0]}x{runtime.fixed_depth_proc_hw[1]}.onnx"
        )

    if args.depth_backend == "tensorrt":
        if runtime.fixed_depth_proc_hw is None:
            raise RuntimeError("Depth TensorRT backend requires fixed nuScenes fast preprocess with a known tensor shape.")
        depth_backend = DepthTensorRTBackend(
            depth_model=depth_model,
            engine_path=args.depth_trt_engine,
            onnx_path=args.depth_onnx_path or str(default_depth_onnx_path),
            input_size=args.depth_input_size,
            fixed_batch_size=fixed_batch_size,
            fixed_input_hw=runtime.fixed_depth_proc_hw,
        )
    elif args.depth_backend == "onnxruntime":
        if runtime.fixed_depth_proc_hw is None:
            raise RuntimeError("Depth ONNX Runtime backend requires fixed nuScenes fast preprocess with a known tensor shape.")
        depth_backend = DepthONNXRuntimeBackend(
            depth_model=depth_model,
            onnx_path=args.depth_onnx_path or str(default_depth_onnx_path),
            input_size=args.depth_input_size,
            fixed_batch_size=fixed_batch_size,
            fixed_input_hw=runtime.fixed_depth_proc_hw,
        )
    else:
        depth_backend = DepthPyTorchBackend(
            depth_model,
            input_size=args.depth_input_size,
            enable_amp=args.enable_depth_amp,
            amp_dtype_name=args.depth_amp_dtype,
            compile_depth=args.compile_depth,
            compile_mode=args.compile_mode,
            enable_cuda_graphs=args.enable_cuda_graphs,
            cuda_graph_warmup_iters=args.cuda_graph_warmup_iters,
        )
    runtime.depth_backend_summary = depth_backend.summary
    if args.enable_cuda_graphs and torch.cuda.is_available():
        runtime.model_cuda_graph_runner = ModelCudaGraphRunner(
            model,
            amp_enabled=args.enable_model_amp,
            amp_dtype_name=args.model_amp_dtype,
            warmup_iters=args.cuda_graph_warmup_iters,
        )
    return model, depth_backend, runtime


def print_runtime_summary(args, runtime: TestRuntime, applied_profile_overrides: Dict[str, object]):
    print(format_effective_test_profile(args, applied_profile_overrides))
    dataset_name = "sjtu" if args.sjtu_test else "nuScenes"
    metrics_mode = (
        f"compute_scale_metrics={args.compute_scale_metrics}"
        if args.scale_only
        else (
            f"compute_scale_orientation_metrics={args.compute_scale_orientation_metrics} "
            f"compute_scene_mid_err={args.compute_scene_mid_err}"
        )
    )
    print(
        "[Runtime] "
        f"dataset={dataset_name} "
        f"image_size={tuple(args.image_size)} "
        f"depth_input_size={args.depth_input_size} "
        f"range_size=({runtime.range_h}, {runtime.range_w}) "
        f"save_visualizations={args.save_visualizations} "
        f"scale_only={args.scale_only} "
        f"{metrics_mode} "
        f"depth_guided_sampling={args.rvt_depth_guided_sampling}"
    )
    print(
        "[Accel] "
        f"nusc_io_prefetch={args.enable_nusc_io_prefetch} "
        f"fast_nusc_preprocess={args.enable_fast_nusc_preprocess} "
        f"depth_backend={runtime.depth_backend_summary} "
        f"depth_amp={args.enable_depth_amp}({args.depth_amp_dtype}) "
        f"depth_pre_workers={args.num_depth_preprocess_workers} "
        f"mapping_backend={runtime.mapping_range_backend} "
        f"model_amp={args.enable_model_amp}({args.model_amp_dtype}) "
        f"compile_model_submodules={args.compile_model_submodules} "
        f"cuda_graphs={args.enable_cuda_graphs} "
        f"cache_geom_constants={args.cache_geom_constants} "
        f"xformers_available={runtime.xformers_available}"
    )
    if args.sjtu_test:
        print("[SJTU] Online undistortion enabled. Metrics are disabled because scene_18 has no GT.")


def create_metrics_accumulators(
    args,
    test_entries: list,
) -> Tuple[Optional[ScaleMetricsAccumulator], Optional[Table3MetricsAccumulator], Optional[SceneMidErrAccumulator]]:
    scale_metrics_acc = None
    orientation_metrics_acc = None
    scene_mid_err_acc = None

    if args.scale_only:
        if args.compute_scale_metrics:
            scale_metrics_acc = ScaleMetricsAccumulator()
            scale_metrics_acc.sample_count = len(test_entries)
        return scale_metrics_acc, orientation_metrics_acc, scene_mid_err_acc

    if args.compute_scale_orientation_metrics:
        if args.sjtu_test:
            print("[INFO] Scale & orientation metrics are only supported for nuScenes test with GT. Skipping metrics for SJTU.")
        else:
            orientation_metrics_acc = Table3MetricsAccumulator()
            orientation_metrics_acc.sample_count = len(test_entries)
            if args.compute_scene_mid_err:
                scene_mid_err_acc = SceneMidErrAccumulator()
    return scale_metrics_acc, orientation_metrics_acc, scene_mid_err_acc


def load_sjtu_camera_frame(channel: str, camera_data: Dict[str, object], sensor_meta: Dict[str, object]):
    image_path = resolve_repo_local_path(camera_data["filename"])
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load SJTU image: {image_path}")

    image_undist, K_undist, _ = undistort_sjtu_bgr_image(
        image_bgr,
        sensor_meta["K"],
        sensor_meta.get("dist", []),
    )
    R = np.asarray(sensor_meta["R"], dtype=np.float32)
    t = np.asarray(sensor_meta["t"], dtype=np.float32).reshape(3)
    R_l2c = R.T.astype(np.float32)
    t_l2c = (-R_l2c @ t).astype(np.float32)
    image_rgb = cv2.cvtColor(image_undist, cv2.COLOR_BGR2RGB)
    return channel, Image.fromarray(image_rgb), K_undist, R_l2c, t_l2c


def load_sjtu_sample_images(
    idx: int,
    test_entries: list,
    camera_channels: List[str],
    pool: ThreadPoolExecutor,
):
    t0 = time.perf_counter()
    sample_info = deepcopy(test_entries[idx])

    prev_futures = [
        pool.submit(
            load_sjtu_camera_frame,
            ch,
            sample_info["prev_camera_data"][ch],
            sample_info["sensor_metas_prev"][ch],
        )
        for ch in camera_channels
    ]
    curr_futures = [
        pool.submit(
            load_sjtu_camera_frame,
            ch,
            sample_info["curr_camera_data"][ch],
            sample_info["sensor_metas_curr"][ch],
        )
        for ch in camera_channels
    ]

    raw_prev = {}
    raw_curr = {}
    for future in prev_futures:
        ch, image, K_undist, R_l2c, t_l2c = future.result()
        raw_prev[ch] = image
        sample_info["sensor_metas_prev"][ch]["K_undist"] = K_undist
        sample_info["sensor_metas_prev"][ch]["R_l2c"] = R_l2c
        sample_info["sensor_metas_prev"][ch]["t_l2c"] = t_l2c
    for future in curr_futures:
        ch, image, K_undist, R_l2c, t_l2c = future.result()
        raw_curr[ch] = image
        sample_info["sensor_metas_curr"][ch]["K_undist"] = K_undist
        sample_info["sensor_metas_curr"][ch]["R_l2c"] = R_l2c
        sample_info["sensor_metas_curr"][ch]["t_l2c"] = t_l2c

    return sample_info, raw_prev, raw_curr, (time.perf_counter() - t0) * 1000.0


def load_nusc_sample_images(
    idx: int,
    test_entries: list,
    camera_channels: List[str],
    pool: ThreadPoolExecutor,
    image_loader,
    prefetch_pool: Optional[ThreadPoolExecutor],
    prefetched_handles,
):
    t0 = time.perf_counter()
    if prefetch_pool is not None:
        if prefetched_handles is None:
            raise RuntimeError("nuScenes prefetch was enabled but no prefetched handles are available.")
        raw_prev, raw_curr = consume_nusc_image_prefetch(prefetched_handles)
        t1 = time.perf_counter()
        next_idx = idx + 1
        next_prefetched_handles = None
        if next_idx < len(test_entries):
            next_prefetched_handles = submit_nusc_image_prefetch(
                prefetch_pool,
                image_loader,
                next_idx,
                test_entries,
                camera_channels,
            )
    else:
        prev_futs = [pool.submit(image_loader, ch, "prev", idx, test_entries) for ch in camera_channels]
        curr_futs = [pool.submit(image_loader, ch, "curr", idx, test_entries) for ch in camera_channels]
        raw_prev = {ch: img for ch, img in (f.result() for f in prev_futs)}
        raw_curr = {ch: img for ch, img in (f.result() for f in curr_futs)}
        t1 = time.perf_counter()
        next_prefetched_handles = None
    return raw_prev, raw_curr, next_prefetched_handles, (t1 - t0) * 1000.0


def prepare_nusc_input_bundle(
    raw_prev: Dict[str, object],
    raw_curr: Dict[str, object],
    camera_channels: List[str],
    pool: ThreadPoolExecutor,
    runtime: TestRuntime,
    device: torch.device,
) -> Tuple[NuscInputBundle, float]:
    t0 = time.perf_counter()
    if args.enable_fast_nusc_preprocess:
        if runtime.fixed_nusc_fast is None:
            raise RuntimeError("Fixed nuScenes fast preprocess spec was not initialized.")
        affine_matrix = runtime.fixed_nusc_fast["affine_matrix"]
        proc_prev = preprocess_nusc_rgb_dict(
            raw_prev,
            camera_channels,
            runtime.fixed_nusc_fast["resize_hw"],
            runtime.fixed_nusc_fast["crop_yx"],
            runtime.fixed_nusc_fast["crop_hw"],
            pool,
        )
        proc_curr = preprocess_nusc_rgb_dict(
            raw_curr,
            camera_channels,
            runtime.fixed_nusc_fast["resize_hw"],
            runtime.fixed_nusc_fast["crop_yx"],
            runtime.fixed_nusc_fast["crop_hw"],
            pool,
        )
        prev_batch, curr_batch, rgb_12 = stack_rgb_pair_dicts_to_device_batches(
            proc_prev, proc_curr, camera_channels, device
        )
    else:
        orig_size = next(iter(raw_prev.values())).size
        affine_params = augmentor.sample_params(orig_size)
        proc_prev, _ = augmentor(raw_prev, affine_params)
        proc_curr, _ = augmentor(raw_curr, affine_params)
        affine_matrix = augmentor.get_affine_matrix(affine_params)
        prev_batch = stack_imgs_to_tensor(proc_prev, camera_channels, device)
        curr_batch = stack_imgs_to_tensor(proc_curr, camera_channels, device)
        rgb_12 = None
    sync_cuda_if_available()
    augment_ms = (time.perf_counter() - t0) * 1000.0
    return NuscInputBundle(
        raw_prev=raw_prev,
        raw_curr=raw_curr,
        proc_prev=proc_prev,
        proc_curr=proc_curr,
        prev_batch=prev_batch,
        curr_batch=curr_batch,
        affine_matrix=affine_matrix,
        rgb_12=rgb_12,
    ), augment_ms


def prepare_sjtu_input_bundle(
    raw_prev: Dict[str, Image.Image],
    raw_curr: Dict[str, Image.Image],
    camera_channels: List[str],
    device: torch.device,
) -> Tuple[NuscInputBundle, float]:
    t0 = time.perf_counter()
    orig_size = next(iter(raw_prev.values())).size
    affine_params = augmentor.sample_params(orig_size)
    proc_prev, _ = augmentor(raw_prev, affine_params)
    proc_curr, _ = augmentor(raw_curr, affine_params)
    affine_matrix = augmentor.get_affine_matrix(affine_params)
    prev_batch = stack_imgs_to_tensor(proc_prev, camera_channels, device)
    curr_batch = stack_imgs_to_tensor(proc_curr, camera_channels, device)
    sync_cuda_if_available()
    augment_ms = (time.perf_counter() - t0) * 1000.0
    return NuscInputBundle(
        raw_prev=raw_prev,
        raw_curr=raw_curr,
        proc_prev=proc_prev,
        proc_curr=proc_curr,
        prev_batch=prev_batch,
        curr_batch=curr_batch,
        affine_matrix=affine_matrix,
        rgb_12=None,
    ), augment_ms


def load_nusc_ground_truth(sample_info: Dict[str, object], need_gt: bool):
    if not need_gt or sample_info.get("gt_map_path") is None:
        return None

    gt_range_path = resolve_nusc_path(sample_info["gt_map_path"]) / "range_image_curr.npy"
    gt_item = np.load(gt_range_path, allow_pickle=True).item()
    gt_scale_map = np.asarray(gt_item["scale"], dtype=np.float32)
    gt_risk_map = np.asarray(gt_item["risk_score"], dtype=np.float32)
    valid_mask = (gt_scale_map > 0.3) & (gt_scale_map < 3.0)
    return {
        "scale": gt_scale_map,
        "risk": gt_risk_map,
        "valid_mask": valid_mask,
    }


def build_sensor_metas_batch(
    sample_info: Dict[str, object],
    affine_matrix: np.ndarray,
    device: torch.device,
    dataset_key: str,
):
    sensor_metas = {
        "prev": build_frame_sensor_metas(sample_info, dataset_key, "prev", affine_matrix, idx=None),
        "curr": build_frame_sensor_metas(sample_info, dataset_key, "curr", affine_matrix, idx=None),
    }
    sensor_metas = tensorize_sensor_metas(sensor_metas)
    sensor_metas = _batchify_sensor_metas(sensor_metas)
    return _move_sensor_metas_to_device(sensor_metas, device)


def run_online_depth_and_mapping(
    sample_info: Dict[str, object],
    inputs: NuscInputBundle,
    camera_channels: List[str],
    pool: ThreadPoolExecutor,
    depth_backend,
    runtime: TestRuntime,
    device: torch.device,
    dataset_key: str,
):
    sync_cuda_if_available()
    t0 = time.perf_counter()
    depth_list_12 = depth_backend.infer_depth_pair(
        inputs,
        camera_channels,
        num_preprocess_workers=args.num_depth_preprocess_workers,
    )
    prev_depth_map = {ch: depth for ch, depth in zip(camera_channels, depth_list_12[:len(camera_channels)])}
    curr_depth_map = {ch: depth for ch, depth in zip(camera_channels, depth_list_12[len(camera_channels):])}
    prev_depth_batch = None
    curr_depth_batch = None
    if args.rvt_depth_guided_sampling:
        prev_depth_batch = stack_depth_maps_to_device_batch(prev_depth_map, camera_channels, device)
        curr_depth_batch = stack_depth_maps_to_device_batch(curr_depth_map, camera_channels, device)
    sync_cuda_if_available()
    depth_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    prev_camera_futs = [
        pool.submit(
            build_camera_mapping_fast,
            sample_info,
            dataset_key,
            "prev",
            prev_depth_map[ch],
            inputs.affine_matrix,
            ch,
            cam_idx,
            args.mapping_pixel_stride,
            True,
        )
        for cam_idx, ch in enumerate(camera_channels)
    ]
    curr_camera_futs = [
        pool.submit(
            build_camera_mapping_fast,
            sample_info,
            dataset_key,
            "curr",
            curr_depth_map[ch],
            inputs.affine_matrix,
            ch,
            cam_idx,
            args.mapping_pixel_stride,
            True,
        )
        for cam_idx, ch in enumerate(camera_channels)
    ]

    prev_camera_results = []
    curr_camera_results = []
    transform_setup_samples = []
    camera_backproject_samples = []
    for future in prev_camera_futs:
        points, pix, timing = future.result()
        prev_camera_results.append((points, pix))
        transform_setup_samples.append(timing["transform_setup_ms"])
        camera_backproject_samples.append(timing["camera_backproject_ms"])
    for future in curr_camera_futs:
        points, pix, timing = future.result()
        curr_camera_results.append((points, pix))
        transform_setup_samples.append(timing["transform_setup_ms"])
        camera_backproject_samples.append(timing["camera_backproject_ms"])

    mapping_detail = init_mapping_detail_totals()
    mapping_detail["transform_setup_ms"] = max(transform_setup_samples) if transform_setup_samples else 0.0
    mapping_detail["camera_backproject_ms"] = max(camera_backproject_samples) if camera_backproject_samples else 0.0

    prev_finalize_future = pool.submit(
        finalize_frame_mapping_fast,
        prev_camera_results,
        runtime.range_h,
        runtime.range_w,
        True,
        runtime.mapping_range_backend,
    )
    curr_finalize_future = pool.submit(
        finalize_frame_mapping_fast,
        curr_camera_results,
        runtime.range_h,
        runtime.range_w,
        True,
        runtime.mapping_range_backend,
    )
    _, proj_pix_prev, prev_finalize_timing = prev_finalize_future.result()
    _, proj_pix_curr, curr_finalize_timing = curr_finalize_future.result()

    mapping_detail["range_project_ms"] = max(
        prev_finalize_timing["range_project_ms"],
        curr_finalize_timing["range_project_ms"],
    )
    mapping_detail["hole_fill_ms"] = max(
        prev_finalize_timing["hole_fill_ms"],
        curr_finalize_timing["hole_fill_ms"],
    )

    proj_pix_prev_batch = to_device_batch(proj_pix_prev, device, dtype=np.int64)
    proj_pix_curr_batch = to_device_batch(proj_pix_curr, device, dtype=np.int64)
    sensor_metas = build_sensor_metas_batch(sample_info, inputs.affine_matrix, device, dataset_key=dataset_key)
    sync_cuda_if_available()
    mapping_ms = (time.perf_counter() - t1) * 1000.0
    return (
        proj_pix_prev_batch,
        proj_pix_curr_batch,
        prev_depth_batch,
        curr_depth_batch,
        sensor_metas,
        mapping_detail,
        depth_ms,
        mapping_ms,
    )


def run_nusc_model_forward(
    model: torch.nn.Module,
    inputs: NuscInputBundle,
    proj_pix_prev_batch: torch.Tensor,
    proj_pix_curr_batch: torch.Tensor,
    prev_depth_batch: Optional[torch.Tensor],
    curr_depth_batch: Optional[torch.Tensor],
    sensor_metas,
    runtime: TestRuntime,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    if args.empty_cache_before_model and torch.cuda.is_available():
        torch.cuda.empty_cache()
    sync_cuda_if_available()
    t0 = time.perf_counter()
    model_kwargs = dict(
        img_prev=inputs.prev_batch,
        img_curr=inputs.curr_batch,
        depth_prev=prev_depth_batch,
        depth_curr=curr_depth_batch,
        proj_pix_prev=proj_pix_prev_batch,
        proj_pix_curr=proj_pix_curr_batch,
        sensor_metas=sensor_metas,
        attn_type=args.attn_type,
        attn_splits_list=args.attn_splits_list,
        corr_radius_list=args.corr_radius_list,
        prop_radius_list=args.prop_radius_list,
        num_reg_refine=args.num_reg_refine,
        scale_only=args.scale_only,
    )
    with torch.inference_mode():
        if runtime.model_cuda_graph_runner is not None:
            scale_pred, risk_pred = runtime.model_cuda_graph_runner.run(**model_kwargs)
        else:
            with autocast_context(args.enable_model_amp, args.model_amp_dtype):
                scale_pred, risk_pred = model.forward(**model_kwargs)
    sync_cuda_if_available()
    return scale_pred, risk_pred, (time.perf_counter() - t0) * 1000.0


def materialize_prediction_arrays(
    scale_pred: torch.Tensor,
    risk_pred: Optional[torch.Tensor],
    gt_bundle,
    need_pred_arrays: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not need_pred_arrays:
        return None, None

    scale_prediction_array = scale_pred[0].squeeze(0).cpu().numpy()
    risk_prediction_array = None
    if risk_pred is not None:
        risk_prediction_array = risk_pred[0].squeeze(0).cpu().numpy()
    if gt_bundle is not None:
        scale_prediction_array = align_prediction_to_shape(scale_prediction_array, gt_bundle["scale"].shape)
        if risk_prediction_array is not None:
            risk_prediction_array = align_prediction_to_shape(risk_prediction_array, gt_bundle["risk"].shape)
    return scale_prediction_array, risk_prediction_array


def update_nusc_metrics(
    sample_info: Dict[str, object],
    gt_bundle,
    scale_prediction_array: Optional[np.ndarray],
    risk_prediction_array: Optional[np.ndarray],
    scale_metrics_acc: Optional[ScaleMetricsAccumulator],
    orientation_metrics_acc: Optional[Table3MetricsAccumulator],
    scene_mid_err_acc: Optional[SceneMidErrAccumulator],
) -> float:
    t0 = time.perf_counter()
    if gt_bundle is not None and scale_prediction_array is not None:
        delta_t = float(sample_info.get("time_diff_cam_us", 500000)) / 1e6
        if scale_metrics_acc is not None:
            scale_metrics_acc.gt_sample_count += 1
            scale_metrics_acc.update(
                scale_pred=scale_prediction_array,
                scale_gt=gt_bundle["scale"],
                valid_mask=gt_bundle["valid_mask"],
                delta_t=delta_t,
            )
        if orientation_metrics_acc is not None and risk_prediction_array is not None:
            orientation_metrics_acc.gt_sample_count += 1
            orientation_metrics_acc.update(
                scale_pred=scale_prediction_array,
                scale_gt=gt_bundle["scale"],
                orientation_pred=risk_prediction_array,
                orientation_gt=gt_bundle["risk"],
                valid_mask=gt_bundle["valid_mask"],
                delta_t=delta_t,
            )
            if scene_mid_err_acc is not None:
                scene_mid_err_acc.update(
                    scene_id=sample_info["scene_indice"],
                    scale_pred=scale_prediction_array,
                    scale_gt=gt_bundle["scale"],
                    valid_mask=gt_bundle["valid_mask"],
                )
    return (time.perf_counter() - t0) * 1000.0


def save_nusc_outputs(
    idx: int,
    scene_indice: str,
    runtime: TestRuntime,
    inputs: NuscInputBundle,
    gt_bundle,
    scale_prediction_array: Optional[np.ndarray],
    risk_prediction_array: Optional[np.ndarray],
):
    t0 = time.perf_counter()
    if args.save_visualizations:
        if args.scale_only:
            pred_scale_rgb = make_scale_analysis_rgb(scale_prediction_array)
            save_rgb_image(os.path.join(runtime.output_dir, f"pred_scale_analysis_{idx}.png"), pred_scale_rgb)
        else:
            save_pred_visualizations(
                output_dir=runtime.output_dir,
                idx=idx,
                scale_map=scale_prediction_array,
                risk_map=risk_prediction_array,
            )
        if gt_bundle is not None:
            if args.scale_only:
                gt_scale_rgb = make_scale_analysis_rgb(gt_bundle["scale"], valid_mask=gt_bundle["valid_mask"])
                save_rgb_image(os.path.join(runtime.output_dir, f"gt_scale_analysis_{idx}.png"), gt_scale_rgb)
            else:
                save_gt_analysis_visualizations(
                    output_dir=runtime.output_dir,
                    idx=idx,
                    gt_scale_map=gt_bundle["scale"],
                    gt_risk_map=gt_bundle["risk"],
                    gt_valid_mask=gt_bundle["valid_mask"],
                )

        concat_prev = np.concatenate([inputs.proc_prev[ch] for ch in runtime.camera_channels], axis=1)
        concat_prev = concat_prev.astype(np.uint8)
        Image.fromarray(concat_prev).save(os.path.join(runtime.output_dir, f"concat_prev_{idx}.png"))
    vis_ms = (time.perf_counter() - t0) * 1000.0

    if args.save_pred_npy:
        pred_npy_subdir = os.path.join(args.pred_npy_dir, f"pred_npy_{os.path.basename(runtime.output_dir)}")
        os.makedirs(pred_npy_subdir, exist_ok=True)
        pred_data = {"scale_pred": scale_prediction_array}
        if not args.scale_only:
            pred_data["risk_pred"] = risk_prediction_array
        np.save(os.path.join(pred_npy_subdir, f"scene_{scene_indice}_pred_{idx}.npy"), pred_data)

    return vis_ms


def save_sjtu_outputs(
    idx: int,
    runtime: TestRuntime,
    inputs: NuscInputBundle,
    scale_prediction_array: Optional[np.ndarray],
    risk_prediction_array: Optional[np.ndarray],
):
    t0 = time.perf_counter()
    if args.save_visualizations:
        if args.scale_only:
            pred_scale_rgb = make_scale_analysis_rgb(scale_prediction_array)
            save_rgb_image(os.path.join(runtime.output_dir, f"pred_scale_analysis_{idx}.png"), pred_scale_rgb)
        else:
            save_pred_visualizations(
                output_dir=runtime.output_dir,
                idx=idx,
                scale_map=scale_prediction_array,
                risk_map=risk_prediction_array,
            )

        concat_prev = np.concatenate([inputs.proc_prev[ch] for ch in runtime.camera_channels], axis=1)
        concat_prev = concat_prev.astype(np.uint8)
        Image.fromarray(concat_prev).save(os.path.join(runtime.output_dir, f"concat_prev_{idx}.png"))

    if args.save_pred_npy:
        pred_npy_subdir = os.path.join(args.pred_npy_dir, f"pred_npy_{os.path.basename(runtime.output_dir)}")
        os.makedirs(pred_npy_subdir, exist_ok=True)
        pred_data = {"scale_pred": scale_prediction_array}
        if not args.scale_only:
            pred_data["risk_pred"] = risk_prediction_array
        np.save(os.path.join(pred_npy_subdir, f"pred_{idx}.npy"), pred_data)

    return (time.perf_counter() - t0) * 1000.0


def run_nusc_test(
    model: torch.nn.Module,
    depth_backend,
    test_entries: list,
    runtime: TestRuntime,
    scale_metrics_acc: Optional[ScaleMetricsAccumulator],
    orientation_metrics_acc: Optional[Table3MetricsAccumulator],
    scene_mid_err_acc: Optional[SceneMidErrAccumulator],
):
    timing_totals = init_timing_totals()
    timing_records = init_timing_records()
    mapping_detail_totals = init_mapping_detail_totals()
    mapping_detail_records = init_mapping_detail_records()
    image_loader = load_nusc_image_np if args.enable_fast_nusc_preprocess else load_nusc_image

    with ThreadPoolExecutor(max_workers=max(1, args.num_io_workers)) as pool:
        prefetch_pool = None
        prefetched_handles = None
        if args.enable_nusc_io_prefetch and len(test_entries) > 0:
            prefetch_pool = ThreadPoolExecutor(max_workers=max(1, args.num_io_workers))
            prefetched_handles = submit_nusc_image_prefetch(
                prefetch_pool,
                image_loader,
                0,
                test_entries,
                runtime.camera_channels,
            )
        try:
            for idx in tqdm(range(len(test_entries)), desc="Processing surround view images"):
                sample_info = test_entries[idx]
                sample_timing = init_timing_totals()

                raw_prev, raw_curr, prefetched_handles, sample_timing["io_ms"] = load_nusc_sample_images(
                    idx,
                    test_entries,
                    runtime.camera_channels,
                    pool,
                    image_loader,
                    prefetch_pool,
                    prefetched_handles,
                )
                inputs, sample_timing["augment_ms"] = prepare_nusc_input_bundle(
                    raw_prev,
                    raw_curr,
                    runtime.camera_channels,
                    pool,
                    runtime,
                    device,
                )
                gt_bundle = load_nusc_ground_truth(
                    sample_info,
                    need_gt=(
                        args.save_visualizations
                        or (scale_metrics_acc is not None)
                        or (orientation_metrics_acc is not None)
                    ),
                )
                (
                    proj_pix_prev_batch,
                    proj_pix_curr_batch,
                    prev_depth_batch,
                    curr_depth_batch,
                    sensor_metas,
                    sample_mapping_detail,
                    sample_timing["depth_ms"],
                    sample_timing["mapping_ms"],
                ) = run_online_depth_and_mapping(
                    sample_info,
                    inputs,
                    runtime.camera_channels,
                    pool,
                    depth_backend,
                    runtime,
                    device,
                    dataset_key="nusc",
                )
                scale_pred, risk_pred, sample_timing["model_ms"] = run_nusc_model_forward(
                    model,
                    inputs,
                    proj_pix_prev_batch,
                    proj_pix_curr_batch,
                    prev_depth_batch,
                    curr_depth_batch,
                    sensor_metas,
                    runtime,
                )
                scale_prediction_array, risk_prediction_array = materialize_prediction_arrays(
                    scale_pred,
                    risk_pred,
                    gt_bundle,
                    need_pred_arrays=(
                        args.save_visualizations
                        or args.save_pred_npy
                        or (scale_metrics_acc is not None)
                        or (orientation_metrics_acc is not None)
                    ),
                )
                sample_timing["metrics_ms"] = update_nusc_metrics(
                    sample_info,
                    gt_bundle,
                    scale_prediction_array,
                    risk_prediction_array,
                    scale_metrics_acc,
                    orientation_metrics_acc,
                    scene_mid_err_acc,
                )
                sample_timing["vis_ms"] = save_nusc_outputs(
                    idx,
                    sample_info["scene_indice"],
                    runtime,
                    inputs,
                    gt_bundle,
                    scale_prediction_array,
                    risk_prediction_array,
                )

                accumulate_timing(timing_totals, sample_timing)
                record_timing_sample(timing_records, sample_timing)
                accumulate_mapping_detail(mapping_detail_totals, sample_mapping_detail)
                record_mapping_detail_sample(mapping_detail_records, sample_mapping_detail)
                if args.count_time:
                    total_ms = sum(sample_timing[key] for key in TIMING_KEYS)
                    print(
                        f"[Timing][{idx}] io={sample_timing['io_ms']:.2f}ms "
                        f"aug={sample_timing['augment_ms']:.2f}ms "
                        f"depth={sample_timing['depth_ms']:.2f}ms "
                        f"mapping={sample_timing['mapping_ms']:.2f}ms "
                        f"model={sample_timing['model_ms']:.2f}ms "
                        f"vis={sample_timing['vis_ms']:.2f}ms "
                        f"metrics={sample_timing['metrics_ms']:.2f}ms "
                        f"total={total_ms:.2f}ms"
                    )
                    print(
                        f"[MappingDetail][{idx}] "
                        f"setup={sample_mapping_detail['transform_setup_ms']:.2f}ms "
                        f"backproject={sample_mapping_detail['camera_backproject_ms']:.2f}ms "
                        f"range={sample_mapping_detail['range_project_ms']:.2f}ms "
                        f"hole_fill={sample_mapping_detail['hole_fill_ms']:.2f}ms "
                        f"backend={runtime.mapping_range_backend}"
                    )

                del inputs.prev_batch, inputs.curr_batch, scale_pred
                if risk_pred is not None:
                    del risk_pred
        finally:
            if prefetch_pool is not None:
                prefetch_pool.shutdown(wait=True)

    print_timing_summary(
        "nuScenes",
        timing_totals,
        timing_records,
        len(test_entries),
        runtime.output_dir,
        mapping_detail_totals,
        mapping_detail_records,
        runtime.mapping_range_backend,
    )


def run_sjtu_test(
    model: torch.nn.Module,
    depth_backend,
    test_entries: list,
    runtime: TestRuntime,
):
    timing_totals = init_timing_totals()
    timing_records = init_timing_records()
    mapping_detail_totals = init_mapping_detail_totals()
    mapping_detail_records = init_mapping_detail_records()

    with ThreadPoolExecutor(max_workers=max(1, args.num_io_workers)) as pool:
        for idx in tqdm(range(len(test_entries)), desc="Processing SJTU scene 18"):
            sample_timing = init_timing_totals()
            sample_info, raw_prev, raw_curr, sample_timing["io_ms"] = load_sjtu_sample_images(
                idx,
                test_entries,
                runtime.camera_channels,
                pool,
            )
            inputs, sample_timing["augment_ms"] = prepare_sjtu_input_bundle(
                raw_prev,
                raw_curr,
                runtime.camera_channels,
                device,
            )
            (
                proj_pix_prev_batch,
                proj_pix_curr_batch,
                prev_depth_batch,
                curr_depth_batch,
                sensor_metas,
                sample_mapping_detail,
                sample_timing["depth_ms"],
                sample_timing["mapping_ms"],
            ) = run_online_depth_and_mapping(
                sample_info,
                inputs,
                runtime.camera_channels,
                pool,
                depth_backend,
                runtime,
                device,
                dataset_key="sjtu",
            )
            scale_pred, risk_pred, sample_timing["model_ms"] = run_nusc_model_forward(
                model,
                inputs,
                proj_pix_prev_batch,
                proj_pix_curr_batch,
                prev_depth_batch,
                curr_depth_batch,
                sensor_metas,
                runtime,
            )
            scale_prediction_array, risk_prediction_array = materialize_prediction_arrays(
                scale_pred,
                risk_pred,
                gt_bundle=None,
                need_pred_arrays=args.save_visualizations or args.save_pred_npy,
            )
            sample_timing["metrics_ms"] = 0.0
            sample_timing["vis_ms"] = save_sjtu_outputs(
                idx,
                runtime,
                inputs,
                scale_prediction_array,
                risk_prediction_array,
            )

            accumulate_timing(timing_totals, sample_timing)
            record_timing_sample(timing_records, sample_timing)
            accumulate_mapping_detail(mapping_detail_totals, sample_mapping_detail)
            record_mapping_detail_sample(mapping_detail_records, sample_mapping_detail)
            if args.count_time:
                total_ms = sum(sample_timing[key] for key in TIMING_KEYS)
                print(
                    f"[Timing][SJTU][{idx}] io={sample_timing['io_ms']:.2f}ms "
                    f"aug={sample_timing['augment_ms']:.2f}ms "
                    f"depth={sample_timing['depth_ms']:.2f}ms "
                    f"mapping={sample_timing['mapping_ms']:.2f}ms "
                    f"model={sample_timing['model_ms']:.2f}ms "
                    f"vis={sample_timing['vis_ms']:.2f}ms "
                    f"metrics={sample_timing['metrics_ms']:.2f}ms "
                    f"total={total_ms:.2f}ms"
                )
                print(
                    f"[MappingDetail][SJTU][{idx}] "
                    f"setup={sample_mapping_detail['transform_setup_ms']:.2f}ms "
                    f"backproject={sample_mapping_detail['camera_backproject_ms']:.2f}ms "
                    f"range={sample_mapping_detail['range_project_ms']:.2f}ms "
                    f"hole_fill={sample_mapping_detail['hole_fill_ms']:.2f}ms "
                    f"backend={runtime.mapping_range_backend}"
                )

            del inputs.prev_batch, inputs.curr_batch, scale_pred
            if risk_pred is not None:
                del risk_pred

    print_timing_summary(
        "SJTU scene_18 (500ms)",
        timing_totals,
        timing_records,
        len(test_entries),
        runtime.output_dir,
        mapping_detail_totals,
        mapping_detail_records,
        runtime.mapping_range_backend,
    )


def finalize_metrics_and_reports(
    args,
    runtime: TestRuntime,
    scale_metrics_acc: Optional[ScaleMetricsAccumulator],
    orientation_metrics_acc: Optional[Table3MetricsAccumulator],
    scene_mid_err_acc: Optional[SceneMidErrAccumulator],
):
    if args.scale_only:
        if scale_metrics_acc is None:
            return
        if scale_metrics_acc.gt_sample_count == 0 or scale_metrics_acc.valid_pixel_count == 0:
            print("[INFO] No GT pixels available for scale metrics. Skipping metric summary and JSON export.")
            return

        metrics_payload = scale_metrics_acc.finalize(args.resume or "", runtime.split_name)
        print_scale_metrics_summary(metrics_payload)
        metrics_path = os.path.join(runtime.output_dir, "scale_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        print(f"[INFO] Saved scale metrics to {metrics_path}")
        return

    if orientation_metrics_acc is None:
        return
    if orientation_metrics_acc.gt_sample_count == 0 or orientation_metrics_acc.valid_pixel_count == 0:
        print("[INFO] No GT pixels available for scale & orientation metrics. Skipping metric summary and JSON export.")
        return

    metrics_payload = orientation_metrics_acc.finalize(args.resume or "", runtime.split_name)
    print_scale_orientation_metrics_summary(metrics_payload)
    metrics_path = os.path.join(runtime.output_dir, "scale_orientation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[INFO] Saved scale & orientation metrics to {metrics_path}")

    if scene_mid_err_acc is None:
        return

    scene_mid_payload = scene_mid_err_acc.finalize(args.resume or "", runtime.split_name)
    print_scene_mid_err_summary(scene_mid_payload)
    scene_mid_json_path = os.path.join(runtime.output_dir, "scene_mid_err.json")
    with open(scene_mid_json_path, "w", encoding="utf-8") as f:
        json.dump(scene_mid_payload, f, indent=2)
    scene_mid_md_path = os.path.join(runtime.output_dir, "scene_mid_err.md")
    with open(scene_mid_md_path, "w", encoding="utf-8") as f:
        f.write(build_scene_mid_err_markdown(scene_mid_payload) + "\n")
    print(f"[INFO] Saved scene-wise MiD Err to {scene_mid_json_path}")
    print(f"[INFO] Saved scene-wise MiD Err table to {scene_mid_md_path}")


def main():
    applied_profile_overrides = resolve_test_profile(args)
    sanitize_accel_flags(args)
    reset_dtype_audit()
    set_dtype_audit_enabled(args.dump_dtype_audit)
    raw_test_entries, split_name = load_test_entries(
        args.test_info_path,
        -1 if args.sjtu_test else args.max_test_samples,
    )
    if args.sjtu_test:
        test_entries = compose_sjtu_500ms_entries(raw_test_entries, args.max_test_samples)
        split_name = f"{split_name}_500ms"
        if args.scale_only and args.compute_scale_metrics:
            print("[SJTU] compute_scale_metrics was requested but will be disabled because no GT is available.")
        if (not args.scale_only) and (args.compute_scale_orientation_metrics or args.compute_scene_mid_err):
            print("[SJTU] orientation metrics / scene_mid_err were requested but will be disabled because scene_18 has no GT.")
        args.compute_scale_metrics = False
        args.compute_scale_orientation_metrics = False
        args.compute_scene_mid_err = False
    else:
        test_entries = raw_test_entries

    model, depth_backend, runtime = build_models_and_runtime(args, test_entries, split_name)
    scale_metrics_acc, orientation_metrics_acc, scene_mid_err_acc = create_metrics_accumulators(args, test_entries)

    print_runtime_summary(args, runtime, applied_profile_overrides)
    if args.sjtu_test:
        run_sjtu_test(model, depth_backend, test_entries, runtime)
    else:
        run_nusc_test(
            model,
            depth_backend,
            test_entries,
            runtime,
            scale_metrics_acc,
            orientation_metrics_acc,
            scene_mid_err_acc,
        )
        finalize_metrics_and_reports(
            args,
            runtime,
            scale_metrics_acc,
            orientation_metrics_acc,
            scene_mid_err_acc,
        )

    if args.dump_dtype_audit:
        dtype_audit_path = os.path.join(runtime.output_dir, "dtype_audit.json")
        dtype_payload = dump_dtype_audit(dtype_audit_path)
        print(f"[INFO] Saved dtype audit to {dtype_audit_path} ({dtype_payload['event_count']} events)")


if __name__ == "__main__":
    main()
