import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Open3D viewer for scene-flow point clouds."
    )
    parser.add_argument(
        "--scene_flow_root",
        type=Path,
        default=Path("/mnt/data/fpttc_ground_truth/0_scene_flow_150_keyframes_float32/scene_flow_key_frames"),
        help="Directory that contains scene-flow sample folders.",
    )
    parser.add_argument(
        "--sample_folder",
        type=str,
        default=None,
        help="Sample folder name or absolute path, e.g. scene_24_...1531886087898421.",
    )
    parser.add_argument(
        "--scene_index",
        type=int,
        default=None,
        help="Optional scene index used together with --pair_index.",
    )
    parser.add_argument(
        "--pair_index",
        type=int,
        default=None,
        help="Optional zero-based pair index within one scene, used together with --scene_index.",
    )
    parser.add_argument(
        "--show",
        type=str,
        default="prev",
        choices=["prev", "curr", "both"],
        help="Initial geometry to show.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=300000,
        help="Randomly sample at most this many points per cloud for interactive viewing.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Open3D render point size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24,
        help="Random seed for point sampling.",
    )
    parser.add_argument(
        "--show_axes",
        action="store_true",
        help="Show a coordinate frame at the origin.",
    )
    return parser.parse_args()


def resolve_sample_dir(scene_flow_root: Path, sample_folder: Optional[str], scene_index: Optional[int],
                       pair_index: Optional[int]) -> Path:
    if sample_folder:
        sample_path = Path(sample_folder)
        if sample_path.is_absolute():
            if not sample_path.is_dir():
                raise FileNotFoundError(f"Sample directory does not exist: {sample_path}")
            return sample_path

        candidate = scene_flow_root / sample_folder
        if not candidate.is_dir():
            raise FileNotFoundError(f"Sample directory does not exist: {candidate}")
        return candidate

    if scene_index is None or pair_index is None:
        raise ValueError("Provide either --sample_folder or both --scene_index and --pair_index.")

    prefix = f"scene_{scene_index}_"
    scene_dirs = sorted(path for path in scene_flow_root.iterdir() if path.is_dir() and path.name.startswith(prefix))
    if not scene_dirs:
        raise FileNotFoundError(f"No scene-flow folders found for scene index {scene_index} under {scene_flow_root}")
    if pair_index < 0 or pair_index >= len(scene_dirs):
        raise IndexError(
            f"--pair_index={pair_index} is out of range for scene {scene_index}. "
            f"Available range: [0, {len(scene_dirs) - 1}]"
        )
    return scene_dirs[pair_index]


def maybe_sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def build_point_cloud(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color(color)
    return pcd


class ViewerState:
    def __init__(self, prev_pcd: o3d.geometry.PointCloud, curr_pcd: o3d.geometry.PointCloud,
                 axes: Optional[o3d.geometry.TriangleMesh], initial_mode: str) -> None:
        self.prev_pcd = prev_pcd
        self.curr_pcd = curr_pcd
        self.axes = axes
        self.mode = initial_mode

    def apply(self, vis: o3d.visualization.VisualizerWithKeyCallback, reset_viewpoint: bool = False) -> None:
        vis.clear_geometries()
        added_geometry = False

        def _add(geometry):
            nonlocal added_geometry
            vis.add_geometry(geometry, reset_bounding_box=(reset_viewpoint and not added_geometry))
            added_geometry = True

        if self.mode in ("prev", "both"):
            _add(self.prev_pcd)
        if self.mode in ("curr", "both"):
            _add(self.curr_pcd)
        if self.axes is not None:
            vis.add_geometry(self.axes, reset_bounding_box=False)

        if reset_viewpoint and added_geometry:
            view_control = vis.get_view_control()
            view_control.set_zoom(0.45)


def register_mode_callback(state: ViewerState, mode: str):
    def _callback(vis):
        state.mode = mode
        state.apply(vis, reset_viewpoint=True)
        return False

    return _callback


def main():
    # Avoid inheriting Qt plugin overrides from other tools such as cv2/mmcv.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_PLUGIN_PATH", None)

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sample_dir = resolve_sample_dir(args.scene_flow_root, args.sample_folder, args.scene_index, args.pair_index)
    prev_points = np.load(sample_dir / "pc_prev.npy")
    curr_points = np.load(sample_dir / "pc_curr.npy")

    prev_points = maybe_sample_points(prev_points, args.max_points, rng)
    curr_points = maybe_sample_points(curr_points, args.max_points, rng)

    prev_pcd = build_point_cloud(prev_points, (1.0, 0.45, 0.1))
    curr_pcd = build_point_cloud(curr_points, (0.1, 0.75, 1.0))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0) if args.show_axes else None

    state = ViewerState(prev_pcd, curr_pcd, axes, args.show)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Scene Flow Viewer - {sample_dir.name}", width=1600, height=900)
    state.apply(vis, reset_viewpoint=True)

    render_option = vis.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.background_color = np.asarray([0.04, 0.04, 0.04], dtype=np.float32)

    vis.register_key_callback(ord("1"), register_mode_callback(state, "prev"))
    vis.register_key_callback(ord("2"), register_mode_callback(state, "curr"))
    vis.register_key_callback(ord("3"), register_mode_callback(state, "both"))

    print(f"Viewing sample: {sample_dir}")
    print(f"Loaded prev={prev_points.shape[0]} points, curr={curr_points.shape[0]} points")
    print("Controls:")
    print("  1: show pc_prev only")
    print("  2: show pc_curr only")
    print("  3: show both")
    print("  Mouse/trackpad: Open3D default orbit/pan/zoom")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
