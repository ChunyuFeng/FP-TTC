# Test.py Simplification Notes

## Summary

On April 15, 2026, the repo's main [test.py](/home/chunyu/WorkSpace/BugStudio/FP-TTC-hardproj-init-local-rvt-v1/test.py) was simplified to keep the core RVT test pipeline easy to read and maintain.

The full-featured version was preserved as:

- [test_full_featured_backup_2026_04_15.py](/home/chunyu/WorkSpace/BugStudio/FP-TTC-hardproj-init-local-rvt-v1/test_full_featured_backup_2026_04_15.py)

Use the backup script if you need the acceleration experiment interfaces again.

## What The Simplified test.py Keeps

- `custom / fastest / accurate` test profiles
- `scale_only=true` scale-only inference and metrics
- `scale_only=false` scale + orientation inference and metrics
- `scene_mid_err` support
- nuScenes test path
- SJTU test path
- RVT-specific `sensor_metas` construction and forward path
- optional `rvt_depth_guided_sampling`
- visualization export
- prediction `.npy` export
- timing summary and mapping breakdown

## What Was Removed From The Main test.py

- `depth_backend` abstraction
- `onnxruntime` depth backend
- `tensorrt` depth backend
- `torch.compile` depth/model hooks
- CUDA Graph capture hooks
- dtype audit logging and JSON dump
- geometry cache CLI hook
- backend-specific sanitization logic

The simplified script now always uses the built-in PyTorch `DepthAnythingV2` path followed by the RVT `FpTTC` forward path.

## Current Mental Model

The main script is now intentionally linear:

1. Load sample images
2. Build the per-sample input bundle
3. Run DepthAnything in PyTorch
4. Build online range-view mapping and `sensor_metas`
5. Run RVT `FpTTC.forward(...)`
6. Materialize predictions when needed
7. Compute metrics, save outputs, and accumulate timing

## Practical Guidance

- If your goal is regular evaluation, use the simplified [test.py](/home/chunyu/WorkSpace/BugStudio/FP-TTC-hardproj-init-local-rvt-v1/test.py).
- If your goal is speed backend experiments, use the backup [test_full_featured_backup_2026_04_15.py](/home/chunyu/WorkSpace/BugStudio/FP-TTC-hardproj-init-local-rvt-v1/test_full_featured_backup_2026_04_15.py).
- Treat the backup file as the experimental branch and the main `test.py` as the stable readable branch.
