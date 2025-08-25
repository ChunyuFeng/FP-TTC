# utils/checkpoint.py
import torch

_ALLOWED_PREFIXES_SCALE = (
    "cnet.",          # CNNEncoder 共享特征
    "featnet.",       # FeatureNet 共享特征
    "corrnet.",       # FlowNet / correlation 共享特征
    "conv_corr.",     # scale 分支使用的 corr 编码器
    "scale_net.",     #  ScaleNet
)
# 注意：不会加载 risk 分支（conv_corr_risk, risk_net）

def _unwrap_state_dict(ckpt):
    """从多种 checkpoint 结构里取到 state_dict"""
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights", "params", "state_dict_ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    # 直接就是 state_dict
    return ckpt

def _strip_prefix(state_dict, prefix="module."):
    return { (k[len(prefix):] if k.startswith(prefix) else k): v
             for k, v in state_dict.items() }

def _filter_to_scale_branch(state_dict, model_state):
    """仅保留 scale 分支与共享子网，且形状匹配"""
    keep = {}
    for k, v in state_dict.items():
        if not k.startswith(_ALLOWED_PREFIXES_SCALE):
            continue
        if k in model_state and model_state[k].shape == v.shape:
            keep[k] = v
    return keep

def load_scale_only_weights(model, ckpt_path, map_location="cpu", verbose=True):
    """
    只把 scale 分支 + 共享骨干的参数加载进当前模型。
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    sd = _unwrap_state_dict(ckpt)
    sd = _strip_prefix(sd, "module.")  # 去掉 DDP 前缀

    model_sd = model.state_dict()
    filtered = _filter_to_scale_branch(sd, model_sd)

    # 实际加载
    missing_before = set(model_sd.keys()) - set(filtered.keys())
    res = model.load_state_dict(filtered, strict=False)

    if verbose:
        loaded_keys = set(filtered.keys())
        print(f"[scale-pretrain] loaded {len(loaded_keys)} params from: {ckpt_path}")
        if hasattr(res, "missing_keys"):
            # 这些 missing_keys 不一定是问题，因为我们只想加载一部分
            miss = [k for k in res.missing_keys if k.startswith(_ALLOWED_PREFIXES_SCALE)]
            if miss:
                print(f"[scale-pretrain] missing (wanted but not found in ckpt): {len(miss)}")
                for k in miss[:20]:
                    print("  -", k)
                if len(miss) > 20:
                    print("  ...")
        if hasattr(res, "unexpected_keys") and res.unexpected_keys:
            # 这些是 ckpt 里有但我们没用到的（或名字不匹配/形状不匹配被过滤）
            unexp = [k for k in res.unexpected_keys if k.startswith(_ALLOWED_PREFIXES_SCALE)]
            if unexp:
                print(f"[scale-pretrain] unexpected (in ckpt but unused): {len(unexp)}")
                for k in unexp[:20]:
                    print("  -", k)
                if len(unexp) > 20:
                    print("  ...")

    return res  # IncompatibleKeys
