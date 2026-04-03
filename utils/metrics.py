import math

import torch
import torch.distributed as dist


SCALE_METRIC_KEYS = (
    'mid_error_sum',
    'valid_count',
    'err_1_count',
    'err_2_count',
    'err_5_count',
)

ORIENTATION_METRIC_KEYS = (
    'mae_sum',
    'acc_count',
    'valid_count',
    'tp',
    'fp',
    'fn',
)


def _final_prediction_map(prediction):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[-1]
    if prediction.dim() == 4 and prediction.shape[1] == 1:
        prediction = prediction.squeeze(1)
    return prediction


def scale_to_ttc(scale, delta_t, eps=1e-6):
    scale = _final_prediction_map(scale)
    return delta_t / (1.0 - scale + eps)


def init_scale_metric_sums():
    return {key: 0.0 for key in SCALE_METRIC_KEYS}


def init_orientation_metric_sums():
    return {key: 0.0 for key in ORIENTATION_METRIC_KEYS}


def add_metric_sums(total_sums, batch_sums):
    return {
        key: float(total_sums.get(key, 0.0)) + float(batch_sums.get(key, 0.0))
        for key in batch_sums
    }


def reduce_metric_sums(metric_sums, device):
    keys = tuple(metric_sums.keys())
    values = torch.tensor(
        [metric_sums[key] for key in keys],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        key: float(value.item())
        for key, value in zip(keys, values)
    }


def compute_scale_metric_sums(
    scale_pred,
    gt_scale_with_mask,
    *,
    delta_t=0.1,
    mid_scale=1e4,
    ttc_thresholds=(1.0, 2.0, 5.0),
    eps=1e-6,
):
    scale_pred = _final_prediction_map(scale_pred)
    gt_scale = gt_scale_with_mask[:, 0, :, :]
    valid_mask = gt_scale_with_mask[:, 1, :, :].bool()

    metric_sums = init_scale_metric_sums()
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return metric_sums

    pred_valid = scale_pred[valid_mask].clamp(min=eps)
    gt_valid = gt_scale[valid_mask].clamp(min=eps)
    log_error = (torch.log(pred_valid) - torch.log(gt_valid)).abs()

    metric_sums['mid_error_sum'] = float((log_error * mid_scale).sum().item())
    metric_sums['valid_count'] = float(valid_count)

    pred_ttc = scale_to_ttc(scale_pred, delta_t, eps=eps)
    gt_ttc = scale_to_ttc(gt_scale, delta_t, eps=eps)

    threshold_to_key = {
        1.0: 'err_1_count',
        2.0: 'err_2_count',
        5.0: 'err_5_count',
    }
    for threshold in ttc_thresholds:
        key = threshold_to_key.get(float(threshold))
        if key is None:
            raise ValueError(f'Unsupported TTC threshold: {threshold}')
        pred_label = (pred_ttc > 0.0) & (pred_ttc < threshold)
        gt_label = (gt_ttc > 0.0) & (gt_ttc < threshold)
        metric_sums[key] = float((pred_label[valid_mask] != gt_label[valid_mask]).sum().item())

    return metric_sums


def finalize_scale_metrics(metric_sums):
    valid_count = float(metric_sums.get('valid_count', 0.0))
    if valid_count <= 0.0:
        return {
            'mid_err': 0.0,
            'err_1': 0.0,
            'err_2': 0.0,
            'err_5': 0.0,
        }

    return {
        'mid_err': float(metric_sums['mid_error_sum']) / valid_count,
        'err_1': float(metric_sums['err_1_count']) / valid_count,
        'err_2': float(metric_sums['err_2_count']) / valid_count,
        'err_5': float(metric_sums['err_5_count']) / valid_count,
    }


def compute_orientation_metric_sums(
    scale_pred,
    orientation_pred,
    gt_scale_with_mask,
    gt_orientation_with_mask,
    *,
    delta_t=0.1,
    tau_theta=math.pi / 12.0,
    tau_t=2.0,
    eps=1e-6,
):
    scale_pred = _final_prediction_map(scale_pred)
    orientation_pred = _final_prediction_map(orientation_pred)

    gt_scale = gt_scale_with_mask[:, 0, :, :]
    scale_mask = gt_scale_with_mask[:, 1, :, :].bool()
    gt_orientation = gt_orientation_with_mask[:, 0, :, :]
    orientation_mask = gt_orientation_with_mask[:, 1, :, :].bool()
    valid_mask = scale_mask & orientation_mask

    metric_sums = init_orientation_metric_sums()
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return metric_sums

    abs_error = (orientation_pred[valid_mask] - gt_orientation[valid_mask]).abs()
    metric_sums['mae_sum'] = float(abs_error.sum().item())
    metric_sums['acc_count'] = float((abs_error <= tau_theta).sum().item())
    metric_sums['valid_count'] = float(valid_count)

    pred_ttc = scale_to_ttc(scale_pred, delta_t, eps=eps)
    gt_ttc = scale_to_ttc(gt_scale, delta_t, eps=eps)

    pred_high_risk = (
        valid_mask
        & (pred_ttc > 0.0)
        & (pred_ttc < tau_t)
        & (orientation_pred <= tau_theta)
    )
    gt_high_risk = (
        valid_mask
        & (gt_ttc > 0.0)
        & (gt_ttc < tau_t)
        & (gt_orientation <= tau_theta)
    )

    metric_sums['tp'] = float((pred_high_risk & gt_high_risk).sum().item())
    metric_sums['fp'] = float((pred_high_risk & ~gt_high_risk).sum().item())
    metric_sums['fn'] = float((~pred_high_risk & gt_high_risk).sum().item())

    return metric_sums


def finalize_orientation_metrics(metric_sums, *, beta=2.0):
    valid_count = float(metric_sums.get('valid_count', 0.0))
    if valid_count <= 0.0:
        return {
            'mae_theta': 0.0,
            'acc_tau_theta': 0.0,
            'hr_precision': 0.0,
            'hr_recall': 0.0,
            'hr_f2': 0.0,
        }

    tp = float(metric_sums.get('tp', 0.0))
    fp = float(metric_sums.get('fp', 0.0))
    fn = float(metric_sums.get('fn', 0.0))

    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0

    beta_sq = beta ** 2
    if precision == 0.0 and recall == 0.0:
        f_beta = 0.0
    else:
        f_beta = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)

    return {
        'mae_theta': float(metric_sums['mae_sum']) / valid_count,
        'acc_tau_theta': float(metric_sums['acc_count']) / valid_count,
        'hr_precision': precision,
        'hr_recall': recall,
        'hr_f2': f_beta,
    }
