#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LABELS = {
    'zero': 'Zero-init',
    'concatfeat': 'ConcatFeat-init',
    'gvb': 'GVB-init',
}

COLORS = {
    'Zero-init': '#0B63CE',
    'ConcatFeat-init': '#FF7A00',
    'GVB-init': '#168A16',
}

ORDER = ['Zero-init', 'ConcatFeat-init', 'GVB-init']


def parse_run_arg(value):
    if '=' in value:
        label, path = value.split('=', 1)
        return label, Path(path)
    return None, Path(value)


def normalize_label(label):
    return LABELS.get(label, label)


def to_float(value):
    if value is None or value == '':
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def read_manifest(run_dir):
    path = run_dir / 'train_manifest.json'
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def read_rows(run_dir, label_override=None):
    path = run_dir / 'epoch_metrics.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing epoch_metrics.csv: {path}')

    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row['epoch'] = int(float(row['epoch']))
            rows.append(row)
    rows.sort(key=lambda item: item['epoch'])

    init_key = ''
    for row in rows:
        init_key = row.get('rvt_query_init') or init_key
    label = normalize_label(label_override) if label_override else LABELS.get(init_key, init_key or run_dir.name)

    return {
        'label': label,
        'run_dir': run_dir,
        'rows': rows,
        'manifest': read_manifest(run_dir),
    }


def moving_average(values, window):
    result = []
    half = window // 2
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        chunk = [v for v in values[start:end] if v is not None]
        result.append(sum(chunk) / len(chunk) if chunk else None)
    return result


def detect_distill_end(run):
    rows = run['rows']
    prev = None
    for row in rows:
        cur = to_float(row.get('distill_scale'))
        if cur is None:
            continue
        if prev is not None and prev > 0.0 and cur <= 0.0:
            return row['epoch']
        prev = cur

    manifest = run.get('manifest') or {}
    branches = manifest.get('branches') or {}
    scale_branch = branches.get('scale') or {}
    if 'stage_a_end' in scale_branch:
        return int(scale_branch['stage_a_end'])
    return None


def detect_plateau(rows, metric='val_mid_err', smooth_window=11, tail_window=40,
                   improve_tol=0.01, final_best_tol=0.05):
    points = [(row['epoch'], to_float(row.get(metric))) for row in rows]
    points = [(e, v) for e, v in points if v is not None]
    if len(points) < tail_window:
        return None, moving_average([v for _, v in points], smooth_window)

    epochs = [e for e, _ in points]
    values = [v for _, v in points]
    smooth = moving_average(values, smooth_window)

    best_so_far = float('inf')
    for idx, value in enumerate(smooth):
        if value is not None:
            best_so_far = min(best_so_far, value)
        if idx + 1 < tail_window or value is None:
            continue
        start_value = smooth[idx + 1 - tail_window]
        if start_value is None:
            continue
        rel_improve = (start_value - value) / max(abs(start_value), 1e-12)
        close_to_best = value <= best_so_far * (1.0 + final_best_tol)
        if rel_improve < improve_tol and close_to_best:
            return epochs[idx], smooth
    return None, smooth


def write_combined_csv(runs, out_path):
    fieldnames = [
        'label',
        'run_dir',
        'epoch',
        'phase',
        'distill_scale',
        'train_loss_total',
        'train_loss_scale',
        'train_loss_feat_distill',
        'train_loss_corr_distill',
        'val_loss_scale',
        'val_mid_err',
        'val_err_1',
        'val_err_2',
        'val_err_5',
    ]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for row in run['rows']:
                writer.writerow({
                    key: (
                        run['label'] if key == 'label'
                        else str(run['run_dir']) if key == 'run_dir'
                        else row.get(key, '')
                    )
                    for key in fieldnames
                })


def plot_metric(runs, metric, ylabel, out_base, distill_ends, plateau_epochs=None):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for run in sorted(runs, key=lambda item: ORDER.index(item['label']) if item['label'] in ORDER else 99):
        xs, ys = [], []
        for row in run['rows']:
            value = to_float(row.get(metric))
            if value is None:
                continue
            xs.append(row['epoch'])
            ys.append(value)
        if not xs:
            continue
        ax.plot(xs, ys, label=run['label'], color=COLORS.get(run['label']), linewidth=2.0)
        if plateau_epochs and plateau_epochs.get(run['label']) is not None:
            plateau = plateau_epochs[run['label']]
            ax.scatter([plateau], [ys[min(range(len(xs)), key=lambda i: abs(xs[i] - plateau))]],
                       color=COLORS.get(run['label']), s=35, zorder=4)

    for distill_end in sorted({x for x in distill_ends if x is not None}):
        ax.axvline(distill_end, color='0.45', linestyle=':', linewidth=1.6)
        ax.text(distill_end + 2, 0.95, f'distill end @ {distill_end}',
                transform=ax.get_xaxis_transform(), color='0.25', va='top')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_base.with_suffix('.png'), dpi=220)
    fig.savefig(out_base.with_suffix('.pdf'))
    plt.close(fig)


def plot_two_panel(runs, out_base, distill_ends, plateau_epochs):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.4), sharex=True)
    specs = [
        ('val_mid_err', 'Validation MiD Error (lower is better)'),
        ('train_loss_scale', 'Train Scale Loss'),
    ]
    for ax, (metric, ylabel) in zip(axes, specs):
        for run in sorted(runs, key=lambda item: ORDER.index(item['label']) if item['label'] in ORDER else 99):
            xs, ys = [], []
            for row in run['rows']:
                value = to_float(row.get(metric))
                if value is None:
                    continue
                xs.append(row['epoch'])
                ys.append(value)
            if xs:
                ax.plot(xs, ys, label=run['label'], color=COLORS.get(run['label']), linewidth=2.0)
                if metric == 'val_mid_err' and plateau_epochs.get(run['label']) is not None:
                    plateau = plateau_epochs[run['label']]
                    closest = min(range(len(xs)), key=lambda i: abs(xs[i] - plateau))
                    ax.scatter([xs[closest]], [ys[closest]], color=COLORS.get(run['label']), s=35, zorder=4)
        for distill_end in sorted({x for x in distill_ends if x is not None}):
            ax.axvline(distill_end, color='0.45', linestyle=':', linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(out_base.with_suffix('.png'), dpi=220)
    fig.savefig(out_base.with_suffix('.pdf'))
    plt.close(fig)


def summarize(runs, out_path):
    summary = []
    for run in runs:
        plateau_epoch, _ = detect_plateau(run['rows'])
        vals = [
            (row['epoch'], to_float(row.get('val_mid_err')))
            for row in run['rows']
            if to_float(row.get('val_mid_err')) is not None
        ]
        best_epoch, best_val = (None, None)
        final_epoch, final_val = (None, None)
        if vals:
            best_epoch, best_val = min(vals, key=lambda item: item[1])
            final_epoch, final_val = vals[-1]
        summary.append({
            'label': run['label'],
            'run_dir': str(run['run_dir']),
            'distill_end_epoch': detect_distill_end(run),
            'plateau_epoch': plateau_epoch,
            'best_epoch': best_epoch,
            'best_val_mid_err': best_val,
            'final_epoch': final_epoch,
            'final_val_mid_err': final_val,
        })

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='Run dirs, optionally label=path')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--prefix', default='query_init_ablation')
    args = parser.parse_args()

    runs = [read_rows(path, label) for label, path in map(parse_run_arg, args.runs)]
    output_dir = Path(args.output_dir) if args.output_dir else runs[0]['run_dir'] / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    distill_ends = [detect_distill_end(run) for run in runs]
    plateau_epochs = {
        run['label']: detect_plateau(run['rows'])[0]
        for run in runs
    }

    write_combined_csv(runs, output_dir / f'{args.prefix}_curves_combined.csv')
    plot_metric(
        runs,
        'val_mid_err',
        'Validation MiD Error (lower is better)',
        output_dir / f'{args.prefix}_val_mid_err_curve',
        distill_ends,
        plateau_epochs,
    )
    plot_metric(
        runs,
        'train_loss_scale',
        'Train Scale Loss',
        output_dir / f'{args.prefix}_train_loss_scale_curve',
        distill_ends,
    )
    plot_metric(
        runs,
        'train_loss_total',
        'Train Total Loss',
        output_dir / f'{args.prefix}_train_loss_total_curve',
        distill_ends,
    )
    plot_two_panel(
        runs,
        output_dir / f'{args.prefix}_val_train_2panel',
        distill_ends,
        plateau_epochs,
    )
    summary = summarize(runs, output_dir / f'{args.prefix}_summary.json')

    for item in summary:
        print(
            f"{item['label']}: distill_end={item['distill_end_epoch']}, "
            f"plateau={item['plateau_epoch']}, "
            f"best=({item['best_epoch']}, {item['best_val_mid_err']}), "
            f"final=({item['final_epoch']}, {item['final_val_mid_err']})"
        )


if __name__ == '__main__':
    main()
