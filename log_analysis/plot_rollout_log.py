import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Use a writable local matplotlib cache by default.
_THIS_DIR = Path(__file__).resolve().parent
_MPL_CACHE_DIR = _THIS_DIR / ".mplconfig"
if "MPLCONFIGDIR" not in os.environ:
    _MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE_DIR)

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = (
    "reward_total",
    "coverage_ratio",
    "collision",
    "reward_area",
    "reward_tv_i",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot rollout metrics from a PPO breakdown JSON log.",
    )
    p.add_argument(
        "--input",
        type=str,
        default="log_analysis/logs/indoor_seed101_baseline.json",
        help="Path to a JSON file that contains a 'rollouts' array.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output PNG path. If empty, uses <input_stem>_plot.png in input folder.",
    )
    p.add_argument(
        "--x-axis",
        type=str,
        default="timesteps",
        choices=("timesteps", "rollout"),
        help="X-axis key in rollout records.",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window. 1 disables smoothing.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display plot window after saving.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional plot title.",
    )
    return p.parse_args()


def _load_rollouts(path: Path) -> List[Dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollouts = payload.get("rollouts", [])
    if not isinstance(rollouts, list) or len(rollouts) == 0:
        raise ValueError(f"No rollout data found in: {path}")
    return rollouts


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    w = max(1, int(window))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    out = np.convolve(values, kernel, mode="same")
    return out


def _select_metrics(rollouts: List[Dict[str, float]]) -> Tuple[str, ...]:
    keys = set()
    for row in rollouts:
        keys.update(row.keys())
    return tuple(m for m in DEFAULT_METRICS if m in keys)


def _get_x_values(rollouts: List[Dict[str, float]], x_axis: str) -> np.ndarray:
    values = [float(r.get(x_axis, float(i + 1))) for i, r in enumerate(rollouts)]
    return np.asarray(values, dtype=np.float64)


def _plot(rollouts: List[Dict[str, float]], x_axis: str, smooth_window: int, title: str):
    metrics = _select_metrics(rollouts)
    if len(metrics) == 0:
        raise ValueError("No known metrics found in rollout records.")

    x = _get_x_values(rollouts, x_axis)
    n = len(metrics)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows), squeeze=False)

    for i, metric in enumerate(metrics):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        y = np.asarray([float(row.get(metric, np.nan)) for row in rollouts], dtype=np.float64)
        ax.plot(x, y, color="#1f77b4", linewidth=1.8, alpha=0.85, label="raw")
        y_s = _moving_average(y, smooth_window)
        if smooth_window > 1:
            ax.plot(x, y_s, color="#d62728", linewidth=2.0, alpha=0.9, label=f"ma({smooth_window})")
        ax.set_title(metric)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        if smooth_window > 1:
            ax.legend(loc="best")

    for i in range(n, rows * cols):
        r = i // cols
        c = i % cols
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    return fig


def main():
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input log not found: {input_path}")

    if args.output.strip():
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rollouts = _load_rollouts(input_path)
    title = args.title.strip() or f"Rollout Metrics: {input_path.name}"
    fig = _plot(
        rollouts=rollouts,
        x_axis=args.x_axis,
        smooth_window=args.smooth_window,
        title=title,
    )
    fig.savefig(output_path, dpi=150)
    print(f"[DONE] saved plot: {output_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
