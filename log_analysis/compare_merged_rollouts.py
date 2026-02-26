import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

# Keep matplotlib cache writable inside repo.
THIS_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = THIS_DIR / ".mplconfig"
if "MPLCONFIGDIR" not in os.environ:
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)

import matplotlib.pyplot as plt


KEY_METRICS = ("coverage_ratio", "reward_total", "reward_area")


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    w = max(1, int(window))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(values, kernel, mode="same")


def infer_global_step(df: pd.DataFrame) -> pd.Series:
    if "global_rollout" in df.columns:
        return df["global_rollout"].astype(np.float64) * 4096.0
    if "timesteps" in df.columns:
        return df["timesteps"].astype(np.float64)
    return pd.Series(np.arange(1, len(df) + 1, dtype=np.float64))


def make_overall_table(base: pd.DataFrame, dtm: pd.DataFrame, tail_ratio: float) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    n_tail = max(1, int(len(base) * tail_ratio))
    base_tail = base.tail(n_tail)
    dtm_tail = dtm.tail(n_tail)

    for metric in KEY_METRICS:
        b_mean = float(base[metric].mean())
        d_mean = float(dtm[metric].mean())
        diff = d_mean - b_mean
        rel = (diff / abs(b_mean) * 100.0) if abs(b_mean) > 1e-12 else np.nan

        b_tail = float(base_tail[metric].mean())
        d_tail = float(dtm_tail[metric].mean())
        tail_diff = d_tail - b_tail
        tail_rel = (tail_diff / abs(b_tail) * 100.0) if abs(b_tail) > 1e-12 else np.nan

        rows.append(
            {
                "metric": metric,
                "baseline_mean": b_mean,
                "dtm_mean": d_mean,
                "diff_dtm_minus_baseline": diff,
                "rel_diff_percent": rel,
                f"baseline_last_{n_tail}_mean": b_tail,
                f"dtm_last_{n_tail}_mean": d_tail,
                f"diff_last_{n_tail}": tail_diff,
                f"rel_diff_last_{n_tail}_percent": tail_rel,
            }
        )
    return pd.DataFrame(rows)


def make_level_table(base: pd.DataFrame, dtm: pd.DataFrame) -> pd.DataFrame:
    req = {"chunk_level", *KEY_METRICS}
    if not req.issubset(base.columns) or not req.issubset(dtm.columns):
        return pd.DataFrame()
    b = base.groupby("chunk_level")[list(KEY_METRICS)].mean().reset_index()
    d = dtm.groupby("chunk_level")[list(KEY_METRICS)].mean().reset_index()
    m = d.merge(b, on="chunk_level", suffixes=("_dtm", "_baseline"))
    for metric in KEY_METRICS:
        m[f"{metric}_diff"] = m[f"{metric}_dtm"] - m[f"{metric}_baseline"]
    return m


def make_pairwise_table(base: pd.DataFrame, dtm: pd.DataFrame) -> pd.DataFrame:
    req = {"global_rollout", *KEY_METRICS}
    if not req.issubset(base.columns) or not req.issubset(dtm.columns):
        return pd.DataFrame()
    b = base[["global_rollout", *KEY_METRICS]].copy()
    d = dtm[["global_rollout", *KEY_METRICS]].copy()
    m = d.merge(b, on="global_rollout", suffixes=("_dtm", "_baseline"))
    for metric in KEY_METRICS:
        m[f"{metric}_diff"] = m[f"{metric}_dtm"] - m[f"{metric}_baseline"]
    return m


def plot_timeline(base: pd.DataFrame, dtm: pd.DataFrame, out_path: Path, ma_window: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    x_b = base["global_step"].to_numpy(dtype=np.float64) / 1_000_000.0
    x_d = dtm["global_step"].to_numpy(dtype=np.float64) / 1_000_000.0

    for ax, metric in zip(axes, KEY_METRICS):
        yb = base[metric].to_numpy(dtype=np.float64)
        yd = dtm[metric].to_numpy(dtype=np.float64)
        ax.plot(x_b, yb, color="#1f77b4", alpha=0.18, linewidth=1.0, label="baseline raw")
        ax.plot(x_d, yd, color="#ff7f0e", alpha=0.18, linewidth=1.0, label="dtm raw")
        ax.plot(x_b, moving_average(yb, ma_window), color="#1f77b4", linewidth=2.5, label=f"baseline MA{ma_window}")
        ax.plot(x_d, moving_average(yd, ma_window), color="#ff7f0e", linewidth=2.5, label=f"dtm MA{ma_window}")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("global_step (millions)")
    fig.suptitle("Merged Rollout Comparison: Baseline vs DTM", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_levelwise(level_df: pd.DataFrame, out_path: Path) -> None:
    if level_df.empty:
        return
    levels = level_df["chunk_level"].to_numpy(dtype=int)
    x = np.arange(len(levels), dtype=float)
    width = 0.36

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for ax, metric in zip(axes, KEY_METRICS):
        b = level_df[f"{metric}_baseline"].to_numpy(dtype=float)
        d = level_df[f"{metric}_dtm"].to_numpy(dtype=float)
        ax.bar(x - width / 2.0, b, width=width, color="#1f77b4", alpha=0.85, label="baseline")
        ax.bar(x + width / 2.0, d, width=width, color="#ff7f0e", alpha=0.85, label="dtm")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xticks(x, [f"L{lv}" for lv in levels])
    axes[-1].set_xlabel("curriculum level")
    fig.suptitle("Level-wise Mean Metrics", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pairwise_diffs(pair_df: pd.DataFrame, out_path: Path) -> None:
    if pair_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, KEY_METRICS):
        d = pair_df[f"{metric}_diff"].to_numpy(dtype=float)
        ax.hist(d, bins=40, color="#2ca02c", alpha=0.85)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
        ax.set_title(f"{metric} diff (dtm - baseline)")
        ax.grid(alpha=0.2)
    fig.suptitle("Pairwise Diff Distribution", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_report_md(
    overall_df: pd.DataFrame,
    level_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Merged Rollout Comparison Report")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    for _, row in overall_df.iterrows():
        lines.append(
            "- {}: baseline={:.6f}, dtm={:.6f}, diff={:+.6f} ({:+.2f}%)".format(
                row["metric"],
                row["baseline_mean"],
                row["dtm_mean"],
                row["diff_dtm_minus_baseline"],
                row["rel_diff_percent"],
            )
        )
    lines.append("")

    if not pair_df.empty:
        lines.append("## Pairwise Win Rate (DTM > Baseline)")
        lines.append("")
        for metric in KEY_METRICS:
            win = float((pair_df[f"{metric}_diff"] > 0.0).mean() * 100.0)
            lines.append(f"- {metric}: {win:.2f}%")
        lines.append("")

    if not level_df.empty:
        lines.append("## Level-wise DTM Advantage")
        lines.append("")
        for _, row in level_df.iterrows():
            lv = int(row["chunk_level"])
            c = row["coverage_ratio_diff"]
            r = row["reward_total_diff"]
            a = row["reward_area_diff"]
            lines.append(f"- L{lv}: coverage_diff={c:+.6f}, reward_total_diff={r:+.6f}, reward_area_diff={a:+.6f}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare merged rollout CSV logs (baseline vs dtm).")
    p.add_argument("--baseline", type=str, required=True, help="Path to baseline merged CSV")
    p.add_argument("--dtm", type=str, required=True, help="Path to dtm merged CSV")
    p.add_argument("--out-dir", type=str, default="log_analysis/logs/merged_compare")
    p.add_argument("--ma-window", type=int, default=31)
    p.add_argument("--tail-ratio", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(args.baseline).sort_values("global_rollout").reset_index(drop=True)
    dtm = pd.read_csv(args.dtm).sort_values("global_rollout").reset_index(drop=True)

    baseline["global_step"] = infer_global_step(baseline)
    dtm["global_step"] = infer_global_step(dtm)

    overall_df = make_overall_table(baseline, dtm, args.tail_ratio)
    level_df = make_level_table(baseline, dtm)
    pair_df = make_pairwise_table(baseline, dtm)

    overall_csv = out_dir / "summary_overall.csv"
    level_csv = out_dir / "summary_by_level.csv"
    pair_csv = out_dir / "pairwise_diff.csv"
    report_md = out_dir / "report.md"

    overall_df.to_csv(overall_csv, index=False)
    if not level_df.empty:
        level_df.to_csv(level_csv, index=False)
    if not pair_df.empty:
        pair_df.to_csv(pair_csv, index=False)

    write_report_md(overall_df, level_df, pair_df, report_md)

    timeline_png = out_dir / "timeline_compare_hd.png"
    level_png = out_dir / "levelwise_compare_hd.png"
    diff_png = out_dir / "pairwise_diff_hist_hd.png"
    plot_timeline(baseline, dtm, timeline_png, ma_window=args.ma_window)
    plot_levelwise(level_df, level_png)
    plot_pairwise_diffs(pair_df, diff_png)

    print(f"[DONE] output dir: {out_dir}")
    print(f"[DONE] {overall_csv}")
    if level_csv.exists():
        print(f"[DONE] {level_csv}")
    if pair_csv.exists():
        print(f"[DONE] {pair_csv}")
    print(f"[DONE] {report_md}")
    print(f"[DONE] {timeline_png}")
    if level_png.exists():
        print(f"[DONE] {level_png}")
    if diff_png.exists():
        print(f"[DONE] {diff_png}")


if __name__ == "__main__":
    main()
