import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


TRACK_KEYS = [
    "reward_total",
    "coverage_ratio",
    "collision",
    "reward_area",
    "reward_tv_i",
]


def _load_rollouts(path: str) -> List[Dict[str, float]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    rollouts = data.get("rollouts", [])
    if not isinstance(rollouts, list) or len(rollouts) == 0:
        raise ValueError(f"No rollout data found in: {p}")
    return rollouts


def _mean_record(arr: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in TRACK_KEYS:
        vals = [float(x[key]) for x in arr if key in x]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def _print_summary(name: str, arr: List[Dict[str, float]]):
    mean = _mean_record(arr)
    last = arr[-1]
    print(f"[{name}] rollouts={len(arr)}")
    for key in TRACK_KEYS:
        print(f"  mean_{key}: {mean[key]:.6f}")
    print(
        "  last: "
        f"reward_total={float(last.get('reward_total', float('nan'))):.6f}, "
        f"coverage_ratio={float(last.get('coverage_ratio', float('nan'))):.6f}, "
        f"collision={float(last.get('collision', float('nan'))):.6f}"
    )


def main():
    ap = argparse.ArgumentParser(description="Summarize PPO reward-breakdown logs.")
    ap.add_argument("--baseline", type=str, required=True)
    ap.add_argument("--dtm", type=str, required=True)
    args = ap.parse_args()

    baseline = _load_rollouts(args.baseline)
    dtm = _load_rollouts(args.dtm)

    _print_summary("baseline", baseline)
    _print_summary("dtm", dtm)

    print("[delta last: dtm - baseline]")
    for key in TRACK_KEYS:
        b = float(baseline[-1].get(key, float("nan")))
        d = float(dtm[-1].get(key, float("nan")))
        print(f"  {key}: {d - b:+.6f}")


if __name__ == "__main__":
    main()
