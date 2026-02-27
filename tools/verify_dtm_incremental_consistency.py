import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.observation.cpp.multiscale_observation import (  # noqa: E402
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)


def _make_ground_truth_map(rng: np.random.RandomState, size: int, obstacle_prob: float) -> np.ndarray:
    gt = (rng.rand(size, size) < obstacle_prob).astype(np.int8)
    gt[0, 0] = 0  # keep a guaranteed free start corner
    return gt


def _reveal_patch(observed: np.ndarray, ground_truth: np.ndarray, center_r: int, center_c: int, radius: int) -> None:
    h, w = observed.shape
    r0 = max(0, center_r - radius)
    r1 = min(h, center_r + radius + 1)
    c0 = max(0, center_c - radius)
    c1 = min(w, center_c + radius + 1)
    observed[r0:r1, c0:c1] = ground_truth[r0:r1, c0:c1]


def _dtm_channel_indices(names: List[str]) -> List[int]:
    return [i for i, n in enumerate(names) if n.startswith("dtm_")]


def verify_incremental_consistency(
    *,
    trials: int,
    steps: int,
    map_size: int,
    sensor_range: int,
    obstacle_prob: float,
    config: MultiScaleCPPObservationConfig,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.RandomState(seed)

    total_level_checks = 0
    level_mismatch_counts: Dict[int, int] = {}
    level_max_abs_err: Dict[int, float] = {}

    for trial in range(int(trials)):
        gt = _make_ground_truth_map(rng, int(map_size), float(obstacle_prob))
        observed = np.full_like(gt, -1, dtype=np.int8)
        explored = np.zeros_like(gt, dtype=bool)

        inc_builder = MultiScaleCPPObservationBuilder(config=config, include_dtm=True)

        for _ in range(int(steps)):
            rr = int(rng.randint(0, map_size))
            cc = int(rng.randint(0, map_size))
            _reveal_patch(observed, gt, rr, cc, int(sensor_range))
            explored |= observed == 0

            robot_r = int(rng.randint(0, map_size))
            robot_c = int(rng.randint(0, map_size))
            robot_pos = (robot_r, robot_c)

            levels_inc = inc_builder.build_levels(observed, robot_pos=robot_pos, explored=explored)

            # Full recompute reference: fresh builder, same input.
            ref_builder = MultiScaleCPPObservationBuilder(config=config, include_dtm=True)
            levels_ref = ref_builder.build_levels(observed, robot_pos=robot_pos, explored=explored)

            names = list(inc_builder.channel_names)
            dtm_idx = _dtm_channel_indices(names)
            if not dtm_idx:
                raise RuntimeError("No DTM channels found in channel_names")

            for lv in range(inc_builder.num_levels):
                a = levels_inc[lv][dtm_idx]
                b = levels_ref[lv][dtm_idx]
                if a.shape != b.shape:
                    raise RuntimeError(f"Shape mismatch at level {lv}: {a.shape} vs {b.shape}")
                total_level_checks += 1

                if not np.array_equal(a, b):
                    level_mismatch_counts[lv] = level_mismatch_counts.get(lv, 0) + 1
                    err = float(np.max(np.abs(a - b)))
                    level_max_abs_err[lv] = max(level_max_abs_err.get(lv, 0.0), err)

    mismatch_total = sum(level_mismatch_counts.values())
    return {
        "total_level_checks": int(total_level_checks),
        "mismatch_total": int(mismatch_total),
        "level_mismatch_counts": level_mismatch_counts,
        "level_max_abs_err": level_max_abs_err,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify incremental DTM update matches full recomputation.")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--map-size", type=int, default=32)
    ap.add_argument("--sensor-range", type=int, default=2)
    ap.add_argument("--obstacle-prob", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dtm-output-mode", type=str, default="extent6", choices=["six", "four", "extent6", "port12"])
    ap.add_argument("--dtm-coarse-mode", type=str, default="bfs", choices=["bfs", "aggregate", "aggregate_transfer"])
    ap.add_argument("--dtm-patch-size", type=int, default=7)
    ap.add_argument("--dtm-connectivity", type=int, default=8, choices=[4, 8])
    ap.add_argument("--dtm-min-known-ratio", type=float, default=0.6)
    ap.add_argument("--dtm-patch-min-known-ratio", type=float, default=0.6)
    args = ap.parse_args()

    config = MultiScaleCPPObservationConfig(
        dtm_output_mode=str(args.dtm_output_mode),
        dtm_coarse_mode=str(args.dtm_coarse_mode),
        dtm_patch_size=int(args.dtm_patch_size),
        dtm_connectivity=int(args.dtm_connectivity),
        dtm_min_known_ratio=float(args.dtm_min_known_ratio),
        dtm_patch_min_known_ratio=float(args.dtm_patch_min_known_ratio),
    )

    res = verify_incremental_consistency(
        trials=int(args.trials),
        steps=int(args.steps),
        map_size=int(args.map_size),
        sensor_range=int(args.sensor_range),
        obstacle_prob=float(args.obstacle_prob),
        config=config,
        seed=int(args.seed),
    )
    print(
        f"[VERIFY-INCREMENTAL] checks={res['total_level_checks']} "
        f"mismatch_total={res['mismatch_total']}"
    )
    if res["mismatch_total"] > 0:
        print(f"  level_mismatch_counts={res['level_mismatch_counts']}")
        print(f"  level_max_abs_err={res['level_max_abs_err']}")
        raise SystemExit(1)
    print("[VERIFY-INCREMENTAL] PASS")


if __name__ == "__main__":
    main()
