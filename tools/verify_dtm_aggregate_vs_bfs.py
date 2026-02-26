import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.observation.cpp.directional_traversability import compute_directional_traversability
from learning.observation.cpp.grid_features import (
    FREE_STATE,
    BLOCKED_STATE,
    UNKNOWN_STATE,
    block_reduce_max,
    block_reduce_mean,
    block_reduce_state,
)
from learning.observation.cpp.multiscale_observation import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)


def _fine_state_from_occ(occ: np.ndarray) -> np.ndarray:
    # occupancy coding: -1 unknown, 0 free, 1 obstacle
    out = np.full(occ.shape, BLOCKED_STATE, dtype=np.int8)
    out[occ == -1] = UNKNOWN_STATE
    out[occ == 0] = FREE_STATE
    return out


def _dtm_bfs(
    state_map: np.ndarray,
    known_ratio: np.ndarray,
    patch_size: int,
    connectivity: int,
    output_mode: str,
) -> np.ndarray:
    return compute_directional_traversability(
        state_map,
        known_ratio_map=known_ratio,
        patch_size=patch_size,
        connectivity=connectivity,
        require_fully_known_patch=False,
        min_center_known_ratio=0.6,
        min_patch_known_ratio=0.6,
        uncertain_fill=-1.0,
        unknown_fill=-1.0,
        output_mode=output_mode,
    )


def _dtm_agg_from_fine(dtm_fine: np.ndarray, block: int) -> np.ndarray:
    chans = int(dtm_fine.shape[0])
    reduced = [block_reduce_max(dtm_fine[k], block) for k in range(chans)]
    return np.stack(reduced, axis=0).astype(np.float32)


def _dtm_transfer_from_fine(
    dtm_fine: np.ndarray,
    state_fine: np.ndarray,
    state_coarse: np.ndarray,
    block: int,
    output_mode: str,
) -> np.ndarray:
    cfg = MultiScaleCPPObservationConfig(
        dtm_coarse_mode="aggregate_transfer",
        dtm_output_mode=output_mode,
    )
    builder = MultiScaleCPPObservationBuilder(cfg, include_dtm=True)
    return builder._aggregate_dtm_block_transfer(
        dtm_fine=dtm_fine,
        state_fine=state_fine,
        state_coarse=state_coarse,
        block=block,
    )


def _accuracy(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return float(np.mean(a == b))


def run_one(
    *,
    shape: Tuple[int, int],
    seed: int,
    block: int,
    patch_size: int,
    connectivity: int,
    output_mode: str,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    rng = np.random.RandomState(seed)
    h, w = shape
    # random occupancy: unknown/free/obstacle
    p = rng.rand(h, w)
    occ = np.full((h, w), -1, dtype=np.int32)
    occ[p < 0.60] = 0
    occ[(p >= 0.60) & (p < 0.82)] = 1

    known_free = occ == 0
    known_obst = occ == 1
    unknown = occ == -1
    known_ratio_f = (occ != -1).astype(np.float32)
    state_f = _fine_state_from_occ(occ)
    dtm_f = _dtm_bfs(state_f, known_ratio_f, patch_size, connectivity, output_mode)

    state_c = block_reduce_state(
        known_free,
        known_obst,
        unknown,
        block,
        min_known_ratio=0.6,
    )
    known_ratio_c = block_reduce_mean((occ != -1).astype(np.float32), block)
    dtm_c_bfs = _dtm_bfs(state_c, known_ratio_c, patch_size, connectivity, output_mode)
    dtm_c_agg = _dtm_agg_from_fine(dtm_f, block)
    dtm_c_transfer = _dtm_transfer_from_fine(dtm_f, state_f, state_c, block, output_mode)

    def _triple_accuracy(candidate: np.ndarray) -> Tuple[float, float, float]:
        # 1) all cells exact match
        acc_all = _accuracy(candidate, dtm_c_bfs)
        # 2) known-only cells (exclude unknown=-1 in bfs result)
        mask_known = dtm_c_bfs != -1.0
        acc_known = float(np.mean((candidate[mask_known] == dtm_c_bfs[mask_known]))) if np.any(mask_known) else 1.0
        # 3) certain-cells (both not unknown)
        mask_certain = (dtm_c_bfs != -1.0) & (candidate != -1.0)
        acc_certain = float(np.mean((candidate[mask_certain] == dtm_c_bfs[mask_certain]))) if np.any(mask_certain) else 1.0
        return acc_all, acc_known, acc_certain

    return _triple_accuracy(dtm_c_agg), _triple_accuracy(dtm_c_transfer)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare aggregate coarse DTM vs BFS coarse DTM.")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--patch-size", type=int, default=7)
    ap.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--output-mode",
        type=str,
        choices=["six", "port12"],
        default="six",
        help="DTM output channel mode used for BFS and coarse variants.",
    )
    ap.add_argument(
        "--coarse-mode",
        type=str,
        choices=["aggregate", "aggregate_transfer", "both"],
        default="both",
        help="Which coarse strategy to report against BFS coarse reference.",
    )
    args = ap.parse_args()

    all_acc_agg = []
    known_acc_agg = []
    certain_acc_agg = []

    all_acc_transfer = []
    known_acc_transfer = []
    certain_acc_transfer = []

    for i in range(int(args.trials)):
        agg_acc, transfer_acc = run_one(
            shape=(int(args.height), int(args.width)),
            seed=int(args.seed) + i,
            block=int(args.block),
            patch_size=int(args.patch_size),
            connectivity=int(args.connectivity),
            output_mode=str(args.output_mode),
        )
        a, b, c = agg_acc
        all_acc_agg.append(a)
        known_acc_agg.append(b)
        certain_acc_agg.append(c)

        x, y, z = transfer_acc
        all_acc_transfer.append(x)
        known_acc_transfer.append(y)
        certain_acc_transfer.append(z)

    if args.coarse_mode in {"aggregate", "both"}:
        print(
            "[VERIFY-AGG] "
            f"mode=aggregate output={args.output_mode} trials={args.trials} block={args.block} "
            f"acc_all={np.mean(all_acc_agg):.4f} "
            f"acc_known={np.mean(known_acc_agg):.4f} "
            f"acc_certain={np.mean(certain_acc_agg):.4f}"
        )
    if args.coarse_mode in {"aggregate_transfer", "both"}:
        print(
            "[VERIFY-AGG] "
            f"mode=aggregate_transfer output={args.output_mode} trials={args.trials} block={args.block} "
            f"acc_all={np.mean(all_acc_transfer):.4f} "
            f"acc_known={np.mean(known_acc_transfer):.4f} "
            f"acc_certain={np.mean(certain_acc_transfer):.4f}"
        )


if __name__ == "__main__":
    main()
