import argparse
from collections import deque
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.observation.cpp.directional_traversability import compute_directional_traversability
from learning.observation.cpp.grid_features import (
    BLOCKED_STATE,
    FREE_STATE,
    UNKNOWN_STATE,
    block_reduce_state,
)
from learning.observation.cpp.multiscale_observation import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)


def _bfs_any(passable: np.ndarray, sources: Sequence[Tuple[int, int]], targets: Sequence[Tuple[int, int]]) -> bool:
    h, w = passable.shape
    src = [(r, c) for r, c in sources if 0 <= r < h and 0 <= c < w and bool(passable[r, c])]
    tgt = {(r, c) for r, c in targets if 0 <= r < h and 0 <= c < w and bool(passable[r, c])}
    if not src or not tgt:
        return False
    if any(p in tgt for p in src):
        return True

    q = deque(src)
    vis = set(src)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        r, c = q.popleft()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if (nr, nc) in vis or not bool(passable[nr, nc]):
                continue
            if (nr, nc) in tgt:
                return True
            vis.add((nr, nc))
            q.append((nr, nc))
    return False


def _side_cells(h: int, w: int, side: str) -> List[Tuple[int, int]]:
    if side == "up":
        return [(0, c) for c in range(w)]
    if side == "right":
        return [(r, w - 1) for r in range(h)]
    if side == "down":
        return [(h - 1, c) for c in range(w)]
    if side == "left":
        return [(r, 0) for r in range(h)]
    raise ValueError(f"Unknown side: {side}")


_PORT12_PAIRS = (
    ("up", "right"),
    ("up", "down"),
    ("up", "left"),
    ("right", "up"),
    ("right", "down"),
    ("right", "left"),
    ("down", "up"),
    ("down", "right"),
    ("down", "left"),
    ("left", "up"),
    ("left", "right"),
    ("left", "down"),
)


def _ref_port12_flags(passable_block: np.ndarray) -> np.ndarray:
    h, w = passable_block.shape
    out = np.zeros(12, dtype=np.float32)
    for i, (a, b) in enumerate(_PORT12_PAIRS):
        out[i] = 1.0 if _bfs_any(passable_block, _side_cells(h, w, a), _side_cells(h, w, b)) else 0.0
    return out


def _run(
    *,
    trials: int,
    map_size: int,
    block: int,
    patch_size: int,
    obstacle_prob: float,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)

    cfg = MultiScaleCPPObservationConfig(
        dtm_output_mode="port12",
        dtm_coarse_mode="aggregate_transfer",
        dtm_patch_size=patch_size,
        dtm_connectivity=8,
        dtm_require_fully_known_patch=False,
        dtm_min_known_ratio=0.6,
        dtm_patch_min_known_ratio=0.6,
    )
    builder = MultiScaleCPPObservationBuilder(cfg, include_dtm=True)

    total = 0
    mismatches = 0
    certain_total = 0
    certain_mismatches = 0

    for _ in range(int(trials)):
        h = int(map_size)
        w = int(map_size)
        occ = np.where(rng.rand(h, w) < float(obstacle_prob), 1, 0).astype(np.int32)

        state = np.full((h, w), BLOCKED_STATE, dtype=np.int8)
        state[occ == 0] = FREE_STATE
        known_ratio = np.ones((h, w), dtype=np.float32)

        dtm_fine = compute_directional_traversability(
            state,
            known_ratio_map=known_ratio,
            patch_size=int(patch_size),
            connectivity=8,
            require_fully_known_patch=False,
            min_center_known_ratio=0.0,
            min_patch_known_ratio=0.0,
            uncertain_fill=-1.0,
            unknown_fill=-1.0,
            output_mode="port12",
        )

        state_coarse = block_reduce_state(
            occ == 0,
            occ == 1,
            np.zeros_like(occ, dtype=bool),
            int(block),
            min_known_ratio=0.6,
        )
        got = builder._aggregate_dtm_block_transfer(
            dtm_fine=dtm_fine,
            state_fine=state,
            state_coarse=state_coarse,
            block=int(block),
        )

        ch, cw = got.shape[1], got.shape[2]
        ref = np.zeros_like(got, dtype=np.float32)
        for r in range(ch):
            rs = r * block
            re = min(h, rs + block)
            for c in range(cw):
                cs = c * block
                ce = min(w, cs + block)
                st = int(state_coarse[r, c])
                if st == int(UNKNOWN_STATE):
                    ref[:, r, c] = -1.0
                elif st == int(BLOCKED_STATE):
                    ref[:, r, c] = 0.0
                else:
                    ref[:, r, c] = _ref_port12_flags(occ[rs:re, cs:ce] == 0)

        eq = got == ref
        total += int(eq.size)
        mismatches += int(np.count_nonzero(~eq))

        # known-only is everything here, but keep explicit for symmetry with other scripts.
        certain = ref != -1.0
        certain_total += int(np.count_nonzero(certain))
        certain_mismatches += int(np.count_nonzero((~eq) & certain))

    acc_all = 1.0 - (float(mismatches) / float(max(1, total)))
    acc_certain = 1.0 - (float(certain_mismatches) / float(max(1, certain_total)))
    return acc_all, acc_certain


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare aggregate_transfer vs full-side BFS reference on blocks.")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--map-size", type=int, default=32)
    ap.add_argument("--blocks", type=str, default="2,4,8")
    ap.add_argument("--patch-sizes", type=str, default="1,3,5,7")
    ap.add_argument("--obstacle-prob", type=float, default=0.28)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    blocks = [int(x.strip()) for x in str(args.blocks).split(",") if x.strip()]
    patches = [int(x.strip()) for x in str(args.patch_sizes).split(",") if x.strip()]
    if not blocks or not patches:
        raise ValueError("blocks/patch-sizes must not be empty")

    print(
        f"[FULLSIDE-BFS] map={args.map_size} trials={args.trials} obstacle_prob={args.obstacle_prob} "
        f"seed={args.seed}"
    )
    for p in patches:
        print(f"  patch_size={p}")
        for b in blocks:
            acc_all, acc_certain = _run(
                trials=int(args.trials),
                map_size=int(args.map_size),
                block=int(b),
                patch_size=int(p),
                obstacle_prob=float(args.obstacle_prob),
                seed=int(args.seed) + int(p) * 100 + int(b),
            )
            print(
                f"    block={b} acc_all={acc_all:.8f} acc_certain={acc_certain:.8f}"
            )


if __name__ == "__main__":
    main()
