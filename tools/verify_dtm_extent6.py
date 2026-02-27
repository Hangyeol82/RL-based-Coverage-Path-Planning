import argparse
from collections import deque
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.observation.cpp.directional_traversability import (
    compute_directional_traversability,
)
from learning.observation.cpp.grid_features import BLOCKED_STATE, FREE_STATE, UNKNOWN_STATE


GridPos = Tuple[int, int]


def _neighbors(connectivity: int) -> Sequence[GridPos]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    raise ValueError("connectivity must be 4 or 8")


def _crop_with_block_pad(arr: np.ndarray, center: GridPos, size: int) -> np.ndarray:
    h, w = arr.shape
    cr, cc = center
    out = np.full((size, size), BLOCKED_STATE, dtype=arr.dtype)
    r0 = cr - size // 2
    c0 = cc - size // 2
    for rr in range(size):
        sr = r0 + rr
        if sr < 0 or sr >= h:
            continue
        for cc_out in range(size):
            sc = c0 + cc_out
            if sc < 0 or sc >= w:
                continue
            out[rr, cc_out] = arr[sr, sc]
    return out


def _side_cells(h: int, w: int, side: str):
    if side == "left":
        return [(r, 0) for r in range(h)]
    if side == "right":
        return [(r, w - 1) for r in range(h)]
    if side == "up":
        return [(0, c) for c in range(w)]
    if side == "down":
        return [(h - 1, c) for c in range(w)]
    raise ValueError(f"Unknown side: {side}")


def _reach_ratio_between_sets(
    passable: np.ndarray,
    *,
    sources: Iterable[GridPos],
    targets: Iterable[GridPos],
    connectivity: int,
) -> float:
    h, w = passable.shape
    src = [(r, c) for r, c in sources if 0 <= r < h and 0 <= c < w and bool(passable[r, c])]
    tgt = [(r, c) for r, c in targets if 0 <= r < h and 0 <= c < w and bool(passable[r, c])]
    if not src or not tgt:
        return 0.0
    q = deque(src)
    vis = set(src)
    deltas = _neighbors(connectivity)
    while q:
        r, c = q.popleft()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if (nr, nc) in vis or not bool(passable[nr, nc]):
                continue
            vis.add((nr, nc))
            q.append((nr, nc))
    reachable_targets = sum(1 for p in tgt if p in vis)
    return float(reachable_targets) / float(len(tgt))


def _ref_extent6(passable: np.ndarray, connectivity: int) -> np.ndarray:
    h, w = passable.shape
    lr = _reach_ratio_between_sets(
        passable,
        sources=_side_cells(h, w, "left"),
        targets=_side_cells(h, w, "right"),
        connectivity=connectivity,
    )
    ud = _reach_ratio_between_sets(
        passable,
        sources=_side_cells(h, w, "up"),
        targets=_side_cells(h, w, "down"),
        connectivity=connectivity,
    )
    # Quadrant-based diagonal ratios (same semantics as runtime code).
    nw = list({*(_side_cells(h, w, "up") + _side_cells(h, w, "left"))})
    se = list({*(_side_cells(h, w, "down") + _side_cells(h, w, "right"))})
    ne = list({*(_side_cells(h, w, "up") + _side_cells(h, w, "right"))})
    sw = list({*(_side_cells(h, w, "down") + _side_cells(h, w, "left"))})
    nw_se = _reach_ratio_between_sets(passable, sources=nw, targets=se, connectivity=connectivity)
    se_nw = _reach_ratio_between_sets(passable, sources=se, targets=nw, connectivity=connectivity)
    ne_sw = _reach_ratio_between_sets(passable, sources=ne, targets=sw, connectivity=connectivity)
    sw_ne = _reach_ratio_between_sets(passable, sources=sw, targets=ne, connectivity=connectivity)
    return np.asarray([lr, ud, nw_se, se_nw, ne_sw, sw_ne], dtype=np.float32)


def verify_known_exact(
    *,
    trials: int,
    shape: GridPos,
    patch_size: int,
    connectivity: int,
    seed: int,
) -> Tuple[int, int, float]:
    rng = np.random.RandomState(seed)
    h, w = shape
    checks = 0
    mismatches = 0
    max_abs_err = 0.0
    for _ in range(int(trials)):
        state = np.where(rng.rand(h, w) < 0.68, FREE_STATE, BLOCKED_STATE).astype(np.int8)
        known_ratio = np.ones((h, w), dtype=np.float32)
        dtm = compute_directional_traversability(
            state,
            known_ratio_map=known_ratio,
            patch_size=patch_size,
            connectivity=connectivity,
            require_fully_known_patch=False,
            min_center_known_ratio=0.0,
            min_patch_known_ratio=0.0,
            uncertain_fill=-1.0,
            unknown_fill=-1.0,
            output_mode="extent6",
        )
        p = int(patch_size)
        if p % 2 == 0:
            p -= 1
        p = max(1, p)
        center = p // 2
        for r in range(h):
            for c in range(w):
                local = _crop_with_block_pad(state, center=(r, c), size=p)
                expected = np.zeros(6, dtype=np.float32)
                if local[center, center] == FREE_STATE:
                    expected = _ref_extent6(local == FREE_STATE, connectivity)
                got = dtm[:, r, c]
                err = float(np.max(np.abs(got - expected)))
                max_abs_err = max(max_abs_err, err)
                checks += 1
                if not np.allclose(got, expected, atol=1e-6):
                    mismatches += 1
    return checks, mismatches, max_abs_err


def verify_unknown_invariants(
    *,
    trials: int,
    shape: GridPos,
    patch_size: int,
    connectivity: int,
    seed: int,
) -> Tuple[int, int, int, int]:
    rng = np.random.RandomState(seed)
    h, w = shape
    total = 0
    invalid_range = 0
    blocked_nonzero = 0
    unknown_not_neg1 = 0

    probs = np.asarray([0.2, 0.6, 0.2], dtype=np.float64)  # unknown, free, blocked
    states = np.asarray([UNKNOWN_STATE, FREE_STATE, BLOCKED_STATE], dtype=np.int8)
    for _ in range(int(trials)):
        idx = rng.choice(3, size=(h, w), p=probs)
        state = states[idx]
        known_ratio = (state != UNKNOWN_STATE).astype(np.float32)
        dtm = compute_directional_traversability(
            state,
            known_ratio_map=known_ratio,
            patch_size=patch_size,
            connectivity=connectivity,
            require_fully_known_patch=False,
            min_center_known_ratio=0.6,
            min_patch_known_ratio=0.6,
            uncertain_fill=-1.0,
            unknown_fill=-1.0,
            output_mode="extent6",
        )
        for r in range(h):
            for c in range(w):
                v = dtm[:, r, c]
                total += int(v.size)
                invalid_range += int(np.count_nonzero((v != -1.0) & ((v < 0.0) | (v > 1.0))))
                if int(state[r, c]) == int(BLOCKED_STATE):
                    blocked_nonzero += int(np.count_nonzero(v != 0.0))
                if int(state[r, c]) == int(UNKNOWN_STATE):
                    # unknown center must not become hard 0/1 under conservative defaults
                    unknown_not_neg1 += int(np.count_nonzero(v != -1.0))
    return total, invalid_range, blocked_nonzero, unknown_not_neg1


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify extent6 DTM against independent BFS reference and invariants.")
    ap.add_argument("--trials-known", type=int, default=20)
    ap.add_argument("--trials-unknown", type=int, default=20)
    ap.add_argument("--height", type=int, default=17)
    ap.add_argument("--width", type=int, default=19)
    ap.add_argument("--patch-size", type=int, default=7)
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--strict-unknown-center",
        action="store_true",
        help="Fail if unknown-center cells produce any value other than -1.",
    )
    args = ap.parse_args()

    shape = (int(args.height), int(args.width))
    checks, mismatches, max_abs_err = verify_known_exact(
        trials=int(args.trials_known),
        shape=shape,
        patch_size=int(args.patch_size),
        connectivity=int(args.connectivity),
        seed=int(args.seed),
    )
    print(
        f"[VERIFY extent6 known] checks={checks} mismatches={mismatches} "
        f"max_abs_err={max_abs_err:.8f}"
    )
    if mismatches > 0:
        raise SystemExit(1)

    total, invalid_range, blocked_nonzero, unknown_not_neg1 = verify_unknown_invariants(
        trials=int(args.trials_unknown),
        shape=shape,
        patch_size=int(args.patch_size),
        connectivity=int(args.connectivity),
        seed=int(args.seed) + 999,
    )
    print(
        f"[VERIFY extent6 unknown] values={total} invalid_range={invalid_range} "
        f"blocked_nonzero={blocked_nonzero} unknown_not_neg1={unknown_not_neg1}"
    )
    if invalid_range > 0 or blocked_nonzero > 0:
        raise SystemExit(1)
    if args.strict_unknown_center and unknown_not_neg1 > 0:
        raise SystemExit(1)
    print("[VERIFY extent6] PASS")


if __name__ == "__main__":
    main()
