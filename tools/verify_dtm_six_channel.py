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
from learning.observation.cpp.grid_features import BLOCKED_STATE, FREE_STATE


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


def _bfs_any_reach(
    passable: np.ndarray,
    *,
    sources: Iterable[GridPos],
    targets: Iterable[GridPos],
    connectivity: int,
) -> bool:
    h, w = passable.shape
    src = [(r, c) for r, c in sources if 0 <= r < h and 0 <= c < w and bool(passable[r, c])]
    tgt = {(r, c) for r, c in targets if 0 <= r < h and 0 <= c < w and bool(passable[r, c])}
    if not src or not tgt:
        return False
    if any(p in tgt for p in src):
        return True

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
            if (nr, nc) in tgt:
                return True
            vis.add((nr, nc))
            q.append((nr, nc))
    return False


def _ref_flags(passable: np.ndarray, connectivity: int, output_mode: str) -> np.ndarray:
    h, w = passable.shape
    left = [(r, 0) for r in range(h)]
    right = [(r, w - 1) for r in range(h)]
    up = [(0, c) for c in range(w)]
    down = [(h - 1, c) for c in range(w)]

    mode = str(output_mode).strip().lower()
    if mode == "six":
        flags = np.zeros(6, dtype=np.float32)
        flags[0] = 1.0 if _bfs_any_reach(passable, sources=left, targets=right, connectivity=connectivity) else 0.0
        flags[1] = 1.0 if _bfs_any_reach(passable, sources=up, targets=down, connectivity=connectivity) else 0.0
        flags[2] = 1.0 if _bfs_any_reach(passable, sources=[(0, 0)], targets=[(h - 1, w - 1)], connectivity=connectivity) else 0.0
        flags[3] = 1.0 if _bfs_any_reach(passable, sources=[(h - 1, w - 1)], targets=[(0, 0)], connectivity=connectivity) else 0.0
        flags[4] = 1.0 if _bfs_any_reach(passable, sources=[(0, w - 1)], targets=[(h - 1, 0)], connectivity=connectivity) else 0.0
        flags[5] = 1.0 if _bfs_any_reach(passable, sources=[(h - 1, 0)], targets=[(0, w - 1)], connectivity=connectivity) else 0.0
        return flags
    if mode == "port12":
        side_cells = {
            "up": up,
            "right": right,
            "down": down,
            "left": left,
        }
        pairs = (
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
        flags = np.zeros(12, dtype=np.float32)
        for k, (a, b) in enumerate(pairs):
            flags[k] = 1.0 if _bfs_any_reach(passable, sources=side_cells[a], targets=side_cells[b], connectivity=connectivity) else 0.0
        return flags
    raise ValueError("output_mode must be one of {'six', 'port12'}")


def validate_random(
    *,
    trials: int,
    shape: GridPos,
    patch_size: int,
    connectivity: int,
    seed: int,
    output_mode: str,
) -> Tuple[int, int]:
    rng = np.random.RandomState(seed)
    h, w = shape
    mismatches = 0
    checks = 0

    for _ in range(trials):
        # Known-only state for strict BFS semantic verification.
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
            output_mode=output_mode,
        )

        p = int(patch_size)
        if p % 2 == 0:
            p -= 1
        p = max(1, p)
        center = p // 2

        for r in range(h):
            for c in range(w):
                local = _crop_with_block_pad(state, center=(r, c), size=p)
                out_ch = 6 if output_mode == "six" else 12
                expected = np.zeros(out_ch, dtype=np.float32)
                if local[center, center] == FREE_STATE:
                    expected = _ref_flags(local == FREE_STATE, connectivity=connectivity, output_mode=output_mode)
                got = dtm[:, r, c]
                checks += 1
                if not np.allclose(got, expected, atol=1e-6):
                    mismatches += 1

    return checks, mismatches


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify DTM against independent BFS reference.")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--height", type=int, default=17)
    ap.add_argument("--width", type=int, default=19)
    ap.add_argument("--patch-size", type=int, default=7)
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--output-mode", type=str, default="six", choices=["six", "port12"])
    args = ap.parse_args()

    checks, mismatches = validate_random(
        trials=int(args.trials),
        shape=(int(args.height), int(args.width)),
        patch_size=int(args.patch_size),
        connectivity=int(args.connectivity),
        seed=int(args.seed),
        output_mode=str(args.output_mode),
    )
    print(f"[VERIFY] checks={checks} mismatches={mismatches}")
    if mismatches > 0:
        raise SystemExit(1)
    print(f"[VERIFY] PASS: {args.output_mode} DTM matches BFS reference")


if __name__ == "__main__":
    main()
