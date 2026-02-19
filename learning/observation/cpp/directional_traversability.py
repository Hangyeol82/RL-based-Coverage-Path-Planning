from collections import deque
from typing import Set, Tuple

import numpy as np

from .grid_features import (
    BLOCKED_STATE,
    FREE_STATE,
    UNKNOWN_STATE,
    center_crop_with_pad,
)


GridPos = Tuple[int, int]


def _neighbor_deltas(connectivity: int):
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


def _flood_component(mask: np.ndarray, start: GridPos, connectivity: int) -> Set[GridPos]:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = mask.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w) or not bool(mask[sr, sc]):
        return set()

    q = deque([(sr, sc)])
    visited = {(sr, sc)}
    deltas = _neighbor_deltas(connectivity)

    while q:
        r, c = q.popleft()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if (nr, nc) in visited or not bool(mask[nr, nc]):
                continue
            visited.add((nr, nc))
            q.append((nr, nc))
    return visited


def _touch_flags(component: Set[GridPos], h: int, w: int):
    left = any(c == 0 for _, c in component)
    right = any(c == w - 1 for _, c in component)
    up = any(r == 0 for r, _ in component)
    down = any(r == h - 1 for r, _ in component)
    nw = any((r == 0) or (c == 0) for r, c in component)
    se = any((r == h - 1) or (c == w - 1) for r, c in component)
    ne = any((r == 0) or (c == w - 1) for r, c in component)
    sw = any((r == h - 1) or (c == 0) for r, c in component)
    return left, right, up, down, nw, se, ne, sw


def _normalize_patch_size(patch_size: int, limit: int) -> int:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    p = int(patch_size)
    if p % 2 == 0:
        p -= 1
    p = max(1, p)
    if limit > 0 and p > limit:
        p = limit if limit % 2 == 1 else max(1, limit - 1)
    return p


def compute_directional_traversability(
    state_grid: np.ndarray,
    *,
    patch_size: int = 7,
    connectivity: int = 8,
    require_fully_known_patch: bool = True,
    unknown_fill: float = -1.0,
) -> np.ndarray:
    """
    Compute directional traversability maps for each cell.

    Output shape: (4, H, W), channel order:
    0) left <-> right
    1) up <-> down
    2) northwest <-> southeast
    3) northeast <-> southwest

    Values:
    - unknown_fill for unknown/invalid computation
    - 0.0 not traversable
    - 1.0 traversable
    """
    if state_grid.ndim != 2:
        raise ValueError("state_grid must be 2D")

    h, w = state_grid.shape
    p = _normalize_patch_size(patch_size, limit=max(h, w))
    center = p // 2
    out = np.full((4, h, w), unknown_fill, dtype=np.float32)

    for r in range(h):
        for c in range(w):
            state_center = int(state_grid[r, c])
            if state_center == UNKNOWN_STATE:
                continue

            local = center_crop_with_pad(
                state_grid,
                center=(r, c),
                out_h=p,
                out_w=p,
                pad_value=float(BLOCKED_STATE),
            ).astype(np.int8)

            if require_fully_known_patch and np.any(local == UNKNOWN_STATE):
                continue

            if int(local[center, center]) != FREE_STATE:
                out[:, r, c] = 0.0
                continue

            free_mask = local == FREE_STATE
            comp = _flood_component(free_mask, start=(center, center), connectivity=connectivity)
            if not comp:
                out[:, r, c] = 0.0
                continue

            left, right, up, down, nw, se, ne, sw = _touch_flags(comp, h=p, w=p)
            out[0, r, c] = 1.0 if (left and right) else 0.0
            out[1, r, c] = 1.0 if (up and down) else 0.0
            out[2, r, c] = 1.0 if (nw and se) else 0.0
            out[3, r, c] = 1.0 if (ne and sw) else 0.0

    return out
