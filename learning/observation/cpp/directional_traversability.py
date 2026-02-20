from collections import deque
from typing import Optional, Set, Tuple

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


def _direction_flags_from_mask(
    passable_mask: np.ndarray,
    *,
    start: GridPos,
    connectivity: int,
) -> Tuple[bool, bool, bool, bool]:
    sr, sc = start
    if not bool(passable_mask[sr, sc]):
        return False, False, False, False
    comp = _flood_component(passable_mask, start=start, connectivity=connectivity)
    if not comp:
        return False, False, False, False
    h, w = passable_mask.shape
    left, right, up, down, nw, se, ne, sw = _touch_flags(comp, h=h, w=w)
    return bool(left and right), bool(up and down), bool(nw and se), bool(ne and sw)


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
    known_ratio_map: Optional[np.ndarray] = None,
    patch_size: int = 7,
    connectivity: int = 8,
    require_fully_known_patch: bool = True,
    min_center_known_ratio: float = 0.0,
    min_patch_known_ratio: float = 0.0,
    uncertain_fill: float = -1.0,
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
    if known_ratio_map is not None and known_ratio_map.shape != state_grid.shape:
        raise ValueError("known_ratio_map shape must match state_grid")
    if not (0.0 <= min_center_known_ratio <= 1.0):
        raise ValueError("min_center_known_ratio must be in [0, 1]")
    if not (0.0 <= min_patch_known_ratio <= 1.0):
        raise ValueError("min_patch_known_ratio must be in [0, 1]")

    h, w = state_grid.shape
    p = _normalize_patch_size(patch_size, limit=max(h, w))
    center = p // 2
    out = np.full((4, h, w), unknown_fill, dtype=np.float32)

    for r in range(h):
        for c in range(w):
            state_center = int(state_grid[r, c])

            local = center_crop_with_pad(
                state_grid,
                center=(r, c),
                out_h=p,
                out_w=p,
                pad_value=float(BLOCKED_STATE),
            ).astype(np.int8)

            local_has_unknown = bool(np.any(local == UNKNOWN_STATE))
            if require_fully_known_patch and local_has_unknown:
                continue

            if known_ratio_map is not None:
                center_known_ratio = float(known_ratio_map[r, c])
                local_known_ratio = center_crop_with_pad(
                    known_ratio_map,
                    center=(r, c),
                    out_h=p,
                    out_w=p,
                    pad_value=0.0,
                )
                patch_known_ratio = float(np.mean(local_known_ratio))
            else:
                center_known_ratio = 1.0 if state_center != UNKNOWN_STATE else 0.0
                patch_known_ratio = float(np.mean(local != UNKNOWN_STATE))

            trusted_local = (
                center_known_ratio >= min_center_known_ratio
                and patch_known_ratio >= min_patch_known_ratio
            )

            # High-confidence branch: same as previous strict free-only DTM.
            if trusted_local and state_center == FREE_STATE:
                free_mask = local == FREE_STATE
                lr, ud, nwse, nesw = _direction_flags_from_mask(
                    free_mask,
                    start=(center, center),
                    connectivity=connectivity,
                )
                out[0, r, c] = 1.0 if lr else 0.0
                out[1, r, c] = 1.0 if ud else 0.0
                out[2, r, c] = 1.0 if nwse else 0.0
                out[3, r, c] = 1.0 if nesw else 0.0
                continue

            # Observed blocked center should stay non-traversable.
            if state_center == BLOCKED_STATE:
                out[:, r, c] = 0.0
                continue

            # Uncertainty-aware branch:
            # - pessimistic: unknown treated as blocked
            # - optimistic: unknown treated as free
            # If both agree, that result is certain and used regardless of known ratio.
            pess_mask = local == FREE_STATE
            opt_mask = (local == FREE_STATE) | (local == UNKNOWN_STATE)

            p_lr, p_ud, p_nwse, p_nesw = _direction_flags_from_mask(
                pess_mask,
                start=(center, center),
                connectivity=connectivity,
            )
            o_lr, o_ud, o_nwse, o_nesw = _direction_flags_from_mask(
                opt_mask,
                start=(center, center),
                connectivity=connectivity,
            )

            for ch, p_flag, o_flag in (
                (0, p_lr, o_lr),
                (1, p_ud, o_ud),
                (2, p_nwse, o_nwse),
                (3, p_nesw, o_nesw),
            ):
                if p_flag:
                    out[ch, r, c] = 1.0
                elif not o_flag:
                    out[ch, r, c] = 0.0
                else:
                    # Ambiguous with partial observability: keep unknown.
                    out[ch, r, c] = float(uncertain_fill)

    return out
