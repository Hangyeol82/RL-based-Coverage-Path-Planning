from collections import deque
from typing import Optional, Tuple

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


def _point_connected(
    passable_mask: np.ndarray,
    *,
    start: GridPos,
    goal: GridPos,
    connectivity: int,
) -> bool:
    h, w = passable_mask.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < h and 0 <= sc < w):
        return False
    if not (0 <= gr < h and 0 <= gc < w):
        return False
    if not bool(passable_mask[sr, sc]) or not bool(passable_mask[gr, gc]):
        return False
    if (sr, sc) == (gr, gc):
        return True

    q = deque([(sr, sc)])
    visited = {(sr, sc)}
    deltas = _neighbor_deltas(connectivity)
    while q:
        r, c = q.popleft()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if (nr, nc) in visited or not bool(passable_mask[nr, nc]):
                continue
            if (nr, nc) == (gr, gc):
                return True
            visited.add((nr, nc))
            q.append((nr, nc))
    return False


def _edge_pair_connected(
    passable_mask: np.ndarray,
    *,
    side_a: str,
    side_b: str,
    connectivity: int,
) -> bool:
    h, w = passable_mask.shape
    if h <= 0 or w <= 0:
        return False

    sources = [p for p in _side_cells(h, w, side_a) if bool(passable_mask[p[0], p[1]])]
    if not sources:
        return False
    targets = {p for p in _side_cells(h, w, side_b) if bool(passable_mask[p[0], p[1]])}
    if not targets:
        return False

    for p in sources:
        if p in targets:
            return True

    q = deque(sources)
    visited = set(sources)
    deltas = _neighbor_deltas(connectivity)
    while q:
        r, c = q.popleft()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if (nr, nc) in visited or not bool(passable_mask[nr, nc]):
                continue
            if (nr, nc) in targets:
                return True
            visited.add((nr, nc))
            q.append((nr, nc))
    return False


def _direction_flags_from_mask6(
    passable_mask: np.ndarray,
    *,
    connectivity: int,
) -> Tuple[bool, bool, bool, bool, bool, bool]:
    h, w = passable_mask.shape
    if h <= 0 or w <= 0:
        return False, False, False, False, False, False

    lr = _edge_pair_connected(
        passable_mask,
        side_a="left",
        side_b="right",
        connectivity=connectivity,
    )
    ud = _edge_pair_connected(
        passable_mask,
        side_a="up",
        side_b="down",
        connectivity=connectivity,
    )
    nw_se = _point_connected(
        passable_mask,
        start=(0, 0),
        goal=(h - 1, w - 1),
        connectivity=connectivity,
    )
    se_nw = _point_connected(
        passable_mask,
        start=(h - 1, w - 1),
        goal=(0, 0),
        connectivity=connectivity,
    )
    ne_sw = _point_connected(
        passable_mask,
        start=(0, w - 1),
        goal=(h - 1, 0),
        connectivity=connectivity,
    )
    sw_ne = _point_connected(
        passable_mask,
        start=(h - 1, 0),
        goal=(0, w - 1),
        connectivity=connectivity,
    )
    return lr, ud, nw_se, se_nw, ne_sw, sw_ne


def _direction_flags_from_mask12(
    passable_mask: np.ndarray,
    *,
    connectivity: int,
) -> Tuple[bool, ...]:
    h, w = passable_mask.shape
    if h <= 0 or w <= 0:
        return (False,) * 12

    sides = ("up", "right", "down", "left")
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
    _ = sides  # keep side order explicit for readability
    vals = []
    for side_a, side_b in pairs:
        vals.append(
            _edge_pair_connected(
                passable_mask,
                side_a=side_a,
                side_b=side_b,
                connectivity=connectivity,
            )
        )
    return tuple(vals)


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
    output_mode: str = "six",
    out: Optional[np.ndarray] = None,
    dirty_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute directional traversability maps for each cell.

    Output modes:
    - six:
      shape (6, H, W), channel order:
      0) left <-> right
      1) up <-> down
      2) northwest -> southeast
      3) southeast -> northwest
      4) northeast -> southwest
      5) southwest -> northeast
    - port12:
      shape (12, H, W), channel order:
      0) up->right, 1) up->down, 2) up->left,
      3) right->up, 4) right->down, 5) right->left,
      6) down->up, 7) down->right, 8) down->left,
      9) left->up, 10) left->right, 11) left->down

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
    mode = str(output_mode).strip().lower()
    if mode not in {"six", "port12"}:
        raise ValueError("output_mode must be one of {'six', 'port12'}")

    h, w = state_grid.shape
    p = _normalize_patch_size(patch_size, limit=max(h, w))
    center = p // 2
    out_ch = 6 if mode == "six" else 12
    if out is None:
        out_arr = np.full((out_ch, h, w), unknown_fill, dtype=np.float32)
    else:
        if out.shape != (out_ch, h, w):
            raise ValueError(f"out shape must be ({out_ch}, H, W) matching state_grid")
        out_arr = out.astype(np.float32, copy=False)

    targets = None
    if dirty_mask is not None:
        if dirty_mask.shape != state_grid.shape:
            raise ValueError("dirty_mask shape must match state_grid")
        targets = np.argwhere(dirty_mask)
        if targets.size == 0:
            return out_arr

    if targets is None:
        iterator = ((r, c) for r in range(h) for c in range(w))
    else:
        iterator = ((int(rc[0]), int(rc[1])) for rc in targets)

    for r, c in iterator:
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
            out_arr[:, r, c] = float(unknown_fill)
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

        # High-confidence branch: strict free-only DTM.
        if trusted_local and state_center == FREE_STATE:
            free_mask = local == FREE_STATE
            if mode == "six":
                flags = _direction_flags_from_mask6(
                    free_mask,
                    connectivity=connectivity,
                )
            else:
                flags = _direction_flags_from_mask12(
                    free_mask,
                    connectivity=connectivity,
                )
            for ch, flag in enumerate(flags):
                out_arr[ch, r, c] = 1.0 if flag else 0.0
            continue

        # Observed blocked center should stay non-traversable.
        if state_center == BLOCKED_STATE:
            out_arr[:, r, c] = 0.0
            continue

        # Uncertainty-aware branch:
        # - pessimistic: unknown treated as blocked
        # - optimistic: unknown treated as free
        # If both agree, that result is certain and used regardless of known ratio.
        pess_mask = local == FREE_STATE
        opt_mask = (local == FREE_STATE) | (local == UNKNOWN_STATE)

        if mode == "six":
            pess_flags = _direction_flags_from_mask6(
                pess_mask,
                connectivity=connectivity,
            )
            opt_flags = _direction_flags_from_mask6(
                opt_mask,
                connectivity=connectivity,
            )
        else:
            pess_flags = _direction_flags_from_mask12(
                pess_mask,
                connectivity=connectivity,
            )
            opt_flags = _direction_flags_from_mask12(
                opt_mask,
                connectivity=connectivity,
            )

        for ch, p_flag, o_flag in zip(range(out_ch), pess_flags, opt_flags):
            if p_flag:
                out_arr[ch, r, c] = 1.0
            elif not o_flag:
                out_arr[ch, r, c] = 0.0
            else:
                # Ambiguous with partial observability: keep unknown.
                out_arr[ch, r, c] = float(uncertain_fill)

    return out_arr
