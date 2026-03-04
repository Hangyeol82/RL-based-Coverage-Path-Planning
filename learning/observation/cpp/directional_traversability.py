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


def _component_labels(
    passable_mask: np.ndarray,
    *,
    connectivity: int,
) -> np.ndarray:
    """
    Label connected components on a passable-mask patch.

    Returns:
    - labels: int32 [H, W], -1 for non-passable, otherwise component id >= 0.
    """
    h, w = passable_mask.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    deltas = _neighbor_deltas(connectivity)

    cid = 0
    for r in range(h):
        for c in range(w):
            if (not bool(passable_mask[r, c])) or int(labels[r, c]) >= 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = cid
            while q:
                rr, cc = q.popleft()
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if (not bool(passable_mask[nr, nc])) or int(labels[nr, nc]) >= 0:
                        continue
                    labels[nr, nc] = cid
                    q.append((nr, nc))
            cid += 1
    return labels


def _component_ids_from_cells(
    labels: np.ndarray,
    *,
    cells: list,
) -> np.ndarray:
    ids = []
    h, w = labels.shape
    for r, c in cells:
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        cid = int(labels[r, c])
        if cid >= 0:
            ids.append(cid)
    if not ids:
        return np.empty((0,), dtype=np.int32)
    return np.asarray(ids, dtype=np.int32)


def _components_connected(
    ids_a: np.ndarray,
    ids_b: np.ndarray,
) -> bool:
    if ids_a.size == 0 or ids_b.size == 0:
        return False
    set_a = {int(x) for x in ids_a.tolist()}
    for x in ids_b.tolist():
        if int(x) in set_a:
            return True
    return False


def _point_connected_labels(
    labels: np.ndarray,
    *,
    start: GridPos,
    goal: GridPos,
) -> bool:
    h, w = labels.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < h and 0 <= sc < w):
        return False
    if not (0 <= gr < h and 0 <= gc < w):
        return False
    s = int(labels[sr, sc])
    g = int(labels[gr, gc])
    if s < 0 or g < 0:
        return False
    return s == g


def _reach_ratio_from_component_ids(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
) -> float:
    if source_ids.size == 0 or target_ids.size == 0:
        return 0.0
    src = {int(x) for x in source_ids.tolist()}
    reachable = 0
    total = int(target_ids.size)
    for x in target_ids.tolist():
        if int(x) in src:
            reachable += 1
    return float(reachable) / float(max(1, total))


def _edge_pair_reach_ratio_labels(
    labels: np.ndarray,
    *,
    source_sides: Tuple[str, str],
    target_sides: Tuple[str, str],
) -> float:
    h, w = labels.shape
    if h <= 0 or w <= 0:
        return 0.0
    src_cells = list(
        {
            *(_side_cells(h, w, source_sides[0])),
            *(_side_cells(h, w, source_sides[1])),
        }
    )
    tgt_cells = list(
        {
            *(_side_cells(h, w, target_sides[0])),
            *(_side_cells(h, w, target_sides[1])),
        }
    )
    src_ids = _component_ids_from_cells(labels, cells=src_cells)
    tgt_ids = _component_ids_from_cells(labels, cells=tgt_cells)
    return _reach_ratio_from_component_ids(src_ids, tgt_ids)


def _direction_flags_from_mask6(
    passable_mask: np.ndarray,
    *,
    connectivity: int,
) -> Tuple[bool, bool, bool, bool, bool, bool]:
    h, w = passable_mask.shape
    if h <= 0 or w <= 0:
        return False, False, False, False, False, False

    labels = _component_labels(passable_mask, connectivity=connectivity)
    left_ids = _component_ids_from_cells(labels, cells=_side_cells(h, w, "left"))
    right_ids = _component_ids_from_cells(labels, cells=_side_cells(h, w, "right"))
    up_ids = _component_ids_from_cells(labels, cells=_side_cells(h, w, "up"))
    down_ids = _component_ids_from_cells(labels, cells=_side_cells(h, w, "down"))

    lr = _components_connected(left_ids, right_ids)
    ud = _components_connected(up_ids, down_ids)
    nw_se = _point_connected_labels(labels, start=(0, 0), goal=(h - 1, w - 1))
    se_nw = _point_connected_labels(labels, start=(h - 1, w - 1), goal=(0, 0))
    ne_sw = _point_connected_labels(labels, start=(0, w - 1), goal=(h - 1, 0))
    sw_ne = _point_connected_labels(labels, start=(h - 1, 0), goal=(0, w - 1))
    return lr, ud, nw_se, se_nw, ne_sw, sw_ne


def _direction_extent_from_mask6(
    passable_mask: np.ndarray,
    *,
    connectivity: int,
) -> Tuple[float, float, float, float, float, float]:
    h, w = passable_mask.shape
    if h <= 0 or w <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    labels = _component_labels(passable_mask, connectivity=connectivity)
    left_ids = _component_ids_from_cells(labels, cells=_side_cells(h, w, "left"))
    right_ids = _component_ids_from_cells(labels, cells=_side_cells(h, w, "right"))
    up_ids = _component_ids_from_cells(labels, cells=_side_cells(h, w, "up"))
    down_ids = _component_ids_from_cells(labels, cells=_side_cells(h, w, "down"))

    lr = _reach_ratio_from_component_ids(left_ids, right_ids)
    ud = _reach_ratio_from_component_ids(up_ids, down_ids)
    # Diagonal extents use quadrant-to-opposite-quadrant reachability.
    nw_se = _edge_pair_reach_ratio_labels(
        labels,
        source_sides=("up", "left"),
        target_sides=("down", "right"),
    )
    se_nw = _edge_pair_reach_ratio_labels(
        labels,
        source_sides=("down", "right"),
        target_sides=("up", "left"),
    )
    ne_sw = _edge_pair_reach_ratio_labels(
        labels,
        source_sides=("up", "right"),
        target_sides=("down", "left"),
    )
    sw_ne = _edge_pair_reach_ratio_labels(
        labels,
        source_sides=("down", "left"),
        target_sides=("up", "right"),
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

    labels = _component_labels(passable_mask, connectivity=connectivity)
    side_ids = {
        "up": _component_ids_from_cells(labels, cells=_side_cells(h, w, "up")),
        "right": _component_ids_from_cells(labels, cells=_side_cells(h, w, "right")),
        "down": _component_ids_from_cells(labels, cells=_side_cells(h, w, "down")),
        "left": _component_ids_from_cells(labels, cells=_side_cells(h, w, "left")),
    }

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
        vals.append(_components_connected(side_ids[side_a], side_ids[side_b]))
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
    - extent6:
      shape (6, H, W), same channel order as six.
      Values are continuous traversability extents in [0, 1].
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
    if mode not in {"six", "extent6", "port12"}:
        raise ValueError("output_mode must be one of {'six', 'extent6', 'port12'}")

    h, w = state_grid.shape
    p = _normalize_patch_size(patch_size, limit=max(h, w))
    center = p // 2
    out_ch = 12 if mode == "port12" else 6
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
            elif mode == "extent6":
                flags = _direction_extent_from_mask6(
                    free_mask,
                    connectivity=connectivity,
                )
            else:
                flags = _direction_flags_from_mask12(
                    free_mask,
                    connectivity=connectivity,
                )
            for ch, flag in enumerate(flags):
                if mode == "extent6":
                    out_arr[ch, r, c] = float(flag)
                else:
                    out_arr[ch, r, c] = 1.0 if bool(flag) else 0.0
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
        elif mode == "extent6":
            pess_flags = _direction_extent_from_mask6(
                pess_mask,
                connectivity=connectivity,
            )
            opt_flags = _direction_extent_from_mask6(
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
            p_val = float(p_flag)
            o_val = float(o_flag)
            if abs(p_val - o_val) <= 1e-9:
                out_arr[ch, r, c] = p_val
            else:
                # Ambiguous with partial observability: keep unknown.
                out_arr[ch, r, c] = float(uncertain_fill)

    return out_arr
