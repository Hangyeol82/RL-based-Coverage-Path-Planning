from typing import Tuple

import numpy as np


UNKNOWN_STATE = -1
BLOCKED_STATE = 0
FREE_STATE = 1


def extract_known_masks(
    occupancy: np.ndarray,
    *,
    unknown_value: int = -1,
    obstacle_value: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert occupancy grid to online masks.

    Returns:
    - known_free: bool
    - known_obstacle: bool
    - unknown: bool
    """
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be 2D")
    known_obstacle = occupancy == obstacle_value
    known = occupancy != unknown_value
    known_free = known & (~known_obstacle)
    unknown = ~known
    return known_free, known_obstacle, unknown


def compute_frontier_map(known_free: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """
    Frontier is known-free cells adjacent (4-neighborhood) to unknown cells.
    Obstacles are excluded by construction.
    """
    if known_free.shape != unknown.shape:
        raise ValueError("known_free and unknown shapes must match")

    adj_unknown = np.zeros_like(unknown, dtype=bool)
    adj_unknown[:-1, :] |= unknown[1:, :]
    adj_unknown[1:, :] |= unknown[:-1, :]
    adj_unknown[:, :-1] |= unknown[:, 1:]
    adj_unknown[:, 1:] |= unknown[:, :-1]
    return known_free & adj_unknown


def center_crop_with_pad(
    arr: np.ndarray,
    center: Tuple[int, int],
    out_h: int,
    out_w: int,
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")

    h, w = arr.shape
    cr, cc = center
    out = np.full((out_h, out_w), pad_value, dtype=np.float32)
    r0 = cr - out_h // 2
    c0 = cc - out_w // 2

    for rr in range(out_h):
        sr = r0 + rr
        if sr < 0 or sr >= h:
            continue
        for cc_out in range(out_w):
            sc = c0 + cc_out
            if sc < 0 or sc >= w:
                continue
            out[rr, cc_out] = float(arr[sr, sc])
    return out


def _iterate_blocks(h: int, w: int, block: int):
    ch = (h + block - 1) // block
    cw = (w + block - 1) // block
    for r in range(ch):
        rs = r * block
        re = min(h, rs + block)
        for c in range(cw):
            cs = c * block
            ce = min(w, cs + block)
            yield r, c, rs, re, cs, ce, ch, cw


def block_reduce_mean(arr: np.ndarray, block: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    if block <= 0:
        raise ValueError("block must be positive")

    h, w = arr.shape
    ch = (h + block - 1) // block
    cw = (w + block - 1) // block
    out = np.zeros((ch, cw), dtype=np.float32)

    for r, c, rs, re, cs, ce, _, _ in _iterate_blocks(h, w, block):
        out[r, c] = float(np.mean(arr[rs:re, cs:ce]))
    return out


def block_reduce_max(arr: np.ndarray, block: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    if block <= 0:
        raise ValueError("block must be positive")

    h, w = arr.shape
    ch = (h + block - 1) // block
    cw = (w + block - 1) // block
    out = np.zeros((ch, cw), dtype=np.float32)

    for r, c, rs, re, cs, ce, _, _ in _iterate_blocks(h, w, block):
        out[r, c] = float(np.max(arr[rs:re, cs:ce]))
    return out


def block_reduce_state(
    known_free: np.ndarray,
    known_obstacle: np.ndarray,
    unknown: np.ndarray,
    block: int,
    *,
    min_known_ratio: float = 1.0,
) -> np.ndarray:
    """
    Build a coarse state grid:
    - UNKNOWN_STATE (-1): contains any unknown fine cells
    - FREE_STATE (1): fully known and contains only free fine cells
    - BLOCKED_STATE (0): all remaining known cases (obstacle/mixed)
    """
    if not (known_free.shape == known_obstacle.shape == unknown.shape):
        raise ValueError("known_free, known_obstacle, unknown shapes must match")
    if block <= 0:
        raise ValueError("block must be positive")
    if not (0.0 < min_known_ratio <= 1.0):
        raise ValueError("min_known_ratio must be in (0, 1]")

    h, w = known_free.shape
    ch = (h + block - 1) // block
    cw = (w + block - 1) // block
    out = np.full((ch, cw), BLOCKED_STATE, dtype=np.int8)

    for r, c, rs, re, cs, ce, _, _ in _iterate_blocks(h, w, block):
        free_count = int(np.count_nonzero(known_free[rs:re, cs:ce]))
        obst_count = int(np.count_nonzero(known_obstacle[rs:re, cs:ce]))
        unknown_count = int(np.count_nonzero(unknown[rs:re, cs:ce]))
        cell_count = (re - rs) * (ce - cs)
        known_ratio = 1.0 - (float(unknown_count) / float(max(1, cell_count)))

        if known_ratio < min_known_ratio:
            out[r, c] = UNKNOWN_STATE
        elif free_count > 0 and obst_count == 0:
            out[r, c] = FREE_STATE
        else:
            out[r, c] = BLOCKED_STATE
    return out


def _global_edges(size: int, out_size: int) -> np.ndarray:
    if out_size <= 0:
        raise ValueError("out_size must be positive")
    if size <= 0:
        return np.zeros(out_size + 1, dtype=np.int32)
    edges = np.linspace(0, size, out_size + 1, dtype=np.int32)
    edges[-1] = size
    return edges


def _safe_slice(start: int, end: int, size: int) -> Tuple[int, int]:
    if size <= 0:
        return 0, 0
    s = int(np.clip(start, 0, size - 1))
    e = int(np.clip(end, s + 1, size))
    return s, e


def global_reduce_mean(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    h, w = arr.shape
    row_edges = _global_edges(h, out_h)
    col_edges = _global_edges(w, out_w)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    if h == 0 or w == 0:
        return out

    for r in range(out_h):
        rs, re = _safe_slice(row_edges[r], row_edges[r + 1], h)
        for c in range(out_w):
            cs, ce = _safe_slice(col_edges[c], col_edges[c + 1], w)
            out[r, c] = float(np.mean(arr[rs:re, cs:ce]))
    return out


def global_reduce_max(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    h, w = arr.shape
    row_edges = _global_edges(h, out_h)
    col_edges = _global_edges(w, out_w)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    if h == 0 or w == 0:
        return out

    for r in range(out_h):
        rs, re = _safe_slice(row_edges[r], row_edges[r + 1], h)
        for c in range(out_w):
            cs, ce = _safe_slice(col_edges[c], col_edges[c + 1], w)
            out[r, c] = float(np.max(arr[rs:re, cs:ce]))
    return out


def global_reduce_state(
    known_free: np.ndarray,
    known_obstacle: np.ndarray,
    unknown: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    min_known_ratio: float = 1.0,
) -> np.ndarray:
    if not (known_free.shape == known_obstacle.shape == unknown.shape):
        raise ValueError("known_free, known_obstacle, unknown shapes must match")
    if not (0.0 < min_known_ratio <= 1.0):
        raise ValueError("min_known_ratio must be in (0, 1]")

    h, w = known_free.shape
    row_edges = _global_edges(h, out_h)
    col_edges = _global_edges(w, out_w)
    out = np.full((out_h, out_w), BLOCKED_STATE, dtype=np.int8)
    if h == 0 or w == 0:
        return out

    for r in range(out_h):
        rs, re = _safe_slice(row_edges[r], row_edges[r + 1], h)
        for c in range(out_w):
            cs, ce = _safe_slice(col_edges[c], col_edges[c + 1], w)
            free_count = int(np.count_nonzero(known_free[rs:re, cs:ce]))
            obst_count = int(np.count_nonzero(known_obstacle[rs:re, cs:ce]))
            unknown_count = int(np.count_nonzero(unknown[rs:re, cs:ce]))
            cell_count = (re - rs) * (ce - cs)
            known_ratio = 1.0 - (float(unknown_count) / float(max(1, cell_count)))
            if known_ratio < min_known_ratio:
                out[r, c] = UNKNOWN_STATE
            elif free_count > 0 and obst_count == 0:
                out[r, c] = FREE_STATE
            else:
                out[r, c] = BLOCKED_STATE
    return out
