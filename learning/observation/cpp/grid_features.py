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


def compute_frontier_map(covered: np.ndarray, known_obstacle: np.ndarray) -> np.ndarray:
    """
    Frontier is the one-cell band between covered space and not-yet-covered
    non-obstacle space, matching the rl-cpp paper/repo convention.
    """
    if covered.shape != known_obstacle.shape:
        raise ValueError("covered and known_obstacle shapes must match")

    covered = np.asarray(covered, dtype=bool) & (~np.asarray(known_obstacle, dtype=bool))
    known_obstacle = np.asarray(known_obstacle, dtype=bool)

    padded = np.pad(covered, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    dilated = np.zeros_like(covered, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            dilated |= padded[dr : dr + covered.shape[0], dc : dc + covered.shape[1]]

    free_uncovered = (~covered) & (~known_obstacle)
    return dilated & free_uncovered


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


def _block_shape(size: int, block: int) -> int:
    return (size + block - 1) // block


def _block_cell_counts(h: int, w: int, block: int) -> np.ndarray:
    ch = _block_shape(h, block)
    cw = _block_shape(w, block)
    row_sizes = np.full(ch, block, dtype=np.int32)
    col_sizes = np.full(cw, block, dtype=np.int32)
    if ch > 0 and (h % block) != 0:
        row_sizes[-1] = h - (ch - 1) * block
    if cw > 0 and (w % block) != 0:
        col_sizes[-1] = w - (cw - 1) * block
    return row_sizes[:, None] * col_sizes[None, :]


def _reshape_blocks(
    arr: np.ndarray,
    block: int,
    *,
    pad_value,
    dtype,
) -> np.ndarray:
    h, w = arr.shape
    ch = _block_shape(h, block)
    cw = _block_shape(w, block)
    work = np.asarray(arr, dtype=dtype)
    pad_h = ch * block - h
    pad_w = cw * block - w
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(
            work,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_value,
        )
    else:
        padded = work
    return padded.reshape(ch, block, cw, block).transpose(0, 2, 1, 3)


def block_reduce_mean(arr: np.ndarray, block: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    if block <= 0:
        raise ValueError("block must be positive")

    h, w = arr.shape
    blocks = _reshape_blocks(arr, block, pad_value=0.0, dtype=np.float32)
    sums = blocks.sum(axis=(2, 3), dtype=np.float32)
    counts = _block_cell_counts(h, w, block).astype(np.float32)
    return sums / counts


def block_reduce_max(arr: np.ndarray, block: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    if block <= 0:
        raise ValueError("block must be positive")

    blocks = _reshape_blocks(
        arr,
        block,
        pad_value=np.finfo(np.float32).min,
        dtype=np.float32,
    )
    return blocks.max(axis=(2, 3))


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
    ch = _block_shape(h, block)
    cw = _block_shape(w, block)
    out = np.full((ch, cw), BLOCKED_STATE, dtype=np.int8)

    free_counts = _reshape_blocks(known_free, block, pad_value=0, dtype=np.int32).sum(axis=(2, 3), dtype=np.int32)
    obst_counts = _reshape_blocks(known_obstacle, block, pad_value=0, dtype=np.int32).sum(
        axis=(2, 3), dtype=np.int32
    )
    unknown_counts = _reshape_blocks(unknown, block, pad_value=0, dtype=np.int32).sum(axis=(2, 3), dtype=np.int32)
    cell_counts = _block_cell_counts(h, w, block)
    known_ratio = 1.0 - (unknown_counts.astype(np.float32) / cell_counts.astype(np.float32))

    unknown_mask = known_ratio < float(min_known_ratio)
    free_mask = (~unknown_mask) & (free_counts > 0) & (obst_counts == 0)
    out[unknown_mask] = UNKNOWN_STATE
    out[free_mask] = FREE_STATE
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
