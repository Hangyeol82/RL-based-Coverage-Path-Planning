from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .large_obstacle_grid import (
    _clear_start,
    _free_is_connected,
    _reachable_mask,
    build_large_obstacle_grid_map,
)


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class MacroDetailPreset:
    level: int
    trap_count: int
    min_rect_half_extent: int
    max_rect_half_extent: int
    min_detail_size: int
    max_detail_size: int
    min_trap_size: int
    max_trap_size: int
    target_detail_ratio_of_base_free: float
    max_detail_ratio_of_base_free: float
    overlap_limit: float = 0.04


PRESETS: Dict[int, MacroDetailPreset] = {
    1: MacroDetailPreset(
        1,
        trap_count=0,
        min_rect_half_extent=1,
        max_rect_half_extent=2,
        min_detail_size=4,
        max_detail_size=8,
        min_trap_size=18,
        max_trap_size=26,
        target_detail_ratio_of_base_free=0.020,
        max_detail_ratio_of_base_free=0.024,
    ),
    2: MacroDetailPreset(
        2,
        trap_count=1,
        min_rect_half_extent=1,
        max_rect_half_extent=2,
        min_detail_size=5,
        max_detail_size=10,
        min_trap_size=20,
        max_trap_size=32,
        target_detail_ratio_of_base_free=0.035,
        max_detail_ratio_of_base_free=0.041,
    ),
    3: MacroDetailPreset(
        3,
        trap_count=2,
        min_rect_half_extent=1,
        max_rect_half_extent=3,
        min_detail_size=5,
        max_detail_size=12,
        min_trap_size=24,
        max_trap_size=38,
        target_detail_ratio_of_base_free=0.093,
        max_detail_ratio_of_base_free=0.103,
    ),
    4: MacroDetailPreset(
        4,
        trap_count=3,
        min_rect_half_extent=1,
        max_rect_half_extent=3,
        min_detail_size=6,
        max_detail_size=14,
        min_trap_size=24,
        max_trap_size=38,
        target_detail_ratio_of_base_free=0.165,
        max_detail_ratio_of_base_free=0.180,
    ),
}


def _rect_mask(size: int, r: int, c: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    r0 = int(np.clip(r, 0, size))
    c0 = int(np.clip(c, 0, size))
    r1 = int(np.clip(r0 + max(0, int(h)), 0, size))
    c1 = int(np.clip(c0 + max(0, int(w)), 0, size))
    if r1 > r0 and c1 > c0:
        mask[r0:r1, c0:c1] = True
    return mask


def _circle_mask(size: int, cy: int, cx: int, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    return (yy - int(cy)) * (yy - int(cy)) + (xx - int(cx)) * (xx - int(cx)) <= int(radius) * int(radius)


def _trap_pocket_mask(
    size: int,
    r: int,
    c: int,
    h: int,
    w: int,
    thickness: int,
    open_side: str,
    gap_width: int,
) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    t = max(1, int(thickness))
    gap = max(2, int(gap_width))
    mask |= _rect_mask(size, r, c, t, w)
    mask |= _rect_mask(size, r + h - t, c, t, w)
    mask |= _rect_mask(size, r, c, h, t)
    mask |= _rect_mask(size, r, c + w - t, h, t)

    if open_side in {"left", "right"}:
        center = r + h // 2
        g0 = int(np.clip(center - gap // 2, r + t, r + h - t - gap))
        if open_side == "left":
            mask[g0 : g0 + gap, c : c + t] = False
        else:
            mask[g0 : g0 + gap, c + w - t : c + w] = False
    else:
        center = c + w // 2
        g0 = int(np.clip(center - gap // 2, c + t, c + w - t - gap))
        if open_side == "up":
            mask[r : r + t, g0 : g0 + gap] = False
        else:
            mask[r + h - t : r + h, g0 : g0 + gap] = False
    return mask


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return 0, 0, 0, 0
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _bbox_mask(shape: Tuple[int, int], bbox: Tuple[int, int, int, int], pad: int = 0) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    rows, cols = shape
    out = np.zeros(shape, dtype=bool)
    r0 = max(0, int(r0) - int(pad))
    c0 = max(0, int(c0) - int(pad))
    r1 = min(rows, int(r1) + int(pad))
    c1 = min(cols, int(c1) + int(pad))
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = True
    return out


def _try_place_detail(
    grid: np.ndarray,
    candidate: np.ndarray,
    *,
    preset: MacroDetailPreset,
    start: GridPos,
    base_free_mask: np.ndarray,
) -> bool:
    if int(np.count_nonzero(candidate)) <= 0:
        return False
    area = int(np.count_nonzero(candidate))
    overlap = int(np.count_nonzero(candidate & (grid == 1)))
    if float(overlap) / float(max(1, area)) > float(preset.overlap_limit):
        return False
    test = grid.copy()
    test[candidate] = 1
    _clear_start(test, radius=4)
    detail_cells = int(np.count_nonzero((test == 1) & base_free_mask))
    base_free_cells = int(np.count_nonzero(base_free_mask))
    detail_ratio = float(detail_cells) / float(max(1, base_free_cells))
    if detail_ratio > float(preset.max_detail_ratio_of_base_free):
        return False
    if not _free_is_connected(test, start=start):
        return False
    grid[:, :] = test
    return True


def _detail_ratio_of_base_free(grid: np.ndarray, base_free_mask: np.ndarray) -> float:
    detail_cells = int(np.count_nonzero((grid == 1) & base_free_mask))
    base_free_cells = int(np.count_nonzero(base_free_mask))
    return float(detail_cells) / float(max(1, base_free_cells))


def _sample_detail_mask(size: int, rng: np.random.RandomState, preset: MacroDetailPreset, kind: str) -> np.ndarray:
    margin = max(6, size // 18)
    if kind == "block":
        if bool(rng.rand() < 0.35):
            h = int(rng.randint(preset.min_detail_size, preset.max_detail_size + 1))
            w = int(rng.randint(preset.min_detail_size, preset.max_detail_size + 1))
            r = int(rng.randint(margin, max(margin + 1, size - h - margin)))
            c = int(rng.randint(margin, max(margin + 1, size - w - margin)))
            radius = max(2, min(h, w) // 2)
            return _circle_mask(size, r + h // 2, c + w // 2, radius)

        # Match the rectangle size scale used by the paper shapegrid generator:
        # half extents 1-2 for L1/L2 and 1-3 for L3/L4.
        hh = int(rng.randint(preset.min_rect_half_extent, preset.max_rect_half_extent + 1))
        hw = int(rng.randint(preset.min_rect_half_extent, preset.max_rect_half_extent + 1))
        cy = int(rng.randint(margin + hh, max(margin + hh + 1, size - margin - hh)))
        cx = int(rng.randint(margin + hw, max(margin + hw + 1, size - margin - hw)))
        return _rect_mask(size, cy - hh, cx - hw, 2 * hh + 1, 2 * hw + 1)

    h = int(rng.randint(preset.min_trap_size, preset.max_trap_size + 1))
    w = int(rng.randint(preset.min_trap_size, preset.max_trap_size + 1))
    r = int(rng.randint(margin, max(margin + 1, size - h - margin)))
    c = int(rng.randint(margin, max(margin + 1, size - w - margin)))
    thickness = 1
    gap = int(rng.randint(2, max(3, min(h, w) // 4 + 1)))
    open_side = ("left", "right", "up", "down")[int(rng.randint(0, 4))]
    return _trap_pocket_mask(size, r, c, h, w, thickness, open_side, gap)


def _place_many(
    grid: np.ndarray,
    *,
    rng: np.random.RandomState,
    preset: MacroDetailPreset,
    kind: str,
    count: int,
    start: GridPos,
    base_free_mask: np.ndarray,
) -> int:
    placed = 0
    trap_reserved = np.zeros_like(grid, dtype=bool)
    for _ in range(max(0, int(count)) * 80):
        if placed >= int(count):
            break
        candidate = _sample_detail_mask(grid.shape[0], rng, preset, kind)
        reserve = None
        if kind == "trap":
            # Pocket walls are thin, so two pockets can visually overlap even
            # without many wall cells intersecting. Reserve the full pocket box.
            reserve = _bbox_mask(candidate.shape, _mask_bbox(candidate), pad=2)
            if np.any(reserve & trap_reserved):
                continue
        if _try_place_detail(
            grid,
            candidate,
            preset=preset,
            start=start,
            base_free_mask=base_free_mask,
        ):
            if reserve is not None:
                trap_reserved |= reserve
            placed += 1
    return placed


def _place_blocks_to_target_ratio(
    grid: np.ndarray,
    *,
    rng: np.random.RandomState,
    preset: MacroDetailPreset,
    start: GridPos,
    base_free_mask: np.ndarray,
) -> int:
    placed = 0
    base_free_cells = int(np.count_nonzero(base_free_mask))
    target_cells = int(np.ceil(float(preset.target_detail_ratio_of_base_free) * float(base_free_cells)))
    # Small shapegrid-style blocks have variable area, so use a generous attempt
    # budget and stop once the desired density is reached.
    max_attempts = max(200, target_cells * 3)
    for _ in range(max_attempts):
        if _detail_ratio_of_base_free(grid, base_free_mask) >= float(preset.target_detail_ratio_of_base_free):
            break
        candidate = _sample_detail_mask(grid.shape[0], rng, preset, "block")
        if _try_place_detail(
            grid,
            candidate,
            preset=preset,
            start=start,
            base_free_mask=base_free_mask,
        ):
            placed += 1
    return placed


def build_macro_detail_grid_map(
    *,
    size: int,
    seed: int,
    level: int = 2,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
):
    """Build a map from a large-obstacle base plus difficulty-controlled details."""

    size = max(24, int(size))
    level = int(level)
    if level not in PRESETS:
        raise ValueError(f"level must be one of {sorted(PRESETS)}; got {level}")
    preset = PRESETS[level]
    rng = np.random.RandomState(int(seed))
    start = (0, 0)

    grid, base_meta = build_large_obstacle_grid_map(
        size=size,
        seed=int(seed) + 7919,
        return_metadata=True,
    )
    grid = np.asarray(grid, dtype=np.int32).copy()
    base_free_mask = grid == 0

    traps = _place_many(
        grid,
        rng=rng,
        preset=preset,
        kind="trap",
        count=preset.trap_count,
        start=start,
        base_free_mask=base_free_mask,
    )
    blocks = _place_blocks_to_target_ratio(
        grid,
        rng=rng,
        preset=preset,
        start=start,
        base_free_mask=base_free_mask,
    )
    if ensure_start_clear:
        _clear_start(grid, radius=4)

    base_free_cells = int(np.count_nonzero(base_free_mask))
    detail_obstacle_cells = int(np.count_nonzero((grid == 1) & base_free_mask))

    meta = {
        "family": "macro_detail_grid",
        "size": int(size),
        "seed": int(seed),
        "level": int(level),
        "base_obstacle_count": int(base_meta["obstacle_count_placed"]),
        "target_detail_ratio_of_base_free": float(preset.target_detail_ratio_of_base_free),
        "max_detail_ratio_of_base_free": float(preset.max_detail_ratio_of_base_free),
        "block_count_placed": int(blocks),
        "trap_count_target": int(preset.trap_count),
        "trap_count_placed": int(traps),
        "base_free_cells": int(base_free_cells),
        "detail_obstacle_cells": int(detail_obstacle_cells),
        "detail_obstacle_ratio_of_base_free": float(detail_obstacle_cells)
        / float(max(1, base_free_cells)),
        "obstacle_ratio": float(np.count_nonzero(grid == 1)) / float(grid.size),
        "reachable_free_ratio": float(np.count_nonzero(_reachable_mask(grid, start=start)))
        / float(max(1, np.count_nonzero(grid == 0))),
    }
    return (grid, meta) if return_metadata else grid
