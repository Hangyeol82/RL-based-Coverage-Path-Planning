from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class LargeObstacleBaseConfig:
    min_free_ratio: float = 0.58
    max_obstacle_ratio: float = 0.42
    one_obstacle_prob: float = 0.45
    overlap_limit: float = 0.02
    edge_clip_prob: float = 0.45
    min_visible_fraction: float = 0.58


DEFAULT_CONFIG = LargeObstacleBaseConfig()


def _neighbors(r: int, c: int, rows: int, cols: int) -> Iterable[GridPos]:
    if r > 0:
        yield r - 1, c
    if r + 1 < rows:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < cols:
        yield r, c + 1


def _reachable_mask(grid: np.ndarray, start: GridPos = (0, 0)) -> np.ndarray:
    rows, cols = grid.shape
    sr, sc = start
    seen = np.zeros_like(grid, dtype=bool)
    if not (0 <= sr < rows and 0 <= sc < cols) or int(grid[sr, sc]) != 0:
        return seen
    stack = [(int(sr), int(sc))]
    seen[sr, sc] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, rows, cols):
            if seen[nr, nc] or int(grid[nr, nc]) != 0:
                continue
            seen[nr, nc] = True
            stack.append((nr, nc))
    return seen


def _free_is_connected(grid: np.ndarray, start: GridPos = (0, 0)) -> bool:
    free = np.asarray(grid) == 0
    free_count = int(np.count_nonzero(free))
    if free_count <= 0:
        return False
    seen = _reachable_mask(grid, start=start)
    return int(np.count_nonzero(seen & free)) == free_count


def _clear_start(grid: np.ndarray, radius: int = 4) -> None:
    rr = min(grid.shape[0], max(1, int(radius) + 1))
    cc = min(grid.shape[1], max(1, int(radius) + 1))
    grid[:rr, :cc] = 0


def _rect_mask(size: int, r: int, c: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    r0 = int(np.clip(r, 0, size))
    c0 = int(np.clip(c, 0, size))
    r1 = int(np.clip(r0 + max(0, int(h)), 0, size))
    c1 = int(np.clip(c0 + max(0, int(w)), 0, size))
    if r1 > r0 and c1 > c0:
        mask[r0:r1, c0:c1] = True
    return mask


def _compound_obstacle_mask(
    *,
    size: int,
    rng: np.random.RandomState,
    large: bool,
    config: LargeObstacleBaseConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    margin = max(7, size // 12)
    if large:
        min_radius = size * 0.20
        max_radius = size * 0.30
    else:
        min_radius = size * 0.14
        max_radius = size * 0.22

    rx = float(rng.uniform(min_radius, max_radius))
    ry = float(rng.uniform(min_radius, max_radius))
    # Keep the one-obstacle case larger, but not so dominant that it becomes
    # the entire task.
    if large and bool(rng.rand() < 0.5):
        rx *= float(rng.uniform(1.02, 1.12))
    else:
        ry *= float(rng.uniform(1.02, 1.12))

    clipped = bool(rng.rand() < float(config.edge_clip_prob))
    edge_axis = str(("top", "bottom", "left", "right")[int(rng.randint(0, 4))])
    if clipped:
        # Slightly enlarge edge-clipped blobs so the visible part keeps a
        # comparable scale while the implicit shape continues outside the map.
        rx *= float(rng.uniform(1.06, 1.20))
        ry *= float(rng.uniform(1.06, 1.20))

    bound = int(np.ceil(max(rx, ry) * 1.15)) + margin
    if bound * 2 >= size:
        bound = max(margin + 1, size // 3)

    def sample_center() -> Tuple[float, float]:
        if not clipped:
            return (
                float(rng.randint(bound, max(bound + 1, size - bound))),
                float(rng.randint(bound, max(bound + 1, size - bound))),
            )
        inner = max(margin, int(np.ceil(max(rx, ry) * 0.35)))
        outer = max(inner + 1, int(np.ceil(max(rx, ry) * 0.90)))
        cy0 = float(rng.randint(inner, max(inner + 1, size - inner)))
        cx0 = float(rng.randint(inner, max(inner + 1, size - inner)))
        if edge_axis == "top":
            cy0 = float(rng.uniform(-0.08 * ry, outer))
        elif edge_axis == "bottom":
            cy0 = float(rng.uniform(size - outer, size + 0.08 * ry))
        elif edge_axis == "left":
            cx0 = float(rng.uniform(-0.08 * rx, outer))
        else:
            cx0 = float(rng.uniform(size - outer, size + 0.08 * rx))
        return cy0, cx0

    cy, cx = sample_center()
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    ca, sa = np.cos(angle), np.sin(angle)
    dx = xx.astype(np.float32) + 0.5 - cx
    dy = yy.astype(np.float32) + 0.5 - cy
    x = ca * dx + sa * dy
    y = -sa * dx + ca * dy

    segments = 32
    profile = rng.uniform(0.82, 1.16, size=segments).astype(np.float32)
    for _ in range(4):
        profile = (np.roll(profile, 1) + profile + np.roll(profile, -1)) / 3.0
    theta = (np.arctan2(y / max(1.0, ry), x / max(1.0, rx)) + 2.0 * np.pi) % (
        2.0 * np.pi
    )
    pos = theta * float(segments) / (2.0 * np.pi)
    idx0 = np.floor(pos).astype(np.int32) % segments
    idx1 = (idx0 + 1) % segments
    frac = pos - np.floor(pos)
    boundary = profile[idx0] * (1.0 - frac) + profile[idx1] * frac

    exponent = float(rng.uniform(3.0, 5.5))
    norm = (np.abs(x / max(1.0, rx)) ** exponent + np.abs(y / max(1.0, ry)) ** exponent) ** (
        1.0 / exponent
    )
    mask = norm <= boundary
    if not clipped:
        return mask

    # Reject candidates that are effectively outside the map. The reference
    # area is computed with the same implicit blob on a padded virtual canvas.
    pad = int(np.ceil(max(rx, ry) * 1.4)) + 3
    v_size = size + 2 * pad
    vyy, vxx = np.mgrid[:v_size, :v_size]
    vdx = vxx.astype(np.float32) + 0.5 - (cx + pad)
    vdy = vyy.astype(np.float32) + 0.5 - (cy + pad)
    vx = ca * vdx + sa * vdy
    vy = -sa * vdx + ca * vdy
    vtheta = (np.arctan2(vy / max(1.0, ry), vx / max(1.0, rx)) + 2.0 * np.pi) % (
        2.0 * np.pi
    )
    vpos = vtheta * float(segments) / (2.0 * np.pi)
    vidx0 = np.floor(vpos).astype(np.int32) % segments
    vidx1 = (vidx0 + 1) % segments
    vfrac = vpos - np.floor(vpos)
    vboundary = profile[vidx0] * (1.0 - vfrac) + profile[vidx1] * vfrac
    vnorm = (
        np.abs(vx / max(1.0, rx)) ** exponent + np.abs(vy / max(1.0, ry)) ** exponent
    ) ** (1.0 / exponent)
    full_area = int(np.count_nonzero(vnorm <= vboundary))
    visible_area = int(np.count_nonzero(mask))
    if visible_area <= 0:
        return np.zeros((size, size), dtype=bool)
    visible_fraction = float(visible_area) / float(max(1, full_area))
    if visible_fraction < float(config.min_visible_fraction):
        return np.zeros((size, size), dtype=bool)
    return mask


def _try_place(
    grid: np.ndarray,
    candidate: np.ndarray,
    *,
    config: LargeObstacleBaseConfig,
    start: GridPos,
) -> bool:
    if int(np.count_nonzero(candidate)) <= 0:
        return False
    area = int(np.count_nonzero(candidate))
    overlap = int(np.count_nonzero(candidate & (grid == 1)))
    if float(overlap) / float(max(1, area)) > float(config.overlap_limit):
        return False
    test = grid.copy()
    test[candidate] = 1
    _clear_start(test, radius=4)
    obs_ratio = float(np.count_nonzero(test == 1)) / float(test.size)
    free_ratio = 1.0 - obs_ratio
    if obs_ratio > float(config.max_obstacle_ratio) or free_ratio < float(config.min_free_ratio):
        return False
    if not _free_is_connected(test, start=start):
        return False
    grid[:, :] = test
    return True


def build_large_obstacle_grid_map(
    *,
    size: int,
    seed: int,
    obstacle_count: Optional[int] = None,
    level: Optional[int] = None,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
):
    """
    Build a base map with one or two large kneaded obstacles.

    This is intentionally not a difficulty curriculum. It creates coarse map
    diversity before smaller structured generators are mixed in.
    """

    del level  # Kept only for backwards-compatible callers.
    size = max(24, int(size))
    rng = np.random.RandomState(int(seed))
    config = DEFAULT_CONFIG
    start = (0, 0)
    grid = np.zeros((size, size), dtype=np.int32)

    if obstacle_count is None:
        count = 1 if rng.rand() < float(config.one_obstacle_prob) else 2
    else:
        count = int(np.clip(int(obstacle_count), 1, 2))

    placed = 0
    for _ in range(count * 200):
        if placed >= count:
            break
        candidate = _compound_obstacle_mask(size=size, rng=rng, large=(count == 1), config=config)
        if _try_place(grid, candidate, config=config, start=start):
            placed += 1

    if ensure_start_clear:
        _clear_start(grid, radius=4)

    meta = {
        "family": "large_obstacle_grid",
        "size": int(size),
        "seed": int(seed),
        "obstacle_count_target": int(count),
        "obstacle_count_placed": int(placed),
        "obstacle_ratio": float(np.count_nonzero(grid == 1)) / float(grid.size),
        "reachable_free_ratio": float(np.count_nonzero(_reachable_mask(grid, start=start)))
        / float(max(1, np.count_nonzero(grid == 0))),
    }
    return (grid, meta) if return_metadata else grid
