from typing import List, Sequence, Tuple

import numpy as np


def _point_side(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    return (px - bx) * (ay - by) - (ax - bx) * (py - by)


def _inside_triangle(
    px: float,
    py: float,
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> bool:
    d1 = _point_side(px, py, a[1], a[0], b[1], b[0])
    d2 = _point_side(px, py, b[1], b[0], c[1], c[0])
    d3 = _point_side(px, py, c[1], c[0], a[1], a[0])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _shape_mask_rect(h: int, w: int, cy: int, cx: int, hh: int, hw: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (np.abs(yy - cy) <= hh) & (np.abs(xx - cx) <= hw)


def _shape_mask_circle(h: int, w: int, cy: int, cx: int, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) * (yy - cy) + (xx - cx) * (xx - cx) <= radius * radius


def _triangle_vertices(cy: int, cx: int, hh: int, hw: int, orientation: str) -> Tuple[Tuple[float, float], ...]:
    if orientation == "up":
        return ((cy - hh, cx), (cy + hh, cx - hw), (cy + hh, cx + hw))
    if orientation == "down":
        return ((cy + hh, cx), (cy - hh, cx - hw), (cy - hh, cx + hw))
    if orientation == "left":
        return ((cy, cx - hw), (cy - hh, cx + hw), (cy + hh, cx + hw))
    return ((cy, cx + hw), (cy - hh, cx - hw), (cy + hh, cx - hw))


def _shape_mask_triangle(
    h: int,
    w: int,
    cy: int,
    cx: int,
    hh: int,
    hw: int,
    orientation: str,
) -> np.ndarray:
    tri = np.zeros((h, w), dtype=bool)
    a, b, c = _triangle_vertices(cy=cy, cx=cx, hh=hh, hw=hw, orientation=orientation)
    r0 = max(0, int(min(a[0], b[0], c[0])) - 1)
    r1 = min(h, int(max(a[0], b[0], c[0])) + 2)
    c0 = max(0, int(min(a[1], b[1], c[1])) - 1)
    c1 = min(w, int(max(a[1], b[1], c[1])) + 2)
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            if _inside_triangle(rr + 0.5, cc + 0.5, a, b, c):
                tri[rr, cc] = True
    return tri


def _has_neighbor_conflict(
    occupied: np.ndarray,
    candidate: np.ndarray,
    clearance: int,
) -> bool:
    rows, cols = np.where(candidate)
    if rows.size == 0:
        return True
    h, w = occupied.shape
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        r0 = max(0, rr - clearance)
        r1 = min(h, rr + clearance + 1)
        c0 = max(0, cc - clearance)
        c1 = min(w, cc + clearance + 1)
        if np.any(occupied[r0:r1, c0:c1]):
            return True
    return False


def _grid_centers(size: int, step: int, border: int) -> List[Tuple[int, int]]:
    ys: List[int] = []
    xs: List[int] = []
    y = border + step // 2
    while y < size - border:
        ys.append(y)
        y += step
    x = border + step // 2
    while x < size - border:
        xs.append(x)
        x += step
    return [(yy, xx) for yy in ys for xx in xs]


def build_shape_grid_map(
    *,
    size: int,
    seed: int,
    grid_step: int = 10,
    spawn_prob: float = 0.75,
    min_half_extent: int = 2,
    max_half_extent: int = 4,
    jitter: int = 2,
    clearance: int = 1,
    shape_types: Sequence[str] = ("rect", "triangle", "circle"),
    ensure_start_clear: bool = True,
) -> np.ndarray:
    size = int(size)
    grid_step = max(6, int(grid_step))
    spawn_prob = float(np.clip(spawn_prob, 0.05, 1.0))
    min_half_extent = max(1, int(min_half_extent))
    max_half_extent = max(min_half_extent, int(max_half_extent))
    jitter = max(0, int(jitter))
    clearance = max(0, int(clearance))
    types = [t for t in shape_types if t in {"rect", "triangle", "circle"}]
    if not types:
        types = ["rect", "triangle", "circle"]

    rng = np.random.RandomState(seed)
    grid = np.zeros((size, size), dtype=np.int32)
    occupied = grid == 1

    border = max(max_half_extent + clearance + 2, 2)
    anchors = _grid_centers(size=size, step=grid_step, border=border)
    rng.shuffle(anchors)

    for ay, ax in anchors:
        if rng.rand() > spawn_prob:
            continue

        cy = int(np.clip(ay + rng.randint(-jitter, jitter + 1), border, size - border - 1))
        cx = int(np.clip(ax + rng.randint(-jitter, jitter + 1), border, size - border - 1))
        hh = int(rng.randint(min_half_extent, max_half_extent + 1))
        hw = int(rng.randint(min_half_extent, max_half_extent + 1))
        shape = str(types[int(rng.randint(0, len(types)))])

        if shape == "rect":
            candidate = _shape_mask_rect(size, size, cy, cx, hh, hw)
        elif shape == "circle":
            candidate = _shape_mask_circle(size, size, cy, cx, radius=min(hh, hw))
        else:
            orientation = ("up", "down", "left", "right")[int(rng.randint(0, 4))]
            candidate = _shape_mask_triangle(size, size, cy, cx, hh, hw, orientation)

        if np.count_nonzero(candidate) < 6:
            continue
        if _has_neighbor_conflict(occupied, candidate, clearance):
            continue

        grid[candidate] = 1
        occupied = grid == 1

    if ensure_start_clear:
        grid[0, 0] = 0
        if size > 1:
            grid[0, 1] = 0
            grid[1, 0] = 0
    return grid
