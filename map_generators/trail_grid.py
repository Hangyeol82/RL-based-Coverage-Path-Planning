from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .large_obstacle_grid import _clear_start, _free_is_connected, _reachable_mask


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class TrailPreset:
    level: int
    trunk_count: int
    trunk_width: int
    branch_width: int
    branch_count: int
    branch_steps: int
    loop_prob: float
    dead_end_prob: float


PRESETS: Dict[int, TrailPreset] = {
    1: TrailPreset(1, trunk_count=3, trunk_width=8, branch_width=6, branch_count=5, branch_steps=34, loop_prob=0.10, dead_end_prob=0.18),
    2: TrailPreset(2, trunk_count=4, trunk_width=7, branch_width=5, branch_count=9, branch_steps=42, loop_prob=0.18, dead_end_prob=0.26),
    3: TrailPreset(3, trunk_count=5, trunk_width=6, branch_width=5, branch_count=14, branch_steps=50, loop_prob=0.27, dead_end_prob=0.34),
    4: TrailPreset(4, trunk_count=6, trunk_width=5, branch_width=4, branch_count=20, branch_steps=58, loop_prob=0.36, dead_end_prob=0.42),
}


def _neighbors(r: int, c: int, rows: int, cols: int) -> Iterable[GridPos]:
    if r > 0:
        yield r - 1, c
    if r + 1 < rows:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < cols:
        yield r, c + 1


def _carve_disc(grid: np.ndarray, cy: float, cx: float, radius: float) -> None:
    rows, cols = grid.shape
    r = max(1.0, float(radius))
    r0 = max(0, int(np.floor(cy - r - 1)))
    r1 = min(rows, int(np.ceil(cy + r + 2)))
    c0 = max(0, int(np.floor(cx - r - 1)))
    c1 = min(cols, int(np.ceil(cx + r + 2)))
    yy, xx = np.ogrid[r0:r1, c0:c1]
    mask = (yy + 0.5 - cy) * (yy + 0.5 - cy) + (xx + 0.5 - cx) * (xx + 0.5 - cx) <= r * r
    grid[r0:r1, c0:c1][mask] = 0


def _carve_line(grid: np.ndarray, a: GridPos, b: GridPos, width: float) -> List[GridPos]:
    ar, ac = float(a[0]), float(a[1])
    br, bc = float(b[0]), float(b[1])
    dist = max(abs(br - ar), abs(bc - ac), 1.0)
    steps = int(np.ceil(dist * 1.6))
    points: List[GridPos] = []
    for t in np.linspace(0.0, 1.0, steps + 1):
        r = ar * (1.0 - t) + br * t
        c = ac * (1.0 - t) + bc * t
        _carve_disc(grid, r, c, width / 2.0)
        points.append((int(round(r)), int(round(c))))
    return points


def _make_polyline(
    *,
    rng: np.random.RandomState,
    start: GridPos,
    end: GridPos,
    size: int,
    waypoint_count: int,
    jitter: float,
) -> List[GridPos]:
    pts = [start]
    sr, sc = start
    er, ec = end
    for i in range(1, waypoint_count + 1):
        t = float(i) / float(waypoint_count + 1)
        r = sr * (1.0 - t) + er * t + rng.uniform(-jitter, jitter)
        c = sc * (1.0 - t) + ec * t + rng.uniform(-jitter, jitter)
        pts.append((int(np.clip(round(r), 1, size - 2)), int(np.clip(round(c), 1, size - 2))))
    pts.append(end)
    return pts


def _carve_polyline(grid: np.ndarray, pts: List[GridPos], width: float) -> List[GridPos]:
    carved: List[GridPos] = []
    for a, b in zip(pts[:-1], pts[1:]):
        carved.extend(_carve_line(grid, a, b, width))
    return carved


def _polyline_mask(shape: Tuple[int, int], pts: List[GridPos], width: float) -> np.ndarray:
    tmp = np.ones(shape, dtype=np.int32)
    _carve_polyline(tmp, pts, width)
    return tmp == 0


def _try_carve_polyline(
    grid: np.ndarray,
    pts: List[GridPos],
    *,
    width: float,
    max_overlap_ratio: float,
    min_new_cells: int,
) -> bool:
    mask = _polyline_mask(grid.shape, pts, width)
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return False
    existing = int(np.count_nonzero(mask & (grid == 0)))
    new_cells = int(np.count_nonzero(mask & (grid == 1)))
    overlap_ratio = float(existing) / float(max(1, area))
    if new_cells < int(min_new_cells) or overlap_ratio > float(max_overlap_ratio):
        return False
    grid[mask] = 0
    return True


def _random_edge_point(rng: np.random.RandomState, size: int, *, avoid_start: bool = False) -> GridPos:
    margin = max(8, size // 10)
    side = int(rng.randint(0, 4))
    if avoid_start and side in {0, 3}:
        side = int(rng.randint(1, 3))
    if side == 0:
        return 0, int(rng.randint(margin, size - margin))
    if side == 1:
        return size - 1, int(rng.randint(margin, size - margin))
    if side == 2:
        return int(rng.randint(margin, size - margin)), size - 1
    return int(rng.randint(margin, size - margin)), 0


def _edge_point_for_side(rng: np.random.RandomState, size: int, side: int) -> GridPos:
    margin = max(8, size // 10)
    if side == 0:
        return 0, int(rng.randint(margin, size - margin))
    if side == 1:
        return size - 1, int(rng.randint(margin, size - margin))
    if side == 2:
        return int(rng.randint(margin, size - margin)), size - 1
    return int(rng.randint(margin, size - margin)), 0


def _free_cells(grid: np.ndarray) -> np.ndarray:
    return np.argwhere(grid == 0)


def _random_free_point(rng: np.random.RandomState, grid: np.ndarray) -> GridPos:
    free = _free_cells(grid)
    if free.size == 0:
        return 0, 0
    idx = int(rng.randint(0, len(free)))
    return int(free[idx, 0]), int(free[idx, 1])


def _nearest_free_point(grid: np.ndarray, target: GridPos) -> GridPos:
    free = _free_cells(grid)
    if free.size == 0:
        return 0, 0
    tr, tc = int(target[0]), int(target[1])
    d2 = (free[:, 0] - tr) * (free[:, 0] - tr) + (free[:, 1] - tc) * (free[:, 1] - tc)
    idx = int(np.argmin(d2))
    return int(free[idx, 0]), int(free[idx, 1])


def _farthest_blocked_point(
    rng: np.random.RandomState,
    grid: np.ndarray,
    *,
    samples: int,
    edge_prob: float,
) -> GridPos:
    size = int(grid.shape[0])
    free = _free_cells(grid)
    if free.size == 0:
        return _random_edge_point(rng, size, avoid_start=True)
    if len(free) > 900:
        ref = free[rng.choice(len(free), size=900, replace=False)]
    else:
        ref = free

    margin = max(4, size // 18)
    best: GridPos = _random_edge_point(rng, size, avoid_start=True)
    best_score = -1.0
    for _ in range(max(1, int(samples))):
        if bool(rng.rand() < float(edge_prob)):
            cand = _random_edge_point(rng, size, avoid_start=False)
        else:
            cand = (
                int(rng.randint(margin, size - margin)),
                int(rng.randint(margin, size - margin)),
            )
        if int(grid[cand[0], cand[1]]) == 0:
            continue
        d2 = (ref[:, 0] - cand[0]) * (ref[:, 0] - cand[0]) + (ref[:, 1] - cand[1]) * (
            ref[:, 1] - cand[1]
        )
        edge_dist = min(cand[0], cand[1], size - 1 - cand[0], size - 1 - cand[1])
        edge_bonus = float(size - edge_dist) * 0.15
        score = float(np.min(d2)) + edge_bonus
        if score > best_score:
            best_score = score
            best = cand
    return best


def _carve_global_trunks(
    grid: np.ndarray,
    *,
    rng: np.random.RandomState,
    preset: TrailPreset,
    size: int,
    start: GridPos,
) -> int:
    sides = [1, 2, 0, 3]
    rng.shuffle(sides)
    carved = 0
    for i in range(int(preset.trunk_count)):
        for attempt in range(50):
            if i == 0:
                end = _edge_point_for_side(rng, size, sides[i % len(sides)])
                trunk_start = start
            elif attempt < 12:
                end = _edge_point_for_side(rng, size, sides[i % len(sides)])
                trunk_start = _nearest_free_point(grid, end)
            else:
                end = _farthest_blocked_point(rng, grid, samples=120, edge_prob=0.55)
                trunk_start = _nearest_free_point(grid, end)
            trunk = _make_polyline(
                rng=rng,
                start=trunk_start,
                end=end,
                size=size,
                waypoint_count=int(rng.randint(2, 5)),
                jitter=float(size) * 0.09,
            )
            if _try_carve_polyline(
                grid,
                trunk,
                width=float(preset.trunk_width),
                max_overlap_ratio=0.42 if i == 0 else 0.26,
                min_new_cells=max(size // 2, int(preset.trunk_width * size * 0.25)),
            ):
                carved += 1
                break
    return carved


def _carve_branch(
    grid: np.ndarray,
    *,
    rng: np.random.RandomState,
    preset: TrailPreset,
    size: int,
) -> Tuple[bool, str]:
    for _ in range(24):
        end = _farthest_blocked_point(rng, grid, samples=90, edge_prob=0.35)
        start = _nearest_free_point(grid, end)
        pts = _make_polyline(
            rng=rng,
            start=start,
            end=end,
            size=size,
            waypoint_count=int(rng.randint(1, 4)),
            jitter=float(size) * 0.07,
        )
        branch_kind = "dead_end"
        if bool(rng.rand() < float(preset.loop_prob)):
            pts.append(_nearest_free_point(grid, _random_edge_point(rng, size, avoid_start=False)))
            branch_kind = "loop"
        elif not bool(rng.rand() < float(preset.dead_end_prob)):
            pts.append(_nearest_free_point(grid, _farthest_blocked_point(rng, grid, samples=60, edge_prob=0.25)))
            branch_kind = "connector"
        if _try_carve_polyline(
            grid,
            pts,
            width=float(preset.branch_width),
            max_overlap_ratio=0.30,
            min_new_cells=max(size // 4, int(preset.branch_width * size * 0.12)),
        ):
            return True, branch_kind
    return False, "rejected"


def _free_bbox_area_ratio(grid: np.ndarray) -> float:
    free = np.argwhere(grid == 0)
    if free.size == 0:
        return 0.0
    r0, c0 = free.min(axis=0)
    r1, c1 = free.max(axis=0) + 1
    return float((int(r1) - int(r0)) * (int(c1) - int(c0))) / float(grid.size)


def _trail_graph_counts(grid: np.ndarray) -> Tuple[int, int, int]:
    free = grid == 0
    rows, cols = grid.shape
    dead_ends = 0
    junctions = 0
    corridor_cells = 0
    for r, c in zip(*np.where(free)):
        deg = sum(bool(free[nr, nc]) for nr, nc in _neighbors(int(r), int(c), rows, cols))
        if deg <= 1:
            dead_ends += 1
        elif deg >= 3:
            junctions += 1
        else:
            corridor_cells += 1
    return int(junctions), int(dead_ends), int(corridor_cells)


def _thin_dead_ends(grid: np.ndarray, rng: np.random.RandomState, *, passes: int) -> None:
    for _ in range(max(0, int(passes))):
        free = grid == 0
        remove = np.zeros_like(free, dtype=bool)
        rows, cols = grid.shape
        for r, c in zip(*np.where(free)):
            if r < 5 and c < 5:
                continue
            deg = sum(bool(free[nr, nc]) for nr, nc in _neighbors(int(r), int(c), rows, cols))
            if deg <= 1 and bool(rng.rand() < 0.55):
                remove[r, c] = True
        grid[remove] = 1


def build_trail_grid_map(
    *,
    size: int,
    seed: int,
    level: int = 2,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
):
    """Build a high-obstacle map whose free space is a branching trail network."""

    size = max(32, int(size))
    level = int(level)
    if level not in PRESETS:
        raise ValueError(f"level must be one of {sorted(PRESETS)}; got {level}")
    preset = PRESETS[level]
    rng = np.random.RandomState(int(seed))
    start = (0, 0)
    grid = np.ones((size, size), dtype=np.int32)

    trunk_placed = _carve_global_trunks(grid, rng=rng, preset=preset, size=size, start=start)
    if ensure_start_clear:
        _clear_start(grid, radius=max(4, preset.trunk_width))

    branch_placed = 0
    loop_branches = 0
    connector_branches = 0
    dead_end_branches = 0
    max_attempts = int(preset.branch_count * 16)
    for _ in range(max_attempts):
        if branch_placed >= int(preset.branch_count):
            break
        placed, branch_kind = _carve_branch(grid, rng=rng, preset=preset, size=size)
        if placed:
            branch_placed += 1
            if branch_kind == "loop":
                loop_branches += 1
            elif branch_kind == "connector":
                connector_branches += 1
            elif branch_kind == "dead_end":
                dead_end_branches += 1

    # Add a few small rounded trail widenings to avoid every path becoming a
    # uniform tube.
    free = _free_cells(grid)
    for _ in range(max(2, int(level) + 1)):
        if len(free) == 0:
            break
        r, c = free[int(rng.randint(0, len(free)))]
        _carve_disc(grid, float(r), float(c), float(rng.uniform(preset.branch_width, preset.trunk_width + 2)))

    _thin_dead_ends(grid, rng, passes=max(0, 3 - level))
    if ensure_start_clear:
        _clear_start(grid, radius=max(4, preset.trunk_width))

    # If thinning damaged connectivity, keep only the reachable trail network.
    reachable = _reachable_mask(grid, start=start)
    grid[(grid == 0) & (~reachable)] = 1
    junction_count, dead_end_count, corridor_cell_count = _trail_graph_counts(grid)

    meta = {
        "family": "trail_grid",
        "size": int(size),
        "seed": int(seed),
        "level": int(level),
        "trunk_count_target": int(preset.trunk_count),
        "trunk_count_placed": int(trunk_placed),
        "branch_count_target": int(preset.branch_count),
        "branch_count_placed": int(branch_placed),
        "loop_branch_count": int(loop_branches),
        "connector_branch_count": int(connector_branches),
        "dead_end_branch_count": int(dead_end_branches),
        "junction_count": int(junction_count),
        "dead_end_count": int(dead_end_count),
        "corridor_cell_count": int(corridor_cell_count),
        "free_ratio": float(np.count_nonzero(grid == 0)) / float(grid.size),
        "obstacle_ratio": float(np.count_nonzero(grid == 1)) / float(grid.size),
        "free_bbox_area_ratio": float(_free_bbox_area_ratio(grid)),
        "reachable_free_ratio": float(np.count_nonzero(_reachable_mask(grid, start=start)))
        / float(max(1, np.count_nonzero(grid == 0))),
        "free_connected": bool(_free_is_connected(grid, start=start)),
    }
    return (grid, meta) if return_metadata else grid
