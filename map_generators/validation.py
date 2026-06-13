from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class MapValidationStats:
    rows: int
    cols: int
    free_cells: int
    obstacle_cells: int
    free_ratio: float
    obstacle_ratio: float
    free_component_count: int
    largest_free_component_cells: int
    largest_free_component_ratio: float
    start_row: int
    start_col: int
    start_is_free: bool
    start_component_cells: int
    start_component_ratio: float
    unreachable_free_cells_from_start: int

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def parse_map_txt(path: Path) -> np.ndarray:
    rows = [list(map(int, line.split())) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Map file is empty: {path}")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"Map rows have inconsistent widths in {path}: {sorted(widths)}")
    grid = np.asarray(rows, dtype=np.int32)
    if not np.isin(grid, [0, 1]).all():
        raise ValueError(f"Map file must contain only 0 and 1: {path}")
    return grid


def _neighbors(pos: GridPos, shape: Tuple[int, int]) -> Iterable[GridPos]:
    r, c = pos
    rows, cols = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _component_size(grid: np.ndarray, start: GridPos, visited: np.ndarray) -> int:
    if grid[start[0], start[1]] != 0:
        return 0
    q = deque([start])
    visited[start[0], start[1]] = True
    count = 0
    while q:
        cur = q.popleft()
        count += 1
        for nxt in _neighbors(cur, grid.shape):
            nr, nc = nxt
            if visited[nr, nc] or grid[nr, nc] != 0:
                continue
            visited[nr, nc] = True
            q.append(nxt)
    return int(count)


def analyze_grid_map(grid: np.ndarray, *, start: GridPos = (0, 0)) -> MapValidationStats:
    arr = np.asarray(grid, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Map must be a 2D array, got shape={arr.shape}")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("Map must contain only 0 and 1")

    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    free_cells = int(np.count_nonzero(arr == 0))
    obstacle_cells = int(np.count_nonzero(arr == 1))
    total = max(1, rows * cols)

    visited = np.zeros(arr.shape, dtype=bool)
    component_sizes: List[int] = []
    for r, c in np.argwhere(arr == 0):
        rr, cc = int(r), int(c)
        if visited[rr, cc]:
            continue
        component_sizes.append(_component_size(arr, (rr, cc), visited))

    largest = max(component_sizes) if component_sizes else 0
    sr, sc = int(start[0]), int(start[1])
    start_in_bounds = 0 <= sr < rows and 0 <= sc < cols
    start_is_free = bool(start_in_bounds and arr[sr, sc] == 0)
    start_component = 0
    if start_is_free:
        start_visited = np.zeros(arr.shape, dtype=bool)
        start_component = _component_size(arr, (sr, sc), start_visited)

    return MapValidationStats(
        rows=rows,
        cols=cols,
        free_cells=free_cells,
        obstacle_cells=obstacle_cells,
        free_ratio=float(free_cells) / float(total),
        obstacle_ratio=float(obstacle_cells) / float(total),
        free_component_count=int(len(component_sizes)),
        largest_free_component_cells=int(largest),
        largest_free_component_ratio=float(largest) / float(max(1, free_cells)),
        start_row=sr,
        start_col=sc,
        start_is_free=start_is_free,
        start_component_cells=int(start_component),
        start_component_ratio=float(start_component) / float(max(1, free_cells)),
        unreachable_free_cells_from_start=int(max(0, free_cells - start_component)),
    )


def map_passes_paper_checks(
    stats: MapValidationStats,
    *,
    min_start_component_ratio: float = 0.995,
    min_free_ratio: float = 0.50,
    max_obstacle_ratio: float = 0.45,
) -> bool:
    if not stats.start_is_free:
        return False
    if stats.start_component_ratio < float(min_start_component_ratio):
        return False
    if stats.free_ratio < float(min_free_ratio):
        return False
    if stats.obstacle_ratio > float(max_obstacle_ratio):
        return False
    return True
