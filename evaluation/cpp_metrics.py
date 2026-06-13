from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class CPPPathMetrics:
    steps: int
    path_length: int
    free_cells: int
    coverage_cells: int
    coverage_ratio: float
    revisit_count: int
    revisit_ratio: float
    overlap_count: int
    overlap_ratio: float
    unique_positions: int
    invalid_step_count: int
    obstacle_step_count: int
    out_of_bounds_step_count: int
    step_to_90: Optional[int]
    step_to_95: Optional[int]
    step_to_99: Optional[int]
    success_90: bool
    success_95: bool
    success_99: bool

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _as_pos_list(path: Iterable[Sequence[int]]) -> List[GridPos]:
    out: List[GridPos] = []
    for p in path:
        if len(p) < 2:
            raise ValueError(f"Path position must have at least 2 elements: {p}")
        out.append((int(p[0]), int(p[1])))
    return out


def _threshold_key(threshold: float) -> str:
    return f"step_to_{int(round(float(threshold) * 100.0))}"


def compute_cpp_path_metrics(
    grid: np.ndarray,
    path: Iterable[Sequence[int]],
    *,
    thresholds: Sequence[float] = (0.90, 0.95, 0.99),
) -> CPPPathMetrics:
    arr = np.asarray(grid, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape={arr.shape}")
    positions = _as_pos_list(path)
    if not positions:
        positions = []

    free_mask = arr == 0
    free_cells = int(np.count_nonzero(free_mask))
    rows, cols = arr.shape
    visited_free: set[GridPos] = set()
    unique_positions: set[GridPos] = set()
    threshold_steps: Dict[str, Optional[int]] = {
        _threshold_key(float(t)): None for t in thresholds
    }
    revisit_count = 0
    obstacle_step_count = 0
    out_of_bounds_step_count = 0

    def coverage_ratio() -> float:
        return float(len(visited_free)) / float(max(1, free_cells))

    for step, pos in enumerate(positions):
        r, c = pos
        unique_positions.add(pos)
        valid = 0 <= r < rows and 0 <= c < cols
        if not valid:
            if step > 0:
                out_of_bounds_step_count += 1
            continue
        if arr[r, c] != 0:
            if step > 0:
                obstacle_step_count += 1
            continue
        if pos in visited_free:
            if step > 0:
                revisit_count += 1
        else:
            visited_free.add(pos)

        ratio = coverage_ratio()
        for threshold in thresholds:
            key = _threshold_key(float(threshold))
            if threshold_steps[key] is None and ratio >= float(threshold):
                threshold_steps[key] = int(step)

    steps = max(0, len(positions) - 1)
    invalid_step_count = int(obstacle_step_count + out_of_bounds_step_count)
    overlap_count = int(revisit_count)
    overlap_ratio = float(overlap_count) / float(max(1, steps))
    revisit_ratio = float(revisit_count) / float(max(1, steps))

    return CPPPathMetrics(
        steps=int(steps),
        path_length=int(len(positions)),
        free_cells=int(free_cells),
        coverage_cells=int(len(visited_free)),
        coverage_ratio=coverage_ratio(),
        revisit_count=int(revisit_count),
        revisit_ratio=revisit_ratio,
        overlap_count=overlap_count,
        overlap_ratio=overlap_ratio,
        unique_positions=int(len(unique_positions)),
        invalid_step_count=invalid_step_count,
        obstacle_step_count=int(obstacle_step_count),
        out_of_bounds_step_count=int(out_of_bounds_step_count),
        step_to_90=threshold_steps.get("step_to_90"),
        step_to_95=threshold_steps.get("step_to_95"),
        step_to_99=threshold_steps.get("step_to_99"),
        success_90=threshold_steps.get("step_to_90") is not None,
        success_95=threshold_steps.get("step_to_95") is not None,
        success_99=threshold_steps.get("step_to_99") is not None,
    )
