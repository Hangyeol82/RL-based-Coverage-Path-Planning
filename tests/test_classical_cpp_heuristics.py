import numpy as np

from classical_cpp_heuristics import run_online_heuristic
from evaluation.cpp_metrics import compute_cpp_path_metrics


def _smoke_grid() -> np.ndarray:
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[2:6, 3] = 1
    grid[5, 4:7] = 1
    return grid


def test_online_cpp_heuristics_return_valid_paths() -> None:
    grid = _smoke_grid()
    for algorithm in ("nearest_unvisited", "frontier_greedy", "wall_follow", "spiral_stc"):
        result = run_online_heuristic(
            algorithm,
            grid,
            start=(0, 0),
            sensor_range=2,
            max_steps=500,
        )
        metrics = compute_cpp_path_metrics(grid, result.path)

        assert result.path[0] == (0, 0)
        assert metrics.steps <= 500
        assert metrics.invalid_step_count == 0
        assert metrics.coverage_ratio > 0.0
        assert result.stats["algorithm"] == algorithm


def test_spiral_stc_covers_reachable_smoke_grid() -> None:
    grid = _smoke_grid()
    result = run_online_heuristic(
        "spiral_stc",
        grid,
        start=(0, 0),
        sensor_range=2,
        max_steps=500,
    )
    metrics = compute_cpp_path_metrics(grid, result.path)

    assert metrics.coverage_ratio == 1.0
    assert metrics.success_99
