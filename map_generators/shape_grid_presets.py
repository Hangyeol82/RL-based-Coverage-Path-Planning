from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .shape_grid import build_shape_grid_map
from .validation import MapValidationStats, analyze_grid_map, map_passes_paper_checks


@dataclass(frozen=True)
class ShapeGridPreset:
    level: int
    name: str
    grid_step: int
    spawn_prob: float
    min_half_extent: int
    max_half_extent: int
    jitter: int
    clearance: int
    shape_types: Tuple[str, ...]
    pocket_min_half_extent: int | None = None
    pocket_max_half_extent: int | None = None
    pocket_thickness: int = 1
    pocket_count_min: int = 0
    pocket_count_max: int = 0
    pocket_clearance: int = 0


PAPER_SHAPE_GRID_PRESETS: Dict[int, ShapeGridPreset] = {
    1: ShapeGridPreset(
        level=1,
        name="easy",
        grid_step=14,
        spawn_prob=0.55,
        min_half_extent=1,
        max_half_extent=2,
        jitter=1,
        clearance=1,
        shape_types=("rect", "circle"),
    ),
    2: ShapeGridPreset(
        level=2,
        name="easy_plus",
        grid_step=12,
        spawn_prob=0.70,
        min_half_extent=1,
        max_half_extent=2,
        jitter=1,
        clearance=1,
        shape_types=("rect", "circle"),
        pocket_min_half_extent=9,
        pocket_max_half_extent=12,
        pocket_thickness=1,
        pocket_count_min=1,
        pocket_count_max=2,
        pocket_clearance=0,
    ),
    3: ShapeGridPreset(
        level=3,
        name="medium",
        grid_step=10,
        spawn_prob=0.78,
        min_half_extent=1,
        max_half_extent=3,
        jitter=1,
        clearance=1,
        shape_types=("rect", "circle"),
        pocket_min_half_extent=10,
        pocket_max_half_extent=13,
        pocket_thickness=1,
        pocket_count_min=2,
        pocket_count_max=3,
        pocket_clearance=0,
    ),
    4: ShapeGridPreset(
        level=4,
        name="hard",
        grid_step=8,
        spawn_prob=0.86,
        min_half_extent=1,
        max_half_extent=3,
        jitter=1,
        clearance=0,
        shape_types=("rect", "circle"),
        pocket_min_half_extent=10,
        pocket_max_half_extent=14,
        pocket_thickness=1,
        pocket_count_min=3,
        pocket_count_max=4,
        pocket_clearance=0,
    ),
}


def get_paper_shape_grid_preset(level: int) -> ShapeGridPreset:
    lv = int(level)
    if lv not in PAPER_SHAPE_GRID_PRESETS:
        raise ValueError(f"Unsupported shape-grid paper level: {level}")
    return PAPER_SHAPE_GRID_PRESETS[lv]


def build_shape_grid_map_from_preset(
    *,
    size: int,
    seed: int,
    preset: ShapeGridPreset,
    shape_types: Sequence[str] | None = None,
) -> np.ndarray:
    return build_shape_grid_map(
        size=int(size),
        seed=int(seed),
        grid_step=int(preset.grid_step),
        spawn_prob=float(preset.spawn_prob),
        min_half_extent=int(preset.min_half_extent),
        max_half_extent=int(preset.max_half_extent),
        jitter=int(preset.jitter),
        clearance=int(preset.clearance),
        shape_types=tuple(shape_types) if shape_types is not None else tuple(preset.shape_types),
        pocket_min_half_extent=(
            None if preset.pocket_min_half_extent is None else int(preset.pocket_min_half_extent)
        ),
        pocket_max_half_extent=(
            None if preset.pocket_max_half_extent is None else int(preset.pocket_max_half_extent)
        ),
        pocket_thickness=int(preset.pocket_thickness),
        pocket_count_min=int(preset.pocket_count_min),
        pocket_count_max=int(preset.pocket_count_max),
        pocket_clearance=int(preset.pocket_clearance),
        ensure_start_clear=True,
    )


def build_validated_shape_grid_map(
    *,
    size: int,
    seed: int,
    level: int,
    max_retries: int = 64,
    min_start_component_ratio: float = 0.995,
    min_free_ratio: float = 0.50,
    max_obstacle_ratio: float = 0.45,
) -> tuple[np.ndarray, MapValidationStats, int, int]:
    preset = get_paper_shape_grid_preset(level)
    last_grid = None
    last_stats = None
    for attempt in range(max(1, int(max_retries))):
        candidate_seed = int(seed) + attempt * 1_000_003
        grid = build_shape_grid_map_from_preset(size=int(size), seed=candidate_seed, preset=preset)
        stats = analyze_grid_map(grid, start=(0, 0))
        if map_passes_paper_checks(
            stats,
            min_start_component_ratio=float(min_start_component_ratio),
            min_free_ratio=float(min_free_ratio),
            max_obstacle_ratio=float(max_obstacle_ratio),
        ):
            return grid, stats, candidate_seed, attempt
        last_grid = grid
        last_stats = stats

    assert last_grid is not None and last_stats is not None
    raise RuntimeError(
        "Failed to generate a valid shape-grid map "
        f"level={level} seed={seed} after {max_retries} retries; "
        f"last_stats={last_stats.as_dict()}"
    )
