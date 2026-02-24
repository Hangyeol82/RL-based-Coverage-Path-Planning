#!/usr/bin/env python3
"""
Lightweight observation validation for baseline + DTM channels.

Run:
  python tools/validate_observations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure local package imports work when running from tools/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.observation.cpp.directional_traversability import (
    compute_directional_traversability,
)
from learning.observation.cpp.grid_features import (
    BLOCKED_STATE,
    FREE_STATE,
    UNKNOWN_STATE,
    block_reduce_mean,
    block_reduce_state,
    center_crop_with_pad,
    compute_frontier_map,
    extract_known_masks,
)
from learning.observation.cpp.multiscale_observation import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)


def _check(cond: bool, name: str):
    if not cond:
        raise AssertionError(name)
    print(f"[PASS] {name}")


def _check_allclose(a: np.ndarray, b: np.ndarray, name: str, atol: float = 1e-6):
    if not np.allclose(a, b, atol=atol):
        diff = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{name} (max diff={diff})")
    print(f"[PASS] {name}")


def _validate_baseline_level0():
    occupancy = np.array(
        [
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, -1, -1, -1],
            [-1, 0, 1, 0, -1, -1, -1],
            [-1, 0, 0, 0, 0, -1, -1],
            [-1, -1, -1, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    explored = np.zeros_like(occupancy, dtype=bool)
    explored[1, 1] = True
    explored[2, 3] = True
    explored[3, 3] = True
    explored[2, 2] = True  # obstacle; should not appear in coverage channel

    cfg = MultiScaleCPPObservationConfig(local_blocks=(1,), local_window_size=7, global_window_size=4)
    builder = MultiScaleCPPObservationBuilder(cfg, include_dtm=False)
    levels = builder.build_levels(occupancy, robot_pos=(3, 3), explored=explored)
    lvl0 = levels[0]
    _check(lvl0.shape == (3, 7, 7), "baseline level0 shape is (3,7,7)")

    known_free, known_obstacle, unknown = extract_known_masks(occupancy)
    coverage = (explored & known_free).astype(np.float32)
    obstacle = known_obstacle.astype(np.float32)
    frontier = compute_frontier_map(known_free, unknown).astype(np.float32)

    exp_cov = center_crop_with_pad(coverage, center=(3, 3), out_h=7, out_w=7, pad_value=0.0)
    exp_obs = center_crop_with_pad(obstacle, center=(3, 3), out_h=7, out_w=7, pad_value=1.0)
    exp_frn = center_crop_with_pad(frontier, center=(3, 3), out_h=7, out_w=7, pad_value=0.0)

    _check_allclose(lvl0[0], exp_cov, "baseline coverage channel matches expected")
    _check_allclose(lvl0[1], exp_obs, "baseline obstacle channel matches expected")
    _check_allclose(lvl0[2], exp_frn, "baseline frontier channel matches expected")
    _check(float(lvl0[0, 2, 2]) == 0.0, "obstacle cell is never counted as covered")


def _validate_baseline_obstacle_ratio_over_total():
    """
    Validate coarse obstacle channel semantics:
    obstacle_ratio = obstacle_count / total_block_cells
    (unknown cells are treated as non-obstacle in this channel).
    """
    occupancy = np.zeros((6, 6), dtype=np.int32)
    # Build a 2x2 fine block at coarse center (rows 2:4, cols 2:4):
    # [1, 0]
    # [-1, -1]
    occupancy[2, 2] = 1
    occupancy[2, 3] = 0
    occupancy[3, 2] = -1
    occupancy[3, 3] = -1

    explored = np.zeros_like(occupancy, dtype=bool)
    cfg = MultiScaleCPPObservationConfig(
        local_blocks=(2,),
        local_window_size=3,
        global_window_size=3,
    )
    builder = MultiScaleCPPObservationBuilder(cfg, include_dtm=False)
    levels = builder.build_levels(occupancy, robot_pos=(3, 3), explored=explored)
    lvl0 = levels[0]

    # Center coarse cell in 3x3 crop.
    obs_center = float(lvl0[1, 1, 1])
    # total_block_cells=4, obstacle_count=1 => 0.25
    _check(abs(obs_center - 0.25) < 1e-6, "baseline coarse obstacle uses total-cell ratio")


def _validate_dtm_core_logic():
    kr_low = np.full((7, 7), 0.1, dtype=np.float32)
    kr_high = np.full((7, 7), 0.95, dtype=np.float32)

    # Case 1: certainly blocked, independent of known-ratio.
    s1 = np.full((7, 7), BLOCKED_STATE, dtype=np.int8)
    s1[3, 3] = UNKNOWN_STATE
    o1_low = compute_directional_traversability(
        s1,
        known_ratio_map=kr_low,
        patch_size=7,
        require_fully_known_patch=False,
        min_center_known_ratio=0.6,
        min_patch_known_ratio=0.6,
    )
    o1_high = compute_directional_traversability(
        s1,
        known_ratio_map=kr_high,
        patch_size=7,
        require_fully_known_patch=False,
        min_center_known_ratio=0.6,
        min_patch_known_ratio=0.6,
    )
    _check_allclose(o1_low[:, 3, 3], np.zeros(4, dtype=np.float32), "certain blocked -> all 0")
    _check_allclose(o1_high[:, 3, 3], np.zeros(4, dtype=np.float32), "certain blocked unchanged for high known-ratio")
    _check_allclose(o1_low[:, 3, 3], o1_high[:, 3, 3], "certain blocked independent of known-ratio")

    # Case 2: certainly traversable, independent of known-ratio.
    s2 = np.full((7, 7), FREE_STATE, dtype=np.int8)
    o2_low = compute_directional_traversability(
        s2,
        known_ratio_map=kr_low,
        patch_size=7,
        require_fully_known_patch=False,
        min_center_known_ratio=0.6,
        min_patch_known_ratio=0.6,
    )
    o2_high = compute_directional_traversability(
        s2,
        known_ratio_map=kr_high,
        patch_size=7,
        require_fully_known_patch=False,
        min_center_known_ratio=0.6,
        min_patch_known_ratio=0.6,
    )
    _check_allclose(o2_low[:, 3, 3], np.ones(4, dtype=np.float32), "certain traversable -> all 1")
    _check_allclose(o2_high[:, 3, 3], np.ones(4, dtype=np.float32), "certain traversable unchanged for high known-ratio")
    _check_allclose(o2_low[:, 3, 3], o2_high[:, 3, 3], "certain traversable independent of known-ratio")

    # Case 3: ambiguous under partial observation -> unknown(-1) in uncertain directions.
    s3 = np.full((7, 7), BLOCKED_STATE, dtype=np.int8)
    s3[:, 3] = FREE_STATE
    s3[3, :] = UNKNOWN_STATE
    s3[3, 3] = FREE_STATE
    o3 = compute_directional_traversability(
        s3,
        known_ratio_map=kr_low,
        patch_size=7,
        require_fully_known_patch=False,
        min_center_known_ratio=0.6,
        min_patch_known_ratio=0.6,
    )
    center = o3[:, 3, 3]
    _check(center[0] == -1.0, "ambiguous LR is unknown(-1)")
    _check(center[1] == 1.0, "certain UD stays traversable(1)")


def _validate_builder_dtm_level0_alignment():
    occupancy = np.array(
        [
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, -1, -1, -1],
            [-1, 0, 1, 0, -1, -1, -1],
            [-1, 0, 0, 0, 0, -1, -1],
            [-1, -1, -1, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    explored = np.zeros_like(occupancy, dtype=bool)
    explored[3, 3] = True
    cfg = MultiScaleCPPObservationConfig(
        local_blocks=(1,),
        local_window_size=7,
        global_window_size=4,
        dtm_min_known_ratio=0.6,
        dtm_patch_min_known_ratio=0.6,
        dtm_uncertain_fill=-1.0,
    )
    builder = MultiScaleCPPObservationBuilder(cfg, include_dtm=True)
    levels = builder.build_levels(occupancy, robot_pos=(3, 3), explored=explored)
    lvl0 = levels[0]
    _check(lvl0.shape == (7, 7, 7), "dtm level0 shape is (7,7,7)")

    known_free, known_obstacle, unknown = extract_known_masks(occupancy)
    state = block_reduce_state(
        known_free,
        known_obstacle,
        unknown,
        block=1,
        min_known_ratio=cfg.dtm_min_known_ratio,
    )
    known_ratio = block_reduce_mean((~unknown).astype(np.float32), block=1)
    dtm_direct = compute_directional_traversability(
        state,
        known_ratio_map=known_ratio,
        patch_size=cfg.dtm_patch_size,
        connectivity=cfg.dtm_connectivity,
        require_fully_known_patch=cfg.dtm_require_fully_known_patch,
        min_center_known_ratio=cfg.dtm_min_known_ratio,
        min_patch_known_ratio=cfg.dtm_patch_min_known_ratio,
        uncertain_fill=cfg.dtm_uncertain_fill,
        unknown_fill=cfg.dtm_unknown_fill,
    )
    for k in range(4):
        _check_allclose(lvl0[3 + k], dtm_direct[k], f"builder DTM channel {k} matches direct DTM")


def main():
    print("Running observation validation...")
    _validate_baseline_level0()
    _validate_baseline_obstacle_ratio_over_total()
    _validate_dtm_core_logic()
    _validate_builder_dtm_level0_alignment()
    print("\nAll observation checks passed.")


if __name__ == "__main__":
    main()
