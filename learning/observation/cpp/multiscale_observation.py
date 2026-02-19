from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .directional_traversability import compute_directional_traversability
from .grid_features import (
    block_reduce_max,
    block_reduce_mean,
    block_reduce_state,
    center_crop_with_pad,
    compute_frontier_map,
    extract_known_masks,
    global_reduce_max,
    global_reduce_mean,
    global_reduce_state,
)


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class MultiScaleCPPObservationConfig:
    # Robot-centered local levels.
    local_blocks: Tuple[int, ...] = (1, 2, 4, 8, 16)
    local_window_size: int = 7
    # Global level.
    global_window_size: int = 4
    # Occupancy coding.
    unknown_value: int = -1
    obstacle_value: int = 1
    # DTM config.
    dtm_patch_size: int = 7
    dtm_connectivity: int = 8
    dtm_require_fully_known_patch: bool = False
    # Unknown relaxation: if a coarse cell has enough known area, treat it as
    # known for DTM state construction.
    dtm_min_known_ratio: float = 0.6
    dtm_unknown_fill: float = -1.0


class MultiScaleCPPObservationBuilder:
    """
    Build multi-scale observations for CPP ablation experiments.

    Modes
    -----
    - baseline: coverage + obstacle + frontier
    - dtm: baseline + directional traversability maps (LR/UD/NW-SE/NE-SW)

    Output
    ------
    Dictionary level -> tensor [C, H, W]
    """

    _BASELINE_CHANNELS = ("coverage", "obstacle", "frontier")
    _DTM_CHANNELS = ("dtm_lr", "dtm_ud", "dtm_nw_se", "dtm_ne_sw")

    def __init__(
        self,
        config: Optional[MultiScaleCPPObservationConfig] = None,
        *,
        include_dtm: bool = False,
    ):
        self.config = config or MultiScaleCPPObservationConfig()
        self.include_dtm = bool(include_dtm)

    @property
    def num_levels(self) -> int:
        return len(self.config.local_blocks) + 1

    @property
    def channel_names(self) -> Tuple[str, ...]:
        if self.include_dtm:
            return self._BASELINE_CHANNELS + self._DTM_CHANNELS
        return self._BASELINE_CHANNELS

    @property
    def channels_per_level(self) -> int:
        return len(self.channel_names)

    def _build_level_channels(
        self,
        coverage_map: np.ndarray,
        obstacle_map: np.ndarray,
        frontier_map: np.ndarray,
        state_map: Optional[np.ndarray],
        *,
        center: Optional[GridPos],
        out_size: int,
    ) -> np.ndarray:
        if center is not None:
            cov = center_crop_with_pad(coverage_map, center, out_size, out_size, pad_value=0.0)
            obs = center_crop_with_pad(obstacle_map, center, out_size, out_size, pad_value=1.0)
            frn = center_crop_with_pad(frontier_map, center, out_size, out_size, pad_value=0.0)
        else:
            cov = coverage_map.astype(np.float32)
            obs = obstacle_map.astype(np.float32)
            frn = frontier_map.astype(np.float32)

        channels = [cov, obs, frn]

        if self.include_dtm:
            if state_map is None:
                raise RuntimeError("state_map is required when include_dtm=True")
            dtm = compute_directional_traversability(
                state_map,
                patch_size=self.config.dtm_patch_size,
                connectivity=self.config.dtm_connectivity,
                require_fully_known_patch=self.config.dtm_require_fully_known_patch,
                unknown_fill=self.config.dtm_unknown_fill,
            )
            if center is not None:
                for k in range(4):
                    channels.append(
                        center_crop_with_pad(dtm[k], center, out_size, out_size, pad_value=0.0)
                    )
            else:
                channels.extend([dtm[0], dtm[1], dtm[2], dtm[3]])

        return np.stack(channels, axis=0).astype(np.float32)

    def build_levels(
        self,
        occupancy: np.ndarray,
        *,
        robot_pos: GridPos,
        explored: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        if occupancy.ndim != 2:
            raise ValueError("occupancy must be 2D")
        if explored.shape != occupancy.shape:
            raise ValueError("explored shape must match occupancy shape")

        h, w = occupancy.shape
        rr, cc = robot_pos
        if not (0 <= rr < h and 0 <= cc < w):
            raise ValueError(f"robot_pos {robot_pos} is out of bounds {(h, w)}")

        known_free, known_obstacle, unknown = extract_known_masks(
            occupancy,
            unknown_value=self.config.unknown_value,
            obstacle_value=self.config.obstacle_value,
        )
        covered = explored.astype(bool) & known_free
        frontier = compute_frontier_map(known_free, unknown)

        covered_f = covered.astype(np.float32)
        obstacle_f = known_obstacle.astype(np.float32)
        frontier_f = frontier.astype(np.float32)

        levels: Dict[int, np.ndarray] = {}

        # Local robot-centered levels.
        for lv, block in enumerate(self.config.local_blocks):
            cov_coarse = block_reduce_mean(covered_f, block)
            obs_coarse = block_reduce_mean(obstacle_f, block)
            frn_coarse = block_reduce_max(frontier_f, block)
            state_coarse = None
            if self.include_dtm:
                state_coarse = block_reduce_state(
                    known_free,
                    known_obstacle,
                    unknown,
                    block,
                    min_known_ratio=self.config.dtm_min_known_ratio,
                )

            center_coarse = (rr // block, cc // block)
            levels[lv] = self._build_level_channels(
                cov_coarse,
                obs_coarse,
                frn_coarse,
                state_coarse,
                center=center_coarse,
                out_size=self.config.local_window_size,
            )

        # Global non-centered level.
        gsize = self.config.global_window_size
        cov_global = global_reduce_mean(covered_f, gsize, gsize)
        obs_global = global_reduce_mean(obstacle_f, gsize, gsize)
        frn_global = global_reduce_max(frontier_f, gsize, gsize)
        state_global = None
        if self.include_dtm:
            state_global = global_reduce_state(
                known_free,
                known_obstacle,
                unknown,
                gsize,
                gsize,
                min_known_ratio=self.config.dtm_min_known_ratio,
            )

        levels[len(self.config.local_blocks)] = self._build_level_channels(
            cov_global,
            obs_global,
            frn_global,
            state_global,
            center=None,
            out_size=gsize,
        )
        return levels
