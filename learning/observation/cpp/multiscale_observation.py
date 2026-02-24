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
    dtm_patch_min_known_ratio: float = 0.6
    # Ambiguous partial-observation DTM output value.
    # Default is unknown; only certain directions become 0/1.
    dtm_uncertain_fill: float = -1.0
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
        # Incremental DTM cache:
        # - track last occupancy snapshot
        # - update only cells impacted by newly observed occupancy changes
        self._prev_occupancy: Optional[np.ndarray] = None
        self._dtm_cache: Dict[int, np.ndarray] = {}
        if not (0.0 < float(self.config.dtm_min_known_ratio) <= 1.0):
            raise ValueError("dtm_min_known_ratio must be in (0, 1]")
        if not (0.0 < float(self.config.dtm_patch_min_known_ratio) <= 1.0):
            raise ValueError("dtm_patch_min_known_ratio must be in (0, 1]")

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

    def _compute_changed_mask(self, occupancy: np.ndarray) -> np.ndarray:
        # First frame (or map shape change): force full DTM refresh.
        if self._prev_occupancy is None or self._prev_occupancy.shape != occupancy.shape:
            self._dtm_cache.clear()
            changed = np.ones_like(occupancy, dtype=bool)
        else:
            changed = occupancy != self._prev_occupancy
        self._prev_occupancy = occupancy.copy()
        return changed

    def _effective_patch_radius(self, limit: int) -> int:
        p = int(self.config.dtm_patch_size)
        if p <= 0:
            raise ValueError("dtm_patch_size must be positive")
        if p % 2 == 0:
            p -= 1
        p = max(1, p)
        if p > limit:
            p = limit if (limit % 2 == 1) else max(1, limit - 1)
        return p // 2

    def _dilate_mask(self, mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask.astype(bool, copy=False)
        h, w = mask.shape
        out = np.zeros((h, w), dtype=bool)
        src = mask.astype(bool, copy=False)
        for dr in range(-radius, radius + 1):
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            if src_r1 <= src_r0:
                continue
            for dc in range(-radius, radius + 1):
                src_c0 = max(0, -dc)
                src_c1 = min(w, w - dc)
                dst_c0 = max(0, dc)
                dst_c1 = dst_c0 + (src_c1 - src_c0)
                if src_c1 <= src_c0:
                    continue
                out[dst_r0:dst_r1, dst_c0:dst_c1] |= src[src_r0:src_r1, src_c0:src_c1]
        return out

    def _update_dtm_level(
        self,
        *,
        level_id: int,
        state_map: np.ndarray,
        known_ratio_map: np.ndarray,
        changed_mask: np.ndarray,
    ) -> np.ndarray:
        h, w = state_map.shape
        cache = self._dtm_cache.get(level_id, None)
        full_refresh = cache is None or cache.shape != (4, h, w)

        if full_refresh:
            dtm = compute_directional_traversability(
                state_map,
                known_ratio_map=known_ratio_map,
                patch_size=self.config.dtm_patch_size,
                connectivity=self.config.dtm_connectivity,
                require_fully_known_patch=self.config.dtm_require_fully_known_patch,
                min_center_known_ratio=self.config.dtm_min_known_ratio,
                min_patch_known_ratio=self.config.dtm_patch_min_known_ratio,
                uncertain_fill=self.config.dtm_uncertain_fill,
                unknown_fill=self.config.dtm_unknown_fill,
            )
            self._dtm_cache[level_id] = dtm
            return dtm

        if not np.any(changed_mask):
            return cache

        radius = self._effective_patch_radius(limit=max(h, w))
        dirty_mask = self._dilate_mask(changed_mask, radius=radius)

        dtm = compute_directional_traversability(
            state_map,
            known_ratio_map=known_ratio_map,
            patch_size=self.config.dtm_patch_size,
            connectivity=self.config.dtm_connectivity,
            require_fully_known_patch=self.config.dtm_require_fully_known_patch,
            min_center_known_ratio=self.config.dtm_min_known_ratio,
            min_patch_known_ratio=self.config.dtm_patch_min_known_ratio,
            uncertain_fill=self.config.dtm_uncertain_fill,
            unknown_fill=self.config.dtm_unknown_fill,
            out=cache,
            dirty_mask=dirty_mask,
        )
        self._dtm_cache[level_id] = dtm
        return dtm

    def _build_level_channels(
        self,
        coverage_map: np.ndarray,
        obstacle_map: np.ndarray,
        frontier_map: np.ndarray,
        state_map: Optional[np.ndarray],
        known_ratio_map: Optional[np.ndarray],
        dtm_map: Optional[np.ndarray] = None,
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
            if dtm_map is not None:
                dtm = dtm_map
            else:
                if state_map is None:
                    raise RuntimeError("state_map is required when include_dtm=True")
                dtm = compute_directional_traversability(
                    state_map,
                    known_ratio_map=known_ratio_map,
                    patch_size=self.config.dtm_patch_size,
                    connectivity=self.config.dtm_connectivity,
                    require_fully_known_patch=self.config.dtm_require_fully_known_patch,
                    min_center_known_ratio=self.config.dtm_min_known_ratio,
                    min_patch_known_ratio=self.config.dtm_patch_min_known_ratio,
                    uncertain_fill=self.config.dtm_uncertain_fill,
                    unknown_fill=self.config.dtm_unknown_fill,
                )
            if center is not None:
                for k in range(4):
                    channels.append(
                        center_crop_with_pad(
                            dtm[k],
                            center,
                            out_size,
                            out_size,
                            pad_value=self.config.dtm_unknown_fill,
                        )
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
        known_f = (~unknown).astype(np.float32)
        covered = explored.astype(bool) & known_free
        frontier = compute_frontier_map(known_free, unknown)

        covered_f = covered.astype(np.float32)
        obstacle_f = known_obstacle.astype(np.float32)
        frontier_f = frontier.astype(np.float32)
        changed_f = self._compute_changed_mask(occupancy) if self.include_dtm else None
        changed_f_float = changed_f.astype(np.float32) if changed_f is not None else None

        levels: Dict[int, np.ndarray] = {}

        # Local robot-centered levels.
        for lv, block in enumerate(self.config.local_blocks):
            cov_coarse = block_reduce_mean(covered_f, block)
            known_ratio_coarse = block_reduce_mean(known_f, block)
            obs_coarse = block_reduce_mean(obstacle_f, block)
            frn_coarse = block_reduce_max(frontier_f, block)
            state_coarse = None
            dtm_coarse = None
            if self.include_dtm:
                state_coarse = block_reduce_state(
                    known_free,
                    known_obstacle,
                    unknown,
                    block,
                    min_known_ratio=self.config.dtm_min_known_ratio,
                )
                changed_coarse = block_reduce_max(changed_f_float, block) > 0.0
                dtm_coarse = self._update_dtm_level(
                    level_id=lv,
                    state_map=state_coarse,
                    known_ratio_map=known_ratio_coarse,
                    changed_mask=changed_coarse,
                )

            center_coarse = (rr // block, cc // block)
            levels[lv] = self._build_level_channels(
                cov_coarse,
                obs_coarse,
                frn_coarse,
                state_coarse,
                known_ratio_coarse,
                dtm_coarse,
                center=center_coarse,
                out_size=self.config.local_window_size,
            )

        # Global non-centered level.
        gsize = self.config.global_window_size
        cov_global = global_reduce_mean(covered_f, gsize, gsize)
        known_ratio_global = global_reduce_mean(known_f, gsize, gsize)
        obs_global = global_reduce_mean(obstacle_f, gsize, gsize)
        frn_global = global_reduce_max(frontier_f, gsize, gsize)
        state_global = None
        dtm_global = None
        if self.include_dtm:
            state_global = global_reduce_state(
                known_free,
                known_obstacle,
                unknown,
                gsize,
                gsize,
                min_known_ratio=self.config.dtm_min_known_ratio,
            )
            changed_global = global_reduce_max(changed_f_float, gsize, gsize) > 0.0
            dtm_global = self._update_dtm_level(
                level_id=len(self.config.local_blocks),
                state_map=state_global,
                known_ratio_map=known_ratio_global,
                changed_mask=changed_global,
            )

        levels[len(self.config.local_blocks)] = self._build_level_channels(
            cov_global,
            obs_global,
            frn_global,
            state_global,
            known_ratio_global,
            dtm_global,
            center=None,
            out_size=gsize,
        )
        return levels
