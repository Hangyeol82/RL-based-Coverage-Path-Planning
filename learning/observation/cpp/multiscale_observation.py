from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .directional_traversability import compute_directional_traversability
from .grid_features import (
    BLOCKED_STATE,
    FREE_STATE,
    UNKNOWN_STATE,
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
class BoundarySegment:
    side: str
    start: int
    end: int
    component: int


@dataclass(frozen=True)
class BoundaryConnectivity:
    """Boundary-only connectivity for one passability interpretation.

    top/right/bottom/left arrays store component ids along each boundary
    position, or -1 when that boundary position is closed in this
    interpretation.
    """

    height: int
    width: int
    top: np.ndarray
    right: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    segments: Tuple[Tuple[BoundarySegment, ...], ...]
    flags12: np.ndarray
    flags6: np.ndarray


@dataclass(frozen=True)
class BoundarySummary:
    """Ternary DTM summary for one hierarchical cell.

    certain treats unknown as blocked. possible treats unknown as free.
    The final flags are 1 when certain connectivity exists, 0 when possible
    connectivity does not exist, and uncertain_fill otherwise.
    """

    certain: BoundaryConnectivity
    possible: BoundaryConnectivity
    flags12: np.ndarray
    flags6: np.ndarray


class _DSU:
    def __init__(self):
        self.parent: List[int] = []

    def make(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        return idx

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass(frozen=True)
class MultiScaleCPPObservationConfig:
    # Robot-centered local levels.
    local_blocks: Tuple[int, ...] = (1, 2, 4, 8, 16)
    local_window_size: int = 7
    # Final coarse level. It is also robot-centered; global_window_size controls
    # the approximate number of coarse cells that span the full map axis
    # (e.g. 128 / 4 => block size 32).
    global_window_size: int = 4
    global_robot_centered_block: Optional[int] = None
    # Occupancy coding.
    unknown_value: int = -1
    obstacle_value: int = 1
    # Unknown handling for map-observation channels.
    # - keep: keep unknown as-is (-1)
    # - as_free: treat unknown as free for map channels
    # - as_obstacle: treat unknown as obstacle for map channels
    unknown_policy: str = "keep"
    # DTM config.
    dtm_patch_size: int = 7
    dtm_connectivity: int = 4
    dtm_require_fully_known_patch: bool = False
    # Unknown relaxation: if a coarse cell has enough known area, treat it as
    # known for DTM state construction.
    dtm_min_known_ratio: float = 0.6
    dtm_patch_min_known_ratio: float = 0.6
    # Ambiguous partial-observation DTM output value.
    # Default is unknown; only certain directions become 0/1.
    dtm_uncertain_fill: float = -1.0
    dtm_unknown_fill: float = -1.0
    # Coarse-level DTM mode:
    # - bfs: compute DTM independently at each scale using BFS
    # - aggregate: compute BFS only on finest level and aggregate upward with max pooling
    # - aggregate_transfer: compute BFS on finest level, then compose coarse cells
    #   using child-cell transfer graph.
    dtm_coarse_mode: str = "bfs"
    # DTM output channel mode:
    # - six: undirected side-to-side transitions
    #   [U<->R, U<->D, U<->L, R<->D, R<->L, D<->L]
    # - extent6: six side-pair traversability extents in [0,1]
    #   with unknown as -1
    # - two/axis2: LR, UD only
    # - axis2km: LR/UD split into passable(0/1) + known(0/1)
    # - four: legacy projection to LR, UD, and two grouped turn channels
    # - port6: undirected side-to-side transitions
    #   [U<->R, U<->D, U<->L, R<->D, R<->L, D<->L]
    # - port12: directed side-to-side transitions
    #   [U->R, U->D, U->L, R->U, R->D, R->L, D->U, D->R, D->L, L->U, L->R, L->D]
    dtm_output_mode: str = "six"
    # Append per-level robot-in-cell phase channels:
    # [sin(row_phase), cos(row_phase), sin(col_phase), cos(col_phase)].
    # This helps disambiguate where the robot is inside each coarse cell.
    include_cell_phase_channels: bool = False
    # DTM at the finest local level duplicates one-cell occupancy/free-space
    # information already represented by the baseline channels, so omit it
    # from level 0 and keep DTM only for coarser summaries.
    exclude_dtm_level0: bool = True


@dataclass(frozen=True)
class HybridLocalGlobalCPPObservationConfig:
    # Robot-centered high-resolution crop. Must be odd so the robot stays at
    # the center pixel.
    local_crop_size: int = 41
    # Deprecated single-scale option kept for old callers. New hybrid
    # experiments use global_coarse_sizes.
    global_coarse_size: int = 16
    # Full-map multi-scale summary resolutions.
    global_coarse_sizes: Tuple[int, ...] = (64, 32, 16)
    unknown_value: int = -1
    obstacle_value: int = 1
    unknown_policy: str = "keep"
    dtm_patch_size: int = 7
    dtm_connectivity: int = 4
    dtm_require_fully_known_patch: bool = False
    dtm_min_known_ratio: float = 0.6
    dtm_patch_min_known_ratio: float = 0.6
    dtm_uncertain_fill: float = -1.0
    dtm_unknown_fill: float = -1.0
    dtm_output_mode: str = "six"


class MultiScaleCPPObservationBuilder:
    """
    Build multi-scale observations for CPP ablation experiments.

    Modes
    -----
    - baseline: coverage + obstacle + frontier
    - dtm: baseline + directional traversability maps
      (six side-pair transitions or the requested DTM projection)

    Output
    ------
    Dictionary level -> tensor [C, H, W]
    """

    _BASELINE_CHANNELS = ("coverage", "obstacle", "frontier")
    _DTM_CHANNELS_6 = (
        "dtm_u_r",
        "dtm_u_d",
        "dtm_u_l",
        "dtm_r_d",
        "dtm_r_l",
        "dtm_d_l",
    )
    _DTM_CHANNELS_2 = (
        "dtm_lr",
        "dtm_ud",
    )
    _DTM_CHANNELS_4_AXIS2KM = (
        "dtm_lr_pass",
        "dtm_ud_pass",
        "dtm_lr_known",
        "dtm_ud_known",
    )
    _DTM_CHANNELS_EXTENT6 = (
        "dtm_extent_u_r",
        "dtm_extent_u_d",
        "dtm_extent_u_l",
        "dtm_extent_r_d",
        "dtm_extent_r_l",
        "dtm_extent_d_l",
    )
    _DTM_CHANNELS_4 = ("dtm_lr", "dtm_ud", "dtm_turn_ur_dl", "dtm_turn_ul_rd")
    _DTM_CHANNELS_12 = (
        "dtm_u_r",
        "dtm_u_d",
        "dtm_u_l",
        "dtm_r_u",
        "dtm_r_d",
        "dtm_r_l",
        "dtm_d_u",
        "dtm_d_r",
        "dtm_d_l",
        "dtm_l_u",
        "dtm_l_r",
        "dtm_l_d",
    )
    _DTM_CHANNELS_PORT6 = (
        "dtm_u_r",
        "dtm_u_d",
        "dtm_u_l",
        "dtm_r_d",
        "dtm_r_l",
        "dtm_d_l",
    )
    _CELL_PHASE_CHANNELS = (
        "cell_row_sin",
        "cell_row_cos",
        "cell_col_sin",
        "cell_col_cos",
    )

    def __init__(
        self,
        config: Optional[MultiScaleCPPObservationConfig] = None,
        *,
        include_dtm: bool = False,
        profile_enabled: bool = False,
    ):
        self.config = config or MultiScaleCPPObservationConfig()
        self.include_dtm = bool(include_dtm)
        self._profile_enabled = bool(profile_enabled)
        self._profile_totals: Dict[str, float] = {}
        self._profile_calls = 0
        self._coarse_cache: Dict[int, Dict[str, np.ndarray]] = {}
        # Incremental DTM cache:
        # - track last occupancy snapshot
        # - update only cells impacted by newly observed occupancy changes
        self._prev_occupancy: Optional[np.ndarray] = None
        self._prev_explored: Optional[np.ndarray] = None
        self._prev_covered: Optional[np.ndarray] = None
        self._prev_known_obstacle: Optional[np.ndarray] = None
        self._prev_frontier: Optional[np.ndarray] = None
        self._dtm_cache: Dict[int, np.ndarray] = {}
        self._boundary_summary_cache: Dict[int, np.ndarray] = {}
        self._boundary_summary_shape: Optional[Tuple[int, int]] = None
        if not (0.0 < float(self.config.dtm_min_known_ratio) <= 1.0):
            raise ValueError("dtm_min_known_ratio must be in (0, 1]")
        if not (0.0 < float(self.config.dtm_patch_min_known_ratio) <= 1.0):
            raise ValueError("dtm_patch_min_known_ratio must be in (0, 1]")
        if self.config.dtm_coarse_mode not in {"bfs", "aggregate", "aggregate_transfer"}:
            raise ValueError("dtm_coarse_mode must be one of {'bfs', 'aggregate', 'aggregate_transfer'}")
        if self.config.unknown_policy not in {"keep", "as_free", "as_obstacle"}:
            raise ValueError("unknown_policy must be one of {'keep', 'as_free', 'as_obstacle'}")
        output_mode = self._normalize_dtm_output_mode(self.config.dtm_output_mode)
        if output_mode not in {"six", "extent6", "axis2", "axis2km", "four", "port6", "port12"}:
            raise ValueError(
                "dtm_output_mode must be one of {'six', 'extent6', 'two', 'axis2', 'axis2km', 'four', 'port6', 'port12'}"
            )
        if (
            output_mode == "extent6"
            and self.config.dtm_coarse_mode == "aggregate_transfer"
        ):
            raise ValueError(
                "dtm_output_mode='extent6' is not supported with "
                "dtm_coarse_mode='aggregate_transfer'. Use dtm_coarse_mode='bfs'."
            )

    def enable_profiling(self, enabled: bool = True):
        self._profile_enabled = bool(enabled)
        if not enabled:
            self.reset_profile()

    def reset_profile(self):
        self._profile_totals = {}
        self._profile_calls = 0

    def reset_incremental_state(self, *, preserve_static_map: bool = False):
        self._coarse_cache = {}
        self._prev_explored = None
        self._prev_covered = None
        self._prev_frontier = None
        if preserve_static_map:
            return
        self._prev_occupancy = None
        self._prev_known_obstacle = None
        self._dtm_cache.clear()
        self._boundary_summary_cache.clear()
        self._boundary_summary_shape = None

    def profile_snapshot(self, *, reset: bool = False) -> Dict[str, float]:
        snap: Dict[str, float] = {"calls": float(self._profile_calls)}
        snap.update(self._profile_totals)
        if reset:
            self.reset_profile()
        return snap

    def _profile_add(self, key: str, dt: float):
        if not self._profile_enabled:
            return
        self._profile_totals[key] = float(self._profile_totals.get(key, 0.0)) + float(dt)

    @staticmethod
    def _normalize_dtm_output_mode(mode: str) -> str:
        mode_s = str(mode).strip().lower()
        return "axis2" if mode_s == "two" else mode_s

    def _native_dtm_mode(self) -> str:
        # For recursive coarse-cell composition, axis projections need full
        # side-to-side connectivity, not only LR/UD summaries.
        mode = self._normalize_dtm_output_mode(self.config.dtm_output_mode)
        if mode in {"axis2", "axis2km", "port12"}:
            return "port12"
        if mode == "port6":
            return "port6"
        if mode in {"six", "four"}:
            return "six"
        if mode == "extent6":
            return "extent6"
        return "port6"

    def _native_dtm_channels(self) -> int:
        mode = self._native_dtm_mode()
        return 12 if mode == "port12" else 6

    @staticmethod
    def _is_power_of_two(x: int) -> bool:
        x = int(x)
        return x > 0 and (x & (x - 1)) == 0

    @classmethod
    def _power_of_two_exp(cls, x: int) -> Optional[int]:
        x = int(x)
        if not cls._is_power_of_two(x):
            return None
        return int(x.bit_length() - 1)

    @property
    def num_levels(self) -> int:
        return len(self.config.local_blocks) + 1

    def _global_level_block(self, map_shape: Tuple[int, int]) -> int:
        override = self.config.global_robot_centered_block
        if override is not None:
            return max(1, int(override))
        h, w = int(map_shape[0]), int(map_shape[1])
        gsize = max(1, int(self.config.global_window_size))
        max_dim = max(1, h, w)
        return max(1, (max_dim + gsize - 1) // gsize)

    @property
    def channel_names(self) -> Tuple[str, ...]:
        # Backward-compatible representative channel list. When DTM is enabled
        # and level 0 is excluded, coarser levels still use the full list.
        if self.include_dtm and self.config.exclude_dtm_level0 and self.num_levels > 1:
            return self.channel_names_for_level(1)
        return self.channel_names_for_level(0)

    def _dtm_channel_names(self) -> Tuple[str, ...]:
        if self.include_dtm:
            mode = self._normalize_dtm_output_mode(self.config.dtm_output_mode)
            if mode == "six":
                return self._DTM_CHANNELS_6
            elif mode == "extent6":
                return self._DTM_CHANNELS_EXTENT6
            elif mode == "axis2":
                return self._DTM_CHANNELS_2
            elif mode == "axis2km":
                return self._DTM_CHANNELS_4_AXIS2KM
            elif mode == "four":
                return self._DTM_CHANNELS_4
            elif mode == "port6":
                return self._DTM_CHANNELS_PORT6
            else:
                return self._DTM_CHANNELS_12
        return ()

    def _include_dtm_for_level(self, level_id: int) -> bool:
        return bool(self.include_dtm) and not (
            bool(self.config.exclude_dtm_level0) and int(level_id) == 0
        )

    def channel_names_for_level(self, level_id: int) -> Tuple[str, ...]:
        names = self._BASELINE_CHANNELS
        if self._include_dtm_for_level(level_id):
            names = names + self._dtm_channel_names()
        if self.config.include_cell_phase_channels:
            names = names + self._CELL_PHASE_CHANNELS
        return names

    @property
    def channel_names_by_level(self) -> Tuple[Tuple[str, ...], ...]:
        return tuple(self.channel_names_for_level(lv) for lv in range(self.num_levels))

    @property
    def channels_per_level(self) -> int:
        return len(self.channel_names)

    @property
    def channels_per_level_by_level(self) -> Tuple[int, ...]:
        return tuple(len(names) for names in self.channel_names_by_level)

    def _compute_changed_mask(self, occupancy: np.ndarray) -> np.ndarray:
        # First frame (or map shape change): force full DTM refresh.
        if self._prev_occupancy is None or self._prev_occupancy.shape != occupancy.shape:
            self.reset_incremental_state()
            changed = np.ones_like(occupancy, dtype=bool)
        else:
            changed = occupancy != self._prev_occupancy
        self._prev_occupancy = occupancy.copy()
        return changed

    @staticmethod
    def _compute_array_change(
        current: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        curr = np.asarray(current)
        if previous is None or previous.shape != curr.shape:
            return np.ones_like(curr, dtype=bool), curr.copy()
        return curr != previous, curr.copy()

    @staticmethod
    def _dirty_blocks_from_changed(changed_mask: np.ndarray, block: int) -> np.ndarray:
        h, w = changed_mask.shape
        ch = (h + block - 1) // block
        cw = (w + block - 1) // block
        dirty = np.zeros((ch, cw), dtype=bool)
        coords = np.argwhere(changed_mask)
        if coords.size == 0:
            return dirty
        dirty[coords[:, 0] // block, coords[:, 1] // block] = True
        return dirty

    @staticmethod
    def _dirty_global_from_changed(changed_mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        h, w = changed_mask.shape
        dirty = np.zeros((out_h, out_w), dtype=bool)
        coords = np.argwhere(changed_mask)
        if coords.size == 0 or h <= 0 or w <= 0:
            return dirty
        rr = np.minimum(out_h - 1, (coords[:, 0] * out_h) // h)
        cc = np.minimum(out_w - 1, (coords[:, 1] * out_w) // w)
        dirty[rr, cc] = True
        return dirty

    @staticmethod
    def _global_edges(size: int, out_size: int) -> np.ndarray:
        edges = np.linspace(0, size, out_size + 1, dtype=np.int32)
        edges[-1] = size
        return edges

    def _update_block_mean_inplace(
        self,
        cache: np.ndarray,
        source: np.ndarray,
        *,
        block: int,
        dirty_mask: np.ndarray,
    ):
        h, w = source.shape
        for r, c in np.argwhere(dirty_mask):
            rs = int(r) * block
            re = min(h, rs + block)
            cs = int(c) * block
            ce = min(w, cs + block)
            cache[int(r), int(c)] = float(np.mean(source[rs:re, cs:ce]))

    def _update_block_max_inplace(
        self,
        cache: np.ndarray,
        source: np.ndarray,
        *,
        block: int,
        dirty_mask: np.ndarray,
    ):
        h, w = source.shape
        for r, c in np.argwhere(dirty_mask):
            rs = int(r) * block
            re = min(h, rs + block)
            cs = int(c) * block
            ce = min(w, cs + block)
            cache[int(r), int(c)] = float(np.max(source[rs:re, cs:ce]))

    def _update_block_state_inplace(
        self,
        cache: np.ndarray,
        *,
        known_free: np.ndarray,
        known_obstacle: np.ndarray,
        unknown: np.ndarray,
        block: int,
        dirty_mask: np.ndarray,
    ):
        h, w = known_free.shape
        thr = float(self.config.dtm_min_known_ratio)
        for r, c in np.argwhere(dirty_mask):
            rs = int(r) * block
            re = min(h, rs + block)
            cs = int(c) * block
            ce = min(w, cs + block)
            free_count = int(np.count_nonzero(known_free[rs:re, cs:ce]))
            obst_count = int(np.count_nonzero(known_obstacle[rs:re, cs:ce]))
            unknown_count = int(np.count_nonzero(unknown[rs:re, cs:ce]))
            cell_count = (re - rs) * (ce - cs)
            known_ratio = 1.0 - (float(unknown_count) / float(max(1, cell_count)))
            if known_ratio < thr:
                cache[int(r), int(c)] = UNKNOWN_STATE
            elif free_count > 0 and obst_count == 0:
                cache[int(r), int(c)] = FREE_STATE
            else:
                cache[int(r), int(c)] = BLOCKED_STATE

    def _update_global_mean_inplace(
        self,
        cache: np.ndarray,
        source: np.ndarray,
        *,
        dirty_mask: np.ndarray,
    ):
        h, w = source.shape
        out_h, out_w = cache.shape
        row_edges = self._global_edges(h, out_h)
        col_edges = self._global_edges(w, out_w)
        for r, c in np.argwhere(dirty_mask):
            rs, re = int(row_edges[int(r)]), int(row_edges[int(r) + 1])
            cs, ce = int(col_edges[int(c)]), int(col_edges[int(c) + 1])
            cache[int(r), int(c)] = float(np.mean(source[rs:re, cs:ce]))

    def _update_global_max_inplace(
        self,
        cache: np.ndarray,
        source: np.ndarray,
        *,
        dirty_mask: np.ndarray,
    ):
        h, w = source.shape
        out_h, out_w = cache.shape
        row_edges = self._global_edges(h, out_h)
        col_edges = self._global_edges(w, out_w)
        for r, c in np.argwhere(dirty_mask):
            rs, re = int(row_edges[int(r)]), int(row_edges[int(r) + 1])
            cs, ce = int(col_edges[int(c)]), int(col_edges[int(c) + 1])
            cache[int(r), int(c)] = float(np.max(source[rs:re, cs:ce]))

    def _update_global_state_inplace(
        self,
        cache: np.ndarray,
        *,
        known_free: np.ndarray,
        known_obstacle: np.ndarray,
        unknown: np.ndarray,
        dirty_mask: np.ndarray,
    ):
        h, w = known_free.shape
        out_h, out_w = cache.shape
        row_edges = self._global_edges(h, out_h)
        col_edges = self._global_edges(w, out_w)
        thr = float(self.config.dtm_min_known_ratio)
        for r, c in np.argwhere(dirty_mask):
            rs, re = int(row_edges[int(r)]), int(row_edges[int(r) + 1])
            cs, ce = int(col_edges[int(c)]), int(col_edges[int(c) + 1])
            free_count = int(np.count_nonzero(known_free[rs:re, cs:ce]))
            obst_count = int(np.count_nonzero(known_obstacle[rs:re, cs:ce]))
            unknown_count = int(np.count_nonzero(unknown[rs:re, cs:ce]))
            cell_count = (re - rs) * (ce - cs)
            known_ratio = 1.0 - (float(unknown_count) / float(max(1, cell_count)))
            if known_ratio < thr:
                cache[int(r), int(c)] = UNKNOWN_STATE
            elif free_count > 0 and obst_count == 0:
                cache[int(r), int(c)] = FREE_STATE
            else:
                cache[int(r), int(c)] = BLOCKED_STATE

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

    def _compute_level0_native_dtm(
        self,
        state_map: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
        dirty_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Level-0 DTM uses per-cell semantics, not patch-level spanning semantics.

        A single free cell is internally traversable across all exits.
        A blocked cell is non-traversable.
        An unknown cell stays unknown.
        """
        h, w = state_map.shape
        native_mode = self._native_dtm_mode()
        dtm_ch = 12 if native_mode == "port12" else 6
        if out is None:
            dtm = np.full((dtm_ch, h, w), float(self.config.dtm_unknown_fill), dtype=np.float32)
        else:
            if out.shape != (dtm_ch, h, w):
                raise ValueError(f"out shape must be ({dtm_ch}, H, W) matching state_map")
            dtm = out.astype(np.float32, copy=False)

        if dirty_mask is None:
            targets = np.argwhere(np.ones_like(state_map, dtype=bool))
        else:
            if dirty_mask.shape != state_map.shape:
                raise ValueError("dirty_mask shape must match state_map")
            targets = np.argwhere(dirty_mask)
            if targets.size == 0:
                return dtm

        for r, c in targets:
            st = int(state_map[int(r), int(c)])
            if st == int(UNKNOWN_STATE):
                dtm[:, int(r), int(c)] = float(self.config.dtm_unknown_fill)
            elif st == int(FREE_STATE):
                dtm[:, int(r), int(c)] = 1.0
            else:
                dtm[:, int(r), int(c)] = 0.0
        return dtm

    @staticmethod
    def _segments_from_boundary(side: str, values: np.ndarray) -> Tuple[BoundarySegment, ...]:
        arr = np.asarray(values, dtype=np.int32)
        segments: List[BoundarySegment] = []
        i = 0
        while i < int(arr.size):
            comp = int(arr[i])
            if comp < 0:
                i += 1
                continue
            j = i + 1
            while j < int(arr.size) and int(arr[j]) == comp:
                j += 1
            segments.append(BoundarySegment(side=str(side), start=int(i), end=int(j), component=comp))
            i = j
        return tuple(segments)

    @classmethod
    def _segments_for_summary(
        cls,
        *,
        top: np.ndarray,
        right: np.ndarray,
        bottom: np.ndarray,
        left: np.ndarray,
    ) -> Tuple[Tuple[BoundarySegment, ...], ...]:
        return (
            cls._segments_from_boundary("top", top),
            cls._segments_from_boundary("right", right),
            cls._segments_from_boundary("bottom", bottom),
            cls._segments_from_boundary("left", left),
        )

    @staticmethod
    def _side_component_set(values: np.ndarray) -> set:
        return {int(x) for x in np.asarray(values, dtype=np.int32).tolist() if int(x) >= 0}

    @classmethod
    def _flags12_from_boundaries(
        cls,
        *,
        top: np.ndarray,
        right: np.ndarray,
        bottom: np.ndarray,
        left: np.ndarray,
    ) -> np.ndarray:
        side_sets = {
            "up": cls._side_component_set(top),
            "right": cls._side_component_set(right),
            "down": cls._side_component_set(bottom),
            "left": cls._side_component_set(left),
        }
        pairs = (
            ("up", "right"),
            ("up", "down"),
            ("up", "left"),
            ("right", "up"),
            ("right", "down"),
            ("right", "left"),
            ("down", "up"),
            ("down", "right"),
            ("down", "left"),
            ("left", "up"),
            ("left", "right"),
            ("left", "down"),
        )
        flags = np.zeros(12, dtype=np.float32)
        for k, (a, b) in enumerate(pairs):
            flags[k] = 1.0 if bool(side_sets[a] & side_sets[b]) else 0.0
        return flags

    @staticmethod
    def _first_open_component(*values: int) -> int:
        for value in values:
            comp = int(value)
            if comp >= 0:
                return comp
        return -1

    @classmethod
    def _corner_component(
        cls,
        *,
        top: np.ndarray,
        right: np.ndarray,
        bottom: np.ndarray,
        left: np.ndarray,
        corner: str,
    ) -> int:
        t = np.asarray(top, dtype=np.int32)
        r = np.asarray(right, dtype=np.int32)
        b = np.asarray(bottom, dtype=np.int32)
        l = np.asarray(left, dtype=np.int32)
        if corner == "nw":
            return cls._first_open_component(t[0] if t.size else -1, l[0] if l.size else -1)
        if corner == "ne":
            return cls._first_open_component(t[-1] if t.size else -1, r[0] if r.size else -1)
        if corner == "se":
            return cls._first_open_component(b[-1] if b.size else -1, r[-1] if r.size else -1)
        if corner == "sw":
            return cls._first_open_component(b[0] if b.size else -1, l[-1] if l.size else -1)
        raise ValueError(f"unknown corner: {corner}")

    @classmethod
    def _flags6_from_boundaries(
        cls,
        *,
        top: np.ndarray,
        right: np.ndarray,
        bottom: np.ndarray,
        left: np.ndarray,
    ) -> np.ndarray:
        side_sets = {
            "up": cls._side_component_set(top),
            "right": cls._side_component_set(right),
            "down": cls._side_component_set(bottom),
            "left": cls._side_component_set(left),
        }
        flags = np.zeros(6, dtype=np.float32)
        flags[0] = 1.0 if bool(side_sets["up"] & side_sets["right"]) else 0.0
        flags[1] = 1.0 if bool(side_sets["up"] & side_sets["down"]) else 0.0
        flags[2] = 1.0 if bool(side_sets["up"] & side_sets["left"]) else 0.0
        flags[3] = 1.0 if bool(side_sets["right"] & side_sets["down"]) else 0.0
        flags[4] = 1.0 if bool(side_sets["right"] & side_sets["left"]) else 0.0
        flags[5] = 1.0 if bool(side_sets["down"] & side_sets["left"]) else 0.0
        return flags

    def _ternary_flags(
        self,
        *,
        certain_flags: np.ndarray,
        possible_flags: np.ndarray,
    ) -> np.ndarray:
        certain = np.asarray(certain_flags, dtype=np.float32)
        possible = np.asarray(possible_flags, dtype=np.float32)
        flags = np.full(certain.shape, float(self.config.dtm_uncertain_fill), dtype=np.float32)
        flags[certain > 0.0] = 1.0
        flags[possible <= 0.0] = 0.0
        return flags

    def _make_boundary_connectivity(self, open_cell: bool) -> BoundaryConnectivity:
        if bool(open_cell):
            side = np.zeros((1,), dtype=np.int32)
        else:
            side = np.full((1,), -1, dtype=np.int32)
        top = side.copy()
        right = side.copy()
        bottom = side.copy()
        left = side.copy()
        flags12 = self._flags12_from_boundaries(top=top, right=right, bottom=bottom, left=left)
        flags6 = self._flags6_from_boundaries(top=top, right=right, bottom=bottom, left=left)
        return BoundaryConnectivity(
            height=1,
            width=1,
            top=top,
            right=right,
            bottom=bottom,
            left=left,
            segments=self._segments_for_summary(top=top, right=right, bottom=bottom, left=left),
            flags12=flags12,
            flags6=flags6,
        )

    def _make_level0_boundary_summary(self, state: int) -> BoundarySummary:
        st = int(state)
        certain = self._make_boundary_connectivity(st == int(FREE_STATE))
        possible = self._make_boundary_connectivity(st in {int(FREE_STATE), int(UNKNOWN_STATE)})
        flags12 = self._ternary_flags(
            certain_flags=certain.flags12,
            possible_flags=possible.flags12,
        )
        flags6 = self._ternary_flags(
            certain_flags=certain.flags6,
            possible_flags=possible.flags6,
        )
        return BoundarySummary(
            certain=certain,
            possible=possible,
            flags12=flags12,
            flags6=flags6,
        )

    @staticmethod
    def _summary_components(summary: BoundaryConnectivity) -> List[int]:
        comps = set()
        for arr in (summary.top, summary.right, summary.bottom, summary.left):
            comps.update(int(x) for x in np.asarray(arr, dtype=np.int32).tolist() if int(x) >= 0)
        return sorted(comps)

    @staticmethod
    def _summary_side(summary: BoundaryConnectivity, side: str) -> np.ndarray:
        if side == "top":
            return summary.top
        if side == "right":
            return summary.right
        if side == "bottom":
            return summary.bottom
        if side == "left":
            return summary.left
        raise ValueError(f"unknown side: {side}")

    def _compose_boundary_connectivity_2x2(
        self,
        children: Dict[Tuple[int, int], BoundaryConnectivity],
    ) -> BoundaryConnectivity:
        if not children:
            return self._make_boundary_connectivity(False)

        dsu = _DSU()
        node_of: Dict[Tuple[Tuple[int, int], int], int] = {}
        for pos, child in children.items():
            for comp in self._summary_components(child):
                node_of[(pos, int(comp))] = dsu.make()

        def node(pos: Tuple[int, int], comp: int) -> Optional[int]:
            if int(comp) < 0:
                return None
            return node_of.get((pos, int(comp)))

        def union_sides(pos_a: Tuple[int, int], side_a: str, pos_b: Tuple[int, int], side_b: str) -> None:
            a = children.get(pos_a)
            b = children.get(pos_b)
            if a is None or b is None:
                return
            arr_a = self._summary_side(a, side_a)
            arr_b = self._summary_side(b, side_b)
            n = min(int(arr_a.size), int(arr_b.size))
            for i in range(n):
                ca = int(arr_a[i])
                cb = int(arr_b[i])
                if ca < 0 or cb < 0:
                    continue
                na = node(pos_a, ca)
                nb = node(pos_b, cb)
                if na is not None and nb is not None:
                    dsu.union(na, nb)

        # Only shared seams are inspected. This is proportional to parent
        # boundary length, not to the child block area.
        union_sides((0, 0), "right", (0, 1), "left")
        union_sides((1, 0), "right", (1, 1), "left")
        union_sides((0, 0), "bottom", (1, 0), "top")
        union_sides((0, 1), "bottom", (1, 1), "top")

        root_to_parent: Dict[int, int] = {}

        def parent_comp(pos: Tuple[int, int], comp: int) -> int:
            n = node(pos, int(comp))
            if n is None:
                return -1
            root = int(dsu.find(n))
            if root not in root_to_parent:
                root_to_parent[root] = len(root_to_parent)
            return root_to_parent[root]

        def mapped_side(pos: Tuple[int, int], side: str) -> np.ndarray:
            child = children.get(pos)
            if child is None:
                return np.empty((0,), dtype=np.int32)
            arr = self._summary_side(child, side)
            out = np.full(arr.shape, -1, dtype=np.int32)
            for i, comp in enumerate(np.asarray(arr, dtype=np.int32)):
                if int(comp) >= 0:
                    out[int(i)] = parent_comp(pos, int(comp))
            return out

        has_bottom = (1, 0) in children or (1, 1) in children
        has_right = (0, 1) in children or (1, 1) in children
        bottom_row = 1 if has_bottom else 0
        right_col = 1 if has_right else 0

        top = np.concatenate(
            [mapped_side((0, 0), "top"), mapped_side((0, 1), "top")]
        ).astype(np.int32, copy=False)
        bottom = np.concatenate(
            [mapped_side((bottom_row, 0), "bottom"), mapped_side((bottom_row, 1), "bottom")]
        ).astype(np.int32, copy=False)
        left = np.concatenate(
            [mapped_side((0, 0), "left"), mapped_side((1, 0), "left")]
        ).astype(np.int32, copy=False)
        right = np.concatenate(
            [mapped_side((0, right_col), "right"), mapped_side((1, right_col), "right")]
        ).astype(np.int32, copy=False)

        if top.size == 0 and bottom.size > 0:
            top = np.full_like(bottom, -1)
        if bottom.size == 0 and top.size > 0:
            bottom = np.full_like(top, -1)
        if left.size == 0 and right.size > 0:
            left = np.full_like(right, -1)
        if right.size == 0 and left.size > 0:
            right = np.full_like(left, -1)

        height = int(max(left.size, right.size, 1))
        width = int(max(top.size, bottom.size, 1))
        flags12 = self._flags12_from_boundaries(top=top, right=right, bottom=bottom, left=left)
        flags6 = self._flags6_from_boundaries(top=top, right=right, bottom=bottom, left=left)
        return BoundaryConnectivity(
            height=height,
            width=width,
            top=top,
            right=right,
            bottom=bottom,
            left=left,
            segments=self._segments_for_summary(top=top, right=right, bottom=bottom, left=left),
            flags12=flags12,
            flags6=flags6,
        )

    def _compose_boundary_summary_2x2(self, children: Dict[Tuple[int, int], BoundarySummary]) -> BoundarySummary:
        if not children:
            return self._make_level0_boundary_summary(BLOCKED_STATE)
        certain_children = {pos: child.certain for pos, child in children.items()}
        possible_children = {pos: child.possible for pos, child in children.items()}
        certain = self._compose_boundary_connectivity_2x2(certain_children)
        possible = self._compose_boundary_connectivity_2x2(possible_children)
        flags12 = self._ternary_flags(
            certain_flags=certain.flags12,
            possible_flags=possible.flags12,
        )
        flags6 = self._ternary_flags(
            certain_flags=certain.flags6,
            possible_flags=possible.flags6,
        )
        return BoundarySummary(
            certain=certain,
            possible=possible,
            flags12=flags12,
            flags6=flags6,
        )

    def _boundary_summary_grid_shape(self, child_shape: Tuple[int, int]) -> Tuple[int, int]:
        return ((int(child_shape[0]) + 1) // 2, (int(child_shape[1]) + 1) // 2)

    def _compose_boundary_summary_parent_grid(
        self,
        child_grid: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
        dirty_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ph, pw = self._boundary_summary_grid_shape(child_grid.shape)
        if out is None or out.shape != (ph, pw):
            parent = np.empty((ph, pw), dtype=object)
            targets = np.argwhere(np.ones((ph, pw), dtype=bool))
        else:
            parent = out
            if dirty_mask is None:
                targets = np.argwhere(np.ones((ph, pw), dtype=bool))
            else:
                if dirty_mask.shape != (ph, pw):
                    raise ValueError("dirty_mask shape mismatch for boundary summary parent grid")
                targets = np.argwhere(dirty_mask)
                if targets.size == 0:
                    return parent

        ch, cw = child_grid.shape
        for rr, cc in targets:
            r = int(rr)
            c = int(cc)
            children: Dict[Tuple[int, int], BoundarySummary] = {}
            for dr in (0, 1):
                cr = 2 * r + dr
                if cr >= ch:
                    continue
                for dc in (0, 1):
                    cc_child = 2 * c + dc
                    if cc_child >= cw:
                        continue
                    children[(dr, dc)] = child_grid[cr, cc_child]
            parent[r, c] = self._compose_boundary_summary_2x2(children)
        return parent

    def _update_boundary_summary_pyramid(
        self,
        state_fine: np.ndarray,
        *,
        changed_mask: np.ndarray,
        max_power: int,
    ) -> Dict[int, np.ndarray]:
        h, w = state_fine.shape
        max_power = max(0, int(max_power))
        full_refresh = (
            self._boundary_summary_shape != (h, w)
            or 0 not in self._boundary_summary_cache
            or self._boundary_summary_cache[0].shape != (h, w)
        )
        dirty_by_power: Dict[int, np.ndarray] = {}

        if full_refresh:
            level0 = np.empty((h, w), dtype=object)
            for r in range(h):
                for c in range(w):
                    level0[r, c] = self._make_level0_boundary_summary(int(state_fine[r, c]))
            self._boundary_summary_cache = {0: level0}
            self._boundary_summary_shape = (h, w)
            dirty_by_power[0] = np.ones((h, w), dtype=bool)
            prev_dirty = dirty_by_power[0]
            for power in range(1, max_power + 1):
                parent = self._compose_boundary_summary_parent_grid(self._boundary_summary_cache[power - 1])
                self._boundary_summary_cache[power] = parent
                dirty_by_power[power] = np.ones(parent.shape, dtype=bool)
                prev_dirty = dirty_by_power[power]
            return dirty_by_power

        dirty0 = changed_mask.astype(bool, copy=False)
        dirty_by_power[0] = dirty0
        if np.any(dirty0):
            level0 = self._boundary_summary_cache[0]
            for r, c in np.argwhere(dirty0):
                level0[int(r), int(c)] = self._make_level0_boundary_summary(int(state_fine[int(r), int(c)]))

        prev_dirty = dirty0
        for power in range(1, max_power + 1):
            child = self._boundary_summary_cache[power - 1]
            dirty_parent = self._dirty_blocks_from_changed(prev_dirty, 2)
            existing = self._boundary_summary_cache.get(power)
            if existing is None or existing.shape != self._boundary_summary_grid_shape(child.shape):
                parent = self._compose_boundary_summary_parent_grid(child)
                dirty_parent = np.ones(parent.shape, dtype=bool)
            else:
                parent = self._compose_boundary_summary_parent_grid(
                    child,
                    out=existing,
                    dirty_mask=dirty_parent,
                )
            self._boundary_summary_cache[power] = parent
            dirty_by_power[power] = dirty_parent
            prev_dirty = dirty_parent
        return dirty_by_power

    def _dtm_from_boundary_summary_level(
        self,
        *,
        level_id: int,
        power: int,
        dirty_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        summaries = self._boundary_summary_cache[int(power)]
        h, w = summaries.shape
        cache = self._dtm_cache.get(level_id)
        native_mode = self._native_dtm_mode()
        if native_mode not in {"port12", "six"}:
            raise RuntimeError("boundary-summary DTM supports only port12/six native modes")
        dtm_ch = 12 if native_mode == "port12" else 6
        full_refresh = cache is None or cache.shape != (dtm_ch, h, w)
        if full_refresh:
            dtm = np.full((dtm_ch, h, w), float(self.config.dtm_unknown_fill), dtype=np.float32)
            targets = np.argwhere(np.ones((h, w), dtype=bool))
        else:
            dtm = cache
            if dirty_mask is None:
                targets = np.argwhere(np.ones((h, w), dtype=bool))
            else:
                if dirty_mask.shape != (h, w):
                    raise ValueError("dirty_mask shape mismatch for DTM summary cache")
                targets = np.argwhere(dirty_mask)
                if targets.size == 0:
                    return dtm
        for r, c in targets:
            summary = summaries[int(r), int(c)]
            dtm[:, int(r), int(c)] = summary.flags12 if native_mode == "port12" else summary.flags6
        self._dtm_cache[level_id] = dtm
        return dtm

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
        dtm_ch = self._native_dtm_channels()
        full_refresh = cache is None or cache.shape != (dtm_ch, h, w)

        # Level 0 represents single cells. Traversability should therefore be
        # interpreted inside the cell itself, not over a surrounding patch.
        if level_id == 0:
            if full_refresh:
                dtm = self._compute_level0_native_dtm(state_map)
                self._dtm_cache[level_id] = dtm
                return dtm
            if not np.any(changed_mask):
                return cache
            dtm = self._compute_level0_native_dtm(
                state_map,
                out=cache,
                dirty_mask=changed_mask,
            )
            self._dtm_cache[level_id] = dtm
            return dtm

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
                output_mode=self._native_dtm_mode(),
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
            output_mode=self._native_dtm_mode(),
            out=cache,
            dirty_mask=dirty_mask,
        )
        self._dtm_cache[level_id] = dtm
        return dtm

    def _project_dtm_output(self, dtm_map: np.ndarray) -> np.ndarray:
        if dtm_map.ndim != 3:
            raise ValueError("dtm_map must be [C, H, W]")
        mode = self._normalize_dtm_output_mode(self.config.dtm_output_mode)
        native_mode = self._native_dtm_mode()
        expected_ch = 12 if native_mode == "port12" else 6
        if dtm_map.shape[0] != expected_ch:
            raise ValueError(
                f"{native_mode}-channel DTM expected with shape[0] == {expected_ch}"
            )
        if mode == "port12":
            return dtm_map
        if mode == "port6":
            return dtm_map
        if mode == "extent6":
            return dtm_map
        if mode == "six":
            return dtm_map
        if mode == "axis2":
            if dtm_map.shape[0] == 12:
                out = np.empty((2, dtm_map.shape[1], dtm_map.shape[2]), dtype=np.float32)
                out[0] = np.maximum(dtm_map[10], dtm_map[5])  # L->R or R->L
                out[1] = np.maximum(dtm_map[1], dtm_map[6])   # U->D or D->U
                return out
            if dtm_map.shape[0] != 6:
                raise ValueError("two/axis2 projection expects six-, port6-, or port12-channel source DTM")
            out = np.empty((2, dtm_map.shape[1], dtm_map.shape[2]), dtype=np.float32)
            out[0] = dtm_map[4]  # R<->L
            out[1] = dtm_map[1]  # U<->D
            return out
        if mode == "axis2km":
            if dtm_map.shape[0] == 12:
                lr_fwd = dtm_map[10]  # L->R
                lr_rev = dtm_map[5]   # R->L
                ud_fwd = dtm_map[1]   # U->D
                ud_rev = dtm_map[6]   # D->U
                lr = np.maximum(lr_fwd, lr_rev)
                ud = np.maximum(ud_fwd, ud_rev)
                lr_known = np.minimum((lr_fwd >= 0.0).astype(np.float32), (lr_rev >= 0.0).astype(np.float32))
                ud_known = np.minimum((ud_fwd >= 0.0).astype(np.float32), (ud_rev >= 0.0).astype(np.float32))
                out = np.empty((4, dtm_map.shape[1], dtm_map.shape[2]), dtype=np.float32)
                out[0] = (lr > 0.0).astype(np.float32)
                out[1] = (ud > 0.0).astype(np.float32)
                out[2] = lr_known
                out[3] = ud_known
                return out
            if dtm_map.shape[0] != 6:
                raise ValueError("axis2km projection expects six-, port6-, or port12-channel source DTM")
            lr = dtm_map[4]
            ud = dtm_map[1]
            out = np.empty((4, dtm_map.shape[1], dtm_map.shape[2]), dtype=np.float32)
            out[0] = (lr > 0.0).astype(np.float32)   # lr_pass
            out[1] = (ud > 0.0).astype(np.float32)   # ud_pass
            out[2] = (lr >= 0.0).astype(np.float32)  # lr_known
            out[3] = (ud >= 0.0).astype(np.float32)  # ud_known
            return out
        # Legacy 4-channel projection for older models:
        # side-turn directions are merged by max.
        if dtm_map.shape[0] != 6:
            raise ValueError("four-channel projection expects six-channel source DTM")
        out = np.empty((4, dtm_map.shape[1], dtm_map.shape[2]), dtype=np.float32)
        out[0] = dtm_map[4]  # LR
        out[1] = dtm_map[1]  # UD
        out[2] = np.maximum(dtm_map[0], dtm_map[5])  # UR or DL turns
        out[3] = np.maximum(dtm_map[2], dtm_map[3])  # UL or RD turns
        return out

    def _aggregate_dtm_block(self, dtm_fine: np.ndarray, block: int) -> np.ndarray:
        if dtm_fine.ndim != 3:
            raise ValueError("dtm_fine must be [C, H, W]")
        chans = int(dtm_fine.shape[0])
        reduced = [block_reduce_max(dtm_fine[k], block) for k in range(chans)]
        return np.stack(reduced, axis=0).astype(np.float32)

    def _aggregate_dtm_global(self, dtm_fine: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        if dtm_fine.ndim != 3:
            raise ValueError("dtm_fine must be [C, H, W]")
        chans = int(dtm_fine.shape[0])
        reduced = [global_reduce_max(dtm_fine[k], out_h, out_w) for k in range(chans)]
        return np.stack(reduced, axis=0).astype(np.float32)

    def _aggregate_transfer_flags_block(
        self,
        dtm_block: np.ndarray,
        state_block: np.ndarray,
        *,
        dtm_mode: str,
    ) -> np.ndarray:
        expected_ch = 12 if dtm_mode == "port12" else 6
        if dtm_block.ndim != 3 or dtm_block.shape[0] != expected_ch:
            raise ValueError(f"dtm_block must be [{expected_ch}, H, W]")
        if state_block.shape != dtm_block.shape[1:]:
            raise ValueError("state_block shape mismatch")

        bh, bw = state_block.shape
        # port index: 0=N, 1=E, 2=S, 3=W
        num_nodes = bh * bw * 4
        adj = [[] for _ in range(num_nodes)]

        def nid(r: int, c: int, p: int) -> int:
            return ((r * bw + c) * 4) + p

        def add_edge(u: int, v: int):
            adj[u].append(v)

        if dtm_mode in {"six", "port6"}:
            side_related = {
                0: (0, 1, 2),  # up
                1: (0, 3, 4),  # right
                2: (1, 3, 5),  # down
                3: (2, 4, 5),  # left
            }
        else:
            side_related = {
                0: (0, 1, 2, 3, 6, 9),      # up
                1: (0, 3, 4, 5, 7, 10),     # right
                2: (1, 4, 6, 7, 8, 11),     # down
                3: (2, 5, 8, 9, 10, 11),    # left
            }

        def side_known(r: int, c: int, p: int) -> bool:
            ch = dtm_block[:, r, c]
            rel = side_related[p]
            return bool(np.any(ch[list(rel)] >= 0.0))

        def side_open(r: int, c: int, p: int) -> bool:
            ch = dtm_block[:, r, c]
            rel = side_related[p]
            return bool(np.any(ch[list(rel)] > 0.5))

        # Intra-cell transfer edges from 6-channel DTM.
        for r in range(bh):
            for c in range(bw):
                if not np.any(dtm_block[:, r, c] >= 0.0):
                    continue
                ch = dtm_block[:, r, c]
                n = nid(r, c, 0)
                e = nid(r, c, 1)
                s = nid(r, c, 2)
                w = nid(r, c, 3)
                if dtm_mode in {"six", "port6"}:
                    # six/port6 channel order:
                    # 0 U<->R, 1 U<->D, 2 U<->L,
                    # 3 R<->D, 4 R<->L, 5 D<->L
                    if float(ch[0]) > 0.5:
                        add_edge(n, e)
                        add_edge(e, n)
                    if float(ch[1]) > 0.5:
                        add_edge(n, s)
                        add_edge(s, n)
                    if float(ch[2]) > 0.5:
                        add_edge(n, w)
                        add_edge(w, n)
                    if float(ch[3]) > 0.5:
                        add_edge(e, s)
                        add_edge(s, e)
                    if float(ch[4]) > 0.5:
                        add_edge(e, w)
                        add_edge(w, e)
                    if float(ch[5]) > 0.5:
                        add_edge(s, w)
                        add_edge(w, s)
                else:
                    # port12 channel order:
                    # 0 U->R, 1 U->D, 2 U->L,
                    # 3 R->U, 4 R->D, 5 R->L,
                    # 6 D->U, 7 D->R, 8 D->L,
                    # 9 L->U,10 L->R,11 L->D
                    if float(ch[0]) > 0.5:
                        add_edge(n, e)
                    if float(ch[1]) > 0.5:
                        add_edge(n, s)
                    if float(ch[2]) > 0.5:
                        add_edge(n, w)
                    if float(ch[3]) > 0.5:
                        add_edge(e, n)
                    if float(ch[4]) > 0.5:
                        add_edge(e, s)
                    if float(ch[5]) > 0.5:
                        add_edge(e, w)
                    if float(ch[6]) > 0.5:
                        add_edge(s, n)
                    if float(ch[7]) > 0.5:
                        add_edge(s, e)
                    if float(ch[8]) > 0.5:
                        add_edge(s, w)
                    if float(ch[9]) > 0.5:
                        add_edge(w, n)
                    if float(ch[10]) > 0.5:
                        add_edge(w, e)
                    if float(ch[11]) > 0.5:
                        add_edge(w, s)

        # Inter-cell boundary crossings (undirected if both fine cells are free).
        for r in range(bh):
            for c in range(bw):
                if c + 1 < bw and side_open(r, c, 1) and side_open(r, c + 1, 3):
                    u = nid(r, c, 1)      # east of left cell
                    v = nid(r, c + 1, 3)  # west of right cell
                    add_edge(u, v)
                    add_edge(v, u)
                if r + 1 < bh and side_open(r, c, 2) and side_open(r + 1, c, 0):
                    u = nid(r, c, 2)      # south of top cell
                    v = nid(r + 1, c, 0)  # north of bottom cell
                    add_edge(u, v)
                    add_edge(v, u)

        def reachable(src_nodes, tgt_nodes) -> bool:
            src = [x for x in src_nodes if x is not None]
            tgt = {x for x in tgt_nodes if x is not None}
            if not src or not tgt:
                return False
            if any(x in tgt for x in src):
                return True
            q = deque(src)
            vis = set(src)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v in vis:
                        continue
                    if v in tgt:
                        return True
                    vis.add(v)
                    q.append(v)
            return False

        def port_node(r: int, c: int, p: int):
            if r < 0 or r >= bh or c < 0 or c >= bw:
                return None
            if not side_known(r, c, p):
                return None
            if not side_open(r, c, p):
                return None
            return nid(r, c, p)

        side_nodes = {
            "up": [port_node(0, c, 0) for c in range(bw)],
            "right": [port_node(r, bw - 1, 1) for r in range(bh)],
            "down": [port_node(bh - 1, c, 2) for c in range(bw)],
            "left": [port_node(r, 0, 3) for r in range(bh)],
        }

        if dtm_mode in {"six", "port6"}:
            pairs = (
                ("up", "right"),
                ("up", "down"),
                ("up", "left"),
                ("right", "down"),
                ("right", "left"),
                ("down", "left"),
            )
            flags = np.zeros(6, dtype=np.float32)
            for k, (a, b) in enumerate(pairs):
                flags[k] = 1.0 if reachable(side_nodes[a], side_nodes[b]) else 0.0
            return flags

        pairs = (
            ("up", "right"),
            ("up", "down"),
            ("up", "left"),
            ("right", "up"),
            ("right", "down"),
            ("right", "left"),
            ("down", "up"),
            ("down", "right"),
            ("down", "left"),
            ("left", "up"),
            ("left", "right"),
            ("left", "down"),
        )
        flags = np.zeros(12, dtype=np.float32)
        for k, (a, b) in enumerate(pairs):
            flags[k] = 1.0 if reachable(side_nodes[a], side_nodes[b]) else 0.0
        return flags

    def _aggregate_dtm_block_transfer(
        self,
        *,
        dtm_fine: np.ndarray,
        state_fine: np.ndarray,
        state_coarse: np.ndarray,
        block: int,
    ) -> np.ndarray:
        dtm_mode = self._native_dtm_mode()
        expected_ch = 12 if dtm_mode == "port12" else 6
        if dtm_fine.ndim != 3 or dtm_fine.shape[0] != expected_ch:
            raise ValueError(f"dtm_fine must be [{expected_ch}, H, W]")
        h, w = state_fine.shape
        ch = (h + block - 1) // block
        cw = (w + block - 1) // block
        out = np.full((expected_ch, ch, cw), float(self.config.dtm_unknown_fill), dtype=np.float32)
        for r in range(ch):
            rs = r * block
            re = min(h, rs + block)
            for c in range(cw):
                cs = c * block
                ce = min(w, cs + block)
                st = int(state_coarse[r, c])
                if st == int(UNKNOWN_STATE):
                    out[:, r, c] = float(self.config.dtm_unknown_fill)
                    continue
                if st == int(BLOCKED_STATE):
                    out[:, r, c] = 0.0
                    continue
                out[:, r, c] = self._aggregate_transfer_flags_block(
                    dtm_fine[:, rs:re, cs:ce],
                    state_fine[rs:re, cs:ce],
                    dtm_mode=dtm_mode,
                )
        return out

    def _compose_dtm_partition_from_children(
        self,
        *,
        dtm_child: np.ndarray,
        state_child: np.ndarray,
        known_ratio_parent: np.ndarray,
        row_edges: np.ndarray,
        col_edges: np.ndarray,
        out: Optional[np.ndarray] = None,
        dirty_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compose coarse-cell DTM directly from child-cell exits.

        Unlike block_reduce_state()-based coarse DTM, mixed obstacle/free cells are
        not collapsed to BLOCKED before checking connectivity.

        Online semantics:
        - parent known ratio does NOT blank out the whole coarse cell
        - each side-pair is marked passable only if a path is proven through known
          child cells
        - uncertainty is carried by the child-side known/open flags themselves
        """
        dtm_mode = self._native_dtm_mode()
        expected_ch = 12 if dtm_mode == "port12" else 6
        if dtm_child.ndim != 3 or dtm_child.shape[0] != expected_ch:
            raise ValueError(f"dtm_child must be [{expected_ch}, H, W]")
        if state_child.shape != dtm_child.shape[1:]:
            raise ValueError("state_child shape mismatch")

        h, w = state_child.shape
        row_edges = np.asarray(row_edges, dtype=np.int32)
        col_edges = np.asarray(col_edges, dtype=np.int32)
        if row_edges.ndim != 1 or col_edges.ndim != 1:
            raise ValueError("row_edges and col_edges must be 1D")
        if len(row_edges) < 2 or len(col_edges) < 2:
            raise ValueError("row_edges and col_edges must have at least two entries")
        if int(row_edges[0]) != 0 or int(col_edges[0]) != 0:
            raise ValueError("row_edges/col_edges must start at 0")
        if int(row_edges[-1]) != h or int(col_edges[-1]) != w:
            raise ValueError("row_edges/col_edges must end at child grid size")
        ch = len(row_edges) - 1
        cw = len(col_edges) - 1
        if known_ratio_parent.shape != (ch, cw):
            raise ValueError("known_ratio_parent shape mismatch")

        if out is None:
            out_arr = np.full((expected_ch, ch, cw), float(self.config.dtm_unknown_fill), dtype=np.float32)
        else:
            if out.shape != (expected_ch, ch, cw):
                raise ValueError("out shape mismatch in _compose_dtm_partition_from_children")
            out_arr = out.astype(np.float32, copy=False)

        if dirty_mask is None:
            targets = np.argwhere(np.ones((ch, cw), dtype=bool))
        else:
            if dirty_mask.shape != (ch, cw):
                raise ValueError("dirty_mask shape mismatch in _compose_dtm_partition_from_children")
            targets = np.argwhere(dirty_mask)
            if targets.size == 0:
                return out_arr

        for r, c in targets:
            rs = int(row_edges[int(r)])
            re = int(row_edges[int(r) + 1])
            cs = int(col_edges[int(c)])
            ce = int(col_edges[int(c) + 1])
            out_arr[:, int(r), int(c)] = self._aggregate_transfer_flags_block(
                dtm_child[:, rs:re, cs:ce],
                state_child[rs:re, cs:ce],
                dtm_mode=dtm_mode,
            )
        return out_arr

    def _compose_dtm_block_from_children(
        self,
        *,
        dtm_child: np.ndarray,
        state_child: np.ndarray,
        known_ratio_parent: np.ndarray,
        block_r: int,
        block_c: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        dirty_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if block_r <= 0:
            raise ValueError("block_r must be positive")
        if block_c is None:
            block_c = block_r
        if block_c <= 0:
            raise ValueError("block_c must be positive")
        h, w = state_child.shape
        ch = (h + block_r - 1) // block_r
        cw = (w + block_c - 1) // block_c
        row_edges = np.asarray([min(h, i * block_r) for i in range(ch + 1)], dtype=np.int32)
        col_edges = np.asarray([min(w, i * block_c) for i in range(cw + 1)], dtype=np.int32)
        row_edges[-1] = h
        col_edges[-1] = w
        return self._compose_dtm_partition_from_children(
            dtm_child=dtm_child,
            state_child=state_child,
            known_ratio_parent=known_ratio_parent,
            row_edges=row_edges,
            col_edges=col_edges,
            out=out,
            dirty_mask=dirty_mask,
        )

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
        cell_phase: Optional[Tuple[float, float, float, float]] = None,
        include_dtm: Optional[bool] = None,
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

        use_dtm = self.include_dtm if include_dtm is None else bool(include_dtm)
        if use_dtm:
            if dtm_map is not None:
                dtm = self._project_dtm_output(dtm_map)
            else:
                if state_map is None:
                    raise RuntimeError("state_map is required when include_dtm=True")
                dtm_raw = compute_directional_traversability(
                    state_map,
                    known_ratio_map=known_ratio_map,
                    patch_size=self.config.dtm_patch_size,
                    connectivity=self.config.dtm_connectivity,
                    require_fully_known_patch=self.config.dtm_require_fully_known_patch,
                    min_center_known_ratio=self.config.dtm_min_known_ratio,
                    min_patch_known_ratio=self.config.dtm_patch_min_known_ratio,
                    uncertain_fill=self.config.dtm_uncertain_fill,
                    unknown_fill=self.config.dtm_unknown_fill,
                    output_mode=self._native_dtm_mode(),
                )
                dtm = self._project_dtm_output(dtm_raw)
            if center is not None:
                for k in range(int(dtm.shape[0])):
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
                channels.extend([dtm[k] for k in range(int(dtm.shape[0]))])

        if self.config.include_cell_phase_channels:
            if cell_phase is None:
                raise RuntimeError("cell_phase is required when include_cell_phase_channels=True")
            for val in cell_phase:
                channels.append(np.full((out_size, out_size), float(val), dtype=np.float32))

        return np.stack(channels, axis=0).astype(np.float32)

    def _get_local_level_maps(
        self,
        *,
        level_id: int,
        block: int,
        covered_f: np.ndarray,
        obstacle_f: np.ndarray,
        frontier_f: np.ndarray,
        known_f: np.ndarray,
        known_free: np.ndarray,
        known_obstacle: np.ndarray,
        unknown: np.ndarray,
        covered_changed: np.ndarray,
        obstacle_changed: np.ndarray,
        frontier_changed: np.ndarray,
        occupancy_changed: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        h, w = covered_f.shape
        ch = (h + block - 1) // block
        cw = (w + block - 1) // block
        cache = self._coarse_cache.get(level_id, None)
        full_refresh = cache is None
        if (not full_refresh) and (
            cache["coverage"].shape != (ch, cw)
            or cache["obstacle"].shape != (ch, cw)
            or cache["frontier"].shape != (ch, cw)
        ):
            full_refresh = True

        if full_refresh:
            cov = block_reduce_mean(covered_f, block)
            known_ratio = block_reduce_mean(known_f, block)
            obs = block_reduce_mean(obstacle_f, block)
            frn = block_reduce_max(frontier_f, block)
            state = None
            if self.include_dtm:
                state = block_reduce_state(
                    known_free,
                    known_obstacle,
                    unknown,
                    block,
                    min_known_ratio=self.config.dtm_min_known_ratio,
                )
            self._coarse_cache[level_id] = {
                "coverage": cov,
                "known_ratio": known_ratio,
                "obstacle": obs,
                "frontier": frn,
                "state": state,
            }
            return cov, known_ratio, obs, frn, state

        cov = cache["coverage"]
        known_ratio = cache["known_ratio"]
        obs = cache["obstacle"]
        frn = cache["frontier"]
        state = cache.get("state", None)

        dirty_cov = self._dirty_blocks_from_changed(covered_changed, block)
        dirty_obs = self._dirty_blocks_from_changed(obstacle_changed, block)
        dirty_frn = self._dirty_blocks_from_changed(frontier_changed, block)
        dirty_occ = self._dirty_blocks_from_changed(occupancy_changed, block)

        if np.any(dirty_cov):
            self._update_block_mean_inplace(cov, covered_f, block=block, dirty_mask=dirty_cov)
        if np.any(dirty_obs):
            self._update_block_mean_inplace(obs, obstacle_f, block=block, dirty_mask=dirty_obs)
        if np.any(dirty_frn):
            self._update_block_max_inplace(frn, frontier_f, block=block, dirty_mask=dirty_frn)
        if np.any(dirty_occ):
            self._update_block_mean_inplace(known_ratio, known_f, block=block, dirty_mask=dirty_occ)
            if self.include_dtm:
                if state is None or state.shape != (ch, cw):
                    state = block_reduce_state(
                        known_free,
                        known_obstacle,
                        unknown,
                        block,
                        min_known_ratio=self.config.dtm_min_known_ratio,
                    )
                else:
                    self._update_block_state_inplace(
                        state,
                        known_free=known_free,
                        known_obstacle=known_obstacle,
                        unknown=unknown,
                        block=block,
                        dirty_mask=dirty_occ,
                    )
                cache["state"] = state
        return cov, known_ratio, obs, frn, state

    def _get_global_level_maps(
        self,
        *,
        level_id: int,
        out_size: int,
        covered_f: np.ndarray,
        obstacle_f: np.ndarray,
        frontier_f: np.ndarray,
        known_f: np.ndarray,
        known_free: np.ndarray,
        known_obstacle: np.ndarray,
        unknown: np.ndarray,
        covered_changed: np.ndarray,
        obstacle_changed: np.ndarray,
        frontier_changed: np.ndarray,
        occupancy_changed: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        cache = self._coarse_cache.get(level_id, None)
        full_refresh = cache is None
        shape = (out_size, out_size)
        if (not full_refresh) and (
            cache["coverage"].shape != shape
            or cache["obstacle"].shape != shape
            or cache["frontier"].shape != shape
        ):
            full_refresh = True

        if full_refresh:
            cov = global_reduce_mean(covered_f, out_size, out_size)
            known_ratio = global_reduce_mean(known_f, out_size, out_size)
            obs = global_reduce_mean(obstacle_f, out_size, out_size)
            frn = global_reduce_max(frontier_f, out_size, out_size)
            state = None
            if self.include_dtm:
                state = global_reduce_state(
                    known_free,
                    known_obstacle,
                    unknown,
                    out_size,
                    out_size,
                    min_known_ratio=self.config.dtm_min_known_ratio,
                )
            self._coarse_cache[level_id] = {
                "coverage": cov,
                "known_ratio": known_ratio,
                "obstacle": obs,
                "frontier": frn,
                "state": state,
            }
            return cov, known_ratio, obs, frn, state

        cov = cache["coverage"]
        known_ratio = cache["known_ratio"]
        obs = cache["obstacle"]
        frn = cache["frontier"]
        state = cache.get("state", None)
        dirty_cov = self._dirty_global_from_changed(covered_changed, out_size, out_size)
        dirty_obs = self._dirty_global_from_changed(obstacle_changed, out_size, out_size)
        dirty_frn = self._dirty_global_from_changed(frontier_changed, out_size, out_size)
        dirty_occ = self._dirty_global_from_changed(occupancy_changed, out_size, out_size)

        if np.any(dirty_cov):
            self._update_global_mean_inplace(cov, covered_f, dirty_mask=dirty_cov)
        if np.any(dirty_obs):
            self._update_global_mean_inplace(obs, obstacle_f, dirty_mask=dirty_obs)
        if np.any(dirty_frn):
            self._update_global_max_inplace(frn, frontier_f, dirty_mask=dirty_frn)
        if np.any(dirty_occ):
            self._update_global_mean_inplace(known_ratio, known_f, dirty_mask=dirty_occ)
            if self.include_dtm:
                if state is None or state.shape != shape:
                    state = global_reduce_state(
                        known_free,
                        known_obstacle,
                        unknown,
                        out_size,
                        out_size,
                        min_known_ratio=self.config.dtm_min_known_ratio,
                    )
                else:
                    self._update_global_state_inplace(
                        state,
                        known_free=known_free,
                        known_obstacle=known_obstacle,
                        unknown=unknown,
                        dirty_mask=dirty_occ,
                    )
                cache["state"] = state
        return cov, known_ratio, obs, frn, state

    @staticmethod
    def _phase_sincos(coord: int, coarse_size: float) -> Tuple[float, float]:
        # Cyclic encoding removes discontinuity at coarse-cell boundaries.
        denom = max(float(coarse_size), 1e-6)
        frac = ((float(coord) + 0.5) / denom) % 1.0
        theta = 2.0 * np.pi * frac
        return float(np.sin(theta)), float(np.cos(theta))

    def _compute_cell_phase(
        self,
        *,
        robot_pos: GridPos,
        coarse_h: float,
        coarse_w: float,
    ) -> Tuple[float, float, float, float]:
        rr, cc = robot_pos
        row_sin, row_cos = self._phase_sincos(int(rr), float(coarse_h))
        col_sin, col_cos = self._phase_sincos(int(cc), float(coarse_w))
        return row_sin, row_cos, col_sin, col_cos

    def build_cell_phase_features(
        self,
        map_shape: Tuple[int, int],
        *,
        robot_pos: GridPos,
    ) -> np.ndarray:
        if len(map_shape) != 2:
            raise ValueError("map_shape must contain (height, width)")
        h, w = int(map_shape[0]), int(map_shape[1])
        rr, cc = robot_pos
        if not (0 <= rr < h and 0 <= cc < w):
            raise ValueError(f"robot_pos {robot_pos} is out of bounds {(h, w)}")

        features: List[float] = []
        # Level 0 corresponds to a single original grid cell, so robot-in-cell
        # phase is constant and carries no useful relative-position signal.
        for block in self.config.local_blocks[1:]:
            features.extend(
                self._compute_cell_phase(
                    robot_pos=robot_pos,
                    coarse_h=float(block),
                    coarse_w=float(block),
                )
            )
        global_block = self._global_level_block((h, w))
        features.extend(
            self._compute_cell_phase(
                robot_pos=robot_pos,
                coarse_h=float(global_block),
                coarse_w=float(global_block),
            )
        )
        return np.asarray(features, dtype=np.float32)

    def build_levels(
        self,
        occupancy: np.ndarray,
        *,
        robot_pos: GridPos,
        explored: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        t_total = perf_counter() if self._profile_enabled else 0.0
        if occupancy.ndim != 2:
            raise ValueError("occupancy must be 2D")
        if explored.shape != occupancy.shape:
            raise ValueError("explored shape must match occupancy shape")

        h, w = occupancy.shape
        rr, cc = robot_pos
        if not (0 <= rr < h and 0 <= cc < w):
            raise ValueError(f"robot_pos {robot_pos} is out of bounds {(h, w)}")

        # Optional unknown-policy ablation on map channels only.
        occupancy_obs = occupancy
        if self.config.unknown_policy != "keep":
            occupancy_obs = occupancy.copy()
            unknown_mask = occupancy_obs == self.config.unknown_value
            if self.config.unknown_policy == "as_free":
                occupancy_obs[unknown_mask] = 0
            else:
                occupancy_obs[unknown_mask] = self.config.obstacle_value

        t0 = perf_counter() if self._profile_enabled else 0.0
        known_free, known_obstacle, unknown = extract_known_masks(
            occupancy_obs,
            unknown_value=self.config.unknown_value,
            obstacle_value=self.config.obstacle_value,
        )
        covered = explored.astype(bool) & known_free
        frontier = compute_frontier_map(covered, known_obstacle)
        known_f = (~unknown).astype(np.float32)
        if self._profile_enabled:
            self._profile_add("prep_masks_frontier", perf_counter() - t0)

        occupancy_changed = self._compute_changed_mask(occupancy_obs)
        covered_changed, self._prev_covered = self._compute_array_change(covered, self._prev_covered)
        obstacle_changed, self._prev_known_obstacle = self._compute_array_change(
            known_obstacle, self._prev_known_obstacle
        )
        frontier_changed, self._prev_frontier = self._compute_array_change(frontier, self._prev_frontier)
        _, self._prev_explored = self._compute_array_change(explored, self._prev_explored)

        covered_f = covered.astype(np.float32)
        obstacle_f = known_obstacle.astype(np.float32)
        frontier_f = frontier.astype(np.float32)
        changed_f = occupancy_changed if self.include_dtm else None
        dtm_fine = None
        state_fine = None
        use_boundary_summary_dtm = (
            self.include_dtm
            and self.config.dtm_coarse_mode == "bfs"
            and self._native_dtm_mode() in {"port12", "six"}
        )
        boundary_dirty_by_power: Dict[int, np.ndarray] = {}
        block_power: Dict[int, int] = {}
        max_boundary_power = 0
        if use_boundary_summary_dtm:
            for block in self.config.local_blocks:
                exp = self._power_of_two_exp(int(block))
                if exp is not None:
                    block_power[int(block)] = exp
                    max_boundary_power = max(max_boundary_power, exp)
            global_block_tmp = self._global_level_block((h, w))
            gexp = self._power_of_two_exp(global_block_tmp)
            if gexp is not None:
                max_boundary_power = max(max_boundary_power, gexp)
        need_fine_dtm = self.include_dtm and (
            use_boundary_summary_dtm
            or self.config.dtm_coarse_mode in {"aggregate", "aggregate_transfer"}
            or len(self.config.local_blocks) > 1
        )
        if need_fine_dtm:
            t0 = perf_counter() if self._profile_enabled else 0.0
            state_fine = np.full(occupancy.shape, BLOCKED_STATE, dtype=np.int8)
            state_fine[unknown] = UNKNOWN_STATE
            state_fine[known_free] = FREE_STATE
            if use_boundary_summary_dtm:
                boundary_dirty_by_power = self._update_boundary_summary_pyramid(
                    state_fine,
                    changed_mask=occupancy_changed,
                    max_power=max_boundary_power,
                )
                dtm_fine = self._dtm_from_boundary_summary_level(
                    level_id=0,
                    power=0,
                    dirty_mask=boundary_dirty_by_power.get(0),
                )
            else:
                dtm_fine = self._update_dtm_level(
                    level_id=0,
                    state_map=state_fine,
                    known_ratio_map=known_f,
                    changed_mask=changed_f,
                )
            if self._profile_enabled:
                self._profile_add("dtm_fine", perf_counter() - t0)

        levels: Dict[int, np.ndarray] = {}
        local_native_levels = []

        # Local robot-centered levels.
        for lv, block in enumerate(self.config.local_blocks):
            t0 = perf_counter() if self._profile_enabled else 0.0
            cov_coarse, known_ratio_coarse, obs_coarse, frn_coarse, state_coarse = self._get_local_level_maps(
                level_id=lv,
                block=block,
                covered_f=covered_f,
                obstacle_f=obstacle_f,
                frontier_f=frontier_f,
                known_f=known_f,
                known_free=known_free,
                known_obstacle=known_obstacle,
                unknown=unknown,
                covered_changed=covered_changed,
                obstacle_changed=obstacle_changed,
                frontier_changed=frontier_changed,
                occupancy_changed=occupancy_changed,
            )
            if self._profile_enabled:
                self._profile_add("local_reduce", perf_counter() - t0)
            dtm_coarse = None
            if self.include_dtm:
                t1 = perf_counter() if self._profile_enabled else 0.0
                if use_boundary_summary_dtm and int(block) in block_power:
                    dtm_coarse = self._dtm_from_boundary_summary_level(
                        level_id=lv,
                        power=block_power[int(block)],
                        dirty_mask=boundary_dirty_by_power.get(block_power[int(block)]),
                    )
                elif self.config.dtm_coarse_mode in {"aggregate", "aggregate_transfer"}:
                    if block == 1:
                        dtm_coarse = dtm_fine
                    elif self.config.dtm_coarse_mode == "aggregate_transfer":
                        dtm_coarse = self._aggregate_dtm_block_transfer(
                            dtm_fine=dtm_fine,
                            state_fine=state_fine,
                            state_coarse=state_coarse,
                            block=block,
                        )
                    else:
                        dtm_coarse = self._aggregate_dtm_block(dtm_fine, block)
                else:
                    if lv == 0:
                        changed_coarse = self._dirty_blocks_from_changed(occupancy_changed, block)
                        dtm_coarse = self._update_dtm_level(
                            level_id=lv,
                            state_map=state_coarse,
                            known_ratio_map=known_ratio_coarse,
                            changed_mask=changed_coarse,
                        )
                    elif dtm_fine is not None and state_fine is not None:
                        dirty_parent = self._dirty_blocks_from_changed(occupancy_changed, block)
                        cache = self._dtm_cache.get(lv, None)
                        if cache is not None and cache.shape == (
                            self._native_dtm_channels(),
                            state_coarse.shape[0],
                            state_coarse.shape[1],
                        ):
                            dtm_coarse = self._compose_dtm_block_from_children(
                                dtm_child=dtm_fine,
                                state_child=state_fine,
                                known_ratio_parent=known_ratio_coarse,
                                block_r=block,
                                out=cache,
                                dirty_mask=dirty_parent,
                            )
                        else:
                            dtm_coarse = self._compose_dtm_block_from_children(
                                dtm_child=dtm_fine,
                                state_child=state_fine,
                                known_ratio_parent=known_ratio_coarse,
                                block_r=block,
                            )
                        self._dtm_cache[lv] = dtm_coarse
                    else:
                        changed_coarse = self._dirty_blocks_from_changed(occupancy_changed, block)
                        dtm_coarse = self._update_dtm_level(
                            level_id=lv,
                            state_map=state_coarse,
                            known_ratio_map=known_ratio_coarse,
                            changed_mask=changed_coarse,
                        )
                if self._profile_enabled:
                    self._profile_add("local_dtm", perf_counter() - t1)
                local_native_levels.append((block, dtm_coarse, state_coarse))

            center_coarse = (rr // block, cc // block)
            cell_phase_local = self._compute_cell_phase(
                robot_pos=(rr, cc),
                coarse_h=float(block),
                coarse_w=float(block),
            )
            t2 = perf_counter() if self._profile_enabled else 0.0
            levels[lv] = self._build_level_channels(
                cov_coarse,
                obs_coarse,
                frn_coarse,
                state_coarse,
                known_ratio_coarse,
                dtm_coarse,
                center=center_coarse,
                out_size=self.config.local_window_size,
                cell_phase=cell_phase_local,
                include_dtm=self._include_dtm_for_level(lv),
            )
            if self._profile_enabled:
                self._profile_add("local_pack", perf_counter() - t2)

        # Final robot-centered coarse level.
        global_block = self._global_level_block((h, w))
        t0 = perf_counter() if self._profile_enabled else 0.0
        level_id = len(self.config.local_blocks)
        cov_global, known_ratio_global, obs_global, frn_global, state_global = self._get_local_level_maps(
            level_id=level_id,
            block=global_block,
            covered_f=covered_f,
            obstacle_f=obstacle_f,
            frontier_f=frontier_f,
            known_f=known_f,
            known_free=known_free,
            known_obstacle=known_obstacle,
            unknown=unknown,
            covered_changed=covered_changed,
            obstacle_changed=obstacle_changed,
            frontier_changed=frontier_changed,
            occupancy_changed=occupancy_changed,
        )
        if self._profile_enabled:
            self._profile_add("global_reduce", perf_counter() - t0)
        dtm_global = None
        if self.include_dtm:
            t1 = perf_counter() if self._profile_enabled else 0.0
            global_power = self._power_of_two_exp(global_block)
            if use_boundary_summary_dtm and global_power is not None:
                dtm_global = self._dtm_from_boundary_summary_level(
                    level_id=level_id,
                    power=global_power,
                    dirty_mask=boundary_dirty_by_power.get(global_power),
                )
            elif self.config.dtm_coarse_mode in {"aggregate", "aggregate_transfer"}:
                if global_block == 1:
                    dtm_global = dtm_fine
                elif self.config.dtm_coarse_mode == "aggregate_transfer":
                    dtm_global = self._aggregate_dtm_block_transfer(
                        dtm_fine=dtm_fine,
                        state_fine=state_fine,
                        state_coarse=state_global,
                        block=global_block,
                    )
                else:
                    dtm_global = self._aggregate_dtm_block(dtm_fine, global_block)
            else:
                if global_block == 1:
                    changed_global = self._dirty_blocks_from_changed(occupancy_changed, global_block)
                    dtm_global = self._update_dtm_level(
                        level_id=level_id,
                        state_map=state_global,
                        known_ratio_map=known_ratio_global,
                        changed_mask=changed_global,
                    )
                elif dtm_fine is not None and state_fine is not None:
                    dirty_parent = self._dirty_blocks_from_changed(occupancy_changed, global_block)
                    cache = self._dtm_cache.get(level_id, None)
                    if cache is not None and cache.shape == (
                        self._native_dtm_channels(),
                        state_global.shape[0],
                        state_global.shape[1],
                    ):
                        dtm_global = self._compose_dtm_block_from_children(
                            dtm_child=dtm_fine,
                            state_child=state_fine,
                            known_ratio_parent=known_ratio_global,
                            block_r=global_block,
                            out=cache,
                            dirty_mask=dirty_parent,
                        )
                    else:
                        dtm_global = self._compose_dtm_block_from_children(
                            dtm_child=dtm_fine,
                            state_child=state_fine,
                            known_ratio_parent=known_ratio_global,
                            block_r=global_block,
                        )
                    self._dtm_cache[level_id] = dtm_global
                else:
                    changed_global = self._dirty_blocks_from_changed(occupancy_changed, global_block)
                    dtm_global = self._update_dtm_level(
                        level_id=level_id,
                        state_map=state_global,
                        known_ratio_map=known_ratio_global,
                        changed_mask=changed_global,
                    )
            if self._profile_enabled:
                self._profile_add("global_dtm", perf_counter() - t1)

        center_global = (rr // global_block, cc // global_block)
        cell_phase_global = self._compute_cell_phase(
            robot_pos=(rr, cc),
            coarse_h=float(global_block),
            coarse_w=float(global_block),
        )

        t2 = perf_counter() if self._profile_enabled else 0.0
        levels[level_id] = self._build_level_channels(
            cov_global,
            obs_global,
            frn_global,
            state_global,
            known_ratio_global,
            dtm_global,
            center=center_global,
            out_size=self.config.local_window_size,
            cell_phase=cell_phase_global,
            include_dtm=self._include_dtm_for_level(level_id),
        )
        if self._profile_enabled:
            self._profile_add("global_pack", perf_counter() - t2)
            self._profile_add("build_levels_total", perf_counter() - t_total)
            self._profile_calls += 1
        return levels


class HybridLocalGlobalCPPObservationBuilder:
    """
    Hybrid observation for performance-oriented CPP experiments.

    Output
    ------
    {
        "local":  [5, local_crop_size, local_crop_size],
        "global_64": [4 + dtm_channels, 64, 64],
        "global_32": [4 + dtm_channels, 32, 32],
        "global_16": [4 + dtm_channels, 16, 16],
    }

    Local channels keep high-resolution online geometry around the robot:
    known_free, obstacle, unknown, explored, frontier.

    Global channels follow the paper multiscale baseline convention:
    coverage, obstacle, frontier, robot_marker. Optional DTM channels are
    appended only to each global tensor.
    """

    _LOCAL_CHANNELS = (
        "known_free",
        "obstacle",
        "unknown",
        "explored",
        "frontier",
    )
    _GLOBAL_BASE_CHANNELS = (
        "coverage",
        "obstacle_ratio",
        "frontier",
        "robot_marker",
    )
    _DTM_CHANNELS_6 = MultiScaleCPPObservationBuilder._DTM_CHANNELS_6
    _DTM_CHANNELS_2 = MultiScaleCPPObservationBuilder._DTM_CHANNELS_2

    def __init__(
        self,
        config: Optional[HybridLocalGlobalCPPObservationConfig] = None,
        *,
        include_dtm: bool = False,
        profile_enabled: bool = False,
    ):
        self.config = config or HybridLocalGlobalCPPObservationConfig()
        self.include_dtm = bool(include_dtm)
        self._profile_enabled = bool(profile_enabled)
        self._profile_totals: Dict[str, float] = {}
        self._profile_calls = 0
        if int(self.config.local_crop_size) <= 0 or int(self.config.local_crop_size) % 2 == 0:
            raise ValueError("local_crop_size must be a positive odd integer")
        self.global_sizes = tuple(int(s) for s in self.config.global_coarse_sizes)
        if len(self.global_sizes) == 0:
            raise ValueError("global_coarse_sizes must not be empty")
        if any(s <= 0 for s in self.global_sizes):
            raise ValueError("global_coarse_sizes values must be positive")
        if self.config.unknown_policy not in {"keep", "as_free", "as_obstacle"}:
            raise ValueError("unknown_policy must be one of {'keep', 'as_free', 'as_obstacle'}")
        mode = self._normalize_dtm_output_mode(self.config.dtm_output_mode)
        if mode not in {"six", "axis2"}:
            raise ValueError("hybrid DTM supports dtm_output_mode in {'six', 'two', 'axis2'}")
        self._global_dtm_cache: Dict[int, np.ndarray] = {}
        self._dtm_builder = MultiScaleCPPObservationBuilder(
            MultiScaleCPPObservationConfig(
                local_blocks=(1,),
                local_window_size=1,
                global_window_size=max(self.global_sizes),
                unknown_value=int(self.config.unknown_value),
                obstacle_value=int(self.config.obstacle_value),
                unknown_policy=str(self.config.unknown_policy),
                dtm_patch_size=int(self.config.dtm_patch_size),
                dtm_connectivity=int(self.config.dtm_connectivity),
                dtm_require_fully_known_patch=bool(self.config.dtm_require_fully_known_patch),
                dtm_min_known_ratio=float(self.config.dtm_min_known_ratio),
                dtm_patch_min_known_ratio=float(self.config.dtm_patch_min_known_ratio),
                dtm_uncertain_fill=float(self.config.dtm_uncertain_fill),
                dtm_unknown_fill=float(self.config.dtm_unknown_fill),
                dtm_coarse_mode="bfs",
                dtm_output_mode=str(self.config.dtm_output_mode),
                include_cell_phase_channels=False,
                exclude_dtm_level0=False,
            ),
            include_dtm=bool(include_dtm),
            profile_enabled=bool(profile_enabled),
        )

    @staticmethod
    def _normalize_dtm_output_mode(mode: str) -> str:
        mode_s = str(mode).strip().lower()
        return "axis2" if mode_s == "two" else mode_s

    @property
    def local_channel_names(self) -> Tuple[str, ...]:
        return self._LOCAL_CHANNELS

    @property
    def global_channel_names(self) -> Tuple[str, ...]:
        names = self._GLOBAL_BASE_CHANNELS
        if self.include_dtm:
            mode = self._normalize_dtm_output_mode(self.config.dtm_output_mode)
            if mode == "axis2":
                names = names + self._DTM_CHANNELS_2
            else:
                names = names + self._DTM_CHANNELS_6
        return names

    def enable_profiling(self, enabled: bool = True):
        self._profile_enabled = bool(enabled)
        self._dtm_builder.enable_profiling(enabled)
        if not enabled:
            self.reset_profile()

    def reset_profile(self):
        self._profile_totals = {}
        self._profile_calls = 0
        self._dtm_builder.reset_profile()

    def reset_incremental_state(self, *, preserve_static_map: bool = False):
        self._global_dtm_cache = {}
        self._dtm_builder.reset_incremental_state(preserve_static_map=preserve_static_map)

    def profile_snapshot(self, *, reset: bool = False) -> Dict[str, float]:
        snap: Dict[str, float] = {"calls": float(self._profile_calls)}
        snap.update(self._profile_totals)
        dtm_snap = self._dtm_builder.profile_snapshot(reset=reset)
        for key, value in dtm_snap.items():
            snap[f"dtm_{key}"] = float(value)
        if reset:
            self.reset_profile()
        return snap

    def _profile_add(self, key: str, dt: float):
        if not self._profile_enabled:
            return
        self._profile_totals[key] = float(self._profile_totals.get(key, 0.0)) + float(dt)

    def _pad_value_for(self, channel: str) -> float:
        policy = str(self.config.unknown_policy)
        if channel == "known_free":
            return 1.0 if policy == "as_free" else 0.0
        if channel == "obstacle":
            return 1.0 if policy == "as_obstacle" else 0.0
        if channel == "unknown":
            return 0.0 if policy in {"as_free", "as_obstacle"} else 1.0
        return 0.0

    def _apply_unknown_policy(self, occupancy: np.ndarray) -> np.ndarray:
        if self.config.unknown_policy == "keep":
            return occupancy
        out = occupancy.copy()
        unknown_mask = out == int(self.config.unknown_value)
        if self.config.unknown_policy == "as_free":
            out[unknown_mask] = 0
        else:
            out[unknown_mask] = int(self.config.obstacle_value)
        return out

    def _build_local(
        self,
        *,
        known_free: np.ndarray,
        known_obstacle: np.ndarray,
        unknown: np.ndarray,
        explored: np.ndarray,
        frontier: np.ndarray,
        robot_pos: GridPos,
    ) -> np.ndarray:
        size = int(self.config.local_crop_size)
        explored_known = np.asarray(explored, dtype=bool) & np.asarray(known_free, dtype=bool)
        source = {
            "known_free": np.asarray(known_free, dtype=np.float32),
            "obstacle": np.asarray(known_obstacle, dtype=np.float32),
            "unknown": np.asarray(unknown, dtype=np.float32),
            "explored": explored_known.astype(np.float32),
            "frontier": np.asarray(frontier, dtype=np.float32),
        }
        channels = [
            center_crop_with_pad(
                source[name],
                robot_pos,
                size,
                size,
                pad_value=self._pad_value_for(name),
            )
            for name in self._LOCAL_CHANNELS
        ]
        return np.stack(channels, axis=0).astype(np.float32)

    def _build_global_base(
        self,
        *,
        size: int,
        known_free: np.ndarray,
        known_obstacle: np.ndarray,
        explored: np.ndarray,
        frontier: np.ndarray,
        robot_pos: GridPos,
    ) -> np.ndarray:
        explored_known = np.asarray(explored, dtype=bool) & np.asarray(known_free, dtype=bool)
        marker = np.zeros((int(size), int(size)), dtype=np.float32)
        h, w = known_free.shape
        rr, cc = robot_pos
        mr = min(int(size) - 1, max(0, (int(rr) * int(size)) // max(1, int(h))))
        mc = min(int(size) - 1, max(0, (int(cc) * int(size)) // max(1, int(w))))
        marker[mr, mc] = 1.0
        channels = [
            global_reduce_mean(explored_known.astype(np.float32), size, size),
            global_reduce_mean(np.asarray(known_obstacle, dtype=np.float32), size, size),
            global_reduce_max(np.asarray(frontier, dtype=np.float32), size, size),
            marker,
        ]
        return np.stack(channels, axis=0).astype(np.float32)

    @staticmethod
    def _build_state_fine(
        *,
        occupancy_shape: Tuple[int, int],
        known_free: np.ndarray,
        unknown: np.ndarray,
    ) -> np.ndarray:
        state_fine = np.full(occupancy_shape, BLOCKED_STATE, dtype=np.int8)
        state_fine[np.asarray(unknown, dtype=bool)] = UNKNOWN_STATE
        state_fine[np.asarray(known_free, dtype=bool)] = FREE_STATE
        return state_fine

    def _global_power_for_size(self, *, map_shape: Tuple[int, int], size: int) -> Optional[int]:
        h, w = int(map_shape[0]), int(map_shape[1])
        size = int(size)
        if size <= 0 or h <= 0 or w <= 0:
            return None
        if h % size != 0 or w % size != 0:
            return None
        block_h = h // size
        block_w = w // size
        if block_h != block_w:
            return None
        return self._dtm_builder._power_of_two_exp(block_h)

    def _build_boundary_dtm_pyramid(
        self,
        *,
        occupancy_obs: np.ndarray,
        known_free: np.ndarray,
        unknown: np.ndarray,
    ) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
        state_fine = self._build_state_fine(
            occupancy_shape=occupancy_obs.shape,
            known_free=known_free,
            unknown=unknown,
        )
        occupancy_changed = self._dtm_builder._compute_changed_mask(occupancy_obs)
        power_by_size: Dict[int, int] = {}
        for size in self.global_sizes:
            power = self._global_power_for_size(map_shape=occupancy_obs.shape, size=int(size))
            if power is not None:
                power_by_size[int(size)] = int(power)
        if not power_by_size:
            return {}, {}
        dirty_by_power = self._dtm_builder._update_boundary_summary_pyramid(
            state_fine,
            changed_mask=occupancy_changed,
            max_power=max(power_by_size.values()),
        )
        return power_by_size, dirty_by_power

    def _build_global_dtm_from_boundary(
        self,
        *,
        size: int,
        power_by_size: Dict[int, int],
        dirty_by_power: Dict[int, np.ndarray],
    ) -> np.ndarray:
        power = int(power_by_size[int(size)])
        native = self._dtm_builder._dtm_from_boundary_summary_level(
            level_id=10_000 + int(size),
            power=power,
            dirty_mask=dirty_by_power.get(power),
        )
        return self._dtm_builder._project_dtm_output(native).astype(np.float32)

    def _build_dtm_fine(
        self,
        *,
        occupancy_obs: np.ndarray,
        known_free: np.ndarray,
        unknown: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_fine = self._build_state_fine(
            occupancy_shape=occupancy_obs.shape,
            known_free=known_free,
            unknown=unknown,
        )

        occupancy_changed = self._dtm_builder._compute_changed_mask(occupancy_obs)
        known_ratio_fine = (~np.asarray(unknown, dtype=bool)).astype(np.float32)
        dtm_fine = self._dtm_builder._update_dtm_level(
            level_id=0,
            state_map=state_fine,
            known_ratio_map=known_ratio_fine,
            changed_mask=occupancy_changed,
        )
        return dtm_fine, state_fine, occupancy_changed

    def _build_global_dtm_from_fine(
        self,
        *,
        size: int,
        dtm_fine: np.ndarray,
        state_fine: np.ndarray,
        occupancy_changed: np.ndarray,
        known_free: np.ndarray,
        unknown: np.ndarray,
    ) -> np.ndarray:
        known_ratio = global_reduce_mean((~np.asarray(unknown, dtype=bool)).astype(np.float32), size, size)
        h, w = known_free.shape
        row_edges = self._dtm_builder._global_edges(h, size)
        col_edges = self._dtm_builder._global_edges(w, size)
        dirty_global = self._dtm_builder._dirty_global_from_changed(occupancy_changed, size, size)
        expected_ch = self._dtm_builder._native_dtm_channels()
        out = self._global_dtm_cache.get(int(size))
        if out is None or out.shape != (expected_ch, size, size):
            out = None
        native = self._dtm_builder._compose_dtm_partition_from_children(
            dtm_child=dtm_fine,
            state_child=state_fine,
            known_ratio_parent=known_ratio,
            row_edges=row_edges,
            col_edges=col_edges,
            out=out,
            dirty_mask=dirty_global,
        )
        self._global_dtm_cache[int(size)] = native
        return self._dtm_builder._project_dtm_output(native).astype(np.float32)

    def build(
        self,
        occupancy: np.ndarray,
        *,
        robot_pos: GridPos,
        explored: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        t_total = perf_counter() if self._profile_enabled else 0.0
        if occupancy.ndim != 2:
            raise ValueError("occupancy must be 2D")
        if explored.shape != occupancy.shape:
            raise ValueError("explored shape must match occupancy shape")
        h, w = occupancy.shape
        rr, cc = robot_pos
        if not (0 <= rr < h and 0 <= cc < w):
            raise ValueError(f"robot_pos {robot_pos} is out of bounds {(h, w)}")

        t0 = perf_counter() if self._profile_enabled else 0.0
        occupancy_obs = self._apply_unknown_policy(np.asarray(occupancy, dtype=np.int32))
        known_free, known_obstacle, unknown = extract_known_masks(
            occupancy_obs,
            unknown_value=int(self.config.unknown_value),
            obstacle_value=int(self.config.obstacle_value),
        )
        covered = np.asarray(explored, dtype=bool) & known_free
        frontier = compute_frontier_map(covered, known_obstacle)
        if self._profile_enabled:
            self._profile_add("prep_masks_frontier", perf_counter() - t0)

        t0 = perf_counter() if self._profile_enabled else 0.0
        local = self._build_local(
            known_free=known_free,
            known_obstacle=known_obstacle,
            unknown=unknown,
            explored=explored,
            frontier=frontier,
            robot_pos=robot_pos,
        )
        if self._profile_enabled:
            self._profile_add("local_pack", perf_counter() - t0)

        dtm_fine = None
        state_fine = None
        occupancy_changed = None
        boundary_power_by_size: Dict[int, int] = {}
        boundary_dirty_by_power: Dict[int, np.ndarray] = {}
        if self.include_dtm:
            candidate_power_by_size = {
                int(size): self._global_power_for_size(map_shape=occupancy_obs.shape, size=int(size))
                for size in self.global_sizes
            }
            use_boundary_summary = (
                self._dtm_builder._native_dtm_mode() in {"port12", "six"}
                and all(power is not None for power in candidate_power_by_size.values())
            )
            t0 = perf_counter() if self._profile_enabled else 0.0
            if use_boundary_summary:
                boundary_power_by_size, boundary_dirty_by_power = self._build_boundary_dtm_pyramid(
                    occupancy_obs=occupancy_obs,
                    known_free=known_free,
                    unknown=unknown,
                )
                if self._profile_enabled:
                    self._profile_add("dtm_boundary_pyramid", perf_counter() - t0)
            else:
                dtm_fine, state_fine, occupancy_changed = self._build_dtm_fine(
                    occupancy_obs=occupancy_obs,
                    known_free=known_free,
                    unknown=unknown,
                )
                if self._profile_enabled:
                    self._profile_add("dtm_fine", perf_counter() - t0)

        t0 = perf_counter() if self._profile_enabled else 0.0
        out: Dict[str, np.ndarray] = {"local": local.astype(np.float32)}
        for size in self.global_sizes:
            global_base = self._build_global_base(
                size=int(size),
                known_free=known_free,
                known_obstacle=known_obstacle,
                explored=explored,
                frontier=frontier,
                robot_pos=robot_pos,
            )
            global_channels = [global_base]
            if self.include_dtm:
                if int(size) in boundary_power_by_size:
                    global_channels.append(
                        self._build_global_dtm_from_boundary(
                            size=int(size),
                            power_by_size=boundary_power_by_size,
                            dirty_by_power=boundary_dirty_by_power,
                        )
                    )
                elif dtm_fine is None or state_fine is None or occupancy_changed is None:
                    raise RuntimeError("DTM fine state was not initialized")
                else:
                    global_channels.append(
                        self._build_global_dtm_from_fine(
                            size=int(size),
                            dtm_fine=dtm_fine,
                            state_fine=state_fine,
                            occupancy_changed=occupancy_changed,
                            known_free=known_free,
                            unknown=unknown,
                        )
                    )
            out[f"global_{int(size)}"] = np.concatenate(global_channels, axis=0).astype(np.float32)
        if self._profile_enabled:
            self._profile_add("global_pack", perf_counter() - t0)
            self._profile_add("build_hybrid_total", perf_counter() - t_total)
            self._profile_calls += 1

        return out
