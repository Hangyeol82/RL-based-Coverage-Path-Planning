from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Tuple

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
class MultiScaleCPPObservationConfig:
    # Robot-centered local levels.
    local_blocks: Tuple[int, ...] = (1, 2, 4, 8, 16)
    local_window_size: int = 7
    # Global level.
    global_window_size: int = 4
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
    # Coarse-level DTM mode:
    # - bfs: compute DTM independently at each scale using BFS
    # - aggregate: compute BFS only on finest level and aggregate upward with max pooling
    # - aggregate_transfer: compute BFS on finest level, then compose coarse cells
    #   using child-cell transfer graph.
    dtm_coarse_mode: str = "bfs"
    # DTM output channel mode:
    # - six: LR, UD, NW->SE, SE->NW, NE->SW, SW->NE
    # - extent6: LR, UD, NW->SE, SE->NW, NE->SW, SW->NE extents in [0,1]
    #   with unknown as -1
    # - axis2: LR, UD only
    # - axis2km: LR/UD split into passable(0/1) + known(0/1)
    # - four: legacy projection to LR, UD, NW-SE, NE-SW
    # - port12: directed side-to-side transitions
    #   [U->R, U->D, U->L, R->U, R->D, R->L, D->U, D->R, D->L, L->U, L->R, L->D]
    dtm_output_mode: str = "six"
    # Append per-level robot-in-cell phase channels:
    # [sin(row_phase), cos(row_phase), sin(col_phase), cos(col_phase)].
    # This helps disambiguate where the robot is inside each coarse cell.
    include_cell_phase_channels: bool = True


class MultiScaleCPPObservationBuilder:
    """
    Build multi-scale observations for CPP ablation experiments.

    Modes
    -----
    - baseline: coverage + obstacle + frontier
    - dtm: baseline + directional traversability maps
      (LR, UD, NW->SE, SE->NW, NE->SW, SW->NE)

    Output
    ------
    Dictionary level -> tensor [C, H, W]
    """

    _BASELINE_CHANNELS = ("coverage", "obstacle", "frontier")
    _DTM_CHANNELS_6 = (
        "dtm_lr",
        "dtm_ud",
        "dtm_nw_se",
        "dtm_se_nw",
        "dtm_ne_sw",
        "dtm_sw_ne",
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
        "dtm_extent_lr",
        "dtm_extent_ud",
        "dtm_extent_nw_se",
        "dtm_extent_se_nw",
        "dtm_extent_ne_sw",
        "dtm_extent_sw_ne",
    )
    _DTM_CHANNELS_4 = ("dtm_lr", "dtm_ud", "dtm_nw_se", "dtm_ne_sw")
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
        if not (0.0 < float(self.config.dtm_min_known_ratio) <= 1.0):
            raise ValueError("dtm_min_known_ratio must be in (0, 1]")
        if not (0.0 < float(self.config.dtm_patch_min_known_ratio) <= 1.0):
            raise ValueError("dtm_patch_min_known_ratio must be in (0, 1]")
        if self.config.dtm_coarse_mode not in {"bfs", "aggregate", "aggregate_transfer"}:
            raise ValueError("dtm_coarse_mode must be one of {'bfs', 'aggregate', 'aggregate_transfer'}")
        if self.config.unknown_policy not in {"keep", "as_free", "as_obstacle"}:
            raise ValueError("unknown_policy must be one of {'keep', 'as_free', 'as_obstacle'}")
        if self.config.dtm_output_mode not in {"six", "extent6", "axis2", "axis2km", "four", "port12"}:
            raise ValueError(
                "dtm_output_mode must be one of {'six', 'extent6', 'axis2', 'axis2km', 'four', 'port12'}"
            )
        if (
            self.config.dtm_output_mode == "extent6"
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

    def reset_incremental_state(self):
        self._coarse_cache = {}
        self._prev_occupancy = None
        self._prev_explored = None
        self._prev_covered = None
        self._prev_known_obstacle = None
        self._prev_frontier = None
        self._dtm_cache.clear()

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

    def _native_dtm_mode(self) -> str:
        # Keep legacy outputs backed by six-channel DTM.
        if self.config.dtm_output_mode in {"six", "axis2", "axis2km", "four"}:
            return "six"
        if self.config.dtm_output_mode == "extent6":
            return "extent6"
        return "port12"

    def _native_dtm_channels(self) -> int:
        mode = self._native_dtm_mode()
        return 12 if mode == "port12" else 6

    @property
    def num_levels(self) -> int:
        return len(self.config.local_blocks) + 1

    @property
    def channel_names(self) -> Tuple[str, ...]:
        if self.include_dtm:
            if self.config.dtm_output_mode == "six":
                dtm_names = self._DTM_CHANNELS_6
            elif self.config.dtm_output_mode == "extent6":
                dtm_names = self._DTM_CHANNELS_EXTENT6
            elif self.config.dtm_output_mode == "axis2":
                dtm_names = self._DTM_CHANNELS_2
            elif self.config.dtm_output_mode == "axis2km":
                dtm_names = self._DTM_CHANNELS_4_AXIS2KM
            elif self.config.dtm_output_mode == "four":
                dtm_names = self._DTM_CHANNELS_4
            else:
                dtm_names = self._DTM_CHANNELS_12
            names = self._BASELINE_CHANNELS + dtm_names
        else:
            names = self._BASELINE_CHANNELS
        if self.config.include_cell_phase_channels:
            names = names + self._CELL_PHASE_CHANNELS
        return names

    @property
    def channels_per_level(self) -> int:
        return len(self.channel_names)

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
        native_mode = self._native_dtm_mode()
        expected_ch = 12 if native_mode == "port12" else 6
        if dtm_map.shape[0] != expected_ch:
            raise ValueError(
                f"{native_mode}-channel DTM expected with shape[0] == {expected_ch}"
            )
        if self.config.dtm_output_mode == "port12":
            return dtm_map
        if self.config.dtm_output_mode == "extent6":
            return dtm_map
        if self.config.dtm_output_mode == "six":
            return dtm_map
        if self.config.dtm_output_mode == "axis2":
            if dtm_map.shape[0] != 6:
                raise ValueError("axis2 projection expects six-channel source DTM")
            out = np.empty((2, dtm_map.shape[1], dtm_map.shape[2]), dtype=np.float32)
            out[0] = dtm_map[0]  # LR
            out[1] = dtm_map[1]  # UD
            return out
        if self.config.dtm_output_mode == "axis2km":
            if dtm_map.shape[0] != 6:
                raise ValueError("axis2km projection expects six-channel source DTM")
            lr = dtm_map[0]
            ud = dtm_map[1]
            out = np.empty((4, dtm_map.shape[1], dtm_map.shape[2]), dtype=np.float32)
            out[0] = (lr > 0.0).astype(np.float32)   # lr_pass
            out[1] = (ud > 0.0).astype(np.float32)   # ud_pass
            out[2] = (lr >= 0.0).astype(np.float32)  # lr_known
            out[3] = (ud >= 0.0).astype(np.float32)  # ud_known
            return out
        # Legacy 4-channel projection for older models:
        # diagonal directions are merged by max.
        if dtm_map.shape[0] != 6:
            raise ValueError("four-channel projection expects six-channel source DTM")
        out = np.empty((4, dtm_map.shape[1], dtm_map.shape[2]), dtype=np.float32)
        out[0] = dtm_map[0]  # LR
        out[1] = dtm_map[1]  # UD
        out[2] = np.maximum(dtm_map[2], dtm_map[3])  # NW-SE or SE-NW
        out[3] = np.maximum(dtm_map[4], dtm_map[5])  # NE-SW or SW-NE
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
        expected_ch = 6 if dtm_mode == "six" else 12
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

        def is_free(r: int, c: int) -> bool:
            return int(state_block[r, c]) == int(FREE_STATE)

        # Intra-cell transfer edges from 6-channel DTM.
        for r in range(bh):
            for c in range(bw):
                if not is_free(r, c):
                    continue
                ch = dtm_block[:, r, c]
                n = nid(r, c, 0)
                e = nid(r, c, 1)
                s = nid(r, c, 2)
                w = nid(r, c, 3)
                if dtm_mode == "six":
                    # ch0: LR (undirected)
                    if float(ch[0]) > 0.5:
                        add_edge(w, e)
                        add_edge(e, w)
                    # ch1: UD (undirected)
                    if float(ch[1]) > 0.5:
                        add_edge(n, s)
                        add_edge(s, n)
                    # ch2: NW->SE
                    if float(ch[2]) > 0.5:
                        add_edge(n, e)
                        add_edge(w, s)
                    # ch3: SE->NW
                    if float(ch[3]) > 0.5:
                        add_edge(e, n)
                        add_edge(s, w)
                    # ch4: NE->SW
                    if float(ch[4]) > 0.5:
                        add_edge(n, w)
                        add_edge(e, s)
                    # ch5: SW->NE
                    if float(ch[5]) > 0.5:
                        add_edge(w, n)
                        add_edge(s, e)
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
                if not is_free(r, c):
                    continue
                if c + 1 < bw and is_free(r, c + 1):
                    u = nid(r, c, 1)      # east of left cell
                    v = nid(r, c + 1, 3)  # west of right cell
                    add_edge(u, v)
                    add_edge(v, u)
                if r + 1 < bh and is_free(r + 1, c):
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
            if not is_free(r, c):
                return None
            return nid(r, c, p)

        side_nodes = {
            "up": [port_node(0, c, 0) for c in range(bw)],
            "right": [port_node(r, bw - 1, 1) for r in range(bh)],
            "down": [port_node(bh - 1, c, 2) for c in range(bw)],
            "left": [port_node(r, 0, 3) for r in range(bh)],
        }

        if dtm_mode == "six":
            # Diagonal flags use quadrant-cell connectivity (all 4 ports on each corner cell)
            # instead of only corner-facing ports. This captures loop/turn-based cases
            # that still traverse from one corner quadrant to the opposite quadrant.
            def cell_nodes(r: int, c: int):
                return [port_node(r, c, p) for p in range(4)]

            nw_nodes = cell_nodes(0, 0)
            se_nodes = cell_nodes(bh - 1, bw - 1)
            ne_nodes = cell_nodes(0, bw - 1)
            sw_nodes = cell_nodes(bh - 1, 0)

            flags = np.zeros(6, dtype=np.float32)
            flags[0] = 1.0 if (reachable(side_nodes["left"], side_nodes["right"]) or reachable(side_nodes["right"], side_nodes["left"])) else 0.0
            flags[1] = 1.0 if (reachable(side_nodes["up"], side_nodes["down"]) or reachable(side_nodes["down"], side_nodes["up"])) else 0.0
            flags[2] = 1.0 if reachable(nw_nodes, se_nodes) else 0.0
            flags[3] = 1.0 if reachable(se_nodes, nw_nodes) else 0.0
            flags[4] = 1.0 if reachable(ne_nodes, sw_nodes) else 0.0
            flags[5] = 1.0 if reachable(sw_nodes, ne_nodes) else 0.0
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
        expected_ch = 6 if dtm_mode == "six" else 12
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
        known_f = (~unknown).astype(np.float32)
        covered = explored.astype(bool) & known_free
        frontier = compute_frontier_map(known_free, unknown)
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
        if self.include_dtm and self.config.dtm_coarse_mode in {"aggregate", "aggregate_transfer"}:
            t0 = perf_counter() if self._profile_enabled else 0.0
            state_fine = np.full(occupancy.shape, BLOCKED_STATE, dtype=np.int8)
            state_fine[unknown] = UNKNOWN_STATE
            state_fine[known_free] = FREE_STATE
            dtm_fine = self._update_dtm_level(
                level_id=0,
                state_map=state_fine,
                known_ratio_map=known_f,
                changed_mask=changed_f,
            )
            if self._profile_enabled:
                self._profile_add("dtm_fine", perf_counter() - t0)

        levels: Dict[int, np.ndarray] = {}

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
                if self.config.dtm_coarse_mode in {"aggregate", "aggregate_transfer"}:
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
                    changed_coarse = self._dirty_blocks_from_changed(occupancy_changed, block)
                    dtm_coarse = self._update_dtm_level(
                        level_id=lv,
                        state_map=state_coarse,
                        known_ratio_map=known_ratio_coarse,
                        changed_mask=changed_coarse,
                    )
                if self._profile_enabled:
                    self._profile_add("local_dtm", perf_counter() - t1)

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
            )
            if self._profile_enabled:
                self._profile_add("local_pack", perf_counter() - t2)

        # Global non-centered level.
        gsize = self.config.global_window_size
        t0 = perf_counter() if self._profile_enabled else 0.0
        level_id = len(self.config.local_blocks)
        cov_global, known_ratio_global, obs_global, frn_global, state_global = self._get_global_level_maps(
            level_id=level_id,
            out_size=gsize,
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
            if self.config.dtm_coarse_mode in {"aggregate", "aggregate_transfer"}:
                dtm_global = self._aggregate_dtm_global(dtm_fine, gsize, gsize)
            else:
                changed_global = self._dirty_global_from_changed(occupancy_changed, gsize, gsize)
                dtm_global = self._update_dtm_level(
                    level_id=level_id,
                    state_map=state_global,
                    known_ratio_map=known_ratio_global,
                    changed_mask=changed_global,
                )
            if self._profile_enabled:
                self._profile_add("global_dtm", perf_counter() - t1)

        # Global coarse cell size can be non-integer when map size is not divisible by gsize.
        global_cell_h = float(max(1, h)) / float(max(1, gsize))
        global_cell_w = float(max(1, w)) / float(max(1, gsize))
        cell_phase_global = self._compute_cell_phase(
            robot_pos=(rr, cc),
            coarse_h=global_cell_h,
            coarse_w=global_cell_w,
        )

        t2 = perf_counter() if self._profile_enabled else 0.0
        levels[len(self.config.local_blocks)] = self._build_level_channels(
            cov_global,
            obs_global,
            frn_global,
            state_global,
            known_ratio_global,
            dtm_global,
            center=None,
            out_size=gsize,
            cell_phase=cell_phase_global,
        )
        if self._profile_enabled:
            self._profile_add("global_pack", perf_counter() - t2)
            self._profile_add("build_levels_total", perf_counter() - t_total)
            self._profile_calls += 1
        return levels
