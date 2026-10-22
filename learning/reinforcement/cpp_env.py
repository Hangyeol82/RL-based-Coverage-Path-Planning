from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from learning.observation import (
    HybridLocalGlobalCPPObservationBuilder,
    HybridLocalGlobalCPPObservationConfig,
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
    RobotStateObservationBuilder,
    RobotStateObservationConfig,
)
from learning.observation.cpp.grid_features import center_crop_with_pad
from .reward import (
    CPPRewardBreakdown,
    CPPRewardConfig,
    CPPRewardInput,
    compute_cpp_reward,
    total_variation,
)


GridPos = Tuple[int, int]

# 4-connected discrete actions:
# 0: up, 1: right, 2: down, 3: left
ACTION_TO_DELTA: Dict[int, GridPos] = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


@dataclass(frozen=True)
class CPPDiscreteEnvConfig:
    sensor_range: int = 2
    max_steps: Optional[int] = 1500
    collision_ends_episode: bool = False
    stop_on_full_coverage: bool = True
    include_dtm: bool = False
    tv_mode: str = "sym-iso"
    local_tv_mode: str = "patch"
    local_tv_patch_radius: int = 3
    # Known-map safety mask (online-safe):
    # - invalid: out-of-bounds or known obstacle (1)
    # - valid: known free (0) or unknown (-1)
    use_action_mask: bool = False
    # shield: replace invalid known-map actions with a valid fallback.
    # penalty: keep the requested invalid action, leave the robot in place, and
    # apply invalid_action_penalty through the collision reward term.
    action_mask_mode: str = "shield"
    invalid_action_penalty: float = -1.0
    # Optional robot-state augmentation from multi-scale DTM at robot's current cell.
    # Per level: [can_exit_up, can_exit_right, can_exit_down, can_exit_left]
    # Optional validity flag can be appended per level.
    use_boundary_exit_features: bool = False
    boundary_exit_threshold: float = 0.0
    boundary_exit_include_valid: bool = True
    # Append per-scale robot-in-cell phase to robot_state instead of broadcasting
    # it as constant CNN map channels.
    use_cell_phase_features: bool = True
    profile_observation: bool = False
    profile_interval_steps: int = 200
    profile_name: str = ""
    # Offline/full-known maps have static occupancy within an episode reset.
    # Preserving occupancy-derived DTM caches avoids rebuilding full-map DTM
    # every time only coverage/explored state is reset.
    preserve_static_map_observation_cache_on_reset: bool = False
    # Optional heuristic assist for training/evaluation. When the policy fails
    # to add new coverage for N consecutive steps, an online-safe graph search can
    # temporarily route to the nearest observed but uncovered known-free cell.
    heuristic_assist_enabled: bool = False
    heuristic_no_progress_threshold: int = 50
    heuristic_min_coverage: float = 0.0
    heuristic_max_astar_expansions: int = 0
    heuristic_turn_cost: float = 0.0
    heuristic_revisit_cost: float = 0.0
    # v2 matches the local-trap rescue trigger used by offline evaluation.
    # It combines hard no-new-cell timeout, period-2 oscillation, and local
    # known-uncovered exhaustion. "simple" preserves the legacy behavior.
    heuristic_trigger_mode: str = "simple"
    heuristic_period2_enabled: bool = True
    heuristic_period2_recent_new_window: int = 4
    heuristic_period2_streak: int = 2
    heuristic_same_cell_enabled: bool = True
    heuristic_same_cell_stall: int = 12
    heuristic_late_local_radius: int = 0
    heuristic_late_local_max_uncovered: int = 0
    heuristic_reset_sgtv_after_rescue: bool = False
    heuristic_reset_policy_state_after_rescue: bool = False

    maps_observation_mode: str = "multiscale"
    observation: MultiScaleCPPObservationConfig = field(
        default_factory=MultiScaleCPPObservationConfig,
    )
    hybrid_observation: HybridLocalGlobalCPPObservationConfig = field(
        default_factory=HybridLocalGlobalCPPObservationConfig,
    )
    robot_state: RobotStateObservationConfig = field(
        default_factory=RobotStateObservationConfig,
    )
    reward: CPPRewardConfig = field(default_factory=CPPRewardConfig)


class CPPDiscreteEnv:
    """
    Online discrete CPP environment for PPO/BC-RL experiments.

    State representation:
    - true occupancy map: 0 free, 1 obstacle
    - known map for online sensing: -1 unknown, 0 free, 1 obstacle
    - explored mask: visited free cells
    """

    def __init__(
        self,
        grid_map: np.ndarray,
        *,
        start_pos: Optional[GridPos] = None,
        config: Optional[CPPDiscreteEnvConfig] = None,
    ):
        cfg = config or CPPDiscreteEnvConfig()
        self.config = cfg

        grid = np.asarray(grid_map, dtype=np.int32)
        if grid.ndim != 2:
            raise ValueError("grid_map must be 2D")
        if not np.isin(grid, [0, 1]).all():
            raise ValueError("grid_map must contain only 0(free) and 1(obstacle)")
        mode = str(cfg.maps_observation_mode).strip().lower()
        if mode not in {"multiscale", "hybrid_local_global"}:
            raise ValueError("maps_observation_mode must be one of {'multiscale', 'hybrid_local_global'}")
        action_mask_mode = str(cfg.action_mask_mode).strip().lower()
        if action_mask_mode not in {"shield", "penalty"}:
            raise ValueError("action_mask_mode must be one of {'shield', 'penalty'}")
        self._action_mask_mode = action_mask_mode

        self.true_map = grid
        self.rows, self.cols = grid.shape
        self.free_mask = self.true_map == 0
        self.free_total = int(np.count_nonzero(self.free_mask))
        if self.free_total <= 0:
            raise ValueError("grid_map must have at least one free cell")

        self.maps_builder = MultiScaleCPPObservationBuilder(
            self.config.observation,
            include_dtm=self.config.include_dtm,
            profile_enabled=bool(self.config.profile_observation),
        )
        self.hybrid_maps_builder = (
            HybridLocalGlobalCPPObservationBuilder(
                self.config.hybrid_observation,
                include_dtm=self.config.include_dtm,
                profile_enabled=bool(self.config.profile_observation),
            )
            if mode == "hybrid_local_global"
            else None
        )
        self._boundary_maps_builder: Optional[MultiScaleCPPObservationBuilder] = None
        if bool(self.config.use_boundary_exit_features) and (not bool(self.config.include_dtm)):
            # Boundary features can be used as an MLP-side replacement even when
            # DTM channels are not part of the CNN maps input.
            self._boundary_maps_builder = MultiScaleCPPObservationBuilder(
                self.config.observation,
                include_dtm=True,
                profile_enabled=bool(self.config.profile_observation),
            )
        self.robot_state_builder = RobotStateObservationBuilder(self.config.robot_state)
        self._profile_observation = bool(self.config.profile_observation)
        self._profile_interval_steps = max(1, int(self.config.profile_interval_steps))
        self._profile_name = str(self.config.profile_name).strip()
        self._obs_profile_totals: Dict[str, float] = {}
        self._obs_profile_calls = 0

        self._default_start = self._resolve_start(start_pos)
        self.known_map = np.full_like(self.true_map, fill_value=-1, dtype=np.int32)
        self.explored = np.zeros_like(self.true_map, dtype=bool)
        self.coverage_time_map = np.full_like(self.true_map, fill_value=-1, dtype=np.int32)
        self._sgtv_window_counts = np.zeros_like(self.true_map, dtype=np.int16)
        self._sgtv_window_deque: Deque[Tuple[int, int, int, int]] = deque(
            maxlen=max(1, int(self.config.reward.semi_global_tv_horizon)),
        )
        self.current_pos: GridPos = self._default_start
        self.prev_pos: Optional[GridPos] = None
        self.prev_action: Optional[int] = None
        self.path: List[GridPos] = [self.current_pos]
        self.steps = 0
        self.done = False
        self.last_collision = False
        self.overlap_streak = 0
        self.last_reward = CPPRewardBreakdown(
            area=0.0,
            tv_local=0.0,
            tv_global=0.0,
            collision=0.0,
            constant=0.0,
            total=0.0,
        )
        self._milestone_hit_90 = False
        self._milestone_hit_99 = False
        self.recent_new_coverage: Deque[float] = deque(
            maxlen=max(1, int(self.config.robot_state.stagnation_window)),
        )
        self.recent_actions: Deque[int] = deque(
            maxlen=max(1, int(self.config.robot_state.action_history_len)),
        )
        self._heuristic_no_progress_streak = 0
        self._heuristic_cached_actions: Deque[int] = deque()
        self._heuristic_cached_target: Optional[GridPos] = None
        self._heuristic_v2_new_events: Deque[int] = deque(maxlen=32)
        self._heuristic_period2_streak = 0
        self._heuristic_same_cell_stall = 0
        self._heuristic_trigger_reason = "none"
        self._heuristic_rescue_selected_previous_step = False

        self.reset()

    @property
    def action_dim(self) -> int:
        return len(ACTION_TO_DELTA)

    def _resolve_start(self, start_pos: Optional[GridPos]) -> GridPos:
        if start_pos is not None:
            r, c = start_pos
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"start_pos {start_pos} out of bounds")
            if self.true_map[r, c] != 0:
                raise ValueError(f"start_pos {start_pos} is not free")
            return int(r), int(c)

        free = np.argwhere(self.true_map == 0)
        if free.size == 0:
            raise ValueError("No free start cell exists")
        r, c = free[0]
        return int(r), int(c)

    def _sense_at(self, pos: GridPos):
        rr, cc = pos
        rng = max(0, int(self.config.sensor_range))
        r0 = max(0, rr - rng)
        r1 = min(self.rows - 1, rr + rng)
        c0 = max(0, cc - rng)
        c1 = min(self.cols - 1, cc + rng)
        self.known_map[r0 : r1 + 1, c0 : c1 + 1] = self.true_map[r0 : r1 + 1, c0 : c1 + 1]

    def _mark_explored(self, pos: GridPos, *, step_index: Optional[int] = None) -> float:
        rr, cc = pos
        if self.true_map[rr, cc] != 0:
            return 0.0
        if self.explored[rr, cc]:
            return 0.0
        self.explored[rr, cc] = True
        self.coverage_time_map[rr, cc] = int(self.steps if step_index is None else step_index)
        return 1.0

    def _coverage_ratio(self) -> float:
        covered = int(np.count_nonzero(self.explored & self.free_mask))
        return float(covered) / float(max(1, self.free_total))

    def _compute_milestone_reward(
        self,
        *,
        prev_coverage_ratio: float,
        curr_coverage_ratio: float,
        collided: bool,
    ) -> Tuple[float, bool, bool]:
        cfg = self.config.reward
        if (not bool(cfg.milestone_reward_enabled)) or bool(collided):
            return 0.0, False, False

        thr90 = float(cfg.milestone_threshold_90)
        thr99 = float(cfg.milestone_threshold_99)
        if not (0.0 < thr90 < 1.0 and 0.0 < thr99 <= 1.0 and thr90 < thr99):
            return 0.0, False, False

        new_cell_unit = max(0.0, float(cfg.new_cell_reward))
        bonus90 = (
            max(0.0, float(cfg.milestone_lambda_90))
            * max(0.0, 1.0 - thr90)
            * float(self.free_total)
            * new_cell_unit
        )
        bonus99 = (
            max(0.0, float(cfg.milestone_lambda_99))
            * max(0.0, 1.0 - thr99)
            * float(self.free_total)
            * new_cell_unit
        )

        hit90_now = False
        hit99_now = False
        reward = 0.0
        if (not self._milestone_hit_90) and (prev_coverage_ratio < thr90 <= curr_coverage_ratio):
            reward += float(bonus90)
            self._milestone_hit_90 = True
            hit90_now = True
        if (not self._milestone_hit_99) and (prev_coverage_ratio < thr99 <= curr_coverage_ratio):
            reward += float(bonus99)
            self._milestone_hit_99 = True
            hit99_now = True
        return float(reward), bool(hit90_now), bool(hit99_now)

    def _coverage_map_float(self) -> np.ndarray:
        known_free = self.known_map == 0
        return (self.explored & known_free).astype(np.float32)

    def _known_obstacle_map_float(self) -> np.ndarray:
        return (self.known_map == 1).astype(np.float32)

    def _local_patch(self, arr: np.ndarray, center: GridPos, radius: int) -> np.ndarray:
        size = max(1, int(radius) * 2 + 1)
        return center_crop_with_pad(arr, center, size, size, pad_value=0.0)

    def _compute_local_tv(self, center: GridPos) -> float:
        cov = self._coverage_map_float()
        obs = self._known_obstacle_map_float()
        return self._compute_local_tv_from_maps(cov, obs, center)

    def _compute_local_tv_from_maps(
        self,
        coverage_map: np.ndarray,
        obstacle_map: np.ndarray,
        center: GridPos,
    ) -> float:
        radius = max(0, int(self.config.local_tv_patch_radius))
        cov_local = self._local_patch(coverage_map, center, radius)
        size = max(1, radius * 2 + 1)
        obs_local = center_crop_with_pad(obstacle_map, center, size, size, pad_value=1.0)
        return float(
            total_variation(
                cov_local,
                obs_local,
                mode=self.config.tv_mode,
                obstacle_mask=obs_local,
                exclude_obstacle_edges=bool(self.config.reward.tv_exclude_obstacle_edges),
                obstacle_edge_weight=float(self.config.reward.tv_obstacle_edge_weight),
            )
        )

    def _compute_incremental_edge_tv_delta(
        self,
        coverage_before: np.ndarray,
        coverage_after: np.ndarray,
        obstacle_map: np.ndarray,
        center: GridPos,
    ) -> float:
        rr, cc = center
        if not (0 <= rr < self.rows and 0 <= cc < self.cols):
            return 0.0
        if obstacle_map[rr, cc] > 0.5:
            return 0.0

        before_center = bool(coverage_before[rr, cc] > 0.5)
        after_center = bool(coverage_after[rr, cc] > 0.5)
        if before_center == after_center:
            return 0.0

        before_edges = 0.0
        after_edges = 0.0
        for dr, dc in ACTION_TO_DELTA.values():
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue
            if obstacle_map[nr, nc] > 0.5:
                continue
            before_neighbor = bool(coverage_before[nr, nc] > 0.5)
            after_neighbor = bool(coverage_after[nr, nc] > 0.5)
            before_edges += float(before_center != before_neighbor)
            after_edges += float(after_center != after_neighbor)
        return float(after_edges - before_edges)

    def _compute_global_tv(self) -> float:
        cov = self._coverage_map_float()
        obs = self._known_obstacle_map_float()
        return float(
            total_variation(
                cov,
                obs,
                mode=self.config.tv_mode,
                obstacle_mask=obs,
                exclude_obstacle_edges=bool(self.config.reward.tv_exclude_obstacle_edges),
                obstacle_edge_weight=float(self.config.reward.tv_obstacle_edge_weight),
            )
        )

    def _sgtv_window_bounds(self, center: GridPos) -> Tuple[int, int, int, int]:
        size = max(1, int(self.config.reward.semi_global_tv_window_size))
        radius = max(0, size // 2)
        rr, cc = center
        r0 = max(0, int(rr) - radius)
        r1 = min(self.rows, int(rr) + radius + 1)
        c0 = max(0, int(cc) - radius)
        c1 = min(self.cols, int(cc) + radius + 1)
        return r0, r1, c0, c1

    def _reset_semi_global_tv_state(self) -> None:
        if self.coverage_time_map.shape != self.true_map.shape:
            self.coverage_time_map = np.full_like(self.true_map, fill_value=-1, dtype=np.int32)
        else:
            self.coverage_time_map.fill(-1)
        if self._sgtv_window_counts.shape != self.true_map.shape:
            self._sgtv_window_counts = np.zeros_like(self.true_map, dtype=np.int16)
        else:
            self._sgtv_window_counts.fill(0)
        self._sgtv_window_deque = deque(
            maxlen=max(1, int(self.config.reward.semi_global_tv_horizon)),
        )

    def _reset_policy_handoff_state_after_rescue(self) -> None:
        """Expose a fresh policy state after a completed heuristic route."""
        self.prev_pos = None
        self.prev_action = None
        self.recent_actions.clear()

    def _push_semi_global_tv_window(self, center: GridPos) -> None:
        bounds = self._sgtv_window_bounds(center)
        if self._sgtv_window_deque.maxlen is not None and len(self._sgtv_window_deque) >= self._sgtv_window_deque.maxlen:
            old_r0, old_r1, old_c0, old_c1 = self._sgtv_window_deque[0]
            self._sgtv_window_counts[old_r0:old_r1, old_c0:old_c1] -= 1
        self._sgtv_window_deque.append(bounds)
        r0, r1, c0, c1 = bounds
        self._sgtv_window_counts[r0:r1, c0:c1] += 1

    def _semi_global_tv_union_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        if not self._sgtv_window_deque:
            return None
        r0 = min(b[0] for b in self._sgtv_window_deque)
        r1 = max(b[1] for b in self._sgtv_window_deque)
        c0 = min(b[2] for b in self._sgtv_window_deque)
        c1 = max(b[3] for b in self._sgtv_window_deque)
        if r0 >= r1 or c0 >= c1:
            return None
        return int(r0), int(r1), int(c0), int(c1)

    def _masked_tv_non_iso(
        self,
        coverage: np.ndarray,
        active_mask: np.ndarray,
        obstacle_mask: np.ndarray,
    ) -> float:
        x = np.asarray(coverage, dtype=np.float32)
        active = np.asarray(active_mask, dtype=bool)
        obs = np.asarray(obstacle_mask, dtype=bool)
        if x.shape != active.shape or x.shape != obs.shape:
            raise ValueError("coverage, active_mask, and obstacle_mask must have the same shape")
        if x.size == 0:
            return 0.0

        diff_v = np.abs(x[1:, :] - x[:-1, :])
        diff_h = np.abs(x[:, 1:] - x[:, :-1])
        valid_v = active[1:, :] & active[:-1, :]
        valid_h = active[:, 1:] & active[:, :-1]
        touches_obs_v = obs[1:, :] | obs[:-1, :]
        touches_obs_h = obs[:, 1:] | obs[:, :-1]
        w = float(np.clip(float(self.config.reward.tv_obstacle_edge_weight), 0.0, 1.0))
        weight_v = np.where(touches_obs_v, w, 1.0)
        weight_h = np.where(touches_obs_h, w, 1.0)
        return float(np.sum(diff_v * valid_v * weight_v) + np.sum(diff_h * valid_h * weight_h))

    def _compute_semi_global_tv_reward(
        self,
        *,
        curr_step_index: int,
        collided: bool,
    ) -> Dict[str, float]:
        cfg = self.config.reward
        scale = float(cfg.semi_global_tv_reward_scale)
        horizon = max(1, int(cfg.semi_global_tv_horizon))
        gate_horizon_raw = int(getattr(cfg, "semi_global_tv_gate_horizon", 0))
        gate_horizon = horizon if gate_horizon_raw <= 0 else max(1, gate_horizon_raw)
        clip_abs = max(0.0, float(cfg.semi_global_tv_clip))

        def zero_info(
            *,
            recent_new: float = 0.0,
            gate_new: float = 0.0,
            active_cells: float = 0.0,
        ) -> Dict[str, float]:
            return {
                "reward_sg_tv": 0.0,
                "semi_global_tv_old": 0.0,
                "semi_global_tv_new": 0.0,
                "semi_global_tv_delta": 0.0,
                "semi_global_tv_normalized": 0.0,
                "semi_global_tv_clipped": 0.0,
                "semi_global_tv_recent_new_cells": float(recent_new),
                "semi_global_tv_gate_new_cells": float(gate_new),
                "semi_global_tv_active_cells": float(active_cells),
            }

        if scale == 0.0 or clip_abs == 0.0 or bool(collided):
            return zero_info()

        bounds = self._semi_global_tv_union_bounds()
        if bounds is None:
            return zero_info()

        r0, r1, c0, c1 = bounds
        active = self._sgtv_window_counts[r0:r1, c0:c1] > 0
        active_cells = int(np.count_nonzero(active))
        if active_cells <= 0:
            return zero_info()

        time_slice = self.coverage_time_map[r0:r1, c0:c1]
        old_step = int(curr_step_index) - int(horizon)
        gate_old_step = int(curr_step_index) - int(gate_horizon)
        old_cov = ((time_slice >= 0) & (time_slice <= old_step) & active).astype(np.float32)
        new_cov = ((time_slice >= 0) & (time_slice <= int(curr_step_index)) & active).astype(np.float32)
        recent_new = int(
            np.count_nonzero(
                (time_slice >= 0)
                & (time_slice > old_step)
                & (time_slice <= int(curr_step_index))
                & active
            )
        )
        gate_new = int(
            np.count_nonzero(
                (time_slice >= 0)
                & (time_slice > gate_old_step)
                & (time_slice <= int(curr_step_index))
                & active
            )
        )
        if gate_new < max(1, int(cfg.semi_global_tv_min_new_cells)):
            return zero_info(
                recent_new=float(recent_new),
                gate_new=float(gate_new),
                active_cells=float(active_cells),
            )

        obs = self._known_obstacle_map_float()[r0:r1, c0:c1] > 0.5
        tv_old = self._masked_tv_non_iso(old_cov, active, obs)
        tv_new = self._masked_tv_non_iso(new_cov, active, obs)
        delta = float(tv_new - tv_old)
        normalized = delta / float(max(1, recent_new))
        clipped = float(np.clip(normalized, -clip_abs, clip_abs))
        reward = -scale * clipped
        return {
            "reward_sg_tv": float(reward),
            "semi_global_tv_old": float(tv_old),
            "semi_global_tv_new": float(tv_new),
            "semi_global_tv_delta": float(delta),
            "semi_global_tv_normalized": float(normalized),
            "semi_global_tv_clipped": float(clipped),
            "semi_global_tv_recent_new_cells": float(recent_new),
            "semi_global_tv_gate_new_cells": float(gate_new),
            "semi_global_tv_active_cells": float(active_cells),
        }

    def _build_observation(self) -> Dict[str, object]:
        t_total = perf_counter() if self._profile_observation else 0.0
        t0 = perf_counter() if self._profile_observation else 0.0
        use_hybrid_maps = str(self.config.maps_observation_mode).strip().lower() == "hybrid_local_global"
        levels: Optional[Dict[int, np.ndarray]] = None
        hybrid_maps: Optional[Dict[str, np.ndarray]] = None
        if use_hybrid_maps:
            if self.hybrid_maps_builder is None:
                raise RuntimeError("hybrid_maps_builder is not initialized")
            hybrid_maps = self.hybrid_maps_builder.build(
                self.known_map,
                robot_pos=self.current_pos,
                explored=self.explored,
            )
        else:
            levels = self.maps_builder.build_levels(
                self.known_map,
                robot_pos=self.current_pos,
                explored=self.explored,
            )
        if self._profile_observation:
            self._profile_add(
                "env_build_hybrid_maps" if use_hybrid_maps else "env_build_levels",
                perf_counter() - t0,
            )
            if use_hybrid_maps:
                self._merge_builder_profile(
                    self.hybrid_maps_builder.profile_snapshot(reset=True),  # type: ignore[union-attr]
                    prefix="hybrid_maps",
                )
            else:
                self._merge_builder_profile(self.maps_builder.profile_snapshot(reset=True), prefix="maps")
        boundary_exit_features: Optional[np.ndarray] = None
        if self.config.use_boundary_exit_features:
            if use_hybrid_maps:
                raise RuntimeError("boundary_exit_features are not supported with hybrid_local_global maps")
            t0 = perf_counter() if self._profile_observation else 0.0
            if self.config.include_dtm:
                if levels is None:
                    raise RuntimeError("levels are required for boundary exit features")
                boundary_levels = levels
                boundary_maps_builder = self.maps_builder
            else:
                if self._boundary_maps_builder is None:
                    raise RuntimeError("Boundary maps builder is not initialized")
                boundary_maps_builder = self._boundary_maps_builder
                boundary_levels = self._boundary_maps_builder.build_levels(
                    self.known_map,
                    robot_pos=self.current_pos,
                    explored=self.explored,
                )
                if self._profile_observation:
                    self._merge_builder_profile(
                        self._boundary_maps_builder.profile_snapshot(reset=True),
                        prefix="boundary_maps",
                    )
            boundary_exit_features = self._build_boundary_exit_features(
                boundary_levels,
                maps_builder=boundary_maps_builder,
            )
            if self._profile_observation:
                self._profile_add("boundary_exit_features", perf_counter() - t0)
        t0 = perf_counter() if self._profile_observation else 0.0
        robot_extra_parts: List[np.ndarray] = []
        if self.config.use_cell_phase_features:
            robot_extra_parts.append(
                self.maps_builder.build_cell_phase_features(
                    self.known_map.shape,
                    robot_pos=self.current_pos,
                )
            )
        if boundary_exit_features is not None:
            robot_extra_parts.append(np.asarray(boundary_exit_features, dtype=np.float32).reshape(-1))
        robot_extra_features = (
            np.concatenate(robot_extra_parts, axis=0).astype(np.float32)
            if robot_extra_parts
            else None
        )
        robot_state = self.robot_state_builder.build(
            occupancy=self.known_map,
            explored=self.explored,
            robot_pos=self.current_pos,
            prev_pos=self.prev_pos,
            recent_actions=list(self.recent_actions),
            action_mask=(
                self.get_action_mask().astype(np.float32)
                if bool(getattr(self.config.robot_state, "include_action_mask", False))
                else None
            ),
            recent_new_coverage=list(self.recent_new_coverage),
            extra_features=robot_extra_features,
        )
        if self._profile_observation:
            self._profile_add("robot_state", perf_counter() - t0)
            self._profile_add("observation_total", perf_counter() - t_total)
            self._obs_profile_calls += 1
            self._maybe_report_observation_profile()
        obs: Dict[str, object] = {"robot_state": robot_state.astype(np.float32)}
        if use_hybrid_maps:
            if hybrid_maps is None:
                raise RuntimeError("hybrid_maps were not built")
            obs["hybrid_maps"] = hybrid_maps
        else:
            if levels is None:
                raise RuntimeError("levels were not built")
            obs["levels"] = levels
        return obs

    def _profile_add(self, key: str, dt: float):
        if not self._profile_observation:
            return
        self._obs_profile_totals[key] = float(self._obs_profile_totals.get(key, 0.0)) + float(dt)

    def _merge_builder_profile(self, snapshot: Dict[str, float], *, prefix: str):
        if not self._profile_observation:
            return
        for key, value in snapshot.items():
            if key == "calls":
                self._profile_add(f"{prefix}_calls", float(value))
            else:
                self._profile_add(f"{prefix}_{key}", float(value))

    def _maybe_report_observation_profile(self):
        if not self._profile_observation:
            return
        if self._obs_profile_calls % self._profile_interval_steps != 0:
            return
        denom = float(max(1, self._profile_interval_steps))
        prefix = self._profile_name or "obs"
        keys = [
            "observation_total",
            "env_build_levels",
            "maps_build_levels_total",
            "maps_prep_masks_frontier",
            "maps_dtm_fine",
            "maps_local_reduce",
            "maps_local_dtm",
            "maps_local_pack",
            "maps_global_reduce",
            "maps_global_dtm",
            "maps_global_pack",
            "env_build_hybrid_maps",
            "hybrid_maps_build_hybrid_total",
            "hybrid_maps_prep_masks_frontier",
            "hybrid_maps_local_pack",
            "hybrid_maps_global_pack",
            "hybrid_maps_dtm_boundary_pyramid",
            "hybrid_maps_dtm_fine",
            "hybrid_maps_dtm_build_levels_total",
            "hybrid_maps_dtm_prep_masks_frontier",
            "hybrid_maps_dtm_global_reduce",
            "hybrid_maps_dtm_global_dtm",
            "hybrid_maps_dtm_global_pack",
            "boundary_exit_features",
            "robot_state",
        ]
        parts = []
        for key in keys:
            val = self._obs_profile_totals.get(key, None)
            if val is None:
                continue
            parts.append(f"{key}={1000.0 * float(val) / denom:.2f}ms")
        print(
            f"[OBS-PROFILE] {prefix} calls={self._obs_profile_calls} "
            + " ".join(parts),
            flush=True,
        )
        self._obs_profile_totals = {}

    def _dtm_output_mode(self) -> str:
        mode = str(self.config.observation.dtm_output_mode).strip().lower()
        return "axis2" if mode == "two" else mode

    def _dtm_channel_count(self) -> int:
        mode = self._dtm_output_mode()
        if mode in {"six", "extent6", "port6"}:
            return 6
        if mode == "axis2":
            return 2
        if mode == "axis2km":
            return 4
        if mode == "four":
            return 4
        if mode == "port12":
            return 12
        raise ValueError(f"Unsupported dtm_output_mode: {self.config.observation.dtm_output_mode}")

    def _level_cell_index(self, level_id: int) -> GridPos:
        c = int(self.config.observation.local_window_size) // 2
        return int(c), int(c)

    def _exit_scores_from_dtm(self, dtm_values: np.ndarray) -> Tuple[float, float, float, float]:
        mode = self._dtm_output_mode()
        vals = np.maximum(dtm_values.astype(np.float32), 0.0)
        if mode in {"six", "extent6", "port6"}:
            u_r, u_d, u_l, r_d, r_l, d_l = [float(v) for v in vals[:6]]
            # six/port6 channel order:
            # [U<->R, U<->D, U<->L, R<->D, R<->L, D<->L]
            up = max(u_r, u_d, u_l)
            right = max(u_r, r_d, r_l)
            down = max(u_d, r_d, d_l)
            left = max(u_l, r_l, d_l)
            return up, right, down, left
        if mode == "axis2":
            lr, ud = [float(v) for v in vals[:2]]
            # Axis-only exits: no diagonal contribution.
            up = ud
            right = lr
            down = ud
            left = lr
            return up, right, down, left
        if mode == "axis2km":
            lr_pass, ud_pass = [float(v) for v in vals[:2]]
            # Axis-only exits from passable channels; known-mask channels are used
            # only for validity gating.
            up = ud_pass
            right = lr_pass
            down = ud_pass
            left = lr_pass
            return up, right, down, left
        if mode == "four":
            lr, ud, d1, d2 = [float(v) for v in vals[:4]]
            up = max(ud, d1, d2)
            right = max(lr, d1, d2)
            down = max(ud, d1, d2)
            left = max(lr, d1, d2)
            return up, right, down, left
        if mode == "port12":
            (
                u_r,
                u_d,
                u_l,
                r_u,
                r_d,
                r_l,
                d_u,
                d_r,
                d_l,
                l_u,
                l_r,
                l_d,
            ) = [float(v) for v in vals[:12]]
            up = max(r_u, d_u, l_u)
            right = max(u_r, d_r, l_r)
            down = max(u_d, r_d, l_d)
            left = max(u_l, r_l, d_l)
            return up, right, down, left
        raise ValueError(f"Unsupported dtm_output_mode: {self.config.observation.dtm_output_mode}")

    def _dtm_valid_flag(self, dtm_values: np.ndarray) -> float:
        mode = self._dtm_output_mode()
        vals = dtm_values.astype(np.float32)
        if mode == "axis2km":
            # [lr_pass, ud_pass, lr_known, ud_known]
            known = vals[2:4]
            return float(np.all(known >= 0.5))
        return float(np.all(vals >= 0.0))

    def _build_boundary_exit_features(
        self,
        levels: Dict[int, np.ndarray],
        *,
        maps_builder: Optional[MultiScaleCPPObservationBuilder] = None,
    ) -> np.ndarray:
        maps_builder = maps_builder or self.maps_builder
        thr = float(self.config.boundary_exit_threshold)
        if thr < 0.0:
            thr = 0.0
        if thr > 1.0:
            thr = 1.0

        dtm_ch = self._dtm_channel_count()
        out: List[float] = []
        for lv in range(maps_builder.num_levels):
            if not maps_builder._include_dtm_for_level(lv):
                out.extend([0.0, 0.0, 0.0, 0.0])
                if self.config.boundary_exit_include_valid:
                    out.append(0.0)
                continue
            level_arr = np.asarray(levels[lv], dtype=np.float32)
            if level_arr.ndim != 3:
                raise ValueError(f"Level tensor must be [C,H,W], got shape={level_arr.shape}")
            ri, ci = self._level_cell_index(lv)
            dtm = level_arr[3 : 3 + dtm_ch, ri, ci]
            valid = self._dtm_valid_flag(dtm)
            up_s, right_s, down_s, left_s = self._exit_scores_from_dtm(dtm)
            if valid > 0.5:
                out.extend(
                    [
                        1.0 if up_s > thr else 0.0,
                        1.0 if right_s > thr else 0.0,
                        1.0 if down_s > thr else 0.0,
                        1.0 if left_s > thr else 0.0,
                    ]
                )
            else:
                out.extend([0.0, 0.0, 0.0, 0.0])
            if self.config.boundary_exit_include_valid:
                out.append(valid)
        return np.asarray(out, dtype=np.float32)

    def get_action_mask(self) -> np.ndarray:
        """
        Validity mask over 4-connected actions from current state.
        True means the action is currently allowed by known-map safety rules.
        """
        mask = np.zeros(self.action_dim, dtype=np.bool_)
        cr, cc = self.current_pos
        for action, (dr, dc) in ACTION_TO_DELTA.items():
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                continue
            if self.known_map[nr, nc] == 1:
                continue
            mask[action] = True
        return mask

    def _fallback_action_from_mask(self, mask: np.ndarray) -> Optional[int]:
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return None

        # Prefer moving to unexplored known-free, then unknown, then explored known-free.
        cr, cc = self.current_pos
        best_key = None
        best_action = int(valid[0])
        for a in valid:
            action = int(a)
            dr, dc = ACTION_TO_DELTA[action]
            nr, nc = cr + dr, cc + dc
            cell = int(self.known_map[nr, nc])
            if cell == 0 and (not self.explored[nr, nc]):
                tier = 0
            elif cell == -1:
                tier = 1
            else:
                tier = 2
            key = (tier, action)
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action

    def _heuristic_should_activate(self) -> bool:
        if not bool(self.config.heuristic_assist_enabled):
            return False
        self._heuristic_trigger_reason = "none"
        if str(self.config.heuristic_trigger_mode).strip().lower() == "v2":
            window = max(1, int(self.config.heuristic_period2_recent_new_window))
            recent_new = int(sum(list(self._heuristic_v2_new_events)[-window:]))
            reasons: List[str] = []
            if int(self._heuristic_no_progress_streak) >= int(self.config.heuristic_no_progress_threshold):
                reasons.append("no_new_timeout")
            if (
                bool(self.config.heuristic_period2_enabled)
                and
                len(self._heuristic_v2_new_events) >= window
                and recent_new == 0
                and int(self._heuristic_period2_streak) >= int(self.config.heuristic_period2_streak)
            ):
                reasons.append("period2")
            if (
                bool(self.config.heuristic_same_cell_enabled)
                and
                len(self._heuristic_v2_new_events) >= 32
                and int(sum(list(self._heuristic_v2_new_events)[-32:])) == 0
                and int(self._heuristic_same_cell_stall) >= int(self.config.heuristic_same_cell_stall)
            ):
                reasons.append("same_cell")
            radius = int(self.config.heuristic_late_local_radius)
            if radius > 0 and self._coverage_ratio() >= float(self.config.heuristic_min_coverage):
                rr, cc = self.current_pos
                r0, r1 = max(0, rr - radius), min(self.rows, rr + radius + 1)
                c0, c1 = max(0, cc - radius), min(self.cols, cc + radius + 1)
                local_uncovered = int(np.count_nonzero(
                    (self.known_map[r0:r1, c0:c1] == 0) & (~self.explored[r0:r1, c0:c1])
                ))
                if local_uncovered <= int(self.config.heuristic_late_local_max_uncovered):
                    reasons.append("late_local_exhausted")
            self._heuristic_trigger_reason = "+".join(reasons) if reasons else "none"
            return bool(reasons)
        threshold = int(self.config.heuristic_no_progress_threshold)
        if threshold <= 0:
            return False
        if int(self._heuristic_no_progress_streak) < threshold:
            return False
        min_cov = float(self.config.heuristic_min_coverage)
        if min_cov > 0.0 and self._coverage_ratio() < min_cov:
            return False
        return True

    def _action_leads_to_known_uncovered_free(self, action: int) -> bool:
        dr, dc = ACTION_TO_DELTA[int(action)]
        cr, cc = self.current_pos
        nr, nc = cr + dr, cc + dc
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            return False
        return bool(self.known_map[nr, nc] == 0 and not self.explored[nr, nc])

    def _action_leads_to_known_free(self, action: int) -> bool:
        dr, dc = ACTION_TO_DELTA[int(action)]
        cr, cc = self.current_pos
        nr, nc = cr + dr, cc + dc
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            return False
        return bool(self.known_map[nr, nc] == 0)

    def _frontier_mask(self) -> np.ndarray:
        unknown = self.known_map == -1
        adjacent_unknown = np.zeros_like(unknown, dtype=bool)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        return (self.known_map == 0) & adjacent_unknown

    def _unknown_neighbor_action(self) -> Optional[int]:
        cr, cc = self.current_pos
        best_key: Optional[Tuple[int, int, int]] = None
        best_action: Optional[int] = None
        for action, (dr, dc) in ACTION_TO_DELTA.items():
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                continue
            if self.known_map[nr, nc] != -1:
                continue
            unknown_gain = 0
            for _, (ndr, ndc) in ACTION_TO_DELTA.items():
                rr, cc2 = nr + ndr, nc + ndc
                if 0 <= rr < self.rows and 0 <= cc2 < self.cols and self.known_map[rr, cc2] == -1:
                    unknown_gain += 1
            turn_cost = 0 if self.prev_action is None else int(action != int(self.prev_action))
            key = (int(unknown_gain), -turn_cost, -int(action))
            if best_key is None or key > best_key:
                best_key = key
                best_action = int(action)
        return best_action

    def _bfs_action_to_target_mask(
        self,
        target_mask: np.ndarray,
        *,
        target_kind: str,
    ) -> Tuple[Optional[int], Dict[str, object]]:
        target_count = int(np.count_nonzero(target_mask))
        stats: Dict[str, object] = {
            "path_found": 0.0,
            "target_count": float(target_count),
            "path_length": 0.0,
            "astar_expansions": 0.0,
            "target_row": -1.0,
            "target_col": -1.0,
            "target_kind": str(target_kind),
            "search_method": "bfs",
        }
        if target_count <= 0:
            return None, stats

        start = self.current_pos
        sr, sc = start
        if bool(target_mask[sr, sc]) and str(target_kind) == "frontier":
            action = self._unknown_neighbor_action()
            if action is None:
                return None, stats
            stats["path_found"] = 1.0
            stats["path_length"] = 1.0
            stats["target_row"] = float(sr)
            stats["target_col"] = float(sc)
            return int(action), stats

        passable = self.known_map == 0
        if not bool(passable[sr, sc]):
            return None, stats

        max_expansions = int(self.config.heuristic_max_astar_expansions)
        if max_expansions <= 0:
            max_expansions = int(self.rows * self.cols)

        turn_cost = max(0.0, float(self.config.heuristic_turn_cost))
        revisit_cost = max(0.0, float(self.config.heuristic_revisit_cost))
        if turn_cost > 0.0 or revisit_cost > 0.0:
            path, expansions = self._turn_aware_path_to_target(
                target_mask,
                passable=passable,
                max_expansions=max_expansions,
                turn_cost=turn_cost,
                revisit_cost=revisit_cost,
            )
            stats["search_method"] = "turn_aware_astar"
        else:
            path, expansions = self._shortest_low_turn_path_to_target(
                target_mask,
                passable=passable,
                max_expansions=max_expansions,
            )
            stats["search_method"] = "bfs_shortest_low_turn"

        stats["astar_expansions"] = float(expansions)
        if len(path) <= 1:
            return None, stats

        found = path[0]

        path = list(path)
        path.reverse()
        if len(path) <= 1:
            return None, stats

        actions: List[int] = []
        for src, dst in zip(path[:-1], path[1:]):
            dr, dc = int(dst[0] - src[0]), int(dst[1] - src[1])
            action = None
            for a, delta in ACTION_TO_DELTA.items():
                if delta == (dr, dc):
                    action = int(a)
                    break
            if action is None:
                actions = []
                break
            actions.append(int(action))
        if not actions:
            return None, stats
        action = int(actions[0])
        self._heuristic_cached_actions = deque(int(a) for a in actions[1:])
        self._heuristic_cached_target = found

        stats["path_found"] = 1.0
        stats["path_length"] = float(len(path) - 1)
        stats["target_row"] = float(found[0])
        stats["target_col"] = float(found[1])
        return int(action), stats

    def _bfs_path_to_target(
        self,
        target_mask: np.ndarray,
        *,
        passable: np.ndarray,
        max_expansions: int,
    ) -> Tuple[List[GridPos], int]:
        start = self.current_pos
        sr, sc = start
        seen = np.zeros((self.rows, self.cols), dtype=bool)
        seen[sr, sc] = True
        parent: Dict[GridPos, GridPos] = {}
        queue: Deque[GridPos] = deque([(int(sr), int(sc))])

        found: Optional[GridPos] = None
        expansions = 0
        while queue and expansions < int(max_expansions):
            r, c = queue.popleft()
            expansions += 1
            if (r, c) != (sr, sc) and bool(target_mask[r, c]):
                found = (int(r), int(c))
                break

            for _, (dr, dc) in ACTION_TO_DELTA.items():
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                if bool(seen[nr, nc]) or (not bool(passable[nr, nc])):
                    continue
                seen[nr, nc] = True
                parent[(int(nr), int(nc))] = (int(r), int(c))
                queue.append((int(nr), int(nc)))

        if found is None:
            return [], expansions

        path: List[GridPos] = [found]
        cur = found
        while cur != start:
            cur = parent.get(cur, start)
            path.append(cur)
            if cur == start:
                break
        return path, expansions

    def _shortest_low_turn_path_to_target(
        self,
        target_mask: np.ndarray,
        *,
        passable: np.ndarray,
        max_expansions: int,
    ) -> Tuple[List[GridPos], int]:
        """Find a shortest path, breaking equal-length ties by turn count.

        The previous plain BFS kept only one state per cell, so the first
        path discovered was selected even when another path of the same
        length had fewer direction changes.  The previous weighted search
        could instead choose a longer path to avoid turns.  This search uses
        the lexicographic objective (steps, turns), preserving shortest-path
        behavior while making tie-breaking turn-aware.
        """
        start = self.current_pos
        sr, sc = start
        start_prev = -1 if self.prev_action is None else int(self.prev_action)
        start_state = (int(sr), int(sc), int(start_prev))

        # A cell can need a separate state for each incoming action.  The
        # expansion budget therefore scales with the number of action states.
        expansion_limit = max(int(max_expansions), int(self.rows * self.cols * 4))
        best_key: Dict[Tuple[int, int, int], Tuple[int, int]] = {
            start_state: (0, 0)
        }
        parent: Dict[
            Tuple[int, int, int], Tuple[int, int, int]
        ] = {}
        heap: List[Tuple[int, int, int, int, int, int]] = [
            (0, 0, 0, int(sr), int(sc), int(start_prev))
        ]
        expansions = 0
        counter = 0
        found_state: Optional[Tuple[int, int, int]] = None

        while heap and expansions < expansion_limit:
            steps, turns, _, r, c, prev_action = heapq.heappop(heap)
            state = (int(r), int(c), int(prev_action))
            if (int(steps), int(turns)) != best_key.get(state):
                continue
            expansions += 1

            if (r, c) != (sr, sc) and bool(target_mask[r, c]):
                # Heap ordering guarantees the first target is optimal under
                # (steps, turns), including ties across different headings.
                found_state = state
                break

            for action, (dr, dc) in ACTION_TO_DELTA.items():
                nr, nc = int(r + dr), int(c + dc)
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                if not bool(passable[nr, nc]):
                    continue

                next_state = (nr, nc, int(action))
                next_key = (
                    int(steps) + 1,
                    int(turns)
                    + int(int(prev_action) >= 0 and int(action) != int(prev_action)),
                )
                if next_key >= best_key.get(next_state, (float("inf"), float("inf"))):
                    continue
                best_key[next_state] = next_key
                parent[next_state] = state
                counter += 1
                heapq.heappush(
                    heap,
                    (
                        next_key[0],
                        next_key[1],
                        counter,
                        nr,
                        nc,
                        int(action),
                    ),
                )

        if found_state is None:
            return [], expansions

        rev_path: List[GridPos] = [(found_state[0], found_state[1])]
        cur = found_state
        while cur != start_state:
            cur = parent.get(cur, start_state)
            rev_path.append((int(cur[0]), int(cur[1])))
            if cur == start_state:
                break
        return rev_path, expansions

    def _turn_aware_path_to_target(
        self,
        target_mask: np.ndarray,
        *,
        passable: np.ndarray,
        max_expansions: int,
        turn_cost: float,
        revisit_cost: float,
    ) -> Tuple[List[GridPos], int]:
        start = self.current_pos
        sr, sc = start
        start_prev = -1 if self.prev_action is None else int(self.prev_action)
        start_state = (int(sr), int(sc), int(start_prev))
        best_cost = {start_state: 0.0}
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        heap: List[Tuple[float, int, int, int, int]] = [(0.0, 0, int(sr), int(sc), int(start_prev))]
        expansions = 0
        counter = 0
        found_state: Optional[Tuple[int, int, int]] = None

        while heap and expansions < int(max_expansions):
            cost, _, r, c, prev_action = heapq.heappop(heap)
            state = (int(r), int(c), int(prev_action))
            if cost > best_cost.get(state, float("inf")):
                continue
            expansions += 1
            if (r, c) != (sr, sc) and bool(target_mask[r, c]):
                found_state = state
                break

            for action, (dr, dc) in ACTION_TO_DELTA.items():
                nr, nc = int(r + dr), int(c + dc)
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                if not bool(passable[nr, nc]):
                    continue
                extra = 1.0
                if int(prev_action) >= 0 and int(action) != int(prev_action):
                    extra += float(turn_cost)
                if bool(self.explored[nr, nc]) and not bool(target_mask[nr, nc]):
                    extra += float(revisit_cost)
                next_state = (nr, nc, int(action))
                next_cost = float(cost) + float(extra)
                if next_cost >= best_cost.get(next_state, float("inf")):
                    continue
                best_cost[next_state] = next_cost
                parent[next_state] = state
                counter += 1
                heapq.heappush(heap, (next_cost, counter, nr, nc, int(action)))

        if found_state is None:
            return [], expansions

        rev_path: List[GridPos] = [(found_state[0], found_state[1])]
        cur = found_state
        while cur != start_state:
            cur = parent.get(cur, start_state)
            rev_path.append((int(cur[0]), int(cur[1])))
            if cur == start_state:
                break
        return rev_path, expansions

    def _nearest_known_uncovered_astar_action(self) -> Tuple[Optional[int], Dict[str, object]]:
        while self._heuristic_cached_actions:
            cached_action = int(self._heuristic_cached_actions.popleft())
            if self._action_leads_to_known_free(cached_action):
                target_count = int(np.count_nonzero((self.known_map == 0) & (~self.explored)))
                return cached_action, {
                    "path_found": 1.0,
                    "target_count": float(target_count),
                    "path_length": float(len(self._heuristic_cached_actions) + 1),
                    "astar_expansions": 0.0,
                    "target_row": float(self._heuristic_cached_target[0]) if self._heuristic_cached_target else -1.0,
                    "target_col": float(self._heuristic_cached_target[1]) if self._heuristic_cached_target else -1.0,
                    "target_kind": "cached",
                }
        self._heuristic_cached_target = None

        known_uncovered = (self.known_map == 0) & (~self.explored)
        action, stats = self._bfs_action_to_target_mask(
            known_uncovered,
            target_kind="known_uncovered",
        )
        if action is not None:
            return action, stats

        frontier = self._frontier_mask()
        frontier_action, frontier_stats = self._bfs_action_to_target_mask(
            frontier,
            target_kind="frontier",
        )
        if frontier_action is not None:
            return frontier_action, frontier_stats

        return None, stats if int(float(stats.get("target_count", 0.0))) > 0 else frontier_stats

    def reset(self, *, start_pos: Optional[GridPos] = None) -> Dict[str, object]:
        self.current_pos = self._resolve_start(start_pos) if start_pos is not None else self._default_start
        self.prev_pos = None
        self.prev_action = None
        self.steps = 0
        self.done = False
        self.last_collision = False
        self.overlap_streak = 0
        self.path = [self.current_pos]
        self.recent_new_coverage.clear()
        self.recent_actions.clear()
        self._milestone_hit_90 = False
        self._milestone_hit_99 = False
        self._heuristic_no_progress_streak = 0
        self._heuristic_cached_actions.clear()
        self._heuristic_cached_target = None
        self._heuristic_v2_new_events.clear()
        self._heuristic_period2_streak = 0
        self._heuristic_same_cell_stall = 0
        self._heuristic_trigger_reason = "none"
        self._heuristic_rescue_selected_previous_step = False

        self.known_map.fill(-1)
        self.explored.fill(False)
        self._reset_semi_global_tv_state()
        preserve_static_map_cache = bool(self.config.preserve_static_map_observation_cache_on_reset)
        self.maps_builder.reset_incremental_state(preserve_static_map=preserve_static_map_cache)
        if self.hybrid_maps_builder is not None:
            self.hybrid_maps_builder.reset_incremental_state(
                preserve_static_map=preserve_static_map_cache,
            )
        if self._boundary_maps_builder is not None:
            self._boundary_maps_builder.reset_incremental_state(
                preserve_static_map=preserve_static_map_cache,
            )
        self._sense_at(self.current_pos)
        self._mark_explored(self.current_pos, step_index=0)
        self._push_semi_global_tv_window(self.current_pos)

        self.last_reward = CPPRewardBreakdown(
            area=0.0,
            tv_local=0.0,
            tv_global=0.0,
            collision=0.0,
            constant=0.0,
            total=0.0,
        )
        return self._build_observation()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")
        requested_action = int(action)
        if requested_action not in ACTION_TO_DELTA:
            raise ValueError(f"Invalid action {action}. Expected one of {sorted(ACTION_TO_DELTA)}")

        action_mask = self.get_action_mask()
        executed_action = requested_action
        action_overridden = False
        masked_invalid_action = bool(
            self.config.use_action_mask and (not bool(action_mask[requested_action]))
        )
        invalid_action_penalized = False
        if masked_invalid_action and self._action_mask_mode == "shield":
            fallback = self._fallback_action_from_mask(action_mask)
            if fallback is not None:
                executed_action = int(fallback)
                action_overridden = True
        elif masked_invalid_action and self._action_mask_mode == "penalty":
            invalid_action_penalized = True

        heuristic_streak_before = int(self._heuristic_no_progress_streak)
        heuristic_active = self._heuristic_should_activate()
        heuristic_selected = False
        heuristic_overridden = False
        heuristic_stats: Dict[str, object] = {
            "path_found": 0.0,
            "target_count": 0.0,
            "path_length": 0.0,
            "astar_expansions": 0.0,
            "target_row": -1.0,
            "target_col": -1.0,
            "target_kind": "none",
        }
        if heuristic_active:
            heuristic_action, heuristic_stats = self._nearest_known_uncovered_astar_action()
            if heuristic_action is not None:
                previous_executed_action = int(executed_action)
                executed_action = int(heuristic_action)
                heuristic_selected = True
                heuristic_overridden = int(executed_action) != previous_executed_action
                action_overridden = bool(action_overridden or heuristic_overridden)
                if heuristic_overridden:
                    invalid_action_penalized = False

        cr, cc = self.current_pos
        prev_action = self.prev_action
        forced_turn = False
        if prev_action is not None:
            pdr, pdc = ACTION_TO_DELTA[int(prev_action)]
            pr, pc = cr + pdr, cc + pdc
            can_keep_heading = (
                0 <= pr < self.rows
                and 0 <= pc < self.cols
                and self.true_map[pr, pc] == 0
            )
            forced_turn = not can_keep_heading

        dr, dc = ACTION_TO_DELTA[executed_action]
        nr, nc = cr + dr, cc + dc

        collided = False
        prev = self.current_pos
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.true_map[nr, nc] == 0:
            self.current_pos = (int(nr), int(nc))
        else:
            collided = True

        self.last_collision = collided
        self.prev_pos = prev
        self._sense_at(self.current_pos)

        was_explored = bool(self.explored[self.current_pos]) if not collided else False
        prev_coverage_cells = int(np.count_nonzero(self.explored & self.free_mask))
        # TV_i compares the same local window before/after this transition.
        # The window is centered at the post-action robot position, matching
        # the local observation used to choose the next action.
        tv_center = self.current_pos
        coverage_before = self._coverage_map_float()
        obstacle_snapshot = self._known_obstacle_map_float()
        curr_step_index = int(self.steps) + 1
        newly_visited = self._mark_explored(self.current_pos, step_index=curr_step_index)
        coverage_after = self._coverage_map_float()
        local_tv_mode = str(self.config.local_tv_mode).strip().lower()
        if local_tv_mode in {"edge", "incremental_edge", "cell_edge"}:
            local_tv_old = 0.0
            local_tv_new = self._compute_incremental_edge_tv_delta(
                coverage_before,
                coverage_after,
                obstacle_snapshot,
                tv_center,
            )
        elif local_tv_mode == "patch":
            local_tv_old = self._compute_local_tv_from_maps(
                coverage_before,
                obstacle_snapshot,
                tv_center,
            )
            local_tv_new = self._compute_local_tv_from_maps(
                coverage_after,
                obstacle_snapshot,
                tv_center,
            )
        else:
            raise ValueError("local_tv_mode must be one of {'patch', 'edge'}")
        global_tv = self._compute_global_tv()
        self._push_semi_global_tv_window(self.current_pos)
        semi_global_tv_info = self._compute_semi_global_tv_reward(
            curr_step_index=curr_step_index,
            collided=collided,
        )
        reward_sg_tv = float(semi_global_tv_info.get("reward_sg_tv", 0.0))
        coverage_cells = int(np.count_nonzero(self.explored & self.free_mask))
        prev_coverage_ratio = float(prev_coverage_cells) / float(max(1, self.free_total))
        curr_coverage_ratio = float(coverage_cells) / float(max(1, self.free_total))

        rew_in = CPPRewardInput(
            newly_visited=float(newly_visited),
            local_tv_old=local_tv_old,
            local_tv_new=local_tv_new,
            global_tv=global_tv,
            coverage_pixels=float(coverage_cells),
            collided=collided,
            collision_reward_override=(
                float(self.config.invalid_action_penalty)
                if collided and invalid_action_penalized
                else None
            ),
        )
        base_reward = compute_cpp_reward(rew_in, self.config.reward)
        reward_turn = 0.0
        if (not collided) and prev_action is not None and executed_action != int(prev_action):
            reward_turn = float(self.config.reward.turn_change_penalty)

        overlap_event = (not collided) and was_explored
        if overlap_event:
            self.overlap_streak += 1
        else:
            self.overlap_streak = 0

        reward_overlap = 0.0
        if overlap_event:
            reward_overlap = float(self.config.reward.revisit_penalty)
            if bool(self.config.reward.overlap_streak_enabled):
                grace = max(0, int(self.config.reward.overlap_streak_grace))
                increment = max(0.0, float(self.config.reward.overlap_streak_increment))
                base_abs = abs(float(self.config.reward.revisit_penalty))
                max_abs = max(
                    base_abs,
                    max(0.0, float(self.config.reward.overlap_streak_max_abs)),
                )
                effective_streak = max(0, int(self.overlap_streak) - grace)
                penalty_abs = min(base_abs + increment * float(effective_streak), max_abs)
                reward_overlap = -float(penalty_abs)

        reward_milestone, milestone_hit_90_now, milestone_hit_99_now = self._compute_milestone_reward(
            prev_coverage_ratio=prev_coverage_ratio,
            curr_coverage_ratio=curr_coverage_ratio,
            collided=collided,
        )

        total_reward = float(
            base_reward.total
            + reward_turn
            + reward_overlap
            + reward_milestone
            + reward_sg_tv
        )
        self.last_reward = CPPRewardBreakdown(
            area=float(base_reward.area),
            tv_local=float(base_reward.tv_local),
            tv_global=float(base_reward.tv_global),
            collision=float(base_reward.collision),
            constant=float(base_reward.constant),
            total=total_reward,
        )
        self.recent_new_coverage.append(float(newly_visited))
        if (not collided) and float(newly_visited) > 0.0:
            self._heuristic_no_progress_streak = 0
            self._heuristic_cached_actions.clear()
            self._heuristic_cached_target = None
        else:
            self._heuristic_no_progress_streak += 1

        self.steps += 1
        self.path.append(self.current_pos)
        self._heuristic_v2_new_events.append(int((not collided) and float(newly_visited) > 0.0))
        self._heuristic_rescue_selected_previous_step = bool(heuristic_selected)
        if len(self.path) >= 3 and self.path[-1] == self.path[-3] and self.path[-1] != self.path[-2]:
            self._heuristic_period2_streak += 1
        else:
            self._heuristic_period2_streak = 0
        if len(self.path) >= 2 and self.path[-1] == self.path[-2]:
            self._heuristic_same_cell_stall += 1
        else:
            self._heuristic_same_cell_stall = 0
        recorded_action = -1 if collided else int(executed_action)
        if not collided:
            self.prev_action = int(executed_action)
        self.recent_actions.append(int(recorded_action))

        # A cached route ends once the target is reached (new coverage clears
        # the remaining cache). Reset before constructing the observation that
        # is returned to the policy, not after it has already chosen an action.
        policy_handoff_reset = bool(
            heuristic_selected
            and not self._heuristic_cached_actions
            and bool(self.config.heuristic_reset_policy_state_after_rescue)
        )
        sgtv_reset_after_rescue = False
        if policy_handoff_reset:
            self._reset_policy_handoff_state_after_rescue()
            if bool(self.config.heuristic_reset_sgtv_after_rescue):
                self._reset_semi_global_tv_state()
                self._push_semi_global_tv_window(self.current_pos)
                sgtv_reset_after_rescue = True

        done_reason = ""
        truncated = False
        done = False

        if collided and self.config.collision_ends_episode:
            done = True
            done_reason = "collision"

        if (not done) and self.config.stop_on_full_coverage and coverage_cells >= self.free_total:
            done = True
            done_reason = "coverage_complete"

        if (not done) and (self.config.max_steps is not None) and self.steps >= int(self.config.max_steps):
            done = True
            truncated = True
            done_reason = "max_steps"

        self.done = done
        obs = self._build_observation()

        heuristic_target_kind = str(heuristic_stats.get("target_kind", "none"))
        info = {
            **self.last_reward.as_dict(),
            "collision": bool(collided),
            "action_requested": int(requested_action),
            "action_executed": int(executed_action),
            "action_overridden": bool(action_overridden),
            "invalid_action": bool(masked_invalid_action),
            "invalid_action_penalized": bool(collided and invalid_action_penalized),
            "invalid_action_penalty": (
                float(self.config.invalid_action_penalty)
                if collided and invalid_action_penalized
                else 0.0
            ),
            "heuristic_assist_enabled": bool(self.config.heuristic_assist_enabled),
            "heuristic_trigger_mode": str(self.config.heuristic_trigger_mode),
            "heuristic_trigger_reason": str(self._heuristic_trigger_reason),
            "heuristic_active": bool(heuristic_active),
            "heuristic_selected": bool(heuristic_selected),
            "heuristic_overridden": bool(heuristic_overridden),
            "heuristic_policy_handoff_reset": bool(policy_handoff_reset),
            "heuristic_sgtv_reset_after_rescue": bool(sgtv_reset_after_rescue),
            "heuristic_path_found": bool(float(heuristic_stats.get("path_found", 0.0)) > 0.5),
            "heuristic_no_progress_streak_before": int(heuristic_streak_before),
            "heuristic_no_progress_streak": int(self._heuristic_no_progress_streak),
            "heuristic_known_uncovered_count": int(heuristic_stats.get("target_count", 0.0)),
            "heuristic_path_length": float(heuristic_stats.get("path_length", 0.0)),
            "heuristic_astar_expansions": int(heuristic_stats.get("astar_expansions", 0.0)),
            "heuristic_target_row": int(heuristic_stats.get("target_row", -1.0)),
            "heuristic_target_col": int(heuristic_stats.get("target_col", -1.0)),
            "heuristic_target_kind": heuristic_target_kind,
            "heuristic_target_known_uncovered": bool(
                heuristic_selected and heuristic_target_kind == "known_uncovered"
            ),
            "heuristic_target_frontier": bool(heuristic_selected and heuristic_target_kind == "frontier"),
            "heuristic_target_cached": bool(heuristic_selected and heuristic_target_kind == "cached"),
            "forced_turn": bool(forced_turn),
            "action_mask_valid_count": int(np.count_nonzero(action_mask)),
            "action_mask": [int(v) for v in action_mask.astype(np.int8)],
            "recorded_action": int(recorded_action),
            "revisited_cell": bool((not collided) and was_explored),
            "overlap_streak": int(self.overlap_streak),
            "reward_turn": float(reward_turn),
            "reward_overlap": float(reward_overlap),
            "reward_milestone": float(reward_milestone),
            **semi_global_tv_info,
            "milestone_hit_90_now": bool(milestone_hit_90_now),
            "milestone_hit_99_now": bool(milestone_hit_99_now),
            "milestone_hit_90": bool(self._milestone_hit_90),
            "milestone_hit_99": bool(self._milestone_hit_99),
            "coverage_cells": int(coverage_cells),
            "free_cells": int(self.free_total),
            "coverage_ratio": float(self._coverage_ratio()),
            "steps": int(self.steps),
            "done_reason": done_reason,
            "truncated": bool(truncated),
            "position": self.current_pos,
        }
        return obs, float(self.last_reward.total), bool(done), info
