from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage as scipy_ndimage
except Exception:
    scipy_ndimage = None

from learning.observation import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
    RobotStateObservationBuilder,
    RobotStateObservationConfig,
)
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

_HOLE_LABEL_STRUCTURE = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class CPPDiscreteEnvConfig:
    sensor_range: int = 2
    max_steps: Optional[int] = 1500
    collision_ends_episode: bool = False
    stop_on_full_coverage: bool = True
    include_dtm: bool = False
    tv_mode: str = "sym-iso"
    local_tv_patch_radius: int = 3
    # Known-map safety mask (online-safe):
    # - invalid: out-of-bounds or known obstacle (1)
    # - valid: known free (0) or unknown (-1)
    use_action_mask: bool = False
    # Optional robot-state augmentation from multi-scale DTM at robot's current cell.
    # Per level: [can_exit_up, can_exit_right, can_exit_down, can_exit_left]
    # Optional validity flag can be appended per level.
    use_boundary_exit_features: bool = False
    boundary_exit_threshold: float = 0.0
    boundary_exit_include_valid: bool = True
    heuristic_loop_window: int = 20
    heuristic_no_progress_k: int = 20
    heuristic_force_loop_k: int = 30
    heuristic_unique_threshold: int = 4
    heuristic_force_only: bool = False
    heuristic_override: bool = False
    track_hole_metrics: bool = False
    profile_observation: bool = False
    profile_interval_steps: int = 200
    profile_name: str = ""

    observation: MultiScaleCPPObservationConfig = field(
        default_factory=MultiScaleCPPObservationConfig,
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

        self.true_map = grid
        self.rows, self.cols = grid.shape
        self.free_mask = self.true_map == 0
        self.free_total = int(np.count_nonzero(self.free_mask))
        self.covered_free_count = 0
        if self.free_total <= 0:
            raise ValueError("grid_map must have at least one free cell")

        self.maps_builder = MultiScaleCPPObservationBuilder(
            self.config.observation,
            include_dtm=self.config.include_dtm,
            profile_enabled=bool(self.config.profile_observation),
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
            hole=0.0,
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
        self.recent_positions: Deque[GridPos] = deque(
            maxlen=max(4, int(self.config.heuristic_loop_window)),
        )
        self.visit_counts = np.zeros_like(self.true_map, dtype=np.int32)
        self.no_progress_streak = 0
        self._hole_component_data: Optional[Dict[str, np.ndarray]] = None
        self._hole_stats_cache: Optional[Dict[str, float]] = None
        self._hole_stats_pos: Optional[GridPos] = None

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

    def _mark_explored(self, pos: GridPos) -> float:
        rr, cc = pos
        if self.true_map[rr, cc] != 0:
            return 0.0
        if self.explored[rr, cc]:
            return 0.0
        self.explored[rr, cc] = True
        self.covered_free_count += 1
        return 1.0

    def _coverage_ratio(self) -> float:
        return float(self.covered_free_count) / float(max(1, self.free_total))

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

        area_unit = max(0.0, float(cfg.newly_visited_reward_scale))
        bonus90 = (
            max(0.0, float(cfg.milestone_lambda_90))
            * max(0.0, 1.0 - thr90)
            * float(self.free_total)
            * area_unit
        )
        bonus99 = (
            max(0.0, float(cfg.milestone_lambda_99))
            * max(0.0, 1.0 - thr99)
            * float(self.free_total)
            * area_unit
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
        rr, cc = center
        r0 = max(0, rr - radius)
        r1 = min(self.rows, rr + radius + 1)
        c0 = max(0, cc - radius)
        c1 = min(self.cols, cc + radius + 1)
        return arr[r0:r1, c0:c1]

    def _compute_local_tv(self, center: GridPos) -> float:
        if float(self.config.reward.local_tv_reward_scale) == 0.0:
            return 0.0
        radius = max(0, int(self.config.local_tv_patch_radius))
        rr, cc = center
        r0 = max(0, rr - radius)
        r1 = min(self.rows, rr + radius + 1)
        c0 = max(0, cc - radius)
        c1 = min(self.cols, cc + radius + 1)
        known_local = self.known_map[r0:r1, c0:c1]
        explored_local = self.explored[r0:r1, c0:c1]
        cov_local = ((known_local == 0) & explored_local).astype(np.float32, copy=False)
        obs_local = (known_local == 1).astype(np.float32, copy=False)
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

    def _compute_global_tv(self) -> float:
        if float(self.config.reward.global_tv_reward_scale) == 0.0:
            return 0.0
        known_free = self.known_map == 0
        cov = (self.explored & known_free).astype(np.float32, copy=False)
        obs = (self.known_map == 1).astype(np.float32, copy=False)
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

    def _build_observation(self) -> Dict[str, object]:
        t_total = perf_counter() if self._profile_observation else 0.0
        t0 = perf_counter() if self._profile_observation else 0.0
        levels = self.maps_builder.build_levels(
            self.known_map,
            robot_pos=self.current_pos,
            explored=self.explored,
        )
        if self._profile_observation:
            self._profile_add("env_build_levels", perf_counter() - t0)
            self._merge_builder_profile(self.maps_builder.profile_snapshot(reset=True), prefix="maps")
        boundary_exit_features: Optional[np.ndarray] = None
        if self.config.use_boundary_exit_features:
            t0 = perf_counter() if self._profile_observation else 0.0
            if self.config.include_dtm:
                boundary_levels = levels
            else:
                if self._boundary_maps_builder is None:
                    raise RuntimeError("Boundary maps builder is not initialized")
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
            boundary_exit_features = self._build_boundary_exit_features(boundary_levels)
            if self._profile_observation:
                self._profile_add("boundary_exit_features", perf_counter() - t0)
        extra_features: List[float] = []
        if boundary_exit_features is not None:
            extra_features.extend(boundary_exit_features.astype(np.float32).tolist())
        if bool(self.config.robot_state.include_heuristic_signals):
            extra_features.extend(self._heuristic_signal_vector().tolist())
        if bool(self.config.robot_state.include_hole_signals):
            extra_features.extend(self._hole_signal_vector().tolist())
        t0 = perf_counter() if self._profile_observation else 0.0
        robot_state = self.robot_state_builder.build(
            occupancy=self.known_map,
            explored=self.explored,
            robot_pos=self.current_pos,
            prev_pos=self.prev_pos,
            recent_actions=list(self.recent_actions),
            recent_new_coverage=list(self.recent_new_coverage),
            extra_features=extra_features,
        )
        if self._profile_observation:
            self._profile_add("robot_state", perf_counter() - t0)
            self._profile_add("observation_total", perf_counter() - t_total)
            self._obs_profile_calls += 1
            self._maybe_report_observation_profile()
        return {
            "levels": levels,
            "robot_state": robot_state.astype(np.float32),
        }

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

    def _dtm_channel_count(self) -> int:
        mode = str(self.config.observation.dtm_output_mode).strip().lower()
        if mode in {"six", "extent6"}:
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
        local_count = len(self.config.observation.local_blocks)
        if level_id < local_count:
            c = int(self.config.observation.local_window_size) // 2
            return int(c), int(c)
        gsize = int(self.config.observation.global_window_size)
        rr, cc = self.current_pos
        gr = min(gsize - 1, max(0, int((float(rr) * float(gsize)) / float(max(1, self.rows)))))
        gc = min(gsize - 1, max(0, int((float(cc) * float(gsize)) / float(max(1, self.cols)))))
        return int(gr), int(gc)

    def _exit_scores_from_dtm(self, dtm_values: np.ndarray) -> Tuple[float, float, float, float]:
        mode = str(self.config.observation.dtm_output_mode).strip().lower()
        vals = np.maximum(dtm_values.astype(np.float32), 0.0)
        if mode in {"six", "extent6"}:
            lr, ud, nw_se, se_nw, ne_sw, sw_ne = [float(v) for v in vals[:6]]
            # Exit side can be reached if any transition path points to that side.
            up = max(ud, se_nw, sw_ne)
            right = max(lr, nw_se, sw_ne)
            down = max(ud, nw_se, ne_sw)
            left = max(lr, se_nw, ne_sw)
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
        mode = str(self.config.observation.dtm_output_mode).strip().lower()
        vals = dtm_values.astype(np.float32)
        if mode == "axis2km":
            # [lr_pass, ud_pass, lr_known, ud_known]
            known = vals[2:4]
            return float(np.all(known >= 0.5))
        return float(np.all(vals >= 0.0))

    def _build_boundary_exit_features(self, levels: Dict[int, np.ndarray]) -> np.ndarray:
        thr = float(self.config.boundary_exit_threshold)
        if thr < 0.0:
            thr = 0.0
        if thr > 1.0:
            thr = 1.0

        dtm_ch = self._dtm_channel_count()
        out: List[float] = []
        for lv in range(self.maps_builder.num_levels):
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

    def _recent_unique_positions(self) -> int:
        if len(self.recent_positions) == 0:
            return 0
        return int(len(set(self.recent_positions)))

    def _cycle2_detected(self) -> bool:
        if len(self.recent_positions) < 4:
            return False
        p = list(self.recent_positions)
        return bool(p[-1] == p[-3] and p[-2] == p[-4])

    def _loop_status(self) -> Dict[str, float]:
        window = max(4, int(self.config.heuristic_loop_window))
        trigger = max(1, int(self.config.heuristic_no_progress_k))
        force_trigger = max(trigger, int(self.config.heuristic_force_loop_k))
        unique_threshold = max(1, int(self.config.heuristic_unique_threshold))
        unique_recent = self._recent_unique_positions()
        cycle2 = self._cycle2_detected()
        low_support = unique_recent <= unique_threshold
        force_loop = self.no_progress_streak >= force_trigger
        if bool(self.config.heuristic_force_only):
            loop_detected = bool(force_loop)
        else:
            loop_detected = bool(force_loop or cycle2 or (self.no_progress_streak >= trigger and low_support))
        no_progress_norm = float(
            np.clip(float(self.no_progress_streak) / float(max(1, trigger)), 0.0, 1.0)
        )
        recent_unique_fraction = float(
            np.clip(float(unique_recent) / float(max(1, min(window, len(self.recent_positions)))), 0.0, 1.0)
        )
        return {
            "no_progress_streak": float(self.no_progress_streak),
            "recent_unique_positions": float(unique_recent),
            "cycle2_detected": float(cycle2),
            "force_loop_detected": float(force_loop),
            "loop_detected": float(loop_detected),
            "no_progress_norm": no_progress_norm,
            "recent_unique_fraction": recent_unique_fraction,
        }

    def _zero_hole_stats(self) -> Dict[str, float]:
        return {
            "coverage_hole_count": 0.0,
            "coverage_hole_known_mass": 0.0,
            "coverage_hole_open_mass": 0.0,
        }

    def _hole_metrics_enabled(self) -> bool:
        cfg = getattr(self, "config", None)
        if cfg is None:
            return True
        return bool(
            bool(cfg.robot_state.include_hole_signals)
            or float(cfg.reward.coverage_hole_penalty_scale) > 0.0
            or bool(getattr(cfg, "track_hole_metrics", False))
        )

    def _hole_risk_enabled(self) -> bool:
        cfg = getattr(self, "config", None)
        if cfg is None:
            return True
        return bool(
            bool(cfg.robot_state.include_hole_signals)
            or float(cfg.reward.coverage_hole_penalty_scale) > 0.0
        )

    def _coverage_hole_component_data(self) -> Dict[str, np.ndarray]:
        """
        Build connected-component metadata for the current known-map open space.

        blocker := covered free cells U known obstacles U boundary
        open    := unknown cells U known-free uncovered cells
        """
        known_free = self.known_map == 0
        uncovered_known = known_free & (~self.explored)
        unknown = self.known_map == -1
        open_base = np.logical_or(unknown, uncovered_known)

        if scipy_ndimage is not None:
            labels, num_labels = scipy_ndimage.label(open_base, structure=_HOLE_LABEL_STRUCTURE)
            if num_labels <= 0:
                comp_id = np.full((self.rows, self.cols), -1, dtype=np.int32)
                return {
                    "open_base": open_base,
                    "comp_id": comp_id,
                    "comp_known_mass": np.zeros((0,), dtype=np.int32),
                    "comp_open_mass": np.zeros((0,), dtype=np.int32),
                    "comp_touches_boundary": np.zeros((0,), dtype=bool),
                }
            labels = labels.astype(np.int32, copy=False)
            comp_id = np.where(labels > 0, labels - 1, -1).astype(np.int32, copy=False)
            open_counts = np.bincount(labels.ravel(), minlength=num_labels + 1)
            known_counts = np.bincount(labels[uncovered_known], minlength=num_labels + 1)
            boundary_labels = np.unique(
                np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
            )
            touches_boundary = np.zeros((num_labels + 1,), dtype=bool)
            touches_boundary[boundary_labels] = True
            return {
                "open_base": open_base,
                "comp_id": comp_id,
                "comp_known_mass": known_counts[1:].astype(np.int32, copy=False),
                "comp_open_mass": open_counts[1:].astype(np.int32, copy=False),
                "comp_touches_boundary": touches_boundary[1:].astype(bool, copy=False),
            }

        comp_id = np.full((self.rows, self.cols), -1, dtype=np.int32)
        comp_known_mass: List[int] = []
        comp_open_mass: List[int] = []
        comp_touches_boundary: List[bool] = []
        q: Deque[GridPos] = deque()
        next_id = 0

        for sr, sc in np.argwhere(open_base):
            sr_i = int(sr)
            sc_i = int(sc)
            if comp_id[sr_i, sc_i] >= 0:
                continue
            comp_id[sr_i, sc_i] = next_id
            q.append((sr_i, sc_i))
            known_mass = 0
            open_mass = 0
            touches_boundary = bool(
                sr_i == 0 or sr_i == (self.rows - 1) or sc_i == 0 or sc_i == (self.cols - 1)
            )
            while q:
                cr, cc = q.popleft()
                open_mass += 1
                if uncovered_known[cr, cc]:
                    known_mass += 1
                for dr, dc in ACTION_TO_DELTA.values():
                    nr, nc = cr + dr, cc + dc
                    if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                        continue
                    if (not open_base[nr, nc]) or comp_id[nr, nc] >= 0:
                        continue
                    comp_id[nr, nc] = next_id
                    if nr == 0 or nr == (self.rows - 1) or nc == 0 or nc == (self.cols - 1):
                        touches_boundary = True
                    q.append((nr, nc))
            comp_known_mass.append(known_mass)
            comp_open_mass.append(open_mass)
            comp_touches_boundary.append(touches_boundary)
            next_id += 1

        return {
            "open_base": open_base,
            "comp_id": comp_id,
            "comp_known_mass": np.asarray(comp_known_mass, dtype=np.int32),
            "comp_open_mass": np.asarray(comp_open_mass, dtype=np.int32),
            "comp_touches_boundary": np.asarray(comp_touches_boundary, dtype=bool),
        }

    def _refresh_hole_cache(self, robot_pos: Optional[GridPos] = None) -> Dict[str, float]:
        rp = self.current_pos if robot_pos is None else robot_pos
        if not self._hole_metrics_enabled():
            stats = self._zero_hole_stats()
            self._hole_component_data = None
            self._hole_stats_cache = stats
            self._hole_stats_pos = rp
            return stats
        data = self._coverage_hole_component_data()
        stats = self._coverage_hole_stats(rp, data)
        self._hole_component_data = data
        self._hole_stats_cache = stats
        self._hole_stats_pos = rp
        return stats

    def _current_hole_stats(self) -> Dict[str, float]:
        if not self._hole_metrics_enabled():
            return self._zero_hole_stats()
        if self._hole_stats_cache is None or self._hole_stats_pos != self.current_pos:
            return self._refresh_hole_cache(self.current_pos)
        return self._hole_stats_cache

    def _coverage_hole_stats(
        self,
        robot_pos: Optional[GridPos] = None,
        component_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Detect sealed coverage holes from the online known map only.

        blocker := covered free cells U known obstacles U boundary
        open    := unknown cells U known-free uncovered cells U {robot current cell}

        Any open connected component that is not connected to the robot cell and
        contains at least one known-free uncovered cell is treated as a
        confirmed coverage hole candidate.
        """
        rp = self.current_pos if robot_pos is None else robot_pos
        rr, cc = rp
        data = component_data or self._coverage_hole_component_data()
        open_base = data["open_base"]
        comp_id = data["comp_id"]
        comp_known_mass = data["comp_known_mass"]
        comp_open_mass = data["comp_open_mass"]
        comp_touches_boundary = data["comp_touches_boundary"]

        active_components = set()
        if 0 <= rr < self.rows and 0 <= cc < self.cols:
            if open_base[rr, cc]:
                cid = int(comp_id[rr, cc])
                if cid >= 0:
                    active_components.add(cid)
            for dr, dc in ACTION_TO_DELTA.values():
                nr, nc = rr + dr, cc + dc
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                if not open_base[nr, nc]:
                    continue
                cid = int(comp_id[nr, nc])
                if cid >= 0:
                    active_components.add(cid)

        hole_count = 0.0
        hole_known_mass = 0.0
        hole_open_mass = 0.0
        for cid, known_mass in enumerate(comp_known_mass.tolist()):
            if known_mass <= 0 or cid in active_components or bool(comp_touches_boundary[cid]):
                continue
            hole_count += 1.0
            hole_known_mass += float(known_mass)
            hole_open_mass += float(comp_open_mass[cid])

        return {
            "coverage_hole_count": float(hole_count),
            "coverage_hole_known_mass": float(hole_known_mass),
            "coverage_hole_open_mass": float(hole_open_mass),
        }

    def _heuristic_signal_vector(self) -> np.ndarray:
        stats = self._loop_status()
        return np.asarray(
            [
                float(stats["no_progress_norm"]),
                float(stats["recent_unique_fraction"]),
                float(stats["cycle2_detected"]),
                float(stats["loop_detected"]),
            ],
            dtype=np.float32,
        )

    def _hole_signal_vector(self) -> np.ndarray:
        if not self._hole_risk_enabled():
            return np.zeros(4, dtype=np.float32)
        component_data = self._hole_component_data
        if component_data is None:
            self._refresh_hole_cache(self.current_pos)
            component_data = self._hole_component_data
        if component_data is None:
            return np.zeros(4, dtype=np.float32)
        current_count = float(self._current_hole_stats()["coverage_hole_count"])
        risk = np.zeros(4, dtype=np.float32)
        for action in range(4):
            dr, dc = ACTION_TO_DELTA[action]
            nr, nc = self.current_pos[0] + dr, self.current_pos[1] + dc
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                continue
            if self.known_map[nr, nc] == 1:
                continue
            next_count = float(
                self._coverage_hole_stats((int(nr), int(nc)), component_data)["coverage_hole_count"]
            )
            risk[action] = 1.0 if next_count > current_count else 0.0
        return risk

    def _heuristic_action_from_mask(self, mask: np.ndarray) -> Optional[int]:
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return None

        cr, cc = self.current_pos
        unknown = self.known_map == -1
        known_free = self.known_map == 0

        # If we already stand on a frontier-approach cell, prefer stepping into
        # an adjacent unknown cell directly.
        direct_unknown: List[int] = []
        for a in valid:
            action = int(a)
            dr, dc = ACTION_TO_DELTA[action]
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and unknown[nr, nc]:
                direct_unknown.append(action)
        if direct_unknown:
            return int(direct_unknown[0])

        # Reachable frontier targets are known-free cells adjacent to unknown.
        frontier_targets = np.zeros_like(known_free, dtype=bool)
        for rr in range(self.rows):
            for cc2 in range(self.cols):
                if not known_free[rr, cc2]:
                    continue
                for dr, dc in ACTION_TO_DELTA.values():
                    nr, nc = rr + dr, cc2 + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and unknown[nr, nc]:
                        frontier_targets[rr, cc2] = True
                        break

        q: Deque[GridPos] = deque([self.current_pos])
        parents: Dict[GridPos, Tuple[GridPos, int]] = {}
        seen = {self.current_pos}
        target: Optional[GridPos] = None

        while q:
            pos = q.popleft()
            if pos != self.current_pos and frontier_targets[pos]:
                target = pos
                break
            pr, pc = pos
            for action, (dr, dc) in ACTION_TO_DELTA.items():
                nr, nc = pr + dr, pc + dc
                nxt = (nr, nc)
                if nxt in seen:
                    continue
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                if not known_free[nr, nc]:
                    continue
                seen.add(nxt)
                parents[nxt] = (pos, int(action))
                q.append(nxt)

        if target is not None:
            cur = target
            first_action = None
            while cur in parents:
                prev_pos, action = parents[cur]
                first_action = int(action)
                if prev_pos == self.current_pos:
                    break
                cur = prev_pos
            if first_action is not None:
                return int(first_action)

        cr, cc = self.current_pos
        best_key = None
        best_action = int(valid[0])
        prev_cell = self.prev_pos
        for a in valid:
            action = int(a)
            dr, dc = ACTION_TO_DELTA[action]
            nr, nc = cr + dr, cc + dc
            cell = int(self.known_map[nr, nc])
            visit_count = int(self.visit_counts[nr, nc])
            backtrack = 1 if (prev_cell is not None and (nr, nc) == prev_cell) else 0
            if cell == 0 and (not self.explored[nr, nc]):
                tier = 0
            elif cell == -1:
                tier = 1
            else:
                tier = 2
            key = (tier, visit_count, backtrack, action)
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action

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
        self.recent_positions.clear()
        self.visit_counts.fill(0)
        self.no_progress_streak = 0
        self._milestone_hit_90 = False
        self._milestone_hit_99 = False

        self.known_map.fill(-1)
        self.explored.fill(False)
        self.covered_free_count = 0
        self.maps_builder.reset_incremental_state()
        if self._boundary_maps_builder is not None:
            self._boundary_maps_builder.reset_incremental_state()
        self._sense_at(self.current_pos)
        self._mark_explored(self.current_pos)
        rr, cc = self.current_pos
        self.visit_counts[rr, cc] = 1
        self.recent_positions.append(self.current_pos)
        if self._hole_metrics_enabled():
            self._refresh_hole_cache(self.current_pos)
        else:
            self._hole_component_data = None
            self._hole_stats_cache = self._zero_hole_stats()
            self._hole_stats_pos = self.current_pos

        self.last_reward = CPPRewardBreakdown(
            area=0.0,
            tv_local=0.0,
            tv_global=0.0,
            collision=0.0,
            constant=0.0,
            hole=0.0,
            total=0.0,
        )
        return self._build_observation()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")
        requested_action = int(action)
        if requested_action not in ACTION_TO_DELTA:
            raise ValueError(f"Invalid action {action}. Expected one of {sorted(ACTION_TO_DELTA)}")

        hole_metrics_enabled = self._hole_metrics_enabled()
        hole_risk_enabled = self._hole_risk_enabled()
        hole_stats_before = self._current_hole_stats() if hole_metrics_enabled else self._zero_hole_stats()
        hole_risk_before = self._hole_signal_vector() if hole_risk_enabled else np.zeros(4, dtype=np.float32)
        action_mask = self.get_action_mask()
        executed_action = requested_action
        action_overridden = False
        if self.config.use_action_mask and (not bool(action_mask[requested_action])):
            fallback = self._fallback_action_from_mask(action_mask)
            if fallback is not None:
                executed_action = int(fallback)
                action_overridden = True
        loop_status_before = self._loop_status()
        heuristic_action_used = False
        if bool(self.config.heuristic_override) and bool(loop_status_before["loop_detected"]):
            fallback = self._heuristic_action_from_mask(action_mask)
            if fallback is not None and int(fallback) != int(executed_action):
                executed_action = int(fallback)
                heuristic_action_used = True

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
        prev_coverage_cells = int(self.covered_free_count)
        local_tv_old = self._compute_local_tv(self.current_pos)
        newly_visited = self._mark_explored(self.current_pos)
        local_tv_new = self._compute_local_tv(self.current_pos)
        global_tv = self._compute_global_tv()
        coverage_cells = int(self.covered_free_count)
        prev_coverage_ratio = float(prev_coverage_cells) / float(max(1, self.free_total))
        curr_coverage_ratio = float(coverage_cells) / float(max(1, self.free_total))

        rew_in = CPPRewardInput(
            newly_visited=float(newly_visited),
            max_newly_visited=1.0,
            local_tv_old=local_tv_old,
            local_tv_new=local_tv_new,
            global_tv=global_tv,
            coverage_pixels=float(coverage_cells),
            collided=collided,
            local_tv_velocity_norm=1.0,
        )
        base_reward = compute_cpp_reward(rew_in, self.config.reward)
        reward_turn = 0.0
        if prev_action is not None and executed_action != int(prev_action):
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

        executed_hole_risk = 0.0
        if 0 <= int(executed_action) < int(hole_risk_before.shape[0]):
            executed_hole_risk = float(hole_risk_before[int(executed_action)])

        if hole_metrics_enabled:
            hole_stats_after = self._refresh_hole_cache(self.current_pos)
            hole_count_delta = max(
                0.0,
                float(hole_stats_after["coverage_hole_count"])
                - float(hole_stats_before["coverage_hole_count"]),
            )
            hole_known_delta = max(
                0.0,
                float(hole_stats_after["coverage_hole_known_mass"])
                - float(hole_stats_before["coverage_hole_known_mass"]),
            )
        else:
            hole_stats_after = self._zero_hole_stats()
            hole_count_delta = 0.0
            hole_known_delta = 0.0
        reward_hole = 0.0
        if (not collided) and executed_hole_risk > 0.0:
            reward_hole = -float(self.config.reward.coverage_hole_penalty_scale) * float(
                executed_hole_risk
            )

        total_reward = float(
            base_reward.total + reward_turn + reward_overlap + reward_milestone + reward_hole
        )
        self.last_reward = CPPRewardBreakdown(
            area=float(base_reward.area),
            tv_local=float(base_reward.tv_local),
            tv_global=float(base_reward.tv_global),
            collision=float(base_reward.collision),
            constant=float(base_reward.constant),
            hole=float(reward_hole),
            total=total_reward,
        )
        self.recent_new_coverage.append(float(newly_visited))
        if newly_visited > 0.0:
            self.no_progress_streak = 0
        else:
            self.no_progress_streak += 1

        self.steps += 1
        self.path.append(self.current_pos)
        self.prev_action = int(executed_action)
        self.recent_actions.append(int(executed_action))
        self.recent_positions.append(self.current_pos)
        rr, cc = self.current_pos
        self.visit_counts[rr, cc] += 1
        loop_status_after = self._loop_status()

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

        info = {
            **self.last_reward.as_dict(),
            "collision": bool(collided),
            "action_requested": int(requested_action),
            "action_executed": int(executed_action),
            "action_overridden": bool(action_overridden),
            "heuristic_action_used": bool(heuristic_action_used),
            "forced_turn": bool(forced_turn),
            "action_mask_valid_count": int(action_mask.sum()),
            "action_mask": [int(v) for v in action_mask.astype(np.int8)],
            "revisited_cell": bool((not collided) and was_explored),
            "overlap_streak": int(self.overlap_streak),
            "no_progress_streak": int(loop_status_after["no_progress_streak"]),
            "recent_unique_positions": int(loop_status_after["recent_unique_positions"]),
            "cycle2_detected": bool(loop_status_after["cycle2_detected"]),
            "force_loop_detected": bool(loop_status_after["force_loop_detected"]),
            "loop_detected": bool(loop_status_after["loop_detected"]),
            "coverage_hole_count": float(hole_stats_after["coverage_hole_count"]),
            "coverage_hole_known_mass": float(hole_stats_after["coverage_hole_known_mass"]),
            "coverage_hole_open_mass": float(hole_stats_after["coverage_hole_open_mass"]),
            "coverage_hole_count_delta": float(hole_count_delta),
            "coverage_hole_known_mass_delta": float(hole_known_delta),
            "hole_risk_vector": [float(v) for v in hole_risk_before.tolist()],
            "hole_executed_action_risk": float(executed_hole_risk),
            "reward_turn": float(reward_turn),
            "reward_overlap": float(reward_overlap),
            "reward_milestone": float(reward_milestone),
            "reward_hole": float(reward_hole),
            "milestone_hit_90_now": bool(milestone_hit_90_now),
            "milestone_hit_99_now": bool(milestone_hit_99_now),
            "milestone_hit_90": bool(self._milestone_hit_90),
            "milestone_hit_99": bool(self._milestone_hit_99),
            "coverage_cells": int(coverage_cells),
            "free_cells": int(self.free_total),
            "coverage_ratio": float(curr_coverage_ratio),
            "steps": int(self.steps),
            "done_reason": done_reason,
            "truncated": bool(truncated),
            "position": self.current_pos,
        }
        return obs, float(self.last_reward.total), bool(done), info
