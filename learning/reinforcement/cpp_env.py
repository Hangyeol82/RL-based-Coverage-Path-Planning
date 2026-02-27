from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

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
        if self.free_total <= 0:
            raise ValueError("grid_map must have at least one free cell")

        self.maps_builder = MultiScaleCPPObservationBuilder(
            self.config.observation,
            include_dtm=self.config.include_dtm,
        )
        self._boundary_maps_builder: Optional[MultiScaleCPPObservationBuilder] = None
        if bool(self.config.use_boundary_exit_features) and (not bool(self.config.include_dtm)):
            # Boundary features can be used as an MLP-side replacement even when
            # DTM channels are not part of the CNN maps input.
            self._boundary_maps_builder = MultiScaleCPPObservationBuilder(
                self.config.observation,
                include_dtm=True,
            )
        self.robot_state_builder = RobotStateObservationBuilder(self.config.robot_state)

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
        radius = max(0, int(self.config.local_tv_patch_radius))
        cov = self._coverage_map_float()
        obs = self._known_obstacle_map_float()
        cov_local = self._local_patch(cov, center, radius)
        obs_local = self._local_patch(obs, center, radius)
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

    def _build_observation(self) -> Dict[str, object]:
        levels = self.maps_builder.build_levels(
            self.known_map,
            robot_pos=self.current_pos,
            explored=self.explored,
        )
        boundary_exit_features: Optional[np.ndarray] = None
        if self.config.use_boundary_exit_features:
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
            boundary_exit_features = self._build_boundary_exit_features(boundary_levels)
        robot_state = self.robot_state_builder.build(
            occupancy=self.known_map,
            explored=self.explored,
            robot_pos=self.current_pos,
            prev_pos=self.prev_pos,
            recent_new_coverage=list(self.recent_new_coverage),
            extra_features=boundary_exit_features,
        )
        return {
            "levels": levels,
            "robot_state": robot_state.astype(np.float32),
        }

    def _dtm_channel_count(self) -> int:
        mode = str(self.config.observation.dtm_output_mode).strip().lower()
        if mode in {"six", "extent6"}:
            return 6
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
            valid = float(np.all(dtm >= 0.0))
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

    def reset(self, *, start_pos: Optional[GridPos] = None) -> Dict[str, object]:
        self.current_pos = self._resolve_start(start_pos) if start_pos is not None else self._default_start
        self.prev_pos = None
        self.prev_action = None
        self.steps = 0
        self.done = False
        self.last_collision = False
        self.path = [self.current_pos]
        self.recent_new_coverage.clear()
        self._milestone_hit_90 = False
        self._milestone_hit_99 = False

        self.known_map.fill(-1)
        self.explored.fill(False)
        self._sense_at(self.current_pos)
        self._mark_explored(self.current_pos)

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
        if self.config.use_action_mask and (not bool(action_mask[requested_action])):
            fallback = self._fallback_action_from_mask(action_mask)
            if fallback is not None:
                executed_action = int(fallback)
                action_overridden = True

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
        local_tv_old = self._compute_local_tv(self.current_pos)
        newly_visited = self._mark_explored(self.current_pos)
        local_tv_new = self._compute_local_tv(self.current_pos)
        global_tv = self._compute_global_tv()
        coverage_cells = int(np.count_nonzero(self.explored & self.free_mask))
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
        if (
            prev_action is not None
            and executed_action != int(prev_action)
            and not forced_turn
        ):
            reward_turn = float(self.config.reward.turn_change_penalty)

        reward_overlap = 0.0
        if (not collided) and was_explored:
            reward_overlap = float(self.config.reward.revisit_penalty)

        reward_milestone, milestone_hit_90_now, milestone_hit_99_now = self._compute_milestone_reward(
            prev_coverage_ratio=prev_coverage_ratio,
            curr_coverage_ratio=curr_coverage_ratio,
            collided=collided,
        )

        total_reward = float(base_reward.total + reward_turn + reward_overlap + reward_milestone)
        self.last_reward = CPPRewardBreakdown(
            area=float(base_reward.area),
            tv_local=float(base_reward.tv_local),
            tv_global=float(base_reward.tv_global),
            collision=float(base_reward.collision),
            constant=float(base_reward.constant),
            total=total_reward,
        )
        self.recent_new_coverage.append(float(newly_visited))

        self.steps += 1
        self.path.append(self.current_pos)
        self.prev_action = int(executed_action)

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
            "forced_turn": bool(forced_turn),
            "action_mask_valid_count": int(np.count_nonzero(action_mask)),
            "action_mask": [int(v) for v in action_mask.astype(np.int8)],
            "revisited_cell": bool((not collided) and was_explored),
            "reward_turn": float(reward_turn),
            "reward_overlap": float(reward_overlap),
            "reward_milestone": float(reward_milestone),
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
