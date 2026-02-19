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
        self.robot_state_builder = RobotStateObservationBuilder(self.config.robot_state)

        self._default_start = self._resolve_start(start_pos)
        self.known_map = np.full_like(self.true_map, fill_value=-1, dtype=np.int32)
        self.explored = np.zeros_like(self.true_map, dtype=bool)
        self.current_pos: GridPos = self._default_start
        self.prev_pos: Optional[GridPos] = None
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
        return float(total_variation(cov_local, obs_local, mode=self.config.tv_mode))

    def _compute_global_tv(self) -> float:
        cov = self._coverage_map_float()
        obs = self._known_obstacle_map_float()
        return float(total_variation(cov, obs, mode=self.config.tv_mode))

    def _build_observation(self) -> Dict[str, object]:
        levels = self.maps_builder.build_levels(
            self.known_map,
            robot_pos=self.current_pos,
            explored=self.explored,
        )
        robot_state = self.robot_state_builder.build(
            occupancy=self.known_map,
            explored=self.explored,
            robot_pos=self.current_pos,
            prev_pos=self.prev_pos,
            recent_new_coverage=list(self.recent_new_coverage),
        )
        return {
            "levels": levels,
            "robot_state": robot_state.astype(np.float32),
        }

    def reset(self, *, start_pos: Optional[GridPos] = None) -> Dict[str, object]:
        self.current_pos = self._resolve_start(start_pos) if start_pos is not None else self._default_start
        self.prev_pos = None
        self.steps = 0
        self.done = False
        self.last_collision = False
        self.path = [self.current_pos]
        self.recent_new_coverage.clear()

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
        if int(action) not in ACTION_TO_DELTA:
            raise ValueError(f"Invalid action {action}. Expected one of {sorted(ACTION_TO_DELTA)}")

        dr, dc = ACTION_TO_DELTA[int(action)]
        cr, cc = self.current_pos
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

        local_tv_old = self._compute_local_tv(self.current_pos)
        newly_visited = self._mark_explored(self.current_pos)
        local_tv_new = self._compute_local_tv(self.current_pos)
        global_tv = self._compute_global_tv()
        coverage_cells = int(np.count_nonzero(self.explored & self.free_mask))

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
        self.last_reward = compute_cpp_reward(rew_in, self.config.reward)
        self.recent_new_coverage.append(float(newly_visited))

        self.steps += 1
        self.path.append(self.current_pos)

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
            "coverage_cells": int(coverage_cells),
            "free_cells": int(self.free_total),
            "coverage_ratio": float(self._coverage_ratio()),
            "steps": int(self.steps),
            "done_reason": done_reason,
            "truncated": bool(truncated),
            "position": self.current_pos,
        }
        return obs, float(self.last_reward.total), bool(done), info
