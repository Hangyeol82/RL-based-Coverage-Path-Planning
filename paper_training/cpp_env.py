from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from learning.reinforcement.cpp_env import CPPDiscreteEnv, CPPDiscreteEnvConfig
from learning.reinforcement.reward import CPPRewardBreakdown

from .hole_observation import HoleObservationCalculator


GridPos = Tuple[int, int]


class PaperCPPDiscreteEnv(CPPDiscreteEnv):
    """Paper-only CPP env extension for hole-risk observation/reward."""

    def __init__(
        self,
        grid_map: np.ndarray,
        *,
        start_pos: Optional[GridPos] = None,
        config: Optional[CPPDiscreteEnvConfig] = None,
        include_hole_signals: bool = False,
        hole_penalty_scale: float = 0.0,
        metric_stagnation_threshold: int = 30,
        metric_loop_window: int = 12,
        grid_map_pool: Optional[Sequence[np.ndarray]] = None,
        episode_map_refresh: bool = False,
        map_refresh_mode: str = "cycle",
        map_refresh_seed: int = 0,
        episode_success_threshold: float = 1.0,
    ):
        self.paper_episode_map_refresh = bool(episode_map_refresh)
        self.paper_map_refresh_mode = str(map_refresh_mode)
        if self.paper_map_refresh_mode not in {"cycle", "random"}:
            raise ValueError("map_refresh_mode must be 'cycle' or 'random'")
        self.paper_episode_success_threshold = float(episode_success_threshold)
        if not (0.0 < self.paper_episode_success_threshold <= 1.0):
            raise ValueError("episode_success_threshold must be in (0, 1]")
        self._paper_map_rng = np.random.default_rng(int(map_refresh_seed))
        self._paper_map_cursor = 0
        self._paper_episode_map_index = 0
        self._paper_grid_map_pool = self._normalize_grid_pool(grid_map, grid_map_pool)
        self.paper_include_hole_signals = bool(include_hole_signals)
        self.paper_hole_penalty_scale = max(0.0, float(hole_penalty_scale))
        self._paper_hole_enabled = self.paper_include_hole_signals or self.paper_hole_penalty_scale > 0.0
        self._paper_hole_calc: Optional[HoleObservationCalculator] = (
            HoleObservationCalculator(require_scipy=True) if self._paper_hole_enabled else None
        )
        self._paper_last_hole_risk = np.zeros(4, dtype=np.float32)
        self._paper_last_hole_stats = None
        self.paper_metric_stagnation_threshold = max(1, int(metric_stagnation_threshold))
        self.paper_metric_loop_window = max(2, int(metric_loop_window))
        self._reset_paper_metrics()
        super().__init__(grid_map=grid_map, start_pos=start_pos, config=config)

    @staticmethod
    def _normalize_grid_pool(
        grid_map: np.ndarray,
        grid_map_pool: Optional[Sequence[np.ndarray]],
    ) -> Tuple[np.ndarray, ...]:
        pool_raw: Sequence[np.ndarray]
        if grid_map_pool is not None and len(grid_map_pool) > 0:
            pool_raw = grid_map_pool
        else:
            pool_raw = (grid_map,)
        pool = []
        base_shape = tuple(np.asarray(grid_map, dtype=np.int32).shape)
        for item in pool_raw:
            arr = np.asarray(item, dtype=np.int32)
            if arr.ndim != 2:
                raise ValueError("Every grid map in the pool must be 2D")
            if tuple(arr.shape) != base_shape:
                raise ValueError(
                    "All grid maps in an episode-refresh pool must have the same shape"
                )
            if not np.isin(arr, [0, 1]).all():
                raise ValueError("Every grid map in the pool must contain only 0 and 1")
            if arr[0, 0] == 1:
                arr = arr.copy()
                arr[0, 0] = 0
            pool.append(arr.copy())
        if not pool:
            raise ValueError("grid_map_pool must not be empty")
        return tuple(pool)

    def _select_next_grid_map(self) -> Tuple[int, np.ndarray]:
        if not self.paper_episode_map_refresh or len(self._paper_grid_map_pool) <= 1:
            return int(self._paper_episode_map_index), self.true_map
        if self.paper_map_refresh_mode == "random":
            idx = int(self._paper_map_rng.integers(0, len(self._paper_grid_map_pool)))
        else:
            idx = int(self._paper_map_cursor % len(self._paper_grid_map_pool))
            self._paper_map_cursor += 1
        return idx, self._paper_grid_map_pool[idx]

    def reseed_map_refresh(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        seed_i = int(seed)
        self._paper_map_rng = np.random.default_rng(seed_i)
        if len(self._paper_grid_map_pool) > 0:
            self._paper_map_cursor = seed_i % len(self._paper_grid_map_pool)

    def _replace_true_map(self, grid_map: np.ndarray) -> None:
        grid = np.asarray(grid_map, dtype=np.int32)
        self.true_map = grid
        self.rows, self.cols = grid.shape
        self.free_mask = self.true_map == 0
        self.free_total = int(np.count_nonzero(self.free_mask))
        if self.free_total <= 0:
            raise ValueError("grid_map must have at least one free cell")
        self._default_start = self._resolve_start(None)
        self.known_map = np.full_like(self.true_map, fill_value=-1, dtype=np.int32)
        self.explored = np.zeros_like(self.true_map, dtype=bool)

    def _reset_paper_metrics(self) -> None:
        self._paper_turn_count = 0
        self._paper_revisit_count = 0
        self._paper_noncollision_steps = 0
        self._paper_no_progress_streak = 0
        self._paper_stagnation_step_count = 0
        self._paper_loop_step_count = 0
        self._paper_loop_or_stagnation_step_count = 0
        self._paper_recent_positions = []
        self._paper_threshold_steps = {0.90: None, 0.95: None, 0.99: None}
        self._paper_last_coverage_cells = 0

    def _update_threshold_steps(self, coverage_ratio: float) -> None:
        for threshold in self._paper_threshold_steps:
            if self._paper_threshold_steps[threshold] is None and float(coverage_ratio) >= threshold:
                self._paper_threshold_steps[threshold] = int(self.steps)

    def _refresh_paper_hole(self) -> np.ndarray:
        if self._paper_hole_calc is None:
            self._paper_last_hole_risk = np.zeros(4, dtype=np.float32)
            self._paper_last_hole_stats = None
            return self._paper_last_hole_risk
        self._paper_hole_calc.refresh(self.known_map, self.explored)
        self._paper_last_hole_stats = self._paper_hole_calc.stats_for_pos(self.current_pos)
        self._paper_last_hole_risk = self._paper_hole_calc.risk_vector(self.current_pos, self.known_map)
        return self._paper_last_hole_risk

    def _build_observation(self) -> Dict[str, object]:
        obs = super()._build_observation()
        if self._paper_hole_enabled:
            risk = self._refresh_paper_hole()
            if self.paper_include_hole_signals:
                robot_state = np.asarray(obs["robot_state"], dtype=np.float32).reshape(-1)
                obs["robot_state"] = np.concatenate([robot_state, risk.astype(np.float32)], axis=0)
        return obs

    def reset(self, *, start_pos: Optional[GridPos] = None) -> Dict[str, object]:
        if start_pos is None and getattr(self, "paper_episode_map_refresh", False):
            map_idx, grid = self._select_next_grid_map()
            if grid is not self.true_map:
                self._replace_true_map(grid)
            self._paper_episode_map_index = int(map_idx)
        self._paper_last_hole_risk = np.zeros(4, dtype=np.float32)
        self._paper_last_hole_stats = None
        self._reset_paper_metrics()
        obs = super().reset(start_pos=start_pos)
        self._paper_last_coverage_cells = int(np.count_nonzero(self.explored & self.free_mask))
        self._paper_recent_positions = [self.current_pos]
        self._update_threshold_steps(self._coverage_ratio())
        return obs

    def step(self, action: int):
        pre_risk = (
            self._paper_last_hole_risk.copy()
            if self._paper_hole_enabled
            else np.zeros(4, dtype=np.float32)
        )
        obs, reward, done, info = super().step(action)

        executed_action = int(info.get("action_executed", int(action)))
        executed_hole_risk = 0.0
        if 0 <= executed_action < int(pre_risk.shape[0]):
            executed_hole_risk = float(pre_risk[executed_action])

        reward_hole = 0.0
        if self.paper_hole_penalty_scale > 0.0 and (not bool(info.get("collision", False))):
            reward_hole = -self.paper_hole_penalty_scale * executed_hole_risk

        if reward_hole != 0.0:
            reward = float(reward) + float(reward_hole)
            last = self.last_reward
            self.last_reward = CPPRewardBreakdown(
                area=float(last.area),
                tv_local=float(last.tv_local),
                tv_global=float(last.tv_global),
                collision=float(last.collision),
                constant=float(last.constant),
                total=float(reward),
            )

        turn_event = 1.0 if abs(float(info.get("reward_turn", 0.0))) > 0.0 else 0.0
        revisit_event = 1.0 if bool(info.get("revisited_cell", False)) else 0.0
        collision = bool(info.get("collision", False))
        coverage_cells = int(info.get("coverage_cells", self._paper_last_coverage_cells))
        newly_covered = (not collision) and coverage_cells > int(self._paper_last_coverage_cells)
        self._paper_last_coverage_cells = coverage_cells

        if turn_event > 0.0:
            self._paper_turn_count += 1
        if revisit_event > 0.0:
            self._paper_revisit_count += 1
        if not collision:
            self._paper_noncollision_steps += 1
        if newly_covered:
            self._paper_no_progress_streak = 0
        else:
            self._paper_no_progress_streak += 1

        position = tuple(info.get("position", self.current_pos))
        loop_event = (not newly_covered) and position in self._paper_recent_positions[-self.paper_metric_loop_window :]
        stagnation_event = self._paper_no_progress_streak >= self.paper_metric_stagnation_threshold
        if loop_event:
            self._paper_loop_step_count += 1
        if stagnation_event:
            self._paper_stagnation_step_count += 1
        if loop_event or stagnation_event:
            self._paper_loop_or_stagnation_step_count += 1
        self._paper_recent_positions.append(position)
        if len(self._paper_recent_positions) > self.paper_metric_loop_window:
            self._paper_recent_positions = self._paper_recent_positions[-self.paper_metric_loop_window :]
        self._update_threshold_steps(float(info.get("coverage_ratio", 0.0)))

        info["hole_risk_up"] = float(pre_risk[0])
        info["hole_risk_right"] = float(pre_risk[1])
        info["hole_risk_down"] = float(pre_risk[2])
        info["hole_risk_left"] = float(pre_risk[3])
        info["executed_hole_risk"] = float(executed_hole_risk)
        info["reward_hole"] = float(reward_hole)
        if self._paper_last_hole_stats is not None:
            info["coverage_hole_count"] = float(self._paper_last_hole_stats.count)
            info["coverage_hole_known_mass"] = float(self._paper_last_hole_stats.known_mass)
            info["coverage_hole_open_mass"] = float(self._paper_last_hole_stats.open_mass)
        if self._paper_hole_calc is not None and self._paper_hole_calc.cache is not None:
            info["hole_refresh_ms"] = float(self._paper_hole_calc.cache.refresh_ms)
            info["hole_risk_ms"] = float(self._paper_hole_calc.last_risk_ms)
        info["episode_map_index"] = float(getattr(self, "_paper_episode_map_index", 0))
        steps = max(1, int(self.steps))
        noncollision_steps = max(1, int(self._paper_noncollision_steps))
        info["turn_event"] = float(turn_event)
        info["turn_count"] = float(self._paper_turn_count)
        info["turn_ratio"] = float(self._paper_turn_count) / float(steps)
        info["revisit_count"] = float(self._paper_revisit_count)
        info["revisit_ratio"] = float(self._paper_revisit_count) / float(noncollision_steps)
        info["overlap_ratio"] = float(self._paper_revisit_count) / float(steps)
        info["no_progress_streak_paper"] = float(self._paper_no_progress_streak)
        info["stagnation_detected"] = float(stagnation_event)
        info["loop_detected"] = float(loop_event)
        info["loop_or_stagnation_detected"] = float(loop_event or stagnation_event)
        info["reward_total"] = float(reward)
        if (
            (not done)
            and self.paper_episode_success_threshold < 1.0
            and float(info.get("coverage_ratio", 0.0)) >= self.paper_episode_success_threshold
        ):
            done = True
            self.done = True
            info["done_reason"] = "coverage_threshold"
            info["truncated"] = False
        if done:
            final_steps = max(1, int(self.steps))
            info["episode_turn_count"] = float(self._paper_turn_count)
            info["episode_turn_ratio"] = float(self._paper_turn_count) / float(final_steps)
            info["episode_revisit_count"] = float(self._paper_revisit_count)
            info["episode_revisit_ratio"] = float(self._paper_revisit_count) / float(noncollision_steps)
            info["episode_overlap_ratio"] = float(self._paper_revisit_count) / float(final_steps)
            info["episode_stagnation_step_count"] = float(self._paper_stagnation_step_count)
            info["episode_loop_step_count"] = float(self._paper_loop_step_count)
            info["episode_loop_or_stagnation_step_count"] = float(
                self._paper_loop_or_stagnation_step_count
            )
            info["episode_loop_or_stagnation_ratio"] = (
                float(self._paper_loop_or_stagnation_step_count) / float(final_steps)
            )
            for suffix, threshold in (("90", 0.90), ("95", 0.95), ("99", 0.99)):
                step_hit = self._paper_threshold_steps[threshold]
                success = step_hit is not None
                info[f"episode_success_{suffix}"] = float(success)
                info[f"episode_step_to_{suffix}"] = float(step_hit) if success else float("nan")
        return obs, float(reward), bool(done), info


class PaperCPPDiscreteGymEnv(gym.Env):
    """Gymnasium wrapper for PaperCPPDiscreteEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_map: np.ndarray,
        *,
        start_pos: Optional[GridPos] = None,
        config: Optional[CPPDiscreteEnvConfig] = None,
        include_hole_signals: bool = False,
        hole_penalty_scale: float = 0.0,
        metric_stagnation_threshold: int = 30,
        metric_loop_window: int = 12,
        grid_map_pool: Optional[Sequence[np.ndarray]] = None,
        episode_map_refresh: bool = False,
        map_refresh_mode: str = "cycle",
        map_refresh_seed: int = 0,
        episode_success_threshold: float = 1.0,
    ):
        super().__init__()
        self.core_env = PaperCPPDiscreteEnv(
            grid_map=grid_map,
            start_pos=start_pos,
            config=config,
            include_hole_signals=include_hole_signals,
            hole_penalty_scale=hole_penalty_scale,
            metric_stagnation_threshold=metric_stagnation_threshold,
            metric_loop_window=metric_loop_window,
            grid_map_pool=grid_map_pool,
            episode_map_refresh=episode_map_refresh,
            map_refresh_mode=map_refresh_mode,
            map_refresh_seed=map_refresh_seed,
            episode_success_threshold=episode_success_threshold,
        )
        self._start_pos = start_pos

        sample = self.core_env.reset(start_pos=start_pos)
        self._level_ids = tuple(sorted(sample["levels"].keys()))

        obs_spaces: Dict[str, spaces.Space] = {}
        for lv in self._level_ids:
            x = np.asarray(sample["levels"][lv], dtype=np.float32)
            obs_spaces[f"level_{lv}"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=x.shape,
                dtype=np.float32,
            )
        rs = np.asarray(sample["robot_state"], dtype=np.float32)
        obs_spaces["robot_state"] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=rs.shape,
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Discrete(self.core_env.action_dim)

    def _convert_obs(self, raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        levels = raw_obs["levels"]
        for lv in self._level_ids:
            out[f"level_{lv}"] = np.asarray(levels[lv], dtype=np.float32)
        out["robot_state"] = np.asarray(raw_obs["robot_state"], dtype=np.float32)
        return out

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        opts = options or {}
        start_pos = opts.get("start_pos", self._start_pos)
        self.core_env.reseed_map_refresh(seed)
        raw = self.core_env.reset(start_pos=start_pos)
        return self._convert_obs(raw), {}

    def step(self, action: int):
        raw_obs, reward, done, info = self.core_env.step(int(action))
        truncated = bool(info.get("truncated", False))
        terminated = bool(done and not truncated)
        return self._convert_obs(raw_obs), float(reward), terminated, truncated, dict(info)

    def action_masks(self) -> np.ndarray:
        return self.core_env.get_action_mask().astype(bool)

    def render(self):
        return None
