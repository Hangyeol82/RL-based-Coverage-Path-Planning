from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import numpy as np

from learning.reinforcement.cpp_env import CPPDiscreteEnvConfig

from .cpp_env import GridPos, PaperCPPDiscreteEnv, PaperCPPDiscreteGymEnv


class OfflinePaperCPPDiscreteEnv(PaperCPPDiscreteEnv):
    """
    Paper environment variant with full-map observations.

    The training loop, reward, action space, and metrics stay identical to the
    online paper environment. Only sensing changes: every sensing call reveals
    the complete true map into known_map, so map-observation builders receive a
    fully known occupancy map at every step.
    """

    def __init__(
        self,
        grid_map: np.ndarray,
        *,
        start_pos: Optional[GridPos] = None,
        config: Optional[CPPDiscreteEnvConfig] = None,
        **kwargs,
    ):
        config = replace(
            config or CPPDiscreteEnvConfig(),
            preserve_static_map_observation_cache_on_reset=True,
        )
        super().__init__(
            grid_map=grid_map,
            start_pos=start_pos,
            config=config,
            **kwargs,
        )

    def _sense_at(self, pos: GridPos):
        self.known_map[:, :] = self.true_map

    def step(self, action: int):
        obs, reward, done, info = super().step(action)
        info["offline_full_map"] = True
        info["offline_known_ratio"] = 1.0
        return obs, reward, done, info


class OfflinePaperCPPDiscreteGymEnv(PaperCPPDiscreteGymEnv):
    """Gymnasium wrapper for OfflinePaperCPPDiscreteEnv."""

    def __init__(
        self,
        grid_map: np.ndarray,
        *,
        start_pos: Optional[GridPos] = None,
        config: Optional[CPPDiscreteEnvConfig] = None,
        include_hole_signals: bool = False,
        hole_penalty_scale: float = 0.0,
        revisit_burden_shaping: bool = False,
        revisit_burden_scale: float = 0.0,
        revisit_burden_gamma: float = 0.99,
        revisit_burden_normalizer: float = 0.0,
        revisit_burden_unreachable_cost: float = 0.0,
        metric_stagnation_threshold: int = 30,
        metric_loop_window: int = 12,
        grid_map_pool: Optional[Sequence[np.ndarray]] = None,
        grid_map_metadata_pool: Optional[Sequence[Dict[str, Any]]] = None,
        episode_map_refresh: bool = False,
        map_refresh_mode: str = "cycle",
        map_refresh_seed: int = 0,
        episode_success_threshold: float = 1.0,
    ):
        gym.Env.__init__(self)
        self.core_env = OfflinePaperCPPDiscreteEnv(
            grid_map=grid_map,
            start_pos=start_pos,
            config=config,
            include_hole_signals=include_hole_signals,
            hole_penalty_scale=hole_penalty_scale,
            revisit_burden_shaping=revisit_burden_shaping,
            revisit_burden_scale=revisit_burden_scale,
            revisit_burden_gamma=revisit_burden_gamma,
            revisit_burden_normalizer=revisit_burden_normalizer,
            revisit_burden_unreachable_cost=revisit_burden_unreachable_cost,
            metric_stagnation_threshold=metric_stagnation_threshold,
            metric_loop_window=metric_loop_window,
            grid_map_pool=grid_map_pool,
            grid_map_metadata_pool=grid_map_metadata_pool,
            episode_map_refresh=episode_map_refresh,
            map_refresh_mode=map_refresh_mode,
            map_refresh_seed=map_refresh_seed,
            episode_success_threshold=episode_success_threshold,
        )
        self._start_pos = start_pos

        sample = self.core_env.reset(start_pos=start_pos)
        self._use_hybrid_maps = "hybrid_maps" in sample
        self._level_ids = tuple(sorted(sample["levels"].keys())) if not self._use_hybrid_maps else ()

        from gymnasium import spaces

        obs_spaces: Dict[str, spaces.Space] = {}
        if self._use_hybrid_maps:
            hybrid_maps = sample["hybrid_maps"]
            for key in hybrid_maps.keys():
                obs_key = self._hybrid_obs_key(str(key))
                x = np.asarray(hybrid_maps[key], dtype=np.float32)
                obs_spaces[obs_key] = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=x.shape,
                    dtype=np.float32,
                )
        else:
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
