from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .cpp_env import CPPDiscreteEnv, CPPDiscreteEnvConfig


GridPos = Tuple[int, int]


class CPPDiscreteGymEnv(gym.Env):
    """
    Gymnasium wrapper for CPPDiscreteEnv so it can be used by SB3 PPO.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_map: np.ndarray,
        *,
        start_pos: Optional[GridPos] = None,
        config: Optional[CPPDiscreteEnvConfig] = None,
    ):
        super().__init__()
        self.core_env = CPPDiscreteEnv(
            grid_map=grid_map,
            start_pos=start_pos,
            config=config,
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
        raw = self.core_env.reset(start_pos=start_pos)
        return self._convert_obs(raw), {}

    def step(self, action: int):
        raw_obs, reward, done, info = self.core_env.step(int(action))
        truncated = bool(info.get("truncated", False))
        terminated = bool(done and not truncated)
        return self._convert_obs(raw_obs), float(reward), terminated, truncated, dict(info)

    # sb3-contrib MaskablePPO looks for this method on envs.
    def action_masks(self) -> np.ndarray:
        return self.core_env.get_action_mask().astype(bool)

    def render(self):
        return None

    def close(self):
        return None
