from dataclasses import dataclass
import math
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class CPPRewardConfig:
    """
    Reward template aligned with rl-cpp:
    R = R_area + R_TV^I + R_TV^G + R_coll + R_const
    """

    newly_visited_reward_scale: float = 1.0
    newly_visited_reward_max: float = 2.0

    local_tv_reward_scale: float = 1.0
    local_tv_reward_max: float = 5.0
    local_tv_normalizer: float = 2.5

    global_tv_reward_scale: float = 0.0
    global_tv_reward_max: float = 5.0
    global_tv_normalizer: float = 4.0

    collision_reward: float = -10.0
    constant_reward: float = -0.1
    constant_reward_always: bool = True


@dataclass(frozen=True)
class CPPRewardInput:
    """
    Required runtime signals for reward computation.
    - newly_visited: number of newly covered cells in this step
    - max_newly_visited: normalization constant for area reward
    - local_tv_old/new: local TV before/after step
    - global_tv: global TV after step
    - coverage_pixels: number of covered free cells after step
    - collided: whether the current step caused a collision
    - local_tv_velocity_norm: optional speed-based TV normalization factor
    """

    newly_visited: float
    max_newly_visited: float
    local_tv_old: float
    local_tv_new: float
    global_tv: float
    coverage_pixels: float
    collided: bool = False
    local_tv_velocity_norm: float = 1.0


@dataclass(frozen=True)
class CPPRewardBreakdown:
    area: float
    tv_local: float
    tv_global: float
    collision: float
    constant: float
    total: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "reward_area": float(self.area),
            "reward_tv_i": float(self.tv_local),
            "reward_tv_g": float(self.tv_global),
            "reward_coll": float(self.collision),
            "reward_const": float(self.constant),
            "reward_total": float(self.total),
        }


def _signed_clip(value: float, max_abs: float) -> float:
    lim = max(0.0, float(max_abs))
    if lim == 0.0:
        return 0.0
    v = float(value)
    if v > lim:
        return lim
    if v < -lim:
        return -lim
    return v


def compute_cpp_reward(inp: CPPRewardInput, cfg: CPPRewardConfig) -> CPPRewardBreakdown:
    """
    Compute reward terms:
      R_area + R_TV^I + R_TV^G + R_coll + R_const
    """

    newly_visited = max(0.0, float(inp.newly_visited))
    collided = bool(inp.collided)

    reward_area = 0.0
    reward_tv_i = 0.0
    reward_tv_g = 0.0

    if not collided:
        denom = max(1.0, float(inp.max_newly_visited))

        reward_area = cfg.newly_visited_reward_scale * newly_visited / denom
        reward_area = min(float(cfg.newly_visited_reward_max), reward_area)

        tv_diff = float(inp.local_tv_new) - float(inp.local_tv_old)
        reward_tv_i = -tv_diff
        reward_tv_i *= float(inp.local_tv_velocity_norm)
        reward_tv_i /= max(1e-8, float(cfg.local_tv_normalizer))
        reward_tv_i *= float(cfg.local_tv_reward_scale)
        reward_tv_i = _signed_clip(reward_tv_i, cfg.local_tv_reward_max)

        coverage_pixels = max(1.0, float(inp.coverage_pixels))
        reward_tv_g = -float(inp.global_tv)
        reward_tv_g /= math.sqrt(coverage_pixels)
        reward_tv_g /= max(1e-8, float(cfg.global_tv_normalizer))
        reward_tv_g *= float(cfg.global_tv_reward_scale)
        reward_tv_g = _signed_clip(reward_tv_g, cfg.global_tv_reward_max)

    reward_coll = float(cfg.collision_reward) if collided else 0.0
    reward_const = (
        float(cfg.constant_reward)
        if cfg.constant_reward_always or newly_visited <= 0.0
        else 0.0
    )

    total = reward_area + reward_tv_i + reward_tv_g + reward_coll + reward_const
    return CPPRewardBreakdown(
        area=float(reward_area),
        tv_local=float(reward_tv_i),
        tv_global=float(reward_tv_g),
        collision=float(reward_coll),
        constant=float(reward_const),
        total=float(total),
    )


def total_variation(
    img: np.ndarray,
    img2: Optional[np.ndarray] = None,
    mode: str = "sym-iso",
) -> float:
    """
    Compatible TV helper adapted from rl-cpp.
    """

    if mode not in {"sym-iso", "non-sym-iso", "non-iso"}:
        raise ValueError(f"Unsupported TV mode: {mode}")

    x = np.asarray(img, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for TV, got shape={x.shape}")

    diff_v = np.abs(x[1:, :] - x[:-1, :])
    diff_h = np.abs(x[:, 1:] - x[:, :-1])

    if img2 is not None:
        y = np.asarray(img2, dtype=np.float32)
        if y.shape != x.shape:
            raise ValueError(f"img2 must match img shape, got {y.shape} vs {x.shape}")
        y = np.maximum(x, y)
        diff_v = np.minimum(diff_v, np.abs(y[1:, :] - y[:-1, :]))
        diff_h = np.minimum(diff_h, np.abs(y[:, 1:] - y[:, :-1]))

    if mode == "sym-iso":
        tv = (
            np.sum(np.sqrt(diff_v[:, 1:] ** 2 + diff_h[1:, :] ** 2))
            + np.sum(np.sqrt(diff_v[:, 1:] ** 2 + diff_h[:-1, :] ** 2))
            + np.sum(np.sqrt(diff_v[:, :-1] ** 2 + diff_h[:-1, :] ** 2))
            + np.sum(np.sqrt(diff_v[:, :-1] ** 2 + diff_h[1:, :] ** 2))
        )
        return float(tv / 4.0)

    if mode == "non-sym-iso":
        tv = np.sum(np.sqrt(diff_v[:, :-1] ** 2 + diff_h[:-1, :] ** 2))
        return float(tv)

    return float(np.sum(diff_v) + np.sum(diff_h))
