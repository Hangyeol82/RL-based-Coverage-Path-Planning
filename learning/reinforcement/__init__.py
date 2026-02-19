"""Reinforcement learning modules (PPO and related components)."""

from .cpp_env import ACTION_TO_DELTA, CPPDiscreteEnv, CPPDiscreteEnvConfig
from .sb3_env import CPPDiscreteGymEnv
from .reward import (
    CPPRewardBreakdown,
    CPPRewardConfig,
    CPPRewardInput,
    compute_cpp_reward,
    total_variation,
)

__all__ = [
    "CPPDiscreteEnv",
    "CPPDiscreteEnvConfig",
    "CPPDiscreteGymEnv",
    "ACTION_TO_DELTA",
    "CPPRewardConfig",
    "CPPRewardInput",
    "CPPRewardBreakdown",
    "compute_cpp_reward",
    "total_variation",
    "ActorCriticPolicy",
    "ActorCriticPolicyConfig",
]


def __getattr__(name: str):
    if name in {"ActorCriticPolicy", "ActorCriticPolicyConfig"}:
        from .policy_ac import ActorCriticPolicy, ActorCriticPolicyConfig

        globals()["ActorCriticPolicy"] = ActorCriticPolicy
        globals()["ActorCriticPolicyConfig"] = ActorCriticPolicyConfig
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
