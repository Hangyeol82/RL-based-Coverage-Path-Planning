from dataclasses import dataclass, field
from typing import Mapping, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from learning.common.encoders import FusedMAPSStateEncoder, FusedMAPSStateEncoderConfig


@dataclass(frozen=True)
class ActorCriticPolicyConfig:
    action_dim: int
    encoder: FusedMAPSStateEncoderConfig = field(default_factory=FusedMAPSStateEncoderConfig)


class ActorCriticPolicy(nn.Module):
    """
    PPO-ready actor-critic network.
    - shared fused encoder
    - policy head (actor)
    - value head (critic)
    """

    def __init__(self, config: ActorCriticPolicyConfig):
        super().__init__()
        if config.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.config = config
        self.encoder = FusedMAPSStateEncoder(config.encoder)
        self.actor_head = nn.Linear(self.encoder.output_dim, config.action_dim)
        self.critic_head = nn.Linear(self.encoder.output_dim, 1)

    def forward(
        self,
        levels: Mapping[int, torch.Tensor],
        robot_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(levels, robot_state)
        logits = self.actor_head(z)
        value = self.critic_head(z).squeeze(-1)
        return logits, value

    def dist_value(
        self,
        levels: Mapping[int, torch.Tensor],
        robot_state: torch.Tensor,
    ) -> Tuple[Categorical, torch.Tensor]:
        logits, value = self.forward(levels, robot_state)
        return Categorical(logits=logits), value

    @torch.no_grad()
    def act(
        self,
        levels: Mapping[int, torch.Tensor],
        robot_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.dist_value(levels, robot_state)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        levels: Mapping[int, torch.Tensor],
        robot_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.dist_value(levels, robot_state)
        log_prob = dist.log_prob(actions.long())
        entropy = dist.entropy()
        return log_prob, entropy, value
