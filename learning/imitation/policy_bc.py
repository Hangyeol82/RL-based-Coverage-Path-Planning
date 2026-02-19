from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn as nn
from torch.distributions import Categorical

from learning.common.encoders import FusedMAPSStateEncoder, FusedMAPSStateEncoderConfig


@dataclass(frozen=True)
class BCPolicyConfig:
    action_dim: int
    encoder: FusedMAPSStateEncoderConfig = field(default_factory=FusedMAPSStateEncoderConfig)


class BCPolicy(nn.Module):
    """
    Behavioral-cloning policy with shared MAPS/state encoder.
    """

    def __init__(self, config: BCPolicyConfig):
        super().__init__()
        if config.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.config = config
        self.encoder = FusedMAPSStateEncoder(config.encoder)
        self.actor_head = nn.Linear(self.encoder.output_dim, config.action_dim)

    def forward(self, levels: Mapping[int, torch.Tensor], robot_state: torch.Tensor) -> torch.Tensor:
        z = self.encoder(levels, robot_state)
        return self.actor_head(z)

    def action_distribution(
        self,
        levels: Mapping[int, torch.Tensor],
        robot_state: torch.Tensor,
    ) -> Categorical:
        logits = self.forward(levels, robot_state)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(
        self,
        levels: Mapping[int, torch.Tensor],
        robot_state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        dist = self.action_distribution(levels, robot_state)
        if deterministic:
            return torch.argmax(dist.logits, dim=-1)
        return dist.sample()

    def loss(
        self,
        levels: Mapping[int, torch.Tensor],
        robot_state: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(levels, robot_state)
        return nn.functional.cross_entropy(logits, target_actions.long())
