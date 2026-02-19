from __future__ import annotations

from typing import Dict, Optional

import torch
from gymnasium import spaces

from learning.common import FusedMAPSStateEncoder, FusedMAPSStateEncoderConfig

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception as exc:  # pragma: no cover
    BaseFeaturesExtractor = object  # type: ignore[assignment]
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None


class MAPSStateFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 features extractor that reuses the project's MAPS+robot-state encoder.
    Expects dict observation with keys:
      - level_0, level_1, ..., level_L
      - robot_state
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        encoder_config: Optional[FusedMAPSStateEncoderConfig] = None,
    ):
        if _SB3_IMPORT_ERROR is not None:
            raise ImportError(
                "stable_baselines3 is required for MAPSStateFeaturesExtractor"
            ) from _SB3_IMPORT_ERROR

        cfg = encoder_config or FusedMAPSStateEncoderConfig()
        encoder = FusedMAPSStateEncoder(cfg)
        super().__init__(observation_space, features_dim=encoder.output_dim)
        self._level_ids = self._parse_level_ids(observation_space)
        self.encoder = encoder

    @staticmethod
    def _parse_level_ids(observation_space: spaces.Dict):
        ids = []
        for key in observation_space.spaces.keys():
            if key.startswith("level_"):
                try:
                    ids.append(int(key.split("_", 1)[1]))
                except ValueError as e:
                    raise ValueError(f"Invalid level key in observation space: {key}") from e
        if not ids:
            raise ValueError("No level_* keys found in observation space")
        ids = sorted(ids)
        expected = list(range(ids[-1] + 1))
        if ids != expected:
            raise ValueError(f"Level keys must be contiguous from 0, got {ids}")
        return tuple(ids)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        levels = {lv: observations[f"level_{lv}"].float() for lv in self._level_ids}
        robot_state = observations["robot_state"].float()
        return self.encoder(levels, robot_state)
