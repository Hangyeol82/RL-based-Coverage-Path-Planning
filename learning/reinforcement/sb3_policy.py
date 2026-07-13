from __future__ import annotations

from typing import Dict, Optional

import torch
from gymnasium import spaces

from learning.common import (
    FusedMAPSStateEncoder,
    FusedMAPSStateEncoderConfig,
    HybridLocalGlobalEncoder,
    HybridLocalGlobalEncoderConfig,
)

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


class HybridLocalGlobalFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 features extractor for hybrid local/global CPP observations.
    Expects dict observation with keys:
      - local_map
      - global_map_64, global_map_32, global_map_16, ...
      - robot_state
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        encoder_config: Optional[HybridLocalGlobalEncoderConfig] = None,
    ):
        if _SB3_IMPORT_ERROR is not None:
            raise ImportError(
                "stable_baselines3 is required for HybridLocalGlobalFeaturesExtractor"
            ) from _SB3_IMPORT_ERROR

        required = {"local_map", "robot_state"}
        missing = required.difference(observation_space.spaces.keys())
        if missing:
            raise ValueError(f"Hybrid observation space missing keys: {sorted(missing)}")
        cfg = encoder_config or HybridLocalGlobalEncoderConfig()
        self._global_sizes = tuple(int(s) for s in cfg.global_sizes)
        global_keys = {f"global_map_{size}" for size in self._global_sizes}
        missing_global = global_keys.difference(observation_space.spaces.keys())
        if missing_global:
            raise ValueError(f"Hybrid observation space missing global keys: {sorted(missing_global)}")
        encoder = HybridLocalGlobalEncoder(cfg)
        super().__init__(observation_space, features_dim=encoder.output_dim)
        self.encoder = encoder

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        global_maps = {
            size: observations[f"global_map_{size}"].float()
            for size in self._global_sizes
        }
        return self.encoder(
            observations["local_map"].float(),
            global_maps,
            observations["robot_state"].float(),
        )
