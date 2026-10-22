from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from gymnasium import spaces

from learning.common import (
    FusedMAPSStateEncoder,
    FusedMAPSStateEncoderConfig,
    HybridLocalGlobalEncoder,
    HybridLocalGlobalEncoderConfig,
)

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
except Exception as exc:  # pragma: no cover
    BaseFeaturesExtractor = object  # type: ignore[assignment]
    MlpExtractor = object  # type: ignore[assignment]
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None

try:
    from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
except Exception as exc:  # pragma: no cover
    RecurrentMultiInputActorCriticPolicy = object  # type: ignore[assignment]
    RNNStates = None  # type: ignore[assignment]
    _SB3_CONTRIB_IMPORT_ERROR = exc
else:
    _SB3_CONTRIB_IMPORT_ERROR = None


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
        self.recurrent_features_dim = int(getattr(encoder, "recurrent_output_dim", encoder.output_dim))
        self.bypass_dim = int(getattr(encoder, "bypass_dim", getattr(encoder, "dtm_bypass_dim", 0)))
        self.dtm_bypass_dim = self.bypass_dim

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


class DtmBypassRecurrentMultiInputPolicy(RecurrentMultiInputActorCriticPolicy):  # type: ignore[misc]
    """
    Recurrent multi-input policy that keeps selected hybrid features outside the LSTM.

    The feature extractor must return [recurrent_features, bypass_features].
    Only recurrent_features are fed to actor/critic LSTMs; bypass features are
    concatenated to the LSTM outputs before the policy/value MLP heads.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        if _SB3_CONTRIB_IMPORT_ERROR is not None:
            raise ImportError(
                "sb3_contrib is required for DtmBypassRecurrentMultiInputPolicy"
            ) from _SB3_CONTRIB_IMPORT_ERROR
        self._dtm_bypass_recurrent_dim = 0
        self._dtm_bypass_dim = 0
        super().__init__(*args, **kwargs)
        self._refresh_bypass_dims()
        if self._dtm_bypass_dim <= 0:
            raise ValueError(
                "DtmBypassRecurrentMultiInputPolicy requires a feature extractor with "
                "bypass_dim > 0"
            )

        current_lr = float(self.optimizer.param_groups[0]["lr"])
        lstm_kwargs = self.lstm_kwargs or {}
        n_lstm_layers = int(self.lstm_hidden_state_shape[0])
        lstm_hidden_size = int(self.lstm_output_dim)
        self.lstm_actor = torch.nn.LSTM(
            self._dtm_bypass_recurrent_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **lstm_kwargs,
        )
        if self.enable_critic_lstm:
            self.lstm_critic = torch.nn.LSTM(
                self._dtm_bypass_recurrent_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **lstm_kwargs,
            )
        elif not self.shared_lstm:
            self.critic = torch.nn.Linear(self._dtm_bypass_recurrent_dim, lstm_hidden_size)

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=current_lr,
            **self.optimizer_kwargs,
        )

    def _refresh_bypass_dims(self) -> None:
        extractor = getattr(self, "features_extractor", None)
        recurrent_dim = int(getattr(extractor, "recurrent_features_dim", self.features_dim))
        bypass_dim = int(getattr(extractor, "bypass_dim", getattr(extractor, "dtm_bypass_dim", 0)))
        self._dtm_bypass_recurrent_dim = recurrent_dim
        self._dtm_bypass_dim = bypass_dim

    def _build_mlp_extractor(self) -> None:
        self._refresh_bypass_dims()
        self.mlp_extractor = MlpExtractor(
            int(self.lstm_output_dim) + int(self._dtm_bypass_dim),
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _split_dtm_bypass(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent_dim = int(self._dtm_bypass_recurrent_dim)
        bypass_dim = int(self._dtm_bypass_dim)
        expected_dim = recurrent_dim + bypass_dim
        if int(features.shape[1]) != expected_dim:
            raise ValueError(
                f"Expected feature dim {expected_dim} for hybrid bypass policy, "
                f"got {int(features.shape[1])}"
            )
        return features[:, :recurrent_dim], features[:, recurrent_dim:]

    @staticmethod
    def _append_bypass(latent: torch.Tensor, bypass: torch.Tensor) -> torch.Tensor:
        return torch.cat([latent, bypass], dim=1)

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: Any,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        pi_recurrent, pi_bypass = self._split_dtm_bypass(pi_features)
        vf_recurrent, vf_bypass = self._split_dtm_bypass(vf_features)

        latent_pi, lstm_states_pi = self._process_sequence(
            pi_recurrent, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_recurrent, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_recurrent)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(self._append_bypass(latent_pi, pi_bypass))
        latent_vf = self.mlp_extractor.forward_critic(self._append_bypass(latent_vf, vf_bypass))

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ):
        features = self.extract_features(obs, self.pi_features_extractor)
        if isinstance(features, tuple):
            features = features[0]
        recurrent_features, bypass_features = self._split_dtm_bypass(features)
        latent_pi, lstm_states = self._process_sequence(
            recurrent_features, lstm_states, episode_starts, self.lstm_actor
        )
        latent_pi = self.mlp_extractor.forward_actor(self._append_bypass(latent_pi, bypass_features))
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        features = self.extract_features(obs, self.vf_features_extractor)
        if isinstance(features, tuple):
            features = features[1]
        recurrent_features, bypass_features = self._split_dtm_bypass(features)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                recurrent_features, lstm_states, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_pi, _ = self._process_sequence(
                recurrent_features, lstm_states, episode_starts, self.lstm_actor
            )
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(recurrent_features)

        latent_vf = self.mlp_extractor.forward_critic(self._append_bypass(latent_vf, bypass_features))
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: Any,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        pi_recurrent, pi_bypass = self._split_dtm_bypass(pi_features)
        vf_recurrent, vf_bypass = self._split_dtm_bypass(vf_features)

        latent_pi, _ = self._process_sequence(pi_recurrent, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_recurrent, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_recurrent)

        latent_pi = self.mlp_extractor.forward_actor(self._append_bypass(latent_pi, pi_bypass))
        latent_vf = self.mlp_extractor.forward_critic(self._append_bypass(latent_vf, vf_bypass))

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
