from __future__ import annotations

from typing import Dict, Generator, NamedTuple, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported


class HeuristicDictRolloutBufferSamples(NamedTuple):
    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    actor_train_mask: th.Tensor


class HeuristicDictRolloutBuffer(DictRolloutBuffer):
    """Dict rollout buffer with an extra per-step mask for actor updates."""

    def reset(self) -> None:
        super().reset()
        self.actor_train_mask = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        actor_train_mask: Optional[np.ndarray] = None,
    ) -> None:
        pos = self.pos
        super().add(obs, action, reward, episode_start, value, log_prob)
        if actor_train_mask is None:
            actor_train_mask = np.ones((self.n_envs,), dtype=np.float32)
        actor_train_mask = np.asarray(actor_train_mask, dtype=np.float32).reshape(self.n_envs)
        self.actor_train_mask[pos] = actor_train_mask

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[HeuristicDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in ["actions", "values", "log_probs", "advantages", "returns"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.actor_train_mask = self.swap_and_flatten(self.actor_train_mask)
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        total = self.buffer_size * self.n_envs
        while start_idx < total:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> HeuristicDictRolloutBufferSamples:
        return HeuristicDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            actor_train_mask=self.to_torch(self.actor_train_mask[batch_inds].flatten()),
        )


class HeuristicAwarePPO(PPO):
    """
    PPO variant that can exclude heuristic-overridden steps from actor/entropy loss
    while still learning the critic from all resulting transitions.
    """

    def __init__(self, *args, exclude_heuristic_actor_steps: bool = False, **kwargs):
        self.exclude_heuristic_actor_steps = bool(exclude_heuristic_actor_steps)
        if self.exclude_heuristic_actor_steps:
            kwargs.setdefault("rollout_buffer_class", HeuristicDictRolloutBuffer)
        super().__init__(*args, **kwargs)
        if self.exclude_heuristic_actor_steps and not isinstance(self.rollout_buffer, HeuristicDictRolloutBuffer):
            raise TypeError("exclude_heuristic_actor_steps requires HeuristicDictRolloutBuffer")

    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: DictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if not self.exclude_heuristic_actor_steps:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        assert self._last_obs is not None, "No previous observation was provided"
        assert isinstance(rollout_buffer, HeuristicDictRolloutBuffer)

        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            actor_train_mask = np.ones((env.num_envs,), dtype=np.float32)
            for idx, info in enumerate(infos):
                if (
                    dones[idx]
                    and info.get("terminal_observation") is not None
                    and info.get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(info["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
                if bool(info.get("heuristic_action_used", False)):
                    actor_train_mask[idx] = 0.0

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                actor_train_mask=actor_train_mask,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:  # type: ignore[override]
        if not self.exclude_heuristic_actor_steps:
            return super().train()

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        actor_mask_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                actor_mask = rollout_data.actor_train_mask > 0.5
                actor_mask_fractions.append(actor_mask.float().mean().item())

                if actor_mask.any():
                    masked_advantages = advantages[actor_mask]
                    masked_ratio = ratio[actor_mask]
                    policy_loss_1 = masked_advantages * masked_ratio
                    policy_loss_2 = masked_advantages * th.clamp(masked_ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    clip_fraction = th.mean((th.abs(masked_ratio - 1) > clip_range).float()).item()

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob[actor_mask])
                    else:
                        entropy_loss = -th.mean(entropy[actor_mask])

                    with th.no_grad():
                        log_ratio = log_prob[actor_mask] - rollout_data.old_log_prob[actor_mask]
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)
                else:
                    policy_loss = ratio.sum() * 0.0
                    clip_fraction = 0.0
                    entropy_loss = log_prob.sum() * 0.0

                pg_losses.append(policy_loss.item())
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if actor_mask.any() and self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if len(approx_kl_divs) > 0 else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/actor_mask_fraction", np.mean(actor_mask_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class HeuristicMaskableDictRolloutBufferSamples(NamedTuple):
    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor
    actor_train_mask: th.Tensor
    teacher_actions: th.Tensor
    teacher_valid_mask: th.Tensor


class HeuristicMaskableDictRolloutBuffer(MaskableDictRolloutBuffer):
    """Maskable dict rollout buffer with actor/train masks and teacher actions."""

    def reset(self) -> None:
        super().reset()
        self.actor_train_mask = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        self.teacher_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.teacher_valid_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, *args, actor_train_mask: Optional[np.ndarray] = None, teacher_actions: Optional[np.ndarray] = None, teacher_valid_mask: Optional[np.ndarray] = None, **kwargs) -> None:  # type: ignore[override]
        pos = self.pos
        super().add(*args, **kwargs)
        if actor_train_mask is None:
            actor_train_mask = np.ones((self.n_envs,), dtype=np.float32)
        if teacher_actions is None:
            teacher_actions = np.zeros((self.n_envs, self.action_dim), dtype=np.float32)
        if teacher_valid_mask is None:
            teacher_valid_mask = np.zeros((self.n_envs,), dtype=np.float32)
        self.actor_train_mask[pos] = np.asarray(actor_train_mask, dtype=np.float32).reshape(self.n_envs)
        self.teacher_actions[pos] = np.asarray(teacher_actions, dtype=np.float32).reshape((self.n_envs, self.action_dim))
        self.teacher_valid_mask[pos] = np.asarray(teacher_valid_mask, dtype=np.float32).reshape(self.n_envs)

    def get(self, batch_size: Optional[int] = None) -> Generator[HeuristicMaskableDictRolloutBufferSamples, None, None]:  # type: ignore[override]
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)
            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
                "actor_train_mask",
                "teacher_actions",
                "teacher_valid_mask",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        total = self.buffer_size * self.n_envs
        while start_idx < total:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> HeuristicMaskableDictRolloutBufferSamples:  # type: ignore[override]
        return HeuristicMaskableDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
            actor_train_mask=self.to_torch(self.actor_train_mask[batch_inds].flatten()),
            teacher_actions=self.to_torch(self.teacher_actions[batch_inds].astype(np.float32, copy=False)),
            teacher_valid_mask=self.to_torch(self.teacher_valid_mask[batch_inds].flatten()),
        )


class HeuristicMaskablePPO(MaskablePPO):
    """
    Maskable PPO variant for heuristic override experiments.
    - exclude mode: heuristic-executed steps are removed from actor/entropy loss
    - bc mode: same exclusion, plus BC loss toward executed heuristic action
    """

    def __init__(
        self,
        *args,
        exclude_heuristic_actor_steps: bool = False,
        heuristic_bc_coef: float = 0.0,
        **kwargs,
    ):
        self.exclude_heuristic_actor_steps = bool(exclude_heuristic_actor_steps)
        self.heuristic_bc_coef = float(heuristic_bc_coef)
        if self.exclude_heuristic_actor_steps or self.heuristic_bc_coef > 0.0:
            kwargs.setdefault("rollout_buffer_class", HeuristicMaskableDictRolloutBuffer)
        super().__init__(*args, **kwargs)
        if (self.exclude_heuristic_actor_steps or self.heuristic_bc_coef > 0.0) and not isinstance(
            self.rollout_buffer, HeuristicMaskableDictRolloutBuffer
        ):
            raise TypeError("Heuristic masked training requires HeuristicMaskableDictRolloutBuffer")

    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: DictRolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        if not (self.exclude_heuristic_actor_steps or self.heuristic_bc_coef > 0.0):
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps, use_masking=use_masking)

        assert isinstance(rollout_buffer, HeuristicMaskableDictRolloutBuffer), "RolloutBuffer doesn't support heuristic masking"
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            actor_train_mask = np.ones((env.num_envs,), dtype=np.float32)
            teacher_actions = np.zeros((env.num_envs, rollout_buffer.action_dim), dtype=np.float32)
            teacher_valid_mask = np.zeros((env.num_envs,), dtype=np.float32)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

                if bool(infos[idx].get("heuristic_action_used", False)):
                    actor_train_mask[idx] = 0.0
                    teacher_valid_mask[idx] = 1.0
                    teacher_actions[idx, 0] = float(int(infos[idx].get("action_executed", actions[idx, 0])))

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
                actor_train_mask=actor_train_mask,
                teacher_actions=teacher_actions,
                teacher_valid_mask=teacher_valid_mask,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:  # type: ignore[override]
        if not (self.exclude_heuristic_actor_steps or self.heuristic_bc_coef > 0.0):
            return super().train()

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, bc_losses = [], [], []
        clip_fractions = []
        actor_mask_fractions = []
        teacher_mask_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                teacher_actions = rollout_data.teacher_actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                    teacher_actions = rollout_data.teacher_actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                actor_mask = rollout_data.actor_train_mask > 0.5
                teacher_mask = rollout_data.teacher_valid_mask > 0.5
                actor_mask_fractions.append(actor_mask.float().mean().item())
                teacher_mask_fractions.append(teacher_mask.float().mean().item())

                if actor_mask.any():
                    masked_advantages = advantages[actor_mask]
                    masked_ratio = ratio[actor_mask]
                    policy_loss_1 = masked_advantages * masked_ratio
                    policy_loss_2 = masked_advantages * th.clamp(masked_ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    clip_fraction = th.mean((th.abs(masked_ratio - 1) > clip_range).float()).item()

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob[actor_mask])
                    else:
                        entropy_loss = -th.mean(entropy[actor_mask])

                    with th.no_grad():
                        log_ratio = log_prob[actor_mask] - rollout_data.old_log_prob[actor_mask]
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)
                else:
                    policy_loss = ratio.sum() * 0.0
                    clip_fraction = 0.0
                    entropy_loss = log_prob.sum() * 0.0

                if teacher_mask.any() and self.heuristic_bc_coef > 0.0:
                    _, teacher_log_prob, _ = self.policy.evaluate_actions(
                        rollout_data.observations,
                        teacher_actions,
                        action_masks=rollout_data.action_masks,
                    )
                    bc_loss = -teacher_log_prob[teacher_mask].mean()
                else:
                    bc_loss = ratio.sum() * 0.0

                pg_losses.append(policy_loss.item())
                clip_fractions.append(clip_fraction)
                bc_losses.append(bc_loss.item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if self.heuristic_bc_coef > 0.0:
                    loss = loss + self.heuristic_bc_coef * bc_loss

                if actor_mask.any() and self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if len(approx_kl_divs) > 0 else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/actor_mask_fraction", np.mean(actor_mask_fractions))
        self.logger.record("train/teacher_mask_fraction", np.mean(teacher_mask_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
