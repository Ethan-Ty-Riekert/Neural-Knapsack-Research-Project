"""priority_pointer_ppo_policy.py - sb3_contrib-compatible policy wrapper
around PriorityPointerActorCritic (see priority_pointer_policy.py), the
Options 2/3 network for the 2026-09-17 action-space-reduction work. Mirrors
Code/policies/pointer_ppo_policy.py's PointerMaskableActorCriticPolicy
exactly (same four MaskablePPO-called methods overridden, for the same
reasons -- see that file's module docstring), differing only in which
network class it builds and which constructor kwargs it forwards.
"""
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from Code.policies.priority_pointer_policy import PriorityPointerActorCritic


class PriorityPointerMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Drop-in MaskablePPO policy using PriorityPointerActorCritic instead of
    a flat MLP. Pass via policy_kwargs: max_jobs, num_machines, num_resources
    must match the env's dimensions; use_atc must match the wrapped
    PriorityOnlyGymSchedulingEnv's own use_atc (Option 2 vs Option 3)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        max_jobs: int,
        num_machines: int,
        num_resources: int,
        use_atc: bool = False,
        embed_dim: int = 128,
        hidden: int = 64,
        **kwargs: Any,
    ):
        self._pointer_kwargs = dict(
            max_jobs=max_jobs,
            num_machines=num_machines,
            num_resources=num_resources,
            use_atc=use_atc,
            embed_dim=embed_dim,
            hidden=hidden,
        )
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self.pointer_net = PriorityPointerActorCritic(**self._pointer_kwargs)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        for key in ("net_arch", "activation_fn", "ortho_init"):
            data.pop(key, None)
        data.update(self._pointer_kwargs)
        return data

    def _dist_and_value(self, obs: PyTorchObs) -> tuple[MaskableDistribution, th.Tensor]:
        logits, values = self.pointer_net(obs)
        return self.action_dist.proba_distribution(action_logits=logits), values

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        distribution, values = self._dist_and_value(obs)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        distribution, values = self._dist_and_value(obs)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()

    def get_distribution(
        self, obs: PyTorchObs, action_masks: np.ndarray | None = None
    ) -> MaskableDistribution:
        distribution, _ = self._dist_and_value(obs)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        _, values = self.pointer_net(obs)
        return values

    def _predict(  # type: ignore[override]
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> th.Tensor:
        return self.get_distribution(observation, action_masks).get_actions(deterministic=deterministic)
