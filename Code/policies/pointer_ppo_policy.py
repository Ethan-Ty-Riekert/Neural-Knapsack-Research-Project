"""pointer_ppo_policy.py

sb3_contrib-compatible policy wrapper around PointerActorCritic (see
pointer_policy.py), so MaskablePPO can use the same pointer/attention
architecture A2C's hand-rolled training loop has used since 2026-08-09.
Motivation: see Future/research/2026-09-16-pointer-network-ppo.md -- every
mechanism tried this project for PPO's persistently poor tardiness (fixed
reward weight, four Lagrangian lambda_max ceilings, a tardiness-directed
hyperparameter search) converged on the same ~1290-1330 band, while A2C
(pointer architecture) sits at ~28. This is the untested remaining variable.

MaskablePPO only calls four methods on its policy during rollout/training
(see sb3_contrib/ppo_mask/ppo_mask.py): forward() (rollout action+value+
logprob), predict_values() (bootstrap value at truncation), evaluate_actions()
(the PPO loss), and -- for inference via the inherited predict() --
get_distribution()/_predict(). This class overrides exactly those four, plus
_build()/_get_constructor_parameters(), and leaves everything else
(obs_to_tensor, predict(), save/load plumbing) to MaskableActorCriticPolicy
unchanged, so this is a drop-in `policy_class` for MaskablePPO the same way
"MlpPolicy" is.
"""

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from Code.policies.pointer_policy import PointerActorCritic


class PointerMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Drop-in MaskablePPO policy using PointerActorCritic instead of a flat MLP.

    Pass via policy_kwargs: max_jobs, num_machines, num_resources are required
    (must match the env's dimensions -- see train_optimized.py, which reads
    them off the built VecEnv via get_attr() rather than hardcoding); embed_dim/
    hidden/clip_c are optional, same defaults as PointerActorCritic itself.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        max_jobs: int,
        num_machines: int,
        num_resources: int,
        embed_dim: int = 128,
        hidden: int = 64,
        clip_c: float = 10.0,
        **kwargs: Any,
    ):
        # Stashed before super().__init__() so _build() (called from within
        # it) can use them.
        self._pointer_kwargs = dict(
            max_jobs=max_jobs,
            num_machines=num_machines,
            num_resources=num_resources,
            embed_dim=embed_dim,
            hidden=hidden,
            clip_c=clip_c,
        )
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """Overridden: build PointerActorCritic instead of the parent's
        mlp_extractor/action_net/value_net, and construct the optimizer over
        exactly its parameters. self.features_extractor (built by __init__
        before this runs) is a parameterless FlattenExtractor for this
        project's flat Box observation space, so never routing anything
        through it costs nothing and keeps it out of the optimizer for free."""
        self.pointer_net = PointerActorCritic(**self._pointer_kwargs)
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
