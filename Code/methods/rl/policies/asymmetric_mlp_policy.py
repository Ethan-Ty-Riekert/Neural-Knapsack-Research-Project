"""asymmetric_mlp_policy.py - MaskablePPO MLP policy whose actor cannot see the critic-only block
(2026-10-06).

The observation's last `critic_dim` entries are the critic-only summary of future arrivals
(Code/core/critic_input.py, an input-dependent baseline: Mao et al. 2019; asymmetric actor-critic:
Pinto et al. 2018). With separate feature extractors (share_features_extractor=False) SB3 routes the
actor through pi_features_extractor and the critic through vf_features_extractor; here the actor's
extractor zeroes that block, so pi(a | obs) depends on the state alone. A zero input contributes nothing
to the first layer, so the actor's output does not depend on the block for any parameter values.
Everything else is MaskableActorCriticPolicy ("MlpPolicy") unchanged.
"""
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ActorBlindExtractor(BaseFeaturesExtractor):
    """Flatten, with the trailing critic-only block set to zero."""

    def __init__(self, observation_space: spaces.Box, critic_dim: int):
        super().__init__(observation_space, int(observation_space.shape[0]))
        self.critic_dim = critic_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = th.flatten(observations, start_dim=1)
        keep = x[:, :x.shape[1] - self.critic_dim]
        return th.cat([keep, th.zeros_like(x[:, keep.shape[1]:])], dim=1)


class AsymmetricMlpPolicy(MaskableActorCriticPolicy):
    """MlpPolicy with the critic-only block hidden from the actor. critic_dim: size of that block."""

    def __init__(self, observation_space, action_space, lr_schedule, critic_dim: int = 0, **kwargs):
        self.critic_dim = critic_dim
        kwargs["share_features_extractor"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # The default extractor (Flatten) has no parameters, so swapping the actor's after the optimizer
        # was built changes no trainable parameter.
        self.features_extractor = self.pi_features_extractor = ActorBlindExtractor(observation_space, critic_dim)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(critic_dim=self.critic_dim)
        return data
