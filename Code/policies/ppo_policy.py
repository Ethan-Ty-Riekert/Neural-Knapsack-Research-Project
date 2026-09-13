"""ppo_policy.py
Proximal Policy Optimisation implementation using sb3_contrib"""

import os
from typing import Dict, Any
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker # for support masking
import numpy as np

def mask_fn(env):
    """Get the action-masked environment"""
    return env.get_action_mask()

def make_maskable_ppo(env, log_dir: str = "./rl_training/logs", policy_kwargs: Dict[str, Any] = None, seed: int = 0, **ppo_kwargs):
    """create a MaskablePPO model with sb3_contrib module with sensible defaults and optional policy kwargs.
    env: wrapped GymSchedulingEnv
    Returns: MaskablePPO instance"""
    os.makedirs(log_dir, exist_ok=True)
    # Wrap with ActionMasker
    masked_env = ActionMasker(env, mask_fn)

    # Default network architecture
    # pi: actor network
    # vf: critic network
    # Both typically use two hidden layers of size 256
    default_policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn = None
    )

    # Allow user overrides
    if policy_kwargs:
        default_policy_kwargs.update(policy_kwargs)

    # Default PPO hyperparameters (as a starting point)
    default_ppo = dict(
        n_steps=2048,            # rollout length
        batch_size=256,          # minibatch size
        learning_rate=3e-4,      # Adam LR
        gamma=0.99,              # discount factor
        gae_lambda=0.95,         # GAE smoothing
        ent_coef=0.01,           # entropy bonus (encourages exploration)
        clip_range=0.2,          # PPO clipping
        n_epochs=10,             # gradient updates per rollout
        verbose=1,
        seed=seed,
        tensorboard_log=log_dir,
        policy_kwargs=default_policy_kwargs,
    )

    # Merge with user-provided PPO kwargs if given
    default_policy_kwargs.update(ppo_kwargs)

    # Create the PPO model to return
    model = MaskablePPO("MlpPolicy", masked_env, **default_ppo) # unload default ppo hyperparameters

    return model

def train_ppo(model, total_timesteps: int = 300000, tb_name: str = "ppo_scheduling"):
    """ Train the PPO agent.
    
    Inputs
    model: MaskablePPO - The agent created make_maskable_ppo()
    
    tb_name: str - TensorBoard run name

    Returns
    model: MaskablePPO - The trained model """

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=tb_name,
        progress_bar=True
    )
    return model