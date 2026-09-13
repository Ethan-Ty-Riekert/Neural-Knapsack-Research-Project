"""train_a2c.py - Standalone A2C training script

A2C (Advantage Actor-Critic) doesn't suffer from PPO's gradient clipping issues,
making it potentially more suitable for large discrete action spaces.
"""

import os
import numpy as np
import argparse
import torch

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.env_config import generate_env_config
from Code.policies.a2c_policy import make_maskable_a2c, train_a2c
from Code.utils.plotting_utils import make_run_dir, LiveTrainingPlotter
from Code.utils.paths import PLOTS_DIR, ENV_CONFIG_A2C_PATH, A2C_MODEL_PATH, ensure_rl_training_dirs
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: GymSchedulingEnv):
    """Action mask function for ActionMasker"""
    return env.get_action_mask()


def make_env(
    seed: int = 0,
    num_jobs=None,
    num_machines=None,
    horizon=None,
    max_jobs=None,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    lambda_3: float = 1.0,
    invalid_penalty: float = 5.0,
    idle_penalty: float = 0.5,
):
    """Environment creation for A2C training."""

    config = generate_env_config(seed=seed)

    # Curriculum overrides
    if num_jobs is not None:
        config["job_durations"] = config["job_durations"][:num_jobs]
        config["job_resources"] = config["job_resources"][:num_jobs, :]
        config["job_deadlines"] = config["job_deadlines"][:num_jobs]
        config["job_weights"] = config["job_weights"][:num_jobs]
        config["num_jobs"] = num_jobs

    if num_machines is not None:
        config["num_machines"] = num_machines

    if horizon is not None:
        config["horizon"] = horizon

    if max_jobs is not None:
        config["max_jobs"] = max_jobs

    # Save config
    ensure_rl_training_dirs()
    np.savez(ENV_CONFIG_A2C_PATH, **config)

    # Create environment
    base_env = SchedulingEnv(
        job_durations=config["job_durations"],
        job_resources=config["job_resources"],
        job_deadlines=config["job_deadlines"],
        job_weights=config["job_weights"],
        num_machines=config["num_machines"],
        machine_capacity=config["machine_capacity"],
        horizon=config["horizon"],
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        invalid_penalty=invalid_penalty,
        idle_penalty=idle_penalty,
    )

    gym_env = GymSchedulingEnv(base_env, max_jobs=max_jobs)
    masked_env = ActionMasker(gym_env, mask_fn)
    monitored_env = Monitor(masked_env)

    return monitored_env


def main():
    # Configuration
    TOTAL_TIMESTEPS = 500_000
    ensure_rl_training_dirs()

    # Live plotter
    plot_run_dir = make_run_dir(str(PLOTS_DIR / "training"), "a2c")
    plotter = LiveTrainingPlotter(save_dir=plot_run_dir)

    # Curriculum learning stages
    MAX_JOBS = 100
    curriculum = [
        {"horizon": 20,  "num_jobs": 15,  "timesteps": 50_000},
        {"horizon": 40,  "num_jobs": 30,  "timesteps": 100_000},
        {"horizon": 60,  "num_jobs": 60,  "timesteps": 150_000},
        {"horizon": 100, "num_jobs": 100, "timesteps": 200_000},
    ]

    print(f"\n{'='*80}")
    print("Training A2C with Curriculum Learning")
    print(f"Total stages: {len(curriculum)}")
    print(f"Total timesteps: {sum(stage['timesteps'] for stage in curriculum)}")
    print(f"{'='*80}\n")

    # Create first environment
    first_env = make_env(
        seed=0,
        horizon=curriculum[0]["horizon"],
        num_jobs=curriculum[0]["num_jobs"],
        max_jobs=MAX_JOBS,
    )

    # Create A2C agent
    agent = make_maskable_a2c(first_env, device="cpu")

    print("A2C Agent Configuration:")
    print(f"  n_steps: {agent.n_steps}")
    print(f"  learning_rate: {agent.lr}")
    print(f"  gamma: {agent.gamma}")
    print(f"  gae_lambda: {agent.lam}")
    print(f"  ent_coef: {agent.ent_coef}")
    print(f"  value_coef: {agent.value_coef}")
    print(f"  max_grad_norm: {agent.max_grad_norm}\n")

    # Curriculum training loop
    for i, stage in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"Stage {i+1}/{len(curriculum)}")
        print(f"  Horizon: {stage['horizon']}")
        print(f"  Jobs: {stage['num_jobs']}")
        print(f"  Timesteps: {stage['timesteps']}")
        print(f"{'='*60}\n")

        # Create environment for this stage
        env = make_env(
            seed=0,
            horizon=stage["horizon"],
            num_jobs=stage["num_jobs"],
            max_jobs=MAX_JOBS,
        )

        # Update agent's environment
        agent.env = env

        # Train
        train_a2c(agent, total_timesteps=stage["timesteps"], plotter=plotter)

        print(f"\nStage {i+1} complete!\n")

    # Save model
    torch.save(agent.model.state_dict(), A2C_MODEL_PATH)
    print(f"\n{'='*80}")
    print(f"A2C training complete!")
    print(f"Model saved to: {A2C_MODEL_PATH}")
    print(f"Training plots saved to: {plot_run_dir}")
    print(f"{'='*80}\n")

    plotter.close()


if __name__ == "__main__":
    main()
