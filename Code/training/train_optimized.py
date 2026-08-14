"""train_optimized.py - Train RL agent using Optuna-optimized hyperparameters

This script loads the best hyperparameters found by Optuna and trains
a full-scale model with curriculum learning.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.env_config import generate_env_config
from Code.utils.plotting_utils import make_run_dir, LiveTrainingPlotter
from Code.utils.paths import LOG_DIR, MODELS_DIR, PLOTS_DIR, OPTUNA_RESULTS_DIR, ENV_CONFIG_PATH, ensure_rl_training_dirs


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
    """Environment creation with tunable reward penalties."""

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
        config["machine_capacity"] = config["machine_capacity"]

    if horizon is not None:
        config["horizon"] = horizon

    if max_jobs is not None:
        config["max_jobs"] = max_jobs

    # Save config for evaluation
    ensure_rl_training_dirs()
    np.savez(ENV_CONFIG_PATH, **config)

    # Create the base environment with tunable penalties
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

    # Wrap in Gym + Masking
    gym_env = GymSchedulingEnv(base_env, max_jobs=max_jobs)
    masked_env = ActionMasker(gym_env, mask_fn)
    monitored_env = Monitor(masked_env)

    return monitored_env


def load_best_params(algorithm: str = "ppo", policy_type: str = "pointer"):
    """Load best hyperparameters from Optuna optimization.

    file_tag mirrors optuna_tune.py::run_optimization()'s output naming: A2C
    saves per-architecture files ("a2c_pointer_best_params.json",
    "a2c_flat_best_params.json"); PPO only has one architecture ("ppo_best_params.json").
    """
    file_tag = f"{algorithm}_{policy_type}" if algorithm == "a2c" else algorithm
    best_params_file = OPTUNA_RESULTS_DIR / f"{file_tag}_best_params.json"

    if not os.path.exists(best_params_file):
        raise FileNotFoundError(
            f"Best parameters file not found: {best_params_file}\n"
            f"Please run optuna_tune.py first to optimize hyperparameters."
        )

    with open(best_params_file, "r") as f:
        params = json.load(f)

    print(f"\nLoaded best hyperparameters from: {best_params_file}")
    print("\nHyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    return params


def train_with_optimized_params(algorithm: str = "ppo", policy_type: str = "pointer", use_curriculum: bool = True):
    """Train agent using optimized hyperparameters from Optuna.

    Args:
        algorithm: "ppo" or "a2c"
        policy_type: "pointer" or "flat" -- A2C only, ignored for PPO. Must match
            which architecture's best-params file to load AND which architecture
            to actually build (see the BUG FIX comment in the A2C branch below).
        use_curriculum: Whether to use curriculum learning
    """

    # Load best hyperparameters
    params = load_best_params(algorithm, policy_type=policy_type)

    # Setup directories. A2C paths are tagged with policy_type -- flat and
    # pointer checkpoints have different state_dict shapes and are not
    # interchangeable, so they must not overwrite each other.
    TOTAL_TIMESTEPS = 500_000 if use_curriculum else 300_000
    PPO_MODEL_PATH = MODELS_DIR / "ppo_scheduling_optimized"
    A2C_MODEL_PATH = MODELS_DIR / f"a2c_{policy_type}_scheduling_optimized.pt"

    ensure_rl_training_dirs()

    # Live plotter
    plot_run_dir = make_run_dir(str(PLOTS_DIR / "training"), f"{algorithm}_optimized")
    plotter = LiveTrainingPlotter(save_dir=plot_run_dir)

    # Extract reward penalties from params
    lambda_1 = params.get("lambda_1", 1.0)
    lambda_2 = params.get("lambda_2", 1.0)
    lambda_3 = params.get("lambda_3", 1.0)
    idle_penalty = params.get("idle_penalty", 0.5)
    invalid_penalty = params.get("invalid_penalty", 5.0)

    # BUG FIX (this session): the Greek-subscript characters here crash with
    # UnicodeEncodeError whenever stdout isn't a UTF-8-capable console --
    # notably when piped/redirected on Windows, where the default codepage is
    # cp1252. Plain ASCII avoids depending on the caller's console encoding.
    print(f"\nReward penalties:")
    print(f"  Machine activation (lambda_1): {lambda_1}")
    print(f"  Tardiness (lambda_2): {lambda_2}")
    print(f"  Hotspot (lambda_3): {lambda_3}")
    print(f"  Idle penalty: {idle_penalty}")
    print(f"  Invalid penalty: {invalid_penalty}\n")

    # ==================== PPO TRAINING ====================
    if algorithm == "ppo":
        # Build network architecture
        layer_size = params.get("layer_size", 256)
        n_layers = params.get("n_layers", 2)
        activation = params.get("activation", "tanh")

        if n_layers == 2:
            net_arch = dict(pi=[layer_size, layer_size], vf=[layer_size, layer_size])
        else:
            net_arch = dict(
                pi=[layer_size, layer_size, layer_size],
                vf=[layer_size, layer_size, layer_size]
            )

        activation_fn = nn.Tanh if activation == "tanh" else nn.ReLU

        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
        )

        # Curriculum or single-stage training
        if use_curriculum:
            MAX_JOBS = 100
            curriculum = [
                {"horizon": 20,  "num_jobs": 15,  "timesteps": 50_000},
                {"horizon": 40,  "num_jobs": 30,  "timesteps": 100_000},
                {"horizon": 60,  "num_jobs": 60,  "timesteps": 150_000},
                {"horizon": 100, "num_jobs": 100, "timesteps": 200_000},
            ]
        else:
            MAX_JOBS = 100
            curriculum = [
                {"horizon": 100, "num_jobs": 100, "timesteps": TOTAL_TIMESTEPS},
            ]

        # Create first environment
        first_env = make_env(
            seed=0,
            horizon=curriculum[0]["horizon"],
            num_jobs=curriculum[0]["num_jobs"],
            max_jobs=MAX_JOBS,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            idle_penalty=idle_penalty,
            invalid_penalty=invalid_penalty,
        )

        # Create PPO model with optimized hyperparameters
        model = MaskablePPO(
            "MlpPolicy",
            first_env,
            verbose=1,
            tensorboard_log=str(LOG_DIR),
            learning_rate=params.get("learning_rate", 3e-4),
            n_steps=params.get("n_steps", 2048),
            batch_size=params.get("batch_size", 256),
            n_epochs=params.get("n_epochs", 4),
            gamma=params.get("gamma", 0.99),
            gae_lambda=params.get("gae_lambda", 0.95),
            ent_coef=params.get("ent_coef", 0.05),
            clip_range=params.get("clip_range", 0.2),
            vf_coef=params.get("vf_coef", 0.5),
            max_grad_norm=params.get("max_grad_norm", 0.5),
            seed=0,
            policy_kwargs=policy_kwargs,
        )

        print(f"\n{'='*80}")
        print(f"Training PPO with optimized hyperparameters")
        print(f"Curriculum stages: {len(curriculum)}")
        print(f"Total timesteps: {sum(stage['timesteps'] for stage in curriculum)}")
        print(f"{'='*80}\n")

        # Curriculum training loop
        for i, stage in enumerate(curriculum):
            print(f"\n--- Stage {i+1}/{len(curriculum)} ---")
            print(f"  Horizon: {stage['horizon']}, Jobs: {stage['num_jobs']}, Timesteps: {stage['timesteps']}")

            env = make_env(
                seed=0,
                horizon=stage["horizon"],
                num_jobs=stage["num_jobs"],
                max_jobs=MAX_JOBS,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                lambda_3=lambda_3,
                idle_penalty=idle_penalty,
                invalid_penalty=invalid_penalty,
            )

            model.set_env(env)

            model.learn(
                total_timesteps=stage["timesteps"],
                tb_log_name="ppo_scheduling_optimized",
                progress_bar=True,
                callback=plotter,
                reset_num_timesteps=False,
            )

            # Per-stage checkpoint -- see the matching comment in
            # train_rl_agent.py for rationale (diagnosability without rerunning
            # the whole curriculum).
            stage_ckpt = MODELS_DIR / f"ppo_optimized_stage{i}_h{stage['horizon']}"
            model.save(stage_ckpt)
            print(f"  Saved stage checkpoint: {stage_ckpt}")

        # Save PPO model
        model.save(PPO_MODEL_PATH)
        print(f"\nPPO training complete. Model saved to: {PPO_MODEL_PATH}\n")

    # ==================== A2C TRAINING ====================
    else:
        from Code.policies.a2c_policy import MaskableA2C

        # For A2C, we need to modify the class to accept hyperparameters
        # This is similar to what we did in optuna_tune.py

        if use_curriculum:
            MAX_JOBS = 100
            curriculum = [
                {"horizon": 20,  "num_jobs": 15,  "timesteps": 50_000},
                {"horizon": 40,  "num_jobs": 30,  "timesteps": 100_000},
                {"horizon": 60,  "num_jobs": 60,  "timesteps": 150_000},
                {"horizon": 100, "num_jobs": 100, "timesteps": 200_000},
            ]
        else:
            MAX_JOBS = 100
            curriculum = [
                {"horizon": 100, "num_jobs": 100, "timesteps": TOTAL_TIMESTEPS},
            ]

        # Create first environment
        first_env = make_env(
            seed=0,
            horizon=curriculum[0]["horizon"],
            num_jobs=curriculum[0]["num_jobs"],
            max_jobs=MAX_JOBS,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            idle_penalty=idle_penalty,
            invalid_penalty=invalid_penalty,
        )

        # Create A2C agent.
        # BUG FIX (this session): this used to always call
        # MaskableA2C(first_env, device="cpu") with no policy_type/policy_kwargs,
        # so it silently ignored policy_type entirely (always built the default
        # "pointer" architecture with PointerActorCritic's hardcoded
        # embed_dim=128/hidden=64) and any embed_dim/hidden Optuna actually found
        # were never applied. Now explicitly threads both through.
        policy_kwargs = (
            dict(embed_dim=params.get("embed_dim", 128), hidden=params.get("hidden", 64))
            if policy_type == "pointer" else None
        )
        agent = MaskableA2C(first_env, device="cpu", policy_type=policy_type, policy_kwargs=policy_kwargs)

        # Override hyperparameters
        agent.n_steps = params.get("n_steps", 5)
        agent.gamma = params.get("gamma", 0.99)
        agent.lam = params.get("gae_lambda", 1.0)
        agent.ent_coef = params.get("ent_coef", 0.0)
        agent.value_coef = params.get("value_coef", 0.5)
        agent.max_grad_norm = params.get("max_grad_norm", 0.5)
        agent.lr = params.get("learning_rate", 7e-4)

        # Rebuild optimizer with new learning rate
        agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.lr)

        print(f"\n{'='*80}")
        print(f"Training A2C with optimized hyperparameters")
        print(f"Curriculum stages: {len(curriculum)}")
        print(f"Total timesteps: {sum(stage['timesteps'] for stage in curriculum)}")
        print(f"{'='*80}\n")

        # Curriculum training loop
        for i, stage in enumerate(curriculum):
            print(f"\n--- Stage {i+1}/{len(curriculum)} ---")
            print(f"  Horizon: {stage['horizon']}, Jobs: {stage['num_jobs']}, Timesteps: {stage['timesteps']}")

            env = make_env(
                seed=0,
                horizon=stage["horizon"],
                num_jobs=stage["num_jobs"],
                max_jobs=MAX_JOBS,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                lambda_3=lambda_3,
                idle_penalty=idle_penalty,
                invalid_penalty=invalid_penalty,
            )

            agent.env = env
            agent.train(total_timesteps=stage["timesteps"], plotter=plotter)

            # Per-stage checkpoint -- see the matching comment in
            # train_rl_agent.py for rationale.
            stage_ckpt = MODELS_DIR / f"a2c_{policy_type}_optimized_stage{i}_h{stage['horizon']}.pt"
            torch.save(agent.model.state_dict(), stage_ckpt)
            print(f"  Saved stage checkpoint: {stage_ckpt}")

        # Save A2C model
        torch.save(agent.model.state_dict(), A2C_MODEL_PATH)
        print(f"\nA2C training complete. Model saved to: {A2C_MODEL_PATH}\n")

    plotter.close()
    print(f"Training reward plot saved to: {plot_run_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL agent with Optuna-optimized hyperparameters"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "a2c"],
        help="Algorithm to train"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="pointer",
        choices=["pointer", "flat"],
        help="A2C only: which architecture's best-params file to load and build.",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning (single-stage training)"
    )

    args = parser.parse_args()

    train_with_optimized_params(
        algorithm=args.algo,
        policy_type=args.policy_type,
        use_curriculum=not args.no_curriculum
    )
