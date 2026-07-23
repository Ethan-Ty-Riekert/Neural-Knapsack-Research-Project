"""train_rl_agent.py - Import the gym wrapper, create the environment, train, save and evaluate the model"""
import os
import numpy as np
import argparse

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO # native action masking to help reduce our massive action space (num_jobs * num_machines) by removing invalid actions
from sb3_contrib.common.wrappers import ActionMasker #

from scheduling_env import SchedulingEnv
from gym_scheduling_wrapper import GymSchedulingEnv
from env_config import generate_env_config
from Policies.a2c_policy import make_maskable_a2c, train_a2c
from Policies.ppo_policy import make_maskable_ppo, train_ppo
from plotting_utils import make_run_dir, LiveTrainingPlotter



def mask_fn(env: GymSchedulingEnv):
    """Action mask function for ActionMasker"""
    return env.get_action_mask()

def make_env(seed: int = 0, num_jobs=None, num_machines=None, horizon=None, max_jobs=None):
    """Environment creation using centralised env_config.py and allowing for curriculum learning.

    num_jobs: actual/logical number of jobs generated for this env instance -- may
    vary freely between curriculum stages.
    max_jobs: fixed job-slot capacity used to size GymSchedulingEnv's observation and
    action spaces (see gym_scheduling_wrapper.py). Must stay constant across every
    stage of a curriculum sharing the same model, regardless of num_jobs, since
    model.set_env() requires matching obs/action spaces. Defaults to num_jobs (no
    padding) when not given.
    """

    # Load environment configuration
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
    np.savez("rl_training/models/env_config.npz", **config)


    # Create the base environment
    base_env = SchedulingEnv(
        job_durations=config["job_durations"],
        job_resources=config["job_resources"],
        job_deadlines=config["job_deadlines"],
        job_weights=config["job_weights"],
        num_machines=config["num_machines"],
        machine_capacity=config["machine_capacity"],
        horizon=config["horizon"],
        lambda_1=1.0,
        lambda_2=1.0,
        lambda_3=1.0,
        invalid_penalty=5.0,
    )

    # Wrap in Gym + Masking
    gym_env = GymSchedulingEnv(base_env, max_jobs=max_jobs)
    masked_env = ActionMasker(gym_env, mask_fn)

    # Monitor records per-episode reward/length into info["episode"], which
    # LiveTrainingPlotter reads to build the live reward curve.
    monitored_env = Monitor(masked_env)

    return monitored_env



############################## Generative AI Made ##############################
# Make a training function for my RL agent given my code below: ... #
def main():
    # -----------------------------
    # Training configuration
    # -----------------------------
    TOTAL_TIMESTEPS = 300_000
    RL_DIR = "./rl_training"
    LOG_DIR = f"{RL_DIR}/logs"
    MODEL_DIR = f"{RL_DIR}/models"
    PPO_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_scheduling")
    A2C_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_scheduling.pt")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -----------------------------
    # Choose which RL algorithm to train
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"])
    args = parser.parse_args()

    USE_PPO = (args.algo == "ppo")

    # Live + saved reward plot for this training run (one instance reused across
    # every curriculum stage below so the curve stays continuous).
    plot_run_dir = make_run_dir(f"{RL_DIR}/plots/training", args.algo)
    plotter = LiveTrainingPlotter(save_dir=plot_run_dir)

    # -----------------------------
    # Curriculum definition
    # num_jobs now varies per stage (kept <= horizon so completing every job --
    # and therefore earning the +50 completion bonus in SchedulingEnv.step() -- is
    # reachable at every stage, not just once horizon catches up to num_jobs).
    #
    # This only works because MAX_JOBS below is passed to every make_env() call as
    # a fixed padding capacity: GymSchedulingEnv sizes its observation/action
    # spaces off max_jobs, not the stage's actual num_jobs, and zero-pads/masks out
    # the unused job slots (see gym_scheduling_wrapper.py). That keeps the obs/
    # action space constant across every stage, which model.set_env() requires --
    # without it, varying num_jobs directly changes those space sizes and
    # set_env() raises "Observation spaces do not match".
    # -----------------------------
    MAX_JOBS = 100
    curriculum = [
        {"horizon": 20,  "num_jobs": 15,  "timesteps": 50_000},
        {"horizon": 40,  "num_jobs": 30,  "timesteps": 75_000},
        {"horizon": 60,  "num_jobs": 60,  "timesteps": 100_000},
        {"horizon": 100, "num_jobs": 100, "timesteps": 150_000},
    ]

    # -----------------------------
    # PPO TRAINING
    # -----------------------------
    if USE_PPO:

        # Create FIRST curriculum environment
        first_env = make_env(
            seed=0,
            horizon=curriculum[0]["horizon"],
            num_jobs=curriculum[0]["num_jobs"],
            max_jobs=MAX_JOBS,
        )

        # Create PPO model WITH FIRST ENV
        model = MaskablePPO(
            "MlpPolicy",
            first_env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            # Raised from 0.01: the policy was collapsing onto "always idle"
            # (entropy/approx_kl/policy_gradient_loss all underflowing to 0)
            # well before it discovered the reward for actually placing jobs.
            # A stronger entropy bonus keeps exploration alive for longer.
            ent_coef=0.05,
            clip_range=0.2,
            # Lowered from 10: 10 gradient epochs over a single rollout risks
            # overfitting to (and locking in on) whatever that rollout happened to
            # contain, which compounds the collapse risk on a large action space.
            n_epochs=4,
            seed=0,
            # Default MlpPolicy net_arch is only [64, 64] for both actor and critic,
            # which is a severe bottleneck for an ~840-dim observation and a
            # ~1000-way action space -- see Future/research/ for why this was
            # suspected to contribute to the idle-collapse behaviour.
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=nn.Tanh,
            ),
        )

        # Curriculum training loop
        for stage in curriculum:
            env = make_env(
                seed=0,
                horizon=stage["horizon"],
                num_jobs=stage["num_jobs"],
                max_jobs=MAX_JOBS,
            )

            model.set_env(env)

            print(f"Training stage: {stage}")
            model.learn(
                total_timesteps=stage["timesteps"],
                tb_log_name="ppo_scheduling",
                progress_bar=True,
                callback=plotter,
                reset_num_timesteps=False,
            )

        # Save PPO model
        model.save(PPO_MODEL_PATH)
        print(f"\nPPO training complete. Model saved to: {PPO_MODEL_PATH}\n")

    # -----------------------------
    # A2C TRAINING
    # -----------------------------
    else:
        from Policies.a2c_policy import make_maskable_a2c, train_a2c
        import torch

        # Create FIRST curriculum environment
        first_env = make_env(
            seed=0,
            horizon=curriculum[0]["horizon"],
            num_jobs=curriculum[0]["num_jobs"],
            max_jobs=MAX_JOBS,
        )

        model = make_maskable_a2c(first_env, device="cpu")

        # Curriculum training loop
        for stage in curriculum:
            env = make_env(
                seed=0,
                horizon=stage["horizon"],
                num_jobs=stage["num_jobs"],
                max_jobs=MAX_JOBS,
            )
            model.env = env
            train_a2c(model, total_timesteps=stage["timesteps"], plotter=plotter)

        # Save A2C model
        torch.save(model.model.state_dict(), A2C_MODEL_PATH)
        print(f"\nA2C training complete. Model saved to: {A2C_MODEL_PATH}\n")

    plotter.close()
    print(f"Training reward plot saved to: {plot_run_dir}\n")

    print("View TensorBoard with:")
    print("  tensorboard --logdir ./logs\n")


############################## END AI Made ##############################


if __name__ == "__main__":
    main()