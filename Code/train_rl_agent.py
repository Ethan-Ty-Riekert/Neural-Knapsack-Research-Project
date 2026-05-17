"""Import the gym wrapper, create the environment, train, save and evaluate the model"""
import os
import argparse
import numpy as np

import gymnasium as gym
from sb3_contrib import MaskablePPO # native action masking to help reduce our massive action space (num_jobs * num_machines) by removing invalid actions
from sb3_contrib.common.wrappers import ActionMasker #

from scheduling_env import SchedulingEnv
from gym_scheduling_wrapper import GymSchedulingEnv


def mask_fn(env: GymSchedulingEnv):
    """Action mask function for ActionMasker"""
    return env.get_action_mask()

def make_env(seed:int=0):
    """Environment creation """
    num_jobs = 40
    num_machines = 5
    horizon = 60
    num_resources = 3 # resource dimensions. Directly effects training time

    # np array randomizer
    rng = np.random.default_rng(seed)

    job_durations = rng.integers(1, 6, size=num_jobs)
    job_resources = rng.integers(1, 6, size=(num_jobs, num_resources))
    job_deadlines = rng.integers(10, 80, size=num_jobs)
    job_weights = np.ones(num_jobs)

    fixed_capacity = 20
    machine_capacity = np.array([fixed_capacity] * num_resources) # All machines are the same

    # Creating the base environment to pass into the gym wrapper
    base_env = SchedulingEnv(
        job_durations=job_durations,
        job_resources=job_resources,
        job_deadlines=job_deadlines,
        job_weights=job_weights,
        num_machines=num_machines,
        machine_capacity=machine_capacity,
        horizon=horizon,
        lambda_1=1.0,
        lambda_2=1.0,
        lambda_3=1.0,
        invalid_penalty=5.0,
    )

    # Gym wrapped environment for RL
    gym_env = GymSchedulingEnv(base_env)
    # masked environment for making actions - efficient with this library
    masked_env = ActionMasker(gym_env, mask_fn)

    return masked_env



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
    MODEL_PATH = os.path.join(MODEL_DIR, "ppo_scheduling")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -----------------------------
    # Create environment
    # -----------------------------
    env = make_env(seed=0)

    # -----------------------------
    # Create PPO model
    # -----------------------------
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        seed=0,
    )

    # -----------------------------
    # Train
    # -----------------------------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="ppo_scheduling",
        progress_bar=True,
    )

    # -----------------------------
    # Save model
    # -----------------------------
    model.save(MODEL_PATH)
    env.close()

    print(f"\nTraining complete. Model saved to: {MODEL_PATH}\n")
    print("View TensorBoard with:")
    print("  tensorboard --logdir ./logs\n")

############################## END AI Made ##############################


if __name__ == "__main__":
    main()