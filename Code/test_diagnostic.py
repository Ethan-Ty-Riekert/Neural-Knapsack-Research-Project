"""Diagnostic script to understand why trials are all idling"""
import numpy as np
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from scheduling_env import SchedulingEnv
from gym_scheduling_wrapper import GymSchedulingEnv
from env_config import generate_env_config


def mask_fn(env):
    return env.get_action_mask()


# Create same environment as optuna
config = generate_env_config(seed=0)
num_jobs = 20
config["job_durations"] = config["job_durations"][:num_jobs]
config["job_resources"] = config["job_resources"][:num_jobs, :]
config["job_deadlines"] = config["job_deadlines"][:num_jobs]
config["job_weights"] = config["job_weights"][:num_jobs]
config["num_jobs"] = num_jobs
config["num_machines"] = 5
config["horizon"] = 30

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
    idle_penalty=0.5,
)

max_jobs = 30
gym_env = GymSchedulingEnv(base_env, max_jobs=max_jobs)
masked_env = ActionMasker(gym_env, mask_fn)
env = Monitor(masked_env)

# Test idle action
idle_action = max_jobs * config["num_machines"]
print(f"Idle action index: {idle_action}")
print(f"Total action space: {env.action_space.n}")
print(f"Max_jobs: {max_jobs}, num_machines: {config['num_machines']}")

# Reset and check
obs, info = env.reset()
print(f"\nAction mask sum: {info['action_mask'].sum()}")
print(f"Idle action valid: {info['action_mask'][idle_action]}")

# Take some actions
print("\n--- Taking actions ---")
for i in range(5):
    # Get a valid non-idle action
    valid_actions = np.where(info['action_mask'] == 1)[0]
    print(f"Step {i}: Valid actions: {len(valid_actions)}, includes idle: {idle_action in valid_actions}")

    if len(valid_actions) > 1:
        # Take first non-idle action
        non_idle_actions = [a for a in valid_actions if a != idle_action]
        if non_idle_actions:
            action = non_idle_actions[0]
            print(f"  Taking action {action} (non-idle)")
        else:
            action = idle_action
            print(f"  Only idle available, taking {action}")
    else:
        action = valid_actions[0]
        print(f"  Taking action {action}")

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Reward: {reward:.2f}, Done: {terminated or truncated}")

    if terminated or truncated:
        break

print("\n--- Creating quick PPO agent ---")
import torch.nn as nn

model = MaskablePPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    ent_coef=0.01,
    verbose=0,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh,
    ),
)

# Train briefly
print("Training for 2000 steps...")
model.learn(total_timesteps=2000, progress_bar=False)

# Evaluate
print("\n--- Evaluating trained agent ---")
obs, info = env.reset()
done = False
episode_reward = 0
actions_taken = []

while not done:
    action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
    actions_taken.append(int(action))
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    done = terminated or truncated

print(f"Episode reward: {episode_reward:.2f}")
print(f"Actions taken: {len(actions_taken)}")
print(f"Idle actions: {sum(1 for a in actions_taken if a == idle_action)}")
print(f"Non-idle actions: {sum(1 for a in actions_taken if a != idle_action)}")
print(f"Idle ratio: {sum(1 for a in actions_taken if a == idle_action) / len(actions_taken):.2%}")
print(f"\nFirst 10 actions: {actions_taken[:10]}")
print(f"Action distribution: idle={idle_action} appears {actions_taken.count(idle_action)} times")
