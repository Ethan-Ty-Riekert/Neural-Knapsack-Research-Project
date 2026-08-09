"""Test if high entropy coefficient prevents idle collapse"""
import numpy as np
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import torch.nn as nn

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.env_config import generate_env_config


def mask_fn(env):
    return env.get_action_mask()


def test_entropy_coef(ent_coef, idle_penalty):
    print(f"\n{'='*60}")
    print(f"Testing ent_coef={ent_coef}, idle_penalty={idle_penalty}")
    print(f"{'='*60}")

    # Create environment
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
        idle_penalty=idle_penalty,
    )

    max_jobs = 30
    gym_env = GymSchedulingEnv(base_env, max_jobs=max_jobs)
    masked_env = ActionMasker(gym_env, mask_fn)
    env = Monitor(masked_env)

    # Create model with high entropy
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,  # Fewer epochs to reduce overfitting
        ent_coef=ent_coef,  # HIGH entropy
        verbose=0,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
        ),
    )

    # Train
    print("Training for 10,000 steps...")
    model.learn(total_timesteps=10_000, progress_bar=False)

    # Evaluate
    idle_action = max_jobs * config["num_machines"]
    n_eval = 5
    all_rewards = []
    all_idle_ratios = []

    for ep in range(n_eval):
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

        idle_count = sum(1 for a in actions_taken if a == idle_action)
        idle_ratio = idle_count / len(actions_taken)

        all_rewards.append(episode_reward)
        all_idle_ratios.append(idle_ratio)

        print(f"  Ep {ep+1}: Reward={episode_reward:.2f}, IdleRatio={idle_ratio:.1%}, Actions={len(actions_taken)}")

    mean_reward = np.mean(all_rewards)
    mean_idle_ratio = np.mean(all_idle_ratios)

    print(f"\nMean reward: {mean_reward:.2f}")
    print(f"Mean idle ratio: {mean_idle_ratio:.1%}")
    print(f"Result: {'SUCCESS' if mean_idle_ratio < 0.5 else 'FAILED (collapsed to idle)'}")

    env.close()
    return mean_reward, mean_idle_ratio


# Test different entropy coefficients
print("\n" + "="*60)
print("TESTING ENTROPY COEFFICIENT AND IDLE PENALTY")
print("="*60)

# Test 1: Low entropy (current optuna best)
test_entropy_coef(ent_coef=0.001, idle_penalty=0.79)

# Test 2: Medium entropy
test_entropy_coef(ent_coef=0.01, idle_penalty=0.79)

# Test 3: High entropy
test_entropy_coef(ent_coef=0.05, idle_penalty=0.79)

# Test 4: Very high entropy
test_entropy_coef(ent_coef=0.1, idle_penalty=0.79)

# Test 5: High entropy + high idle penalty
test_entropy_coef(ent_coef=0.1, idle_penalty=2.5)

# Test 6: Extreme entropy
test_entropy_coef(ent_coef=0.3, idle_penalty=2.5)

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("The test results above show which parameter combinations")
print("prevent the policy from collapsing to always-idle behavior.")
print("="*60)
