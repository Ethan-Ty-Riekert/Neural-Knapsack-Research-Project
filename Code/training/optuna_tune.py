"""optuna_tune.py - Hyperparameter optimization using Optuna for PPO and A2C policies

This module implements a comprehensive hyperparameter search for the scheduling environment,
addressing the key issues:
1. Policy collapse to always-idle behavior
2. Large discrete action space exploration
3. Reward structure balance
"""

import os
import optuna
import numpy as np
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.env_config import generate_env_config
from Code.utils.paths import OPTUNA_RESULTS_DIR, OPTUNA_DB_PATH


def mask_fn(env: GymSchedulingEnv):
    """Action mask function for ActionMasker"""
    return env.get_action_mask()


def make_tuning_env(
    seed: int = 0,
    num_jobs: int = 20,
    num_machines: int = 5,
    horizon: int = 30,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    lambda_3: float = 1.0,
    invalid_penalty: float = 5.0,
    idle_penalty: float = 0.5,
):
    """Create environment for hyperparameter tuning.

    Uses smaller problem size for faster evaluation during optimization.
    max_jobs is fixed to allow consistent obs/action space across trials.
    """
    config = generate_env_config(seed=seed)

    # Use smaller problem for tuning (faster evaluation)
    config["job_durations"] = config["job_durations"][:num_jobs]
    config["job_resources"] = config["job_resources"][:num_jobs, :]
    config["job_deadlines"] = config["job_deadlines"][:num_jobs]
    config["job_weights"] = config["job_weights"][:num_jobs]
    config["num_jobs"] = num_jobs
    config["num_machines"] = num_machines
    config["horizon"] = horizon

    # Create base environment with tunable penalties
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

    # Wrap with fixed max_jobs for consistent action space
    max_jobs = 30  # Fixed padding capacity
    gym_env = GymSchedulingEnv(base_env, max_jobs=max_jobs)
    masked_env = ActionMasker(gym_env, mask_fn)
    monitored_env = Monitor(masked_env)

    return monitored_env


def objective_ppo(trial: optuna.Trial):
    """Optuna objective function for PPO hyperparameter optimization.

    This function is called by Optuna for each trial. It:
    1. Samples hyperparameters from the search space
    2. Creates an environment with those hyperparameters
    3. Trains a PPO agent
    4. Returns the mean episode reward (to be maximized)
    """

    # ==================== SEARCH SPACE ====================

    # Network architecture
    layer_size = trial.suggest_categorical("layer_size", [128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 2, 3)
    activation = trial.suggest_categorical("activation", ["tanh", "relu"])

    # PPO hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

    # CRITICAL: Entropy coefficient (exploration vs exploitation)
    # Higher values encourage more exploration, preventing collapse to idle
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)

    # Clipping range
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    # Value function coefficient
    vf_coef = trial.suggest_float("vf_coef", 0.25, 1.0)

    # Gradient clipping
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    # Reward penalties (critical for balancing objectives)
    lambda_1 = trial.suggest_float("lambda_1", 0.5, 2.0)  # machine activation
    # Widened from [0.5, 2.0] (this session): the tardiness term is now
    # T_j/horizon (see SchedulingEnv.reward(), "Issue D" fix), not raw T_j, so
    # it's O(1) like every other term instead of reaching ~90 at horizon=100.
    # The old range was calibrated against that unbounded term; post-fix the
    # useful order of magnitude for lambda_2 relative to the other O(1) terms
    # (flat +3.0 placement bonus, -1 activation, etc.) is genuinely uncertain
    # and worth searching multiplicatively, not just additively.
    lambda_2 = trial.suggest_float("lambda_2", 0.5, 20.0, log=True)  # tardiness
    lambda_3 = trial.suggest_float("lambda_3", 0.5, 2.0)  # hotspot

    # CRITICAL: Idle penalty must be significant enough to discourage idling
    # but not so high that the agent never idles when it should
    idle_penalty = trial.suggest_float("idle_penalty", 0.5, 3.0)
    invalid_penalty = trial.suggest_float("invalid_penalty", 3.0, 10.0)

    # ==================== BUILD NETWORK ARCHITECTURE ====================

    # Create network architecture
    if n_layers == 2:
        net_arch = dict(pi=[layer_size, layer_size], vf=[layer_size, layer_size])
    else:  # n_layers == 3
        net_arch = dict(
            pi=[layer_size, layer_size, layer_size],
            vf=[layer_size, layer_size, layer_size]
        )

    activation_fn = nn.Tanh if activation == "tanh" else nn.ReLU

    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn,
    )

    # ==================== CREATE ENVIRONMENT ====================

    env = make_tuning_env(
        seed=trial.number,  # Different seed per trial
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        idle_penalty=idle_penalty,
        invalid_penalty=invalid_penalty,
    )

    # ==================== CREATE AND TRAIN MODEL ====================

    try:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            clip_range=clip_range,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=trial.number,
        )

        # Train for a fixed number of timesteps
        # Use fewer timesteps for tuning (faster evaluation)
        eval_timesteps = 30_000

        model.learn(total_timesteps=eval_timesteps, progress_bar=False)

        # ==================== EVALUATION ====================

        # Evaluate the trained model
        n_eval_episodes = 10
        episode_rewards = []
        episode_actions = []  # Track action distribution to detect idle-only behavior

        for _ in range(n_eval_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            actions_taken = []

            while not done:
                action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                actions_taken.append(int(action))
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_actions.append(actions_taken)

        mean_reward = np.mean(episode_rewards)

        # ==================== DETECT PATHOLOGICAL BEHAVIOR ====================

        # Check if agent is stuck on idle action
        idle_action = 30 * 5  # max_jobs * num_machines (from make_tuning_env)
        total_actions = sum(len(actions) for actions in episode_actions)
        idle_count = sum(1 for actions in episode_actions for a in actions if a == idle_action)
        idle_ratio = idle_count / max(total_actions, 1)

        # Penalize trials that collapse to always-idle
        if idle_ratio > 0.95:
            # This trial is useless - agent learned to always idle
            mean_reward -= 100.0  # Heavy penalty

        # Report intermediate value for pruning
        trial.report(mean_reward, step=0)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Store additional metrics for analysis
        trial.set_user_attr("idle_ratio", idle_ratio)
        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("std_reward", np.std(episode_rewards))

        return mean_reward

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -1000.0  # Return very poor score for failed trials

    finally:
        env.close()


def objective_a2c(trial: optuna.Trial, policy_type: str = "pointer"):
    """Optuna objective function for A2C hyperparameter optimization.

    A2C doesn't have the same gradient clipping issues as PPO, so it may
    be more suitable for this large action space problem.

    Previously this hand-reconstructed its own flat actor-critic (a monkey-patched
    MaskableA2C.__init__ building a "ModifiedActorCritic") and duplicated the entire
    rollout/loss loop independently of Policies/a2c_policy.py's MaskableA2C.train()
    -- including that method's masked-entropy bug, in a second place, silently.
    Left as-is, it would keep tuning a stale architecture no longer deployed
    (MaskableA2C now defaults to PointerActorCritic; see pointer_policy.py and
    Future/research/2026-08-09-pointer-network-action-head.md) and would need the
    same fixes applied twice. Rebuilt to construct a real MaskableA2C(policy_type=
    ...) and call its own .train(), so there is exactly one A2C training loop
    in the codebase.

    policy_type: "pointer" (default, PointerActorCritic -- searches embed_dim/
    hidden) or "flat" (MaskableActorCritic -- fixed 256,256 trunk, no
    architecture search added in this pass, kept as the A/B baseline). Bind via
    functools.partial when registering with Optuna so each architecture gets its
    own independent study (see run_optimization()) rather than one shared search
    space trading off architecture choice against hyperparameters.
    """
    from Code.policies.a2c_policy import MaskableA2C

    # ==================== SEARCH SPACE ====================

    if policy_type == "pointer":
        # Pointer-network architecture (see pointer_policy.PointerActorCritic)
        embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
        hidden = trial.suggest_categorical("hidden", [32, 64, 128])
        policy_kwargs = dict(embed_dim=embed_dim, hidden=hidden)
    else:
        # Flat MaskableActorCritic has no configurable width/depth in the
        # current code (fixed 256,256 shared trunk) -- no architecture search
        # plumbing added for it in this pass, to keep scope contained.
        policy_kwargs = None

    # A2C hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [5, 10, 20, 50])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)

    # Entropy coefficient (critical for exploration)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)

    # Value coefficient
    value_coef = trial.suggest_float("value_coef", 0.25, 1.0)

    # Gradient clipping
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    # Reward penalties
    lambda_1 = trial.suggest_float("lambda_1", 0.5, 2.0)
    # See the matching comment in objective_ppo: widened for the same reason
    # (tardiness term is now O(1), not O(horizon)).
    lambda_2 = trial.suggest_float("lambda_2", 0.5, 20.0, log=True)
    lambda_3 = trial.suggest_float("lambda_3", 0.5, 2.0)
    idle_penalty = trial.suggest_float("idle_penalty", 0.5, 3.0)
    invalid_penalty = trial.suggest_float("invalid_penalty", 3.0, 10.0)

    # ==================== CREATE ENVIRONMENT ====================

    env = make_tuning_env(
        seed=trial.number,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        idle_penalty=idle_penalty,
        invalid_penalty=invalid_penalty,
    )

    try:
        agent = MaskableA2C(
            env,
            device="cpu",
            policy_type=policy_type,
            policy_kwargs=policy_kwargs,
        )

        # Override hyperparameters sampled above (constructor sets sensible
        # defaults; trial-specific values replace them before training starts)
        agent.n_steps = n_steps
        agent.gamma = gamma
        agent.lam = gae_lambda
        agent.ent_coef = ent_coef
        agent.value_coef = value_coef
        agent.max_grad_norm = max_grad_norm
        agent.lr = learning_rate
        agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=learning_rate)

        eval_timesteps = 30_000
        agent.train(total_timesteps=eval_timesteps)

        # Evaluate deterministically over a handful of fresh episodes
        n_eval_episodes = 10
        episode_rewards = []
        for _ in range(n_eval_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                mask = info["action_mask"]
                # deterministic=True: consistent with the eval-fairness fix in
                # eval_rl_agent.py -- greedy eval gives a less noisy trial-reward
                # signal for Optuna to compare across trials than a stochastic
                # rollout would.
                action = agent.act(obs, mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            episode_rewards.append(ep_reward)

        mean_reward = float(np.mean(episode_rewards))

        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("n_episodes", n_eval_episodes)

        return mean_reward

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -1000.0

    finally:
        env.close()


def run_optimization(
    algorithm: str = "ppo",
    policy_type: str = "pointer",
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: str = None,
    storage: str = None,
):
    """Run Optuna hyperparameter optimization.

    Args:
        algorithm: "ppo" or "a2c"
        policy_type: "pointer" or "flat" -- A2C only, ignored for PPO (which only
            supports the flat MaskablePPO MlpPolicy). Each policy_type gets its
            own independent study/output-file set (see study_name/results below),
            since architecture choice and hyperparameters shouldn't trade off
            against each other inside one shared search.
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (1 = sequential)
        study_name: Name for the study (for persistent storage)
        storage: Database URL for persistent storage (e.g., "sqlite:///optuna.db")

    Returns:
        study: Optuna study object with results
    """

    # result_tag identifies both the study name and the output-file prefix.
    # BUG FIX (this session): both the search space (lambda_2 range, and for A2C
    # the flat/pointer split above) and the reward function's numeric meaning
    # (SchedulingEnv's tardiness-normalisation fix) changed. The "_v2" suffix
    # guarantees these runs get a fresh study under `storage` rather than
    # resuming (load_if_exists=True, below) any pre-fix study history that would
    # otherwise bias the TPE sampler's posterior with results measured against a
    # now-nonexistent objective landscape.
    if algorithm == "a2c":
        result_tag = f"a2c_{policy_type}_v2"
    else:
        result_tag = f"{algorithm}_v2"

    if study_name is None:
        study_name = f"{result_tag}_scheduling_optimization"

    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",  # Maximize reward
        load_if_exists=True,
    )

    # Select objective function
    if algorithm == "ppo":
        objective = objective_ppo
    else:
        import functools
        objective = functools.partial(objective_a2c, policy_type=policy_type)

    print(f"\n{'='*80}")
    print(f"Starting Optuna optimization for {algorithm.upper()}"
          + (f" ({policy_type})" if algorithm == "a2c" else ""))
    print(f"Number of trials: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Study name: {study_name}")
    print(f"{'='*80}\n")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    # Print results
    print(f"\n{'='*80}")
    print("Optimization completed!")
    print(f"{'='*80}\n")

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best reward: {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results. File prefix is the architecture-identifying tag (e.g.
    # "a2c_pointer", "a2c_flat", "ppo") -- NOT result_tag's "_v2" suffix, which
    # only exists to keep the Optuna *study* fresh in storage. train_optimized.py
    # loads "a2c_{policy_type}_best_params.json" (see Code/training/
    # train_optimized.py), so the two names are deliberately different.
    file_tag = f"{algorithm}_{policy_type}" if algorithm == "a2c" else algorithm
    results_dir = OPTUNA_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters
    import json
    best_params_file = os.path.join(results_dir, f"{file_tag}_best_params.json")
    with open(best_params_file, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest parameters saved to: {best_params_file}")

    # Save study dataframe
    df = study.trials_dataframe()
    df_file = os.path.join(results_dir, f"{file_tag}_trials.csv")
    df.to_csv(df_file, index=False)
    print(f"All trials saved to: {df_file}")

    # Create visualization
    try:
        import optuna.visualization as vis

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(results_dir, f"{file_tag}_optimization_history.html"))

        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(results_dir, f"{file_tag}_param_importances.html"))

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(results_dir, f"{file_tag}_parallel_coordinate.html"))

        print(f"Visualizations saved to: {results_dir}")
    except Exception as e:
        print(f"Could not create visualizations: {e}")

    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter optimization for RL scheduling")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"],
                        help="Algorithm to optimize")
    parser.add_argument("--policy-type", type=str, default="pointer", choices=["pointer", "flat"],
                        help="A2C only: 'pointer' (PointerActorCritic) or 'flat' (MaskableActorCritic). "
                             "Each gets its own independent study and output files.")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of trials to run")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel jobs")
    parser.add_argument("--storage", type=str, default=f"sqlite:///{OPTUNA_DB_PATH.as_posix()}",
                        help="Database URL for persistent storage")

    args = parser.parse_args()

    study = run_optimization(
        algorithm=args.algo,
        policy_type=args.policy_type,
        n_trials=args.trials,
        n_jobs=args.jobs,
        storage=args.storage,
    )
