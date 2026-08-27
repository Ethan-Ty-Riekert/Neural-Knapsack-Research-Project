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
    use_potential_shaping: bool = False,
    shaping_gamma: float = 0.99,
    randomize_instances: bool = False,
):
    """Create environment for hyperparameter tuning.

    Uses smaller problem size for faster evaluation during optimization.
    max_jobs is fixed to allow consistent obs/action space across trials.

    randomize_instances: Experiment 2 -- if True, every episode reset() draws a
    fresh random job set (same num_jobs/num_machines/horizon) instead of
    reusing the single `seed`-derived instance for the whole trial. See
    Code/training/train_optimized.py::make_random_instance_resampler() (same
    mechanism, reused here so the tuning distribution matches what
    train_optimized.py --randomize-instances actually trains on -- Eimer,
    Lindauer & Raileanu (2023, arXiv:2306.01324) warn that hyperparameters
    tuned on a mismatched tuning environment can overfit to it and not
    transfer, which is exactly what happened in the S2W5 tardiness-retuning
    experiment; this keeps tuning and deployment distributions aligned instead
    of repeating that mismatch for this experiment too).
    """
    # NOTE: kept as generate-100-then-truncate (not a direct num_jobs=num_jobs
    # call) to preserve the exact job data every prior Optuna run (reward and
    # tardiness modes) has used for a given seed -- changing this would silently
    # change tuning-env content for those modes too, not just this new
    # randomize_instances path. make_random_instance_resampler() (used below,
    # and for actual training) generates directly at the target size instead,
    # which is fine there since every episode is already a brand-new draw with
    # no prior "same seed -> same jobs" expectation to preserve.
    config = generate_env_config(seed=seed)
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
        use_potential_shaping=use_potential_shaping,
        shaping_gamma=shaping_gamma,
    )

    # Wrap with fixed max_jobs for consistent action space.
    # BUG FIX (2026-08-28): this was a hardcoded `max_jobs = 30` regardless of
    # `num_jobs` -- GymSchedulingEnv assumes num_jobs <= max_jobs (job slots
    # beyond num_jobs are padding), so any call with num_jobs > 30 (e.g. the
    # 2026-08-28 deployment-scale Pareto rerun's num_jobs=60) indexed past the
    # padded observation/action space and crashed deep inside
    # PointerActorCritic's obs-splitting logic ("index N is out of bounds"),
    # not with a clear error at the actual boundary. max(30, num_jobs)
    # preserves the exact previous behaviour for every call at num_jobs<=30
    # (every study before tonight) while making a larger requested scale
    # actually work instead of crashing cryptically.
    max_jobs = max(30, num_jobs)
    job_resampler = None
    if randomize_instances:
        from Code.training.train_optimized import make_random_instance_resampler
        job_resampler = make_random_instance_resampler(num_jobs, num_machines, horizon)
    gym_env = GymSchedulingEnv(base_env, max_jobs=max_jobs, job_resampler=job_resampler)
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


TARDINESS_PENALTY_WEIGHT = 50.0
"""Weight applied to mean horizon-normalised tardiness (T_j/H, same convention as
SchedulingEnv.reward()) when optimize_for="tardiness" below. Chosen to be the same
order of magnitude as the environment's own +50 completion bonus (Code/env/
scheduling_env.py::step()) -- i.e. a policy that averages one full horizon-unit of
tardiness per job (T_j/H = 1, as late as a job already scheduled can be) is treated
as costing roughly one whole completion bonus. This is a chosen calibration point,
not a derived optimum -- flagged explicitly per CLAUDE.md's rule that untested
weightings must say so rather than imply they're proven. See
Future/research/training-log.md's S2W5 tardiness-retuning entry for the
before/after this produces.
"""


def objective_a2c(
    trial: optuna.Trial,
    policy_type: str = "pointer",
    optimize_for: str = "reward",
    use_potential_shaping: bool = False,
    randomize_instances: bool = False,
    tuning_num_jobs: int = 20,
    tuning_num_machines: int = 5,
    tuning_horizon: int = 30,
):
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

    optimize_for: "reward" (default, unchanged behaviour -- Optuna's fitness
    metric is mean_reward), "tardiness", or "pareto". All three modes train the
    agent against the *same* env reward (lambda_2 included) -- optimize_for only
    changes which metric(s) Optuna uses to rank/select trials against each
    other, the same way a model can be trained on one loss but selected on a
    different validation metric. This exists because "reward" mode has --
    repeatedly, per Future/research/training-log.md's S2W4/S2W5 entries --
    picked hyperparameters that are reward-competitive with EDF but far worse
    on tardiness/late-jobs; the two objectives are only loosely correlated
    given the current reward formula's O(1) tardiness term vs. its larger
    placement/completion bonuses. "tardiness" mode instead ranks trials by
    `mean_reward - TARDINESS_PENALTY_WEIGHT * mean_tardiness_normalised`, so a
    trial only looks good if it both completes jobs (mean_reward requires
    that, same anti-idle-collapse pressure as before) AND keeps them on time --
    but this is still a single scalarization with one hand-picked weight
    (TARDINESS_PENALTY_WEIGHT), the exact practice Roijers et al. (2013)
    critique for only ever reaching one point on the reward-tardiness Pareto
    front (see Future/research/2026-08-21-rcpo-constrained-tardiness.md
    Section 1, which cites the same critique). "pareto" mode (2026-08-28,
    S2W6) returns the raw (mean_reward, mean_tardiness_normalised) pair
    instead of collapsing them into one number -- see run_optimization()'s
    multi-objective study branch -- so Optuna's own NSGA-II-based multi-
    objective sampler (Deb et al., 2002) finds a genuine non-dominated front
    instead of one arbitrarily-weighted point on it.

    use_potential_shaping: carries forward Experiment 4's result (see
    Future/research/training-log.md's 2026-08-19 entry -- pointer+shaping beat
    EDF on tardiness) into this search, rather than re-deriving it from
    scratch. shaping_gamma is set to this trial's own sampled `gamma` below.

    randomize_instances: Experiment 2 -- if True, every episode within a trial
    draws a fresh random job set (Code/training/train_optimized.py::
    make_random_instance_resampler()) instead of the fixed `seed=trial.number`
    instance. Kept aligned with train_optimized.py --randomize-instances so the
    tuning and deployment distributions match -- see make_tuning_env()'s
    docstring for why that alignment matters (Eimer et al. 2023).

    tuning_num_jobs/tuning_num_machines/tuning_horizon (2026-08-28): instance
    scale for this search, forwarded to make_tuning_env(). Default (20, 5, 30)
    is the scale every Optuna study before tonight used. Added after the
    optimize_for="pareto" mode's first run found a degenerate, single-point
    Pareto front at that default scale (see
    Future/research/2026-08-28-multi-objective-optuna-pareto.md Section 4) --
    to test whether the reward/tardiness trade-off this project keeps finding
    at deployment scale (100 jobs) actually appears once the *tuning* scale is
    pushed closer to it. `eval_timesteps` below is deliberately NOT scaled up
    alongside this -- isolating instance scale as the one changed variable,
    consistent with this project's "change one thing at a time" convention --
    so a larger-scale trial is not automatically a fairer trial, only a
    same-training-budget one at a harder distribution.
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
        num_jobs=tuning_num_jobs,
        num_machines=tuning_num_machines,
        horizon=tuning_horizon,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        idle_penalty=idle_penalty,
        invalid_penalty=invalid_penalty,
        use_potential_shaping=use_potential_shaping,
        shaping_gamma=gamma,
        randomize_instances=randomize_instances,
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

        # Evaluate deterministically over a handful of fresh episodes.
        # base_env unwraps Monitor(ActionMasker(GymSchedulingEnv(SchedulingEnv))) --
        # needed to read .tardiness, which the reward/obs alone don't expose.
        base_env = env.env.env.env
        n_eval_episodes = 10
        episode_rewards = []
        episode_tardiness = []
        episode_late_jobs = []
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
            episode_tardiness.append(float(base_env.tardiness.sum()))
            episode_late_jobs.append(int((base_env.tardiness > 0).sum()))

        mean_reward = float(np.mean(episode_rewards))
        mean_tardiness = float(np.mean(episode_tardiness))
        mean_late_jobs = float(np.mean(episode_late_jobs))
        # Same T_j/H normalisation convention as SchedulingEnv.reward()'s tardiness
        # term, so TARDINESS_PENALTY_WEIGHT is calibrated against a comparable,
        # O(1)-per-job quantity rather than a horizon-dependent raw sum.
        mean_tardiness_norm = mean_tardiness / base_env.horizon
        composite_score = mean_reward - TARDINESS_PENALTY_WEIGHT * mean_tardiness_norm

        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("mean_tardiness", mean_tardiness)
        trial.set_user_attr("mean_late_jobs", mean_late_jobs)
        trial.set_user_attr("composite_score", composite_score)
        trial.set_user_attr("n_episodes", n_eval_episodes)

        if optimize_for == "pareto":
            # Two raw objectives, no scalarization -- see this function's
            # optimize_for docstring. run_optimization() creates the study
            # with directions=["maximize", "minimize"] to match this order.
            return mean_reward, mean_tardiness_norm
        return composite_score if optimize_for == "tardiness" else mean_reward

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        if optimize_for == "pareto":
            # Worst-possible point on both axes, consistent with the
            # single-objective failure sentinel (-1000.0) below -- a failed
            # trial must never look good on either objective.
            return -1000.0, 1000.0
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
    optimize_for: str = "reward",
    use_potential_shaping: bool = False,
    randomize_instances: bool = False,
    tuning_num_jobs: int = 20,
    tuning_num_machines: int = 5,
    tuning_horizon: int = 30,
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
        optimize_for: "reward" (default) or "tardiness" -- A2C only, see
            objective_a2c's docstring. Ignored for PPO (objective_ppo unchanged;
            PPO+pointer integration is separately deferred, see
            Future/research/2026-08-09-pointer-network-action-head.md Section 9).
            Given its own result_tag/output-file suffix so it produces a
            comparable *alternative* best-params file rather than overwriting
            the existing reward-tuned one train_optimized.py already loads by
            default.
        use_potential_shaping / randomize_instances: A2C only, see
            objective_a2c's docstring. Both False by default (unchanged
            behaviour); randomize_instances=True gets its own result_tag/
            output-file suffix ("_randinst") for the same reason optimize_for
            does -- a different fitness landscape needs a fresh study and a
            separate best-params file, not to silently overwrite or resume
            over the fixed-instance one.

    Returns:
        study: Optuna study object with results
    """

    # result_tag identifies both the study name and the output-file prefix.
    # BUG FIX (earlier session): both the search space (lambda_2 range, and for
    # A2C the flat/pointer split above) and the reward function's numeric meaning
    # (SchedulingEnv's tardiness-normalisation fix) changed. The "_v2" suffix
    # guarantees these runs get a fresh study under `storage` rather than
    # resuming (load_if_exists=True, below) any pre-fix study history that would
    # otherwise bias the TPE sampler's posterior with results measured against a
    # now-nonexistent objective landscape. Same reasoning applies to
    # optimize_for="tardiness": the *fitness metric* Optuna ranks trials by has
    # changed (composite_score, not mean_reward), so it also needs a fresh study
    # rather than resuming "_v2"'s reward-ranked trial history.
    tardiness_suffix = "_tardiness" if (algorithm == "a2c" and optimize_for == "tardiness") else ""
    tardiness_suffix = "_pareto" if (algorithm == "a2c" and optimize_for == "pareto") else tardiness_suffix
    randinst_suffix = "_randinst" if (algorithm == "a2c" and randomize_instances) else ""
    # Distinguishes a non-default tuning scale (see objective_a2c's
    # tuning_num_jobs/etc. docstring) so it gets its own study/output files
    # instead of colliding with the default (20, 5, 30) scale's.
    scale_suffix = (f"_scale{tuning_num_jobs}x{tuning_num_machines}x{tuning_horizon}"
                     if (algorithm == "a2c" and (tuning_num_jobs, tuning_num_machines, tuning_horizon) != (20, 5, 30))
                     else "")
    if algorithm == "a2c":
        result_tag = f"a2c_{policy_type}_v2{tardiness_suffix}{randinst_suffix}{scale_suffix}"
    else:
        result_tag = f"{algorithm}_v2"

    if study_name is None:
        study_name = f"{result_tag}_scheduling_optimization"

    # Create study
    is_pareto = (algorithm == "a2c" and optimize_for == "pareto")
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    if is_pareto:
        # No sampler= override here: Optuna auto-selects NSGA-II (Deb et al.,
        # 2002) as the default multi-objective sampler once directions is a
        # list rather than a single direction= string -- TPESampler (used
        # below for single-objective) does not support multiple objectives.
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            pruner=pruner,
            directions=["maximize", "minimize"],  # (mean_reward, mean_tardiness_norm)
            load_if_exists=True,
        )
    else:
        sampler = TPESampler(seed=42)
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
        objective = functools.partial(
            objective_a2c,
            policy_type=policy_type,
            optimize_for=optimize_for,
            use_potential_shaping=use_potential_shaping,
            randomize_instances=randomize_instances,
            tuning_num_jobs=tuning_num_jobs,
            tuning_num_machines=tuning_num_machines,
            tuning_horizon=tuning_horizon,
        )

    print(f"\n{'='*80}")
    print(f"Starting Optuna optimization for {algorithm.upper()}"
          + (f" ({policy_type})" if algorithm == "a2c" else "")
          + (f" [optimize_for={optimize_for}]" if algorithm == "a2c" else "")
          + (f" [shaping={use_potential_shaping}, randinst={randomize_instances}]" if algorithm == "a2c" else ""))
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

    file_tag = f"{algorithm}_{policy_type}{tardiness_suffix}{randinst_suffix}{scale_suffix}" if algorithm == "a2c" else algorithm
    results_dir = OPTUNA_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    import json

    if is_pareto:
        # study.best_trial/.best_value/.best_params don't exist for a
        # multi-objective study (there is no single "best" -- that's the
        # point). study.best_trials is the set of non-dominated trials, i.e.
        # the Pareto front itself.
        pareto_trials = study.best_trials
        print(f"Pareto front: {len(pareto_trials)} non-dominated trials (of {len(study.trials)} total)")
        pareto_sorted = sorted(pareto_trials, key=lambda t: t.values[0], reverse=True)
        for t in pareto_sorted:
            print(f"  trial {t.number}: reward={t.values[0]:.2f}  tardiness_norm={t.values[1]:.4f}")

        pareto_file = os.path.join(results_dir, f"{file_tag}_pareto_front.json")
        with open(pareto_file, "w") as f:
            json.dump([
                {"trial": t.number, "mean_reward": t.values[0], "mean_tardiness_norm": t.values[1],
                 "mean_tardiness": t.user_attrs.get("mean_tardiness"),
                 "mean_late_jobs": t.user_attrs.get("mean_late_jobs"),
                 "params": t.params}
                for t in pareto_sorted
            ], f, indent=2)
        print(f"\nPareto front saved to: {pareto_file}")
    else:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best reward: {study.best_value:.2f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save best parameters. File prefix is the architecture-identifying
        # tag (e.g. "a2c_pointer", "a2c_flat", "ppo") plus "_tardiness" when
        # optimize_for="tardiness" -- NOT result_tag's "_v2" suffix, which
        # only exists to keep the Optuna *study* fresh in storage.
        # train_optimized.py loads "a2c_{policy_type}_best_params.json" by
        # default (see Code/training/train_optimized.py's --params-tag), so
        # the reward-tuned file stays untouched and this produces a
        # separate, comparable alternative.
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

        if is_pareto:
            # plot_optimization_history/plot_param_importances assume a
            # single objective -- plot_pareto_front is the multi-objective
            # equivalent, showing the actual reward/tardiness trade-off
            # surface this mode exists to find.
            fig = vis.plot_pareto_front(study, target_names=["mean_reward", "mean_tardiness_norm"])
            fig.write_html(os.path.join(results_dir, f"{file_tag}_pareto_front.html"))
        else:
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
    parser.add_argument("--optimize-for", type=str, default="reward", choices=["reward", "tardiness", "pareto"],
                        help="A2C only: 'reward' (default, unchanged) ranks trials by mean episode "
                             "reward. 'tardiness' ranks by mean_reward - 50*mean_tardiness_normalised "
                             "instead, to search for hyperparameters that are actually tardiness-"
                             "competitive, not just reward-competitive -- see objective_a2c's docstring. "
                             "'pareto' (2026-08-28) runs a genuine multi-objective study over the raw "
                             "(mean_reward, mean_tardiness_normalised) pair via Optuna's NSGA-II sampler, "
                             "instead of collapsing them into one hand-weighted score -- saves a "
                             "*_pareto_front.json of the non-dominated trial set. Each mode writes to "
                             "separate *_{tardiness,pareto}_best_params.json / _pareto_front.json files, "
                             "not overwriting the default reward-tuned file.")
    parser.add_argument("--use-potential-shaping", action="store_true",
                        help="A2C only: carry forward Experiment 4's shaping win (see "
                             "objective_a2c's docstring) into this search.")
    parser.add_argument("--randomize-instances", action="store_true",
                        help="A2C only, Experiment 2: tune against randomized per-episode job sets "
                             "instead of a fixed instance, matching train_optimized.py "
                             "--randomize-instances -- see make_tuning_env's docstring for why the "
                             "tuning and deployment distributions should match. Writes to a separate "
                             "*_randinst_best_params.json.")
    parser.add_argument("--tuning-num-jobs", type=int, default=20,
                        help="A2C only (2026-08-28): instance scale for the tuning env (default 20, "
                             "every study before tonight's used this). See objective_a2c's "
                             "tuning_num_jobs docstring for why this was added -- non-default values "
                             "get their own study/output files via a _scaleNxMxH suffix.")
    parser.add_argument("--tuning-num-machines", type=int, default=5,
                        help="A2C only: paired with --tuning-num-jobs/--tuning-horizon.")
    parser.add_argument("--tuning-horizon", type=int, default=30,
                        help="A2C only: paired with --tuning-num-jobs/--tuning-num-machines.")

    args = parser.parse_args()

    study = run_optimization(
        algorithm=args.algo,
        policy_type=args.policy_type,
        n_trials=args.trials,
        n_jobs=args.jobs,
        storage=args.storage,
        optimize_for=args.optimize_for,
        use_potential_shaping=args.use_potential_shaping,
        randomize_instances=args.randomize_instances,
        tuning_num_jobs=args.tuning_num_jobs,
        tuning_num_machines=args.tuning_num_machines,
        tuning_horizon=args.tuning_horizon,
    )
