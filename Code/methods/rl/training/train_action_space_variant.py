"""train_action_space_variant.py - Standalone trainer for the three
action-space-reduction variants (2026-09-17 work -- see
Future/research/2026-09-17-action-space-reduction.md and
C:\\Users\\ethan\\.claude\\plans\\encapsulated-questing-cray.md for the full
design/motivation).

Deliberately NOT wired into train_optimized.py's curriculum/Optuna/RCPO
machinery: these are first-pass ~30-minute comparison runs on the single
fixed deployed instance (generate_env_config(seed=0, num_jobs=100,
num_machines=10, horizon=100) -- the same instance every other "final"
offline comparison this session uses), with MaskablePPO and un-tuned but
reasonable defaults, to decide WHICH of the three options (if any) is worth
the investment of a full curriculum-scale validation run before tuning any
one of them further. Run via:

    python -m Code.methods.rl.training.train_action_space_variant --option 1
    python -m Code.methods.rl.training.train_action_space_variant --option 2
    python -m Code.methods.rl.training.train_action_space_variant --option 3 --timesteps 300000

Online mode (2026-09-17 follow-up -- see
Future/research/2026-09-17-heavy-tailed-arrivals.md): --online trains
against OnlineSchedulingEnv/generate_poisson_arrivals instead of the fixed
offline instance, with a fresh realized arrival sequence resampled every
episode (make_online_resampler(), matching train_optimized.py's own online
convention) rather than one fixed instance -- appropriate here since the
online case's entire point is generalizing across arrival realizations, not
memorizing one (this session already found evidence that a strong-looking
fixed-instance online result can be pure memorization, not skill -- see
training-log.md's 2026-09-16 entry).

    python -m Code.methods.rl.training.train_action_space_variant --option 1 --online \\
        --arrival-rate 3 --online-horizon 100 --online-max-jobs 400 \\
        --job-size-distribution lognormal
"""
import argparse
import functools
import json
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from Code.core.scheduling_env import SchedulingEnv
from Code.core.env_config import generate_env_config
from Code.core.online_scheduling_env import OnlineSchedulingEnv
from Code.core.arrival_process import generate_poisson_arrivals
from Code.core.gym_scheduling_wrapper import GymSchedulingEnv
from Code.core.online_gym_wrapper import OnlineGymSchedulingEnv
from Code.methods.rl.action_spaces.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv
from Code.methods.rl.action_spaces.priority_only_gym_wrapper import PriorityOnlyGymSchedulingEnv
from Code.methods.rl.action_spaces.windowed_priority_gym_wrapper import WindowedPriorityGymSchedulingEnv
from Code.methods.rl.action_spaces.action_branching_gym_wrapper import ActionBranchingGymSchedulingEnv
from Code.methods.rl.policies.pointer_ppo_policy import PointerMaskableActorCriticPolicy
from Code.methods.rl.policies.priority_pointer_ppo_policy import PriorityPointerMaskableActorCriticPolicy
from Code.methods.rl.policies.windowed_priority_pointer_ppo_policy import WindowedPriorityPointerMaskableActorCriticPolicy
from Code.methods.rl.policies.action_branching_ppo_policy import ActionBranchingMaskableActorCriticPolicy
from Code.methods.rl.training.train_optimized import (
    make_online_resampler, make_random_instance_resampler, RANDOM_INSTANCE_SEED_CEILING,
    resolve_ppo_rollout_params,
)
from Code.methods.rl.action_spaces.rule_selection_gym_wrapper import (
    RULE_NAMES, rule_kwargs_from_args, is_default_rule_kwargs,
)
from Code.utils.paths import MODELS_DIR, ensure_rl_training_dirs
from Code.core.difficulty import DIFFICULTIES, generate as generate_difficulty, max_job_duration
from Code.core.critic_input import critic_dim_of, wrap_critic_input
from Code.methods.rl.policies.asymmetric_mlp_policy import AsymmetricMlpPolicy
from Code.utils.training_diagnostics import build_diagnostics_callbacks


def mask_fn(env):
    return env.get_action_mask()


def make_base_gym_env(seed=0, reward_mode="legacy", use_potential_shaping=False, shaping_gamma=0.99,
                       randomize_instances=False, job_weight_range=None, objective=None, difficulty=None,
                       extend_horizon=False):
    """The OFFLINE case. seed=0 (default) is the real deployed fixed instance
    -- same seed/dims as exact_solver.py's --fixed-instance and every
    offline PPO/A2C "final" result this session, so tardiness numbers stay
    directly comparable. A different seed builds a distinct held-out
    instance instead (used by eval_action_space_variant.py's
    --randomized-eval, matching eval_rl_agent.py's own held-out-seed
    convention), still with the same dimensions (100 jobs, 10 machines,
    horizon=100).

    randomize_instances=True (for TRAINING, not eval) ignores this seed for
    everything but the env's initial construction -- see below.

    reward_mode/use_potential_shaping (2026-09-17 follow-up, same day as the
    rest of this file -- see Future/research/2026-09-17-action-space-
    reduction.md Section 7): both default to their pre-existing "off"
    values, matching every comparison run so far in this file -- these were
    only ever tested against the OLD (huge) action space and ruled out
    there; whether they help now that the action-space bottleneck itself is
    fixed is untested until a caller passes them explicitly.

    randomize_instances (2026-09-17 follow-up): the offline randomized-
    instance track has tracked the fixed-instance PPO failure almost
    exactly since 2026-09-14 (~1327 vs ~1315-1320 tardiness) -- strong
    historical evidence the failure was never about instance diversity, but
    the same action-space-size bottleneck this file's Options 1/2/3 already
    fix for the fixed instance. Wires in
    Code.methods.rl.training.train_optimized.make_random_instance_resampler() (the
    exact same resampler every prior randomized-instance PPO/A2C run in
    this project used) as GymSchedulingEnv's job_resampler, so every episode
    draws a fresh instance instead of reusing the one built here for
    construction.

    objective / difficulty (2026-09-29, S2W11, variant v2): objective is an ObjectiveConfig used
    with reward_mode="objective" (Code/core/objectives.py); difficulty is a
    Code.core.difficulty.Difficulty -- when given, instances (and the per-episode resampler) come
    from that difficulty preset instead of the default generator.
    """
    if difficulty is not None:
        config = generate_difficulty(difficulty, seed)
    else:
        config = generate_env_config(seed=seed, num_jobs=100, num_machines=10, horizon=100,
                                      job_weight_range=job_weight_range)
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
        reward_mode=reward_mode,
        use_potential_shaping=use_potential_shaping,
        shaping_gamma=shaping_gamma,
        objective=objective,
        extend_horizon=extend_horizon,
    )
    if not randomize_instances:
        job_resampler = None
    elif difficulty is not None:
        job_resampler = make_difficulty_resampler(difficulty, rng_seed=seed)
    else:
        job_resampler = make_random_instance_resampler(config["num_jobs"], config["num_machines"], config["horizon"],
                                                        job_weight_range=job_weight_range)
    return GymSchedulingEnv(base_env, max_jobs=len(config["job_durations"]), job_resampler=job_resampler)


def make_difficulty_resampler(difficulty, rng_seed=0):
    """Fresh instance of a v2 difficulty preset every episode, drawn from training seeds below
    RANDOM_INSTANCE_SEED_CEILING (held-out evaluation seeds start at the ceiling, so training
    never sees them) -- same convention as make_random_instance_resampler()."""
    rng = np.random.default_rng(rng_seed)

    def resample():
        return generate_difficulty(difficulty, int(rng.integers(0, RANDOM_INSTANCE_SEED_CEILING)))
    return resample


def make_online_base_gym_env(arrival_rate, horizon, max_jobs, job_size_distribution,
                              seed=0, num_machines=10, num_resources=4, use_resampler=True,
                              reward_mode="legacy", use_potential_shaping=False, shaping_gamma=0.99,
                              job_weight_range=None, objective=None, difficulty=None, extend_horizon=False):
    """ONLINE case, dynamic Poisson arrivals -- see generate_poisson_arrivals()
    for job_size_distribution. use_resampler=True (default): every episode
    draws a fresh realized arrival sequence (make_online_resampler()) rather
    than reusing the one instance built here for the env's initial
    construction -- see this module's docstring for why that's the right
    default for the online case specifically.

    reward_mode/use_potential_shaping: see make_base_gym_env()'s matching
    docstring. Set once at construction (OnlineSchedulingEnv.__init__) --
    every resample (set_jobs_and_arrivals()) only swaps job/arrival DATA,
    not these env-level reward settings, so they stay in effect across every
    resampled episode automatically."""
    if difficulty is not None:  # v2 difficulty preset: its own rate / max_jobs / sizes / slack
        config = generate_difficulty(difficulty, seed)
        max_jobs = int(config["num_jobs"])
    else:
        config = generate_poisson_arrivals(
            seed=seed, arrival_rate=arrival_rate, horizon=horizon, max_jobs=max_jobs,
            num_machines=num_machines, num_resources=num_resources,
            job_size_distribution=job_size_distribution, job_weight_range=job_weight_range,
        )
    base_env = OnlineSchedulingEnv(
        job_durations=config["job_durations"],
        job_resources=config["job_resources"],
        job_deadlines=config["job_deadlines"],
        job_weights=config["job_weights"],
        num_machines=config["num_machines"],
        machine_capacity=config["machine_capacity"],
        horizon=config["horizon"],
        job_arrival_times=config["job_arrival_times"],
        lambda_1=1.0,
        lambda_2=1.0,
        lambda_3=1.0,
        invalid_penalty=5.0,
        reward_mode=reward_mode,
        use_potential_shaping=use_potential_shaping,
        shaping_gamma=shaping_gamma,
        objective=objective,
        extend_horizon=extend_horizon,
    )
    if not use_resampler:
        job_resampler = None
    elif difficulty is not None:
        job_resampler = make_difficulty_resampler(difficulty, rng_seed=seed)
    else:
        job_resampler = make_online_resampler(arrival_rate, horizon, max_jobs, num_machines, num_resources,
                                              job_size_distribution=job_size_distribution,
                                              job_weight_range=job_weight_range)
    return OnlineGymSchedulingEnv(base_env, max_jobs=max_jobs, job_resampler=job_resampler)


def build_env_and_policy(option: str, full_gym_env=None, window_size=None, window_order="edf",
                          use_atc_feature=False, rule_kwargs=None, policy_arch="flat"):
    """window_size (2026-09-18, S2W10, user-approved): only meaningful for
    option in ("2", "3") -- DeepRM-style bounded action-space window
    (Code/methods/rl/action_spaces/windowed_priority_gym_wrapper.py), Discrete(window_size+1)
    instead of Discrete(max_jobs+1). See that module's docstring for the
    design choices (window_order, backlog scalar). None (default) keeps the
    existing unwindowed Options 2/3 behaviour unchanged.

    window_order (2026-09-21, S2W9 follow-up): "edf" (default, unchanged) or
    "fifo" -- see windowed_priority_gym_wrapper.py's module docstring for why
    "fifo" was added (testing whether the window's EDF-ordering itself, not
    just undertraining, explains the earlier windowed results landing at
    EDF-like performance).

    use_atc_feature (2026-09-23, S2W9 follow-up): only meaningful for
    option == "1" -- appends the same per-job ATC-priority feature Option 3
    already uses (Code/env/obs_atc_feature.py) to Option 1's observation. See
    rule_selection_gym_wrapper.py's module docstring for the motivation
    (observation-informativeness-probe result). Default False keeps existing
    Option 1 checkpoints' observation_space shape unchanged.

    placements (2026-10-05, S2W12): only meaningful for option == "1" -- the
    placement rules offered alongside each priority rule (see
    rule_selection_gym_wrapper.py's module docstring). Default ("FirstFit",) keeps
    existing Option 1 checkpoints' action space unchanged.

    policy_arch (2026-10-05): only meaningful for option == "0" -- the original full
    (job x machine) action space of GymSchedulingEnv, unreduced. "flat" = MLP (v1's 256x256 tanh),
    "pointer" = PointerMaskableActorCriticPolicy (Code/methods/rl/policies/pointer_ppo_policy.py),
    the two architectures every v1 Option-0 PPO/A2C result used."""
    if not is_default_rule_kwargs(rule_kwargs) and option != "1":
        raise ValueError("--rule-placements / --decision-epoch only apply to --option 1.")
    if full_gym_env is None:
        full_gym_env = make_base_gym_env()

    if option == "0":
        if window_size is not None or use_atc_feature:
            raise ValueError("--window-size / --use-atc-feature do not apply to --option 0.")
        env = full_gym_env
        if policy_arch == "pointer":
            policy = PointerMaskableActorCriticPolicy
            policy_kwargs = dict(max_jobs=env.max_jobs, num_machines=env.num_machines,
                                 num_resources=env.num_resources)
        else:
            policy = "MlpPolicy"
            policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh)
    elif option == "1":
        if window_size is not None:
            raise ValueError("--window-size only applies to --option 2/3 (Option 1's action "
                              "space is already the 8-choice rule menu, not job-slot selection).")
        env = RuleSelectionGymSchedulingEnv(full_gym_env, use_atc_feature=use_atc_feature,
                                            **(rule_kwargs or {}))
        policy, policy_kwargs = "MlpPolicy", {}
    elif option in ("2", "3"):
        if use_atc_feature:
            raise ValueError("--use-atc-feature only applies to --option 1 (Options 2/3 already "
                              "have their own --use-atc flag via option == '3').")
        use_atc = option == "3"
        if window_size is not None:
            env = WindowedPriorityGymSchedulingEnv(full_gym_env, window_size=window_size, use_atc=use_atc,
                                                     window_order=window_order)
            policy = WindowedPriorityPointerMaskableActorCriticPolicy
            policy_kwargs = dict(
                window_size=window_size,
                num_machines=env.num_machines,
                num_resources=env.num_resources,
                use_atc=use_atc,
            )
        else:
            env = PriorityOnlyGymSchedulingEnv(full_gym_env, use_atc=use_atc)
            policy = PriorityPointerMaskableActorCriticPolicy
            policy_kwargs = dict(
                max_jobs=env.max_jobs,
                num_machines=env.num_machines,
                num_resources=env.num_resources,
                use_atc=use_atc,
            )
    elif option == "4":
        if window_size is not None:
            raise ValueError("--window-size only applies to --option 2/3.")
        if use_atc_feature:
            raise ValueError("--use-atc-feature only applies to --option 1.")
        env = ActionBranchingGymSchedulingEnv(full_gym_env)
        policy = ActionBranchingMaskableActorCriticPolicy
        policy_kwargs = dict(
            max_jobs=env.max_jobs,
            num_machines=env.num_machines,
            num_resources=env.num_resources,
        )
    else:
        raise ValueError(f"Unknown option {option!r} (expected '1', '2', '3', or '4')")

    layout, critic_dim = full_gym_env.obs_layout, critic_dim_of(full_gym_env)
    if policy != "MlpPolicy" and layout.markov:
        # network policies slice the observation with the same layout (Code/core/obs_layout.py); their
        # value head alone reads the critic-only block (Code/core/critic_input.py)
        policy_kwargs = dict(policy_kwargs, markov=True, lookahead=layout.lookahead, critic_dim=critic_dim)
    elif policy == "MlpPolicy" and critic_dim:
        policy, policy_kwargs = AsymmetricMlpPolicy, dict(policy_kwargs, critic_dim=critic_dim)  # actor blind to it
    monitored = Monitor(ActionMasker(wrap_critic_input(env, full_gym_env), mask_fn))
    return monitored, policy, policy_kwargs


# Per-algorithm defaults for the on-policy update (2026-10-05). PPO = SB3 MaskablePPO defaults.
# A2C is run as the special case of PPO it is (Huang, Dossa, Raffin, Kanervisto & Wang 2022, "A2C is
# a special case of PPO", arXiv:2205.09123): one epoch over one full batch of short rollouts, no
# advantage normalisation, RMSprop, with SB3 A2C's defaults (lr 7e-4, 5 steps per env, lambda 1.0).
# With a single epoch on a single batch the probability ratio is exactly 1, so PPO's clipping never
# activates. This keeps masking, envs, wrappers and evaluation identical between the two algorithms,
# so a PPO-vs-A2C difference is the update rule alone (the confound flagged in report.md 1.6).
ALGO_DEFAULTS = {
    "ppo": dict(learning_rate=3e-4, rollout_size=2048, batch_size=64, n_epochs=10, clip_range=0.2,
                gae_lambda=0.95, vf_coef=0.5, max_grad_norm=0.5),
    "a2c": dict(learning_rate=7e-4, rollout_size=None, batch_size=None, n_epochs=1, clip_range=0.2,
                gae_lambda=1.0, vf_coef=0.5, max_grad_norm=0.5),
}


def resolve_algo_kwargs(args, n_envs):
    """MaskablePPO keyword arguments for --algo, with any explicitly given hyperparameter flag
    overriding the algorithm's default. Returns (kwargs, resolved dict for the sidecar spec)."""
    d = dict(ALGO_DEFAULTS[args.algo])
    for k in d:
        if getattr(args, k, None) is not None:
            d[k] = getattr(args, k)
    if args.algo == "a2c":
        rollout = d["rollout_size"] or 5 * n_envs
        n_steps = max(1, rollout // n_envs)
        batch_size = n_steps * n_envs  # one full batch
        extra = dict(normalize_advantage=False,
                     policy_kwargs_update=dict(optimizer_class=torch.optim.RMSprop,
                                               optimizer_kwargs=dict(alpha=0.99, eps=1e-5, weight_decay=0)))
    else:
        n_steps, batch_size, _ = resolve_ppo_rollout_params(d["rollout_size"], d["batch_size"], n_envs)
        extra = {}
    d.update(n_steps_per_env=n_steps, batch_size=batch_size, rollout_size=n_steps * n_envs)
    kwargs = dict(learning_rate=d["learning_rate"], n_steps=n_steps, batch_size=batch_size,
                  n_epochs=d["n_epochs"], clip_range=d["clip_range"], gae_lambda=d["gae_lambda"],
                  vf_coef=d["vf_coef"], max_grad_norm=d["max_grad_norm"], **extra)
    return kwargs, d


def checkpoint_path(option, tag=None):
    """The one place the final-model filename is defined (training saves here, evaluation and
    run.py load from here)."""
    return MODELS_DIR / f"action_space_option{option}_ppo{f'_{tag}' if tag else ''}.zip"


def write_env_spec(option, tag, spec):
    """Sidecar JSON next to the saved model recording how its env was built (2026-10-05), so an
    evaluator can rebuild the matching action/observation space from the tag alone instead of
    the caller having to repeat every training flag."""
    path = checkpoint_path(option, tag).with_suffix(".json")
    path.write_text(json.dumps(dict(option=option, tag=tag, **spec), indent=2), encoding="utf-8")


def read_env_spec(option, tag=None):
    """The sidecar written by write_env_spec(), or {} for checkpoints trained before it existed
    (their env then has to be described by the caller's flags, as before)."""
    path = checkpoint_path(option, tag).with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def make_full_gym_env(online, base_env_kwargs, seed=0):
    """Single place the base (full-action-space) gym env is built from, for both the
    single-env path in main() and every --n-envs worker -- so a new env option (e.g.
    the v2 objective / difficulty / extend_horizon kwargs) only has to be threaded
    through base_env_kwargs once, and the two paths can never silently diverge.
    base_env_kwargs holds make_online_base_gym_env()'s or make_base_gym_env()'s
    keyword arguments (everything except seed), and must stay picklable for
    SubprocVecEnv (see make_worker_env())."""
    kwargs = dict(base_env_kwargs)
    options = {k: kwargs.pop(k) for k in GymSchedulingEnv.OPTION_KEYS if k in kwargs}
    env = make_online_base_gym_env(seed=seed, **kwargs) if online else make_base_gym_env(seed=seed, **kwargs)
    return env.apply_options(**options)  # work-conserving, repair, full state, look-ahead, scaling, critic input


def make_worker_env(worker_index, option, online, base_env_kwargs, window_size, window_order,
                    use_atc_feature, base_seed, rule_kwargs=None):
    """2026-10-04, S2W11: per-worker env factory for --n-envs > 1 parallel rollout
    (previously this whole file only ever trained on a single env, auto-wrapped by
    SB3 in a DummyVecEnv -- confirmed by grep, unlike train_optimized.py's existing
    build_vec_env()). Deliberately a top-level function taking only picklable
    arguments (not a closure) -- SubprocVecEnv on Windows always spawns workers via
    the "spawn" start method, which pickles the env-constructor callable; the
    exact same constraint train_optimized.py's build_vec_env() docstring already
    documents.

    worker_index varies the online arrival-sequence / randomized-instance seed per
    worker, so the N parallel actors collect genuinely independent episode
    realizations (Schulman et al. 2017, PPO, arXiv:1707.06347, Algorithm 1's N-actor
    rollout scheme) rather than N correlated copies of the same realization. The
    offline FIXED instance (randomize_instances=False) uses base_seed unchanged for
    every worker instead -- every worker must see the IDENTICAL instance there,
    matching build_vec_env()'s own fixed-instance bugfix precedent (a per-worker-
    varying seed there was a real, previously-fixed bug, not a style choice)."""
    per_worker = online or base_env_kwargs.get("randomize_instances", False)
    seed = (base_seed + worker_index) if per_worker else base_seed
    full_gym_env = make_full_gym_env(online, base_env_kwargs, seed=seed)
    env, _, _ = build_env_and_policy(option, full_gym_env=full_gym_env, window_size=window_size,
                                      window_order=window_order, use_atc_feature=use_atc_feature,
                                      rule_kwargs=rule_kwargs)
    return env


def build_parallel_env(n_envs, vec_backend, option, online, base_env_kwargs, window_size,
                       window_order, use_atc_feature, base_seed=0, rule_kwargs=None):
    """n_envs==1 always uses DummyVecEnv (SB3's own default when a single env is
    passed to MaskablePPO -- no behaviour change from before this function existed).
    n_envs>1 with vec_backend="subproc" uses true multi-process rollout (SubprocVecEnv);
    "dummy" runs every sub-env in the calling process (no IPC overhead, but no
    multi-core speedup either) -- see build_vec_env()'s docstring for the same
    backend-choice tradeoff, already established in this project."""
    env_fns = [
        functools.partial(make_worker_env, i, option, online, base_env_kwargs, window_size,
                          window_order, use_atc_feature, base_seed, rule_kwargs)
        for i in range(n_envs)
    ]
    if vec_backend == "subproc" and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def build_parser():
    """The training script's argument parser (also used by the Optuna tuner to parse a trial's argv)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-arch", choices=["flat", "pointer"], default="flat",
                         help="--option 0 only: flat MLP or pointer network over the full action space.")
    parser.add_argument("--option", choices=["0", "1", "2", "3", "4"], required=True,
                         help="1=hyper-heuristic rule selection, 2=raw-feature priority "
                              "learning, 3=ATC-primed priority learning, 4=action-branching "
                              "(learned placement, MultiDiscrete([max_jobs+1, num_machines]) "
                              "-- see Code/methods/rl/action_spaces/action_branching_gym_wrapper.py).")
    parser.add_argument("--timesteps", type=int, default=300_000,
                         help="~30-min comparison scale by this session's own precedent "
                              "(the dense-tardiness-reward quick check used the same "
                              "300k figure and correctly predicted its full-scale result).")
    parser.add_argument("--online", action="store_true",
                         help="Train against the online (dynamic-arrival) case instead of "
                              "the fixed offline instance. Requires --arrival-rate.")
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--online-horizon", type=int, default=100)
    parser.add_argument("--online-max-jobs", type=int, default=400,
                         help="Size generously above arrival_rate*horizon (with margin) "
                              "or arrivals silently truncate into an early burst -- see "
                              "Future/research/2026-09-17-retrospective-cpsat-oracle-and-"
                              "rho-testing.md Section 3.")
    parser.add_argument("--job-size-distribution", choices=["uniform", "lognormal"], default="uniform")
    parser.add_argument("--window-size", type=int, default=None,
                         help="2026-09-18, user-approved: DeepRM-style bounded action-space "
                              "window for --option 2/3 -- Discrete(window_size+1) instead of "
                              "Discrete(max_jobs+1). EDF/FIFO-ordered window + backlog scalar, "
                              "see Code/methods/rl/action_spaces/windowed_priority_gym_wrapper.py. Default None keeps "
                              "the existing unwindowed behaviour.")
    parser.add_argument("--window-order", choices=["edf", "fifo"], default="edf",
                         help="2026-09-21 follow-up: only meaningful with --window-size set. "
                              "Tests whether the window's EDF-ordering itself (not just "
                              "undertraining) explains earlier windowed results landing at "
                              "EDF-like performance -- see windowed_priority_gym_wrapper.py's "
                              "module docstring.")
    parser.add_argument("--reward-mode", choices=["legacy", "dense_tardiness", "objective"], default="legacy",
                         help="2026-09-17 follow-up: dense_tardiness was only tested against the "
                              "old (huge) action space and ruled out there -- untested against "
                              "the winning action-space design until now.")
    parser.add_argument("--objectives", default="tardiness_sq",
                         help="--reward-mode objective only (variant v2): comma list from "
                              "tardiness_sq (default),tardiness,late_count,energy; dropped-job cost is always on.")
    parser.add_argument("--drop-surcharge", type=float, default=None, help="v2: B in ticks (default H)")
    parser.add_argument("--lambda-late", type=float, default=1.0)
    parser.add_argument("--lambda-energy", type=float, default=1.0)
    parser.add_argument("--power-model", default="linear", choices=["linear", "specpower_ml110g5"])
    parser.add_argument("--no-extend-horizon", action="store_true",
                         help="v2: fixed window with dropped jobs instead of the default extended horizon.")
    parser.add_argument("--no-drop-shaping", action="store_true",
                         help="v2: disable potential-based drop-risk shaping (on by default for training).")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default=None,
                         help="v2 difficulty preset (Code/core/difficulty.py) to train on; implies "
                              "--online for online presets and per-episode resampling.")
    parser.add_argument("--gamma", type=float, default=0.99,
                         help="PPO discount factor; also used as the v2 shaping gamma (must match).")
    parser.add_argument("--algo", choices=sorted(ALGO_DEFAULTS), default="ppo",
                         help="2026-10-05: on-policy update rule. 'a2c' = A2C as the special case of PPO "
                              "(Huang et al. 2022, arXiv:2205.09123) on the identical pipeline -- see ALGO_DEFAULTS.")
    # Hyperparameters (default None = the --algo default in ALGO_DEFAULTS); used by the v2 tuning search.
    parser.add_argument("--gae-lambda", type=float, default=None,
                         help="GAE lambda (PPO default 0.95, A2C 1.0). 1.0 = Monte Carlo advantages with a "
                              "value baseline: unbiased credit over the whole episode, at higher variance.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rollout-size", type=int, default=None,
                         help="timesteps collected per update across all envs (PPO default 2048, A2C 5 x n_envs)")
    parser.add_argument("--batch-size", type=int, default=None, help="PPO minibatch size (default 64)")
    parser.add_argument("--n-epochs", type=int, default=None, help="PPO epochs per update (default 10)")
    parser.add_argument("--clip-range", type=float, default=None, help="PPO clip range (default 0.2)")
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--use-potential-shaping", action="store_true",
                         help="Ng/Harada/Russell 1999 potential-based shaping -- same untested "
                              "status as --reward-mode dense_tardiness, see above.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                         help="2026-09-18 follow-up: SB3's MaskablePPO default (0.0) means no "
                              "entropy regularization -- found to let the online case's policy "
                              "entropy decay to near-zero over long runs, collapsing onto a "
                              "single dominant action (see training-log.md's Option 1 online "
                              "900k entry). This project's own precedent for the fix "
                              "(2026-08-09-pointer-network-action-head.md): raised A2C's "
                              "ent_coef 0.0 -> 0.01 for an analogous collapse.")
    parser.add_argument("--job-weight-min", type=int, default=None,
                         help="2026-09-18 follow-up: with --job-weight-max, draws each job's "
                              "weight i.i.d. Uniform{min,...,max-1} instead of the historic "
                              "constant 1.0 (previously dead code for WSPT/ATC's w_j/p_j term). "
                              "Omit both for unchanged legacy behaviour.")
    parser.add_argument("--job-weight-max", type=int, default=None,
                         help="See --job-weight-min (numpy.integers convention: exclusive).")
    parser.add_argument("--randomize-instances", action="store_true",
                         help="Offline only: fresh random job set every episode "
                              "(make_random_instance_resampler()) instead of the one fixed "
                              "instance. 2026-09-17 follow-up -- randomized-instance PPO has "
                              "tracked the fixed-instance failure almost exactly since "
                              "2026-09-14, strong evidence it's the same action-space "
                              "bottleneck, not an instance-diversity problem.")
    parser.add_argument("--save-tag", type=str, default=None,
                         help="Suffix for the saved checkpoint filename (e.g. 'online_lognormal') "
                              "so an online/heavy-tailed run doesn't overwrite the offline "
                              "fixed-instance checkpoint for the same --option.")
    parser.add_argument("--seed", type=int, default=None,
                         help="2026-10-05: MaskablePPO seed (torch/numpy/env-action sampling). Default None keeps "
                              "the previous unseeded behaviour. Instance generation is unaffected, matching the "
                              "project convention that --seed varies only algorithmic randomness.")
    parser.add_argument("--markov-obs", action="store_true",
                         help="2026-10-06: full MDP state of the report's Methodology -- adds each job's "
                              "start time s_j and machine m_j and each machine's activation y_m "
                              "(Code/core/obs_layout.py). Default off keeps the original observation.")
    parser.add_argument("--lookahead", action="store_true",
                         help="2026-10-06, with --markov-obs: each machine's remaining capacity for the next K "
                              "ticks, K = the preset's longest possible job (a function of the state; "
                              "Code/core/obs_layout.py).")
    parser.add_argument("--fixed-scaling", action="store_true",
                         help="2026-10-06, with --markov-obs: job features scaled by fixed constants (durations "
                              "by H, weights by the largest weight, requirements by machine capacity) instead "
                              "of each instance's maxima (GymSchedulingEnv._job_scales).")
    parser.add_argument("--lateness-shaping", action="store_true",
                         help="2026-10-06, extended horizon: potential-based shaping on each unfinished job's "
                              "least remaining lateness (Code/core/objectives.py); optimal policy unchanged "
                              "(Ng et al. 1999), J unaffected.")
    parser.add_argument("--critic-arrivals", action="store_true",
                         help="2026-10-06: the critic (only) also sees a summary of the future arrivals -- an "
                              "input-dependent baseline, unbiased (Code/core/critic_input.py).")
    parser.add_argument("--repair-placement", action="store_true",
                         help="2026-10-05, --option 4 only: if the chosen machine does not fit but another "
                              "does, place the job by FirstFit instead of idling (see the wrapper's step()).")
    parser.add_argument("--work-conserving", action="store_true",
                         help="2026-10-05: non-delay action space -- idle is masked whenever a job can be "
                              "placed (as every heuristic does). Fixes the online failure where priority "
                              "policies idled thousands of ticks and left jobs to the safety cap.")
    parser.add_argument("--decision-epoch", choices=["placement", "tick"], default="placement",
                         help="--option 1 only: one rule choice per job placement (default) or per tick "
                              "(the rule fills the tick, then time advances; online only -- offline the two "
                              "are identical).")
    parser.add_argument("--rule-placements", default="FirstFit",
                         help="2026-10-05: only meaningful with --option 1. Comma-separated placement "
                              "rules offered with each priority rule, e.g. FirstFit,Consolidate "
                              "(menu = rules x placements + idle). Default FirstFit keeps the "
                              "original 8-action menu.")
    parser.add_argument("--use-atc-feature", action="store_true",
                         help="2026-09-23 follow-up: only meaningful with --option 1. Appends the "
                              "same per-job ATC-priority feature Option 3 already uses to Option "
                              "1's observation -- see rule_selection_gym_wrapper.py's module "
                              "docstring (observation-informativeness-probe result) for the "
                              "motivation. Default off keeps existing Option 1 checkpoints' "
                              "observation_space shape unchanged.")
    parser.add_argument("--n-envs", type=int, default=1,
                         help="2026-10-04, S2W11: number of parallel rollout workers. Default 1 "
                              "keeps the exact prior behaviour (single env, no vectorization) for "
                              "every existing checkpoint's reproducibility. This file previously had "
                              "NO multi-env support at all (confirmed by grep against "
                              "train_optimized.py's existing build_vec_env()) -- every online run "
                              "to date trained on a single CPU-bound env regardless of how many "
                              "cores were available.")
    parser.add_argument("--vec-backend", choices=["dummy", "subproc"], default="subproc",
                         help="Only matters with --n-envs > 1. 'subproc' (default): true "
                              "multi-process rollout via SubprocVecEnv. 'dummy': every sub-env "
                              "runs in the calling process (no IPC overhead, no multi-core "
                              "speedup) -- see build_vec_env()'s docstring in train_optimized.py "
                              "for the same tradeoff, already established in this project.")
    parser.add_argument("--diagnostics-interval", type=int, default=None,
                         help="2026-09-20 follow-up: opt-in periodic training-time diagnostics "
                              "(Code/utils/training_diagnostics.py) -- action-distribution "
                              "entropy/frequency logged as TensorBoard scalars, plus a cheap "
                              "held-out tardiness eval, every N timesteps. Default None keeps "
                              "existing behaviour (no callback at all) unchanged.")
    parser.add_argument("--checkpoint-every", type=int, default=25_000,
                         help="Save a checkpoint every N timesteps (2026-09-30: a run killed by the system "
                              "lost all progress because the model was only saved at the end). 0 disables.")
    parser.add_argument("--resume", action="store_true",
                         help="Continue from the latest checkpoint of this --option/--save-tag, training only "
                              "the remaining timesteps.")
    return parser


def env_spec_from_args(args):
    """Everything an evaluator needs to rebuild this model's env (the sidecar spec): one definition
    shared by main()'s sidecar and the tuner's in-process validation evaluation."""
    rule_kwargs = rule_kwargs_from_args(args)
    return dict(
        placements=list(rule_kwargs["placements"]), decision_epoch=rule_kwargs["decision_epoch"],
        use_atc_feature=args.use_atc_feature, window_size=args.window_size, window_order=args.window_order,
        work_conserving=args.work_conserving, repair_placement=args.repair_placement,
        markov_obs=args.markov_obs, policy_arch=args.policy_arch if args.option == "0" else None,
        lookahead=max_job_duration(DIFFICULTIES[args.difficulty]) if args.lookahead else 0,
        fixed_scaling=args.fixed_scaling, critic_arrivals=args.critic_arrivals,
        lateness_shaping=args.lateness_shaping,  # training reward only (provenance); evaluation uses J
    )


def main(argv=None, extra_callbacks=None, save=True):
    """Train one model. argv: the CLI arguments (None = sys.argv). extra_callbacks: SB3 callbacks added
    to training (the Optuna tuner's pruning callback). save=False: return the trained model without
    saving it, its sidecar or TensorBoard logs (tuning trials); the training env is closed either way."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.online and args.arrival_rate is None and args.difficulty is None:
        parser.error("--online requires --arrival-rate")
    if (args.job_weight_min is None) != (args.job_weight_max is None):
        parser.error("--job-weight-min and --job-weight-max must be given together")
    job_weight_range = (
        (args.job_weight_min, args.job_weight_max) if args.job_weight_min is not None else None
    )

    rule_kwargs = rule_kwargs_from_args(args)
    ensure_rl_training_dirs()

    objective = None
    if args.reward_mode == "objective":
        from Code.variants.v2_objectives import objective_config
        objective = objective_config(tuple(o.strip() for o in args.objectives.split(",")), args.drop_surcharge,
                                     args.lambda_late, drop_shaping=not args.no_drop_shaping,
                                     lateness_shaping=args.lateness_shaping,
                                     lambda_energy=args.lambda_energy, power_model=args.power_model)
        objective.shaping_gamma = args.gamma
    difficulty = DIFFICULTIES[args.difficulty] if args.difficulty else None
    extend = args.reward_mode == "objective" and not args.no_extend_horizon  # v2 default (2026-09-29)
    if difficulty is not None:
        args.online = difficulty.case == "online"
        args.randomize_instances = True
        args.arrival_rate = args.arrival_rate or 1.0  # unused: the difficulty preset sets its own rate

    base_env_kwargs = dict(
        reward_mode=args.reward_mode, use_potential_shaping=args.use_potential_shaping,
        shaping_gamma=args.gamma, job_weight_range=job_weight_range,
        objective=objective, difficulty=difficulty, extend_horizon=extend,
        **{k: v for k, v in env_spec_from_args(args).items() if k in GymSchedulingEnv.OPTION_KEYS},
    )
    if args.online:
        base_env_kwargs.update(arrival_rate=args.arrival_rate, horizon=args.online_horizon,
                               max_jobs=args.online_max_jobs,
                               job_size_distribution=args.job_size_distribution)
    else:
        base_env_kwargs.update(randomize_instances=args.randomize_instances)
    full_gym_env = make_full_gym_env(args.online, base_env_kwargs)

    # template_env is ALWAYS built single/un-vectorized -- it's only used for
    # policy/policy_kwargs (identical across every worker) and, below, as the
    # diagnostics callback's source for max_jobs/num_machines (walking
    # template_env.env.env, same chain-walk every other script in this project
    # uses). The actual TRAINING env is this (n_envs==1, unchanged prior
    # behaviour) or the parallel VecEnv built just below (n_envs>1, new).
    template_env, policy, policy_kwargs = build_env_and_policy(
        args.option, full_gym_env=full_gym_env, window_size=args.window_size,
        window_order=args.window_order, use_atc_feature=args.use_atc_feature, rule_kwargs=rule_kwargs,
        policy_arch=args.policy_arch,
    )

    if args.n_envs > 1:
        env = build_parallel_env(
            args.n_envs, args.vec_backend, args.option, args.online, base_env_kwargs,
            args.window_size, args.window_order, args.use_atc_feature, rule_kwargs=rule_kwargs,
        )
        # MaskablePPO's own un-tuned defaults (n_steps=2048, batch_size=64)
        # assume n_envs=1 -- the on-policy buffer size is n_envs*n_steps, so
        # leaving n_steps at 2048 with n_envs>1 would silently grow the buffer
        # (and the real timesteps-per-update) n_envs-fold instead of just
        # speeding up wall-clock throughput. Rescale exactly like
        # train_optimized.py's existing resolve_ppo_rollout_params() already
        # does for its own tuned-hyperparameter case (Schulman et al. 2017,
        # PPO, arXiv:1707.06347, Algorithm 1: N*T, not T alone, is what
        # governs on-policy staleness) -- keeps buffer_size close to the
        # single-env default of 2048 regardless of --n-envs.
        print(f"Parallel rollout: n_envs={args.n_envs} (backend={args.vec_backend})")
    else:
        env = template_env

    algo_kwargs, algo_spec = resolve_algo_kwargs(args, args.n_envs)
    policy_kwargs = dict(policy_kwargs or {}, **algo_kwargs.pop("policy_kwargs_update", {}))
    print(f"Algorithm {args.algo}: " + ", ".join(f"{k}={v}" for k, v in algo_spec.items()))
    model = MaskablePPO(
        policy, env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(MODELS_DIR / "tb_action_space") if save else None,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        seed=args.seed,
        **algo_kwargs,
    )

    print(f"Option {args.option}: online={args.online}, action_space={env.action_space}, "
          f"obs_dim={env.observation_space.shape[0]}, timesteps={args.timesteps}, "
          f"n_envs={args.n_envs} (backend={args.vec_backend if args.n_envs > 1 else 'n/a'})")

    callback = None
    if args.diagnostics_interval is not None:
        held_out_envs = []
        for i in range(5):
            seed = RANDOM_INSTANCE_SEED_CEILING + i
            if args.online:
                held_out_full = make_online_base_gym_env(
                    args.arrival_rate, args.online_horizon, args.online_max_jobs,
                    args.job_size_distribution, seed=seed, use_resampler=False,
                    job_weight_range=job_weight_range, reward_mode=args.reward_mode,
                    objective=objective, difficulty=difficulty, extend_horizon=extend,
                )
            else:
                held_out_full = make_base_gym_env(seed=seed, job_weight_range=job_weight_range,
                                                  reward_mode=args.reward_mode, objective=objective,
                                                  difficulty=difficulty, extend_horizon=extend)
            held_out_full.apply_options(**env_spec_from_args(args))
            held_out_env, _, _ = build_env_and_policy(args.option, full_gym_env=held_out_full,
                                                       window_size=args.window_size,
                                                       window_order=args.window_order,
                                                       use_atc_feature=args.use_atc_feature,
                                                       rule_kwargs=rule_kwargs)
            # build_env_and_policy wraps Monitor(ActionMasker(wrapper, mask_fn))
            # for the TRAINING env (Monitor tracks episode completion for SB3's
            # own info buffer) -- run_episode() (Code/evaluation/
            # eval_action_space_variant.py) expects env.env.env to reach the
            # RAW SchedulingEnv (matching build_eval_env()'s un-Monitor-wrapped
            # ActionMasker(wrapper)), so strip the extra Monitor layer here.
            held_out_envs.append(held_out_env.env)

        # template_env is ALWAYS Monitor(ActionMasker(wrapper, mask_fn)) regardless
        # of --n-envs (env itself may be a VecEnv when n_envs>1, which doesn't
        # have this attribute chain) -- max_jobs/num_machines live on the
        # innermost wrapper (RuleSelectionGymSchedulingEnv etc.), same chain-walk
        # eval_action_space_variant.py's run_episode() uses (base_env = env.env.env).
        inner_env = template_env.env.env
        callback = build_diagnostics_callbacks(
            args.option, held_out_envs=held_out_envs, diagnostics_interval=args.diagnostics_interval,
            rule_names=getattr(inner_env, "rule_names", RULE_NAMES), max_jobs=inner_env.max_jobs,
            num_machines=inner_env.num_machines if args.option == "4" else None,
        )
        print(f"Option {args.option}: diagnostics enabled, interval={args.diagnostics_interval}, "
              f"{len(held_out_envs)} held-out eval instances (seeds {RANDOM_INSTANCE_SEED_CEILING}.."
              f"{RANDOM_INSTANCE_SEED_CEILING + len(held_out_envs) - 1})")

    tag_suffix = f"_{args.save_tag}" if args.save_tag else ""
    save_path = checkpoint_path(args.option, args.save_tag)
    ckpt_dir = MODELS_DIR / "checkpoints" / f"option{args.option}{tag_suffix}"

    reset_num_timesteps = True
    if args.resume:
        ckpts = sorted(ckpt_dir.glob("ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[-2]))
        if not ckpts:
            raise SystemExit(f"--resume: no checkpoints in {ckpt_dir}")
        model = MaskablePPO.load(str(ckpts[-1]), env=env)
        reset_num_timesteps = False
        print(f"Option {args.option}: resumed from {ckpts[-1].name} ({model.num_timesteps} steps done)")

    # build_diagnostics_callbacks() returns a list -- extend, don't nest it (nesting crashed every
    # --diagnostics-interval run once checkpointing was added; fixed 2026-10-05).
    callbacks = list(callback) if callback is not None else []
    callbacks += list(extra_callbacks or [])
    if args.checkpoint_every:
        from stable_baselines3.common.callbacks import CheckpointCallback
        callbacks.append(CheckpointCallback(save_freq=args.checkpoint_every, save_path=str(ckpt_dir),
                                            name_prefix="ckpt"))
    from stable_baselines3.common.callbacks import CallbackList
    remaining = max(0, args.timesteps - (model.num_timesteps if not reset_num_timesteps else 0))

    t0 = time.time()
    try:
        model.learn(total_timesteps=remaining,
                    tb_log_name=f"option{args.option}{tag_suffix}",
                    callback=CallbackList(callbacks) if callbacks else None,
                    reset_num_timesteps=reset_num_timesteps)
    except KeyboardInterrupt:
        interrupted = MODELS_DIR / f"action_space_option{args.option}_ppo{tag_suffix}_interrupted.zip"
        model.save(str(interrupted))
        raise SystemExit(f"interrupted at {model.num_timesteps} steps -- saved to {interrupted}")
    finally:
        env.close()  # frees the SubprocVecEnv workers (also when a tuning trial is pruned)
    elapsed_min = (time.time() - t0) / 60.0

    if not save:
        return model
    model.save(str(save_path))
    write_env_spec(args.option, args.save_tag, dict(
        **env_spec_from_args(args),
        reward_mode=args.reward_mode, objectives=args.objectives if args.reward_mode == "objective" else None,
        difficulty=args.difficulty, online=args.online, timesteps=args.timesteps, n_envs=args.n_envs,
        seed=args.seed, gamma=args.gamma, ent_coef=args.ent_coef, algo=args.algo, algo_hparams=algo_spec,
        train_minutes=round(elapsed_min, 1),
    ))
    print(f"Option {args.option}: trained {args.timesteps} timesteps in {elapsed_min:.1f} min, "
          f"saved to {save_path}")
    return model


if __name__ == "__main__":
    main()
