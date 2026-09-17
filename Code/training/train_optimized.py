"""train_optimized.py - Train RL agent using Optuna-optimized hyperparameters

This script loads the best hyperparameters found by Optuna and trains
a full-scale model with curriculum learning.
"""

import os
import json
import argparse
import functools
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Code.env.scheduling_env import SchedulingEnv
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.env_config import generate_env_config
from Code.env.online_scheduling_env import OnlineSchedulingEnv
from Code.env.online_gym_wrapper import OnlineGymSchedulingEnv
from Code.env.arrival_process import generate_poisson_arrivals
from Code.env.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv
from Code.utils.plotting_utils import make_run_dir, LiveTrainingPlotter
from Code.utils.paths import (
    LOG_DIR, MODELS_DIR, PLOTS_DIR, OPTUNA_RESULTS_DIR, ENV_CONFIG_PATH,
    REPO_ROOT, ensure_rl_training_dirs,
)
from Code.utils.results_log import archive_checkpoint_files
from Code.policies.ppo_lagrangian import PPOLagrangianCallback


def mask_fn(env: GymSchedulingEnv):
    """Action mask function for ActionMasker"""
    return env.get_action_mask()


# Experiment 2 (randomized-instance generalization): training draws per-episode
# job-set seeds from [1, RANDOM_INSTANCE_SEED_CEILING). Held-out evaluation seeds
# (Code/evaluation/eval_rl_agent.py's --randomized-eval) live at/above this
# ceiling, so by construction no training seed can ever coincide with an eval
# seed -- no need to track/exclude a used-seed set at runtime.
RANDOM_INSTANCE_SEED_CEILING = 500_000


def make_random_instance_resampler(num_jobs: int, num_machines: int, horizon: int, job_weight_range=None):
    """Returns a zero-arg callable that draws a fresh random job set (same
    num_jobs/num_machines/horizon every call, per GymSchedulingEnv.set_jobs()'s
    requirement) each time it's invoked -- passed as GymSchedulingEnv's
    job_resampler so every episode reset() gets a genuinely different instance
    instead of reusing the one fixed instance every prior result in this project
    trained/evaluated on. See Future/research/training-log.md's S2W5 entries and
    2026-08-09-pointer-network-action-head.md Section 9/10 for why this is the
    real test of the pointer network's generalization design claim.

    job_weight_range (2026-09-18, S2W10): see
    Code.env.env_config.generate_env_config()'s matching parameter -- None
    (default, unchanged) keeps every resampled instance's weights at 1.0.
    """
    def resample():
        seed = int(np.random.randint(1, RANDOM_INSTANCE_SEED_CEILING))
        return generate_env_config(seed=seed, num_jobs=num_jobs, num_machines=num_machines, horizon=horizon,
                                    job_weight_range=job_weight_range)
    return resample


def make_online_resampler(arrival_rate: float, horizon: int, max_jobs: int, num_machines: int, num_resources: int,
                           job_size_distribution: str = "uniform", job_weight_range=None):
    """Online-case counterpart of make_random_instance_resampler(): returns a
    zero-arg callable that draws a fresh Poisson-arrival instance (job
    content AND arrival times -- see arrival_process.py) each call, passed as
    OnlineGymSchedulingEnv's job_resampler so every episode gets a genuinely
    different realized arrival sequence instead of reusing the one instance
    generated at construction time.

    job_size_distribution: "uniform" (default, unchanged) or "lognormal"
    (2026-09-17, S2W10 heavy-tailed job sizes -- see
    Code/env/arrival_process.py::generate_poisson_arrivals()'s docstring for
    the Google/Azure cluster-trace grounding). Threaded straight through so
    every resampled episode, not just the first instance, uses the chosen
    distribution."""
    def resample():
        seed = int(np.random.randint(1, RANDOM_INSTANCE_SEED_CEILING))
        return generate_poisson_arrivals(
            seed=seed, arrival_rate=arrival_rate, horizon=horizon, max_jobs=max_jobs,
            num_resources=num_resources, num_machines=num_machines,
            job_size_distribution=job_size_distribution, job_weight_range=job_weight_range,
        )
    return resample


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
    use_potential_shaping: bool = False,
    shaping_gamma: float = 0.99,
    randomize_instances: bool = False,
    save_config: bool = True,
    is_online: bool = False,
    arrival_rate: float = None,
    reward_mode: str = "legacy",
    job_size_distribution: str = "uniform",
    job_weight_range=None,
    action_mode: str = "placement",
):
    """Environment creation with tunable reward penalties.

    job_weight_range (2026-09-18, S2W10): see
    Code.env.env_config.generate_env_config()'s matching parameter -- None
    (default, unchanged) keeps every job's weight at 1.0.

    action_mode (2026-09-18, S2W10): "placement" (default, unchanged) is
    every prior behaviour in this file -- the policy picks a raw (job,
    machine) pair. "rule_selection" wraps the built GymSchedulingEnv/
    OnlineGymSchedulingEnv with RuleSelectionGymSchedulingEnv (Option 1,
    Future/research/2026-09-17-action-space-reduction.md) BEFORE masking, so
    curriculum training can use Option 1's Discrete(8) hyper-heuristic
    action space at real (Optuna-tuned-hyperparameter, multi-stage
    curriculum) scale instead of only the standalone
    train_action_space_variant.py trainer's flat-timestep runs. Rebuilt
    fresh at each curriculum stage exactly like every other wrapper here, so
    the action space size (fixed at 8 regardless of num_jobs) stays valid
    across model.set_env() calls the same way max_jobs already does for
    the placement action space.

    reward_mode: "legacy" (default, unchanged) or "dense_tardiness" (new --
    see SchedulingEnv.__init__'s reward_mode docstring for the full
    derivation/citations). Threaded straight through to
    SchedulingEnv/OnlineSchedulingEnv.

    is_online / arrival_rate: build the online (dynamic-arrival) case
    instead -- OnlineSchedulingEnv/OnlineGymSchedulingEnv with Poisson
    arrivals (Code/env/arrival_process.py, arrival_rate required) rather
    than SchedulingEnv/GymSchedulingEnv. `num_jobs` is not meaningful here
    (there is no fixed job count to truncate to -- the realized arrival
    count is itself random) and is ignored; `max_jobs` is the array-capacity
    ceiling (the phantom-padding convention) and, like the offline case,
    must stay constant across curriculum stages for the policy network's
    input dimension to stay valid across model.set_env() calls -- only
    `horizon` is expected to vary per stage. `randomize_instances` reuses
    the same flag/semantics as the offline case: True draws a fresh
    Poisson-arrival instance every episode (make_online_resampler()) instead
    of reusing the one instance generated below.

    use_potential_shaping/shaping_gamma: Solution 3 (Ng, Harada, Russell 1999
    potential-based shaping, see SchedulingEnv._compute_potential()) -- off by
    default. shaping_gamma should match the RL algorithm's own gamma (passed
    explicitly by callers below, not left at this default) so the shaping
    term's policy-invariance guarantee holds under the same discounting the
    algorithm actually uses.

    randomize_instances: Experiment 2 -- if True, every episode reset() draws a
    fresh random job set (same num_jobs/num_machines/horizon as this call, per
    the curriculum stage) instead of reusing the single instance generated by
    `seed` below. `seed`/the initial config below still construct the env's
    *first* job set (needed to construct SchedulingEnv at all), but it gets
    immediately replaced on the first reset() once training starts.

    save_config: whether to write ENV_CONFIG_PATH for this env (read later by
    eval_rl_agent.py). When building an n_envs-way vectorized env for parallel
    rollout collection, every worker process calls make_env() concurrently --
    only one of them should write this shared file, or concurrent writers can
    race and leave eval_rl_agent.py reading a corrupted/truncated .npz. Callers
    building a VecEnv must pass save_config=True for exactly one worker (index
    0) and False for the rest; the single-env call sites below keep the True
    default.
    """

    if is_online:
        if arrival_rate is None:
            raise ValueError("arrival_rate is required when is_online=True")
        online_max_jobs = max_jobs if max_jobs is not None else 100
        online_horizon = horizon if horizon is not None else 100
        config = generate_poisson_arrivals(
            seed=seed, arrival_rate=arrival_rate, horizon=online_horizon, max_jobs=online_max_jobs,
            job_size_distribution=job_size_distribution, job_weight_range=job_weight_range,
        )
        if num_machines is not None:
            config["num_machines"] = num_machines

        if save_config:
            ensure_rl_training_dirs()
            np.savez(ENV_CONFIG_PATH, **config)

        base_env = OnlineSchedulingEnv(
            job_durations=config["job_durations"],
            job_resources=config["job_resources"],
            job_deadlines=config["job_deadlines"],
            job_weights=config["job_weights"],
            num_machines=config["num_machines"],
            machine_capacity=config["machine_capacity"],
            horizon=config["horizon"],
            job_arrival_times=config["job_arrival_times"],
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            invalid_penalty=invalid_penalty,
            idle_penalty=idle_penalty,
            use_potential_shaping=use_potential_shaping,
            shaping_gamma=shaping_gamma,
            reward_mode=reward_mode,
        )
        job_resampler = (
            make_online_resampler(
                arrival_rate, config["horizon"], config["num_jobs"],
                config["num_machines"], config["num_resources"],
                job_size_distribution=job_size_distribution, job_weight_range=job_weight_range,
            )
            if randomize_instances else None
        )
        gym_env = OnlineGymSchedulingEnv(base_env, max_jobs=config["num_jobs"], job_resampler=job_resampler)
        if action_mode == "rule_selection":
            gym_env = RuleSelectionGymSchedulingEnv(gym_env)
        masked_env = ActionMasker(gym_env, mask_fn)
        return Monitor(masked_env)

    config = generate_env_config(seed=seed, job_weight_range=job_weight_range)

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
    if save_config:
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
        use_potential_shaping=use_potential_shaping,
        shaping_gamma=shaping_gamma,
        reward_mode=reward_mode,
    )

    # Wrap in Gym + Masking. job_resampler is None (fixed-instance, previous
    # behaviour) unless randomize_instances=True -- see
    # make_random_instance_resampler() and GymSchedulingEnv.reset().
    job_resampler = (
        make_random_instance_resampler(config["num_jobs"], config["num_machines"], config["horizon"],
                                        job_weight_range=job_weight_range)
        if randomize_instances else None
    )
    gym_env = GymSchedulingEnv(base_env, max_jobs=max_jobs, job_resampler=job_resampler)
    if action_mode == "rule_selection":
        gym_env = RuleSelectionGymSchedulingEnv(gym_env)
    masked_env = ActionMasker(gym_env, mask_fn)
    monitored_env = Monitor(masked_env)

    return monitored_env


def build_vec_env(n_envs: int, vec_backend: str, instance_seed: int, save_config: bool = True, **env_kwargs):
    """Build an n_envs-way vectorized environment for parallel PPO rollout
    collection.

    instance_seed is passed unchanged to every worker's make_env(seed=...) --
    i.e. every worker constructs the exact same fixed job instance (matching
    the single canonical fixed instance, generate_env_config(seed=0), every
    prior fixed-instance result in this project trained/evaluated on).

    BUG FIX (this session): this used to give each worker a *different*
    seed (instance_seed + worker_index), which does the wrong thing for
    ordinary fixed-instance training (randomize_instances=False, the
    default) -- it made each of the n_envs workers train on a different job
    instance concurrently, mixing rollouts from n_envs distinct problems into
    one policy update instead of the intended "thousands of repeated
    exposures to the one instance" (see the 2026-08-09 fixed-instance bugfix
    doc). It also meant --seed values passed for Henderson et al. 2018-style
    multi-seed variance estimation (which should vary only algorithmic
    randomness -- policy init, action sampling -- and hold the task fixed)
    were silently changing the task itself instead. This constant-instance_seed
    version restores the intended "same instance for every worker" semantics
    regardless of n_envs; algorithmic randomness (torch/numpy global seed,
    MaskablePPO's own seed) is a separate parameter at the call site, not this
    function's instance_seed. If a future run uses randomize_instances=True
    with vec_backend="subproc", note that each worker process's own global
    numpy RNG (which make_random_instance_resampler's np.random.randint calls
    read) would need independent seeding to avoid correlated resampling
    streams across workers -- not needed for the fixed-instance case here, or
    for the "dummy" backend (single process, naturally-interleaved shared
    RNG), so it's not implemented until an actual randomize_instances+subproc
    run needs it.

    vec_backend="subproc" uses SubprocVecEnv (true multi-process rollout,
    n_envs>1 only -- Windows always spawns via the "spawn" start method,
    which is why every make_env() thunk here is a picklable functools.partial
    rather than a closure). vec_backend="dummy" (or n_envs==1) uses
    DummyVecEnv, which runs every sub-env in the calling process -- no IPC
    overhead, but no multi-core speedup either. See the plan's Verification
    step: time both backends at the target n_envs on this machine before
    committing to one for the final run, since SchedulingEnv.step() is cheap
    numpy arithmetic and SubprocVecEnv's per-step pipe IPC can offset its own
    parallelism gains if the env is cheap enough relative to that overhead.

    Only worker index 0 ever writes ENV_CONFIG_PATH (make_env's save_config
    param), and only when this function's own save_config=True (the caller's
    responsibility -- see train_with_optimized_params, which only passes
    True for the FINAL curriculum stage).

    BUG FIX (2026-09-14, S2W9): every prior call site (and the single-env
    make_env() calls before this function existed) wrote ENV_CONFIG_PATH on
    EVERY curriculum stage transition, unconditionally -- already flagged as
    a real, not-yet-fixed hazard in Future/research/training-log.md's
    2026-08-28 entry ("training silently corrupts the shared eval instance
    file if run concurrently with eval/baseline work"): any eval/heuristic/
    exact-solver script reading ENV_CONFIG_PATH while a training run is
    mid-curriculum sees whatever small-scale stage (15/30/60 jobs) happens
    to be active at that instant, not the deployed 100-job/horizon=100
    instance, with no error to signal the mismatch. Re-hit this exact hazard
    this session (running eval concurrently with a newly-launched training
    run produced a fully plausible-looking but wrong jobs_scheduled~14-30
    result). Fixed properly now rather than re-deferred: ENV_CONFIG_PATH is
    only written once training reaches its LAST curriculum stage, so the
    shared file only ever reflects the final, full-scale instance once any
    training run has passed that point -- matching what every eval script
    has always assumed it contains.
    """
    def make_worker_env(worker_index: int):
        return make_env(seed=instance_seed, save_config=(worker_index == 0 and save_config), **env_kwargs)

    env_fns = [functools.partial(make_worker_env, i) for i in range(n_envs)]

    if vec_backend == "subproc" and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def resolve_ppo_rollout_params(tuned_n_steps: int, tuned_batch_size: int, n_envs: int):
    """Rescale Optuna-tuned n_steps/batch_size for n_envs-way parallel rollout.

    Optuna's search tuned these values assuming n_envs=1, i.e. it tuned the
    total on-policy rollout buffer size (n_envs * n_steps) to equal
    tuned_n_steps. Per Schulman et al. 2017 ("Proximal Policy Optimization
    Algorithms," arXiv:1707.06347, Algorithm 1), N parallel actors each
    collecting T timesteps per update is the standard PPO rollout scheme, and
    it's N*T -- not T alone -- that governs the on-policy staleness/update
    frequency the Optuna search actually explored. So n_steps must shrink by
    roughly a factor of n_envs to keep N*T close to what was tuned, rather
    than growing the effective buffer n_envs-fold by reusing tuned_n_steps
    per worker unchanged.

    Returns (effective_n_steps, resolved_batch_size, buffer_size). SB3 only
    warns (never errors) when batch_size doesn't evenly divide the buffer
    size, silently dropping part of the last minibatch every epoch -- for a
    multi-hour run that's worth avoiding explicitly, so resolved_batch_size
    is shrunk to the nearest divisor of buffer_size when needed.
    """
    effective_n_steps = max(1, round(tuned_n_steps / n_envs))
    buffer_size = n_envs * effective_n_steps

    resolved_batch_size = min(tuned_batch_size, buffer_size)
    while buffer_size % resolved_batch_size != 0:
        resolved_batch_size -= 1

    return effective_n_steps, resolved_batch_size, buffer_size


def get_git_commit_hash() -> str:
    """Best-effort git commit hash of the current checkout, for tracing a
    later paper figure/table back to the exact code that produced it. Returns
    "unknown" (rather than raising) if git isn't available or REPO_ROOT isn't
    a git checkout, since this is provenance metadata, not something worth
    failing a multi-hour training run over."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def save_run_metadata(path, **fields) -> None:
    """Write this run's resolved config (seed, hyperparameters, git commit,
    etc.) as JSON so an archived checkpoint's exact provenance can be traced
    later for a paper figure/table without rerunning anything."""
    fields.setdefault("git_commit", get_git_commit_hash())
    fields.setdefault("saved_at_utc", datetime.now(timezone.utc).isoformat())
    with open(path, "w") as f:
        json.dump(fields, f, indent=2)


def load_best_params(algorithm: str = "ppo", policy_type: str = "pointer", params_tag: str = None):
    """Load best hyperparameters from Optuna optimization.

    file_tag mirrors optuna_tune.py::run_optimization()'s output naming: A2C
    saves per-architecture files ("a2c_pointer_best_params.json",
    "a2c_flat_best_params.json"); PPO only has one architecture ("ppo_best_params.json").

    params_tag: optional suffix (e.g. "tardiness") to load an alternative params
    file produced by `optuna_tune.py --optimize-for tardiness`
    ("a2c_pointer_tardiness_best_params.json") instead of the default
    reward-tuned one, for comparing the two without overwriting either.
    """
    tag_suffix = f"_{params_tag}" if params_tag else ""
    # ppo+flat is the original, pre-architecture-suffix convention (its file
    # is "ppo_best_params.json", not "ppo_flat_best_params.json") -- keep it
    # unsuffixed so every existing reference to that file keeps working.
    # Every other combination (a2c+pointer, a2c+flat, and the new ppo+pointer)
    # includes policy_type in the tag.
    file_tag = (
        f"{algorithm}{tag_suffix}"
        if (algorithm == "ppo" and policy_type == "flat")
        else f"{algorithm}_{policy_type}{tag_suffix}"
    )
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


def train_with_optimized_params(
    algorithm: str = "ppo",
    policy_type: str = "pointer",
    use_curriculum: bool = True,
    stage4_timesteps: int = 200_000,
    run_tag: str = None,
    params_tag: str = None,
    use_potential_shaping: bool = False,
    randomize_instances: bool = False,
    reward_mode: str = "legacy",
    use_rcpo: bool = False,
    rcpo_alpha: float = 0.0,
    rcpo_lambda_lr: float = 0.01,
    rcpo_lambda_max: float = 50.0,
    rcpo_update_every: int = 5,
    seed: int = 0,
    n_envs: int = None,
    vec_backend: str = "subproc",
    torch_threads: int = None,
    is_online: bool = False,
    arrival_rate: float = None,
    max_jobs_override: int = None,
    job_size_distribution: str = "uniform",
    job_weight_range=None,
    action_mode: str = "placement",
):
    """Train agent using optimized hyperparameters from Optuna.

    job_weight_range / action_mode (2026-09-18, S2W10): see make_env()'s
    matching parameters. action_mode="rule_selection" is only wired through
    the PPO (non-RCPO) curriculum path below -- Option 1 has only ever been
    trained via MaskablePPO this session, so the A2C/RCPO paths are left
    untouched.

    is_online / arrival_rate: online (dynamic-arrival) case -- see
    make_env()'s matching docstring. Curriculum stages still vary `horizon`;
    `max_jobs` (MAX_JOBS below) stays constant across stages exactly as in
    the offline case, since the policy network's input dimension is fixed at
    first construction. Not yet combined with use_rcpo/PPO-Lagrangian or
    --randomize-instances-style Optuna tuning -- those remain untested for
    this mode as of this writing.

    max_jobs_override: replaces the hardcoded MAX_JOBS=100 (which sizes every
    job-slot array capacity, offline and online alike) when given. Matters
    most for the online case: at a given arrival_rate and horizon, the
    expected realized-arrival count can be far more than 100 (e.g.
    ~arrival_rate*horizon on average -- see mathformulation.tex's
    arrival-rate derivation), so a too-small max_jobs silently truncates the
    arrival process rather than testing the intended load. None (default)
    keeps the original, offline-unaffected MAX_JOBS=100 for every existing
    invocation.

    Args:
        algorithm: "ppo" or "a2c"
        policy_type: "pointer" or "flat". For A2C, must match which
            architecture's best-params file to load AND which architecture to
            actually build (see the BUG FIX comment in the A2C branch below).
            For PPO, "flat" is the original sb3_contrib "MlpPolicy" (default,
            unchanged); "pointer" builds PointerMaskableActorCriticPolicy
            (Code/policies/pointer_ppo_policy.py) instead -- see
            Future/research/2026-09-16-pointer-network-ppo.md.
        use_curriculum: Whether to use curriculum learning
        stage4_timesteps: Timestep budget for the final curriculum stage
            (100 jobs, horizon=100). Overridable because the 2026-08-10
            training-log entry found the pointer architecture was still
            improving (not plateaued) at the default 200k -- see
            Future/research/training-log.md, S2W4 entry.
        run_tag: Label for this run's archived checkpoints (see
            Code/utils/results_log.py::archive_checkpoint_files). Defaults to
            "<algo>_<policy_type>_s4-<stage4_timesteps>" if not given. The
            canonical (non-archived) checkpoint at MODELS_DIR is still
            overwritten as before -- the archive is a separate, dated copy.
        params_tag: optional suffix (e.g. "tardiness") to train from an
            alternative Optuna params file produced by
            `optuna_tune.py --optimize-for tardiness`, and save to a
            correspondingly-suffixed checkpoint path so it doesn't overwrite
            the default reward-tuned model -- see load_best_params().
        use_potential_shaping: Experiment 4 (Solution 3, Ng/Harada/Russell 1999
            potential-based shaping) ablation flag. Deliberately kept
            orthogonal to params_tag -- this run still loads the *reward*-tuned
            params (params_tag=None) unless params_tag is also given, so the
            ablation isolates shaping as the only changed variable against the
            S2W4 baseline. shaping_gamma is set to this run's own `gamma`
            hyperparameter, not a separate default -- see make_env().
        randomize_instances: Experiment 2 -- train on a fresh random job set
            every episode (Code/training/train_optimized.py::
            make_random_instance_resampler()) instead of the single fixed
            seed=0 instance every prior run in this project used. NOTE: the
            Optuna hyperparameters loaded here were NOT re-tuned for this
            distribution (that re-tuning is a flagged follow-up, not done in
            this pass) -- results should be read with that caveat.
        use_rcpo: Experiment 5 (Lagrangian-constrained optimization --
            RCPO for A2C, Code/policies/a2c_policy.py; PPO-Lagrangian for
            PPO, Code/policies/ppo_lagrangian.py -- see
            Future/research/2026-08-21-rcpo-constrained-tardiness.md and
            2026-09-14-ppo-lagrangian-and-reward-structure.md). Replaces the
            fixed lambda_2 tardiness weight with a Lagrange multiplier that
            adapts during training toward the constraint
            `E[C(tau)] <= rcpo_alpha`. Warm-started from this run's loaded
            `lambda_2` (params["lambda_2"]) rather than an arbitrary value,
            for both algorithms.
        rcpo_alpha/rcpo_lambda_lr/rcpo_lambda_max/rcpo_update_every: see
            MaskableA2C's docstring (A2C) or PPOLagrangianCallback's
            docstring (PPO) for the grounding of each default -- identical
            constraint/update rule for both.
        seed: RNG seed for this run -- seeds numpy/torch globally, the first
            training instance's generation, and (PPO only) MaskablePPO's own
            seed. Running the "same" config at several seeds and reporting
            mean +/- std is standard RL reproducibility practice (Henderson
            et al. 2018, "Deep Reinforcement Learning that Matters," AAAI).
            Non-zero seeds are folded into path_suffix/run_tag defaults below
            so concurrent seed runs don't overwrite each other's canonical
            checkpoints -- pass distinct --run-tag values too when launching
            seeds in parallel, for clarity in the archive folder names.
        n_envs: number of parallel environment copies for PPO rollout
            collection (ignored for A2C, which has no vectorized rollout --
            see build_vec_env()). Defaults to max(1, os.cpu_count() - 1),
            leaving one core free for the main process (optimizer step,
            plotting, checkpoint I/O).
        vec_backend: "subproc" (true multi-process rollout, the default) or
            "dummy" (single-process, no IPC overhead but no multi-core
            speedup) -- see build_vec_env()'s docstring for when to prefer
            "dummy".
        torch_threads: explicit torch.set_num_threads() override. PyTorch's
            CPU backend defaults to using every logical core for intra-op
            (BLAS) parallelism within a single process -- fine for one
            process, but when running several seeds concurrently (see the
            `seed` docstring above) as separate OS processes, each process
            independently defaulting to all cores causes thread
            oversubscription/contention that can slow every process down
            rather than speed any of them up. Pass e.g. os.cpu_count() //
            num_concurrent_seeds when launching multiple seeds at once.
            None (default) leaves PyTorch's own default in place, correct
            for a single solo run.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)

    # The single canonical fixed job instance every fixed-instance result in
    # this project has trained/evaluated on (generate_env_config(seed=0)).
    # `seed` above controls algorithmic randomness only (policy init, action
    # sampling) -- it must never be threaded into instance generation, or
    # different --seed runs silently become different tasks instead of the
    # same task under different algorithmic randomness (Henderson et al.
    # 2018's multi-seed variance protocol requires the task held fixed).
    FIXED_INSTANCE_SEED = 0

    if n_envs is None:
        n_envs = max(1, (os.cpu_count() or 1) - 1)

    seed_suffix = f"_seed{seed}" if seed != 0 else ""

    if run_tag is None:
        run_tag = (
            f"{algorithm}_{policy_type}_s4-{stage4_timesteps}"
            + (f"_{params_tag}" if params_tag else "")
            + ("_shaped" if use_potential_shaping else "")
            + ("_randinst" if randomize_instances else "")
            + ("_rcpo" if use_rcpo else "")
            + seed_suffix
        )

    # use_rcpo now supports both algorithms: A2C via MaskableA2C's own
    # hand-rolled multiplier update (below), PPO via PPOLagrangianCallback
    # (Code/policies/ppo_lagrangian.py) -- extending RCPO+PPO integration,
    # previously flagged as out of scope (2026-08-21-rcpo-constrained-
    # tardiness.md Section 5), per Future/research/
    # 2026-09-14-ppo-lagrangian-and-reward-structure.md.

    # Load best hyperparameters
    params = load_best_params(algorithm, policy_type=policy_type, params_tag=params_tag)

    # Setup directories. A2C paths are tagged with policy_type -- flat and
    # pointer checkpoints have different state_dict shapes and are not
    # interchangeable, so they must not overwrite each other. params_tag (if
    # given) is also appended, so a tardiness-tuned model doesn't overwrite the
    # default reward-tuned one at the same policy_type. seed_suffix is appended
    # too so concurrent multi-seed runs (see `seed` docstring above) write to
    # distinct canonical checkpoint/stage-checkpoint paths instead of racing
    # on the same file.
    TOTAL_TIMESTEPS = 500_000 if use_curriculum else 300_000
    # BUG FIX (this session): online runs at different arrival_rate values are
    # different problem configurations, but nothing here distinguished them --
    # two concurrent --online runs at different rates (same seed/algo/policy_type)
    # would race on the exact same canonical PPO_MODEL_PATH/A2C_MODEL_PATH.
    # Found via a real 6-way concurrent arrival-rate sweep tonight: no actual
    # corruption happened (verified via checkpoint weight hashes -- pure luck,
    # since the differently-sized runs happened to finish at staggered times),
    # but relying on that would be fragile. arrival_rate is formatted with
    # underscores (not '.') to stay filesystem/glob-safe.
    online_suffix = f"_online{str(arrival_rate).replace('.', '_')}" if is_online else ""
    # reward_mode="dense_tardiness" is a genuinely different training objective
    # (see SchedulingEnv.__init__'s docstring) -- must not overwrite the
    # "legacy" checkpoint at the same config, since the whole point of this
    # experiment is comparing the two.
    reward_mode_suffix = f"_{reward_mode}" if reward_mode != "legacy" else ""
    # action_mode="rule_selection" changes the action space size itself
    # (Discrete(8) vs the placement space) -- a fundamentally different,
    # incompatible checkpoint shape, so it needs its own path exactly like
    # reward_mode_suffix above, not a silent overwrite of the placement-mode
    # checkpoint at the same config.
    action_mode_suffix = f"_{action_mode}" if action_mode != "placement" else ""
    weight_suffix = (
        f"_w{job_weight_range[0]}-{job_weight_range[1]}" if job_weight_range is not None else ""
    )
    path_suffix = (
        (f"_{params_tag}" if params_tag else "")
        + ("_shaped" if use_potential_shaping else "")
        + ("_randinst" if randomize_instances else "")
        + ("_rcpo" if use_rcpo else "")
        + online_suffix
        + reward_mode_suffix
        + action_mode_suffix
        + weight_suffix
        + seed_suffix
    )
    # BUG FIX (this session): unlike A2C's path (below), which has always
    # folded policy_type in, PPO's canonical save path ignored it entirely --
    # harmless while PPO only ever had one architecture, but now that
    # policy_type="pointer" is a real option (see pointer_ppo_policy.py), a
    # flat and a pointer PPO run at the same seed would silently overwrite
    # each other's checkpoint at this same path. Gated to "flat" producing no
    # suffix so every existing "ppo_scheduling_optimized..." path/reference
    # keeps meaning exactly what it always did.
    ppo_policy_suffix = "_pointer" if policy_type == "pointer" else ""
    PPO_MODEL_PATH = MODELS_DIR / f"ppo_scheduling_optimized{ppo_policy_suffix}{path_suffix}"
    A2C_MODEL_PATH = MODELS_DIR / f"a2c_{policy_type}_scheduling_optimized{path_suffix}.pt"

    ensure_rl_training_dirs()

    # Live plotter. Keyed by run_tag (not just algorithm) so concurrent
    # multi-seed runs -- which share the same make_run_dir() second-level
    # timestamp often enough when launched together -- still get distinct
    # plot directories instead of racing on the same CSV/PNG.
    plot_run_dir = make_run_dir(str(PLOTS_DIR / "training"), f"{algorithm}_optimized_{run_tag}")
    plotter = LiveTrainingPlotter(save_dir=plot_run_dir)

    # Extract reward penalties from params
    lambda_1 = params.get("lambda_1", 1.0)
    lambda_2 = params.get("lambda_2", 1.0)
    lambda_3 = params.get("lambda_3", 1.0)
    # shaping_gamma matches this run's own discount factor (see make_env()'s
    # docstring on why that matters for the policy-invariance guarantee to hold).
    shaping_gamma = params.get("gamma", 0.99)
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
        # Network architecture: "flat" builds SB3's stock MlpPolicy net_arch
        # from Optuna-tuned layer_size/n_layers/activation (original
        # behavior, unchanged). "pointer" instead uses
        # PointerMaskableActorCriticPolicy (Code/policies/pointer_ppo_policy.py),
        # the same PointerActorCritic architecture A2C has used since
        # 2026-08-09 -- see Future/research/2026-09-16-pointer-network-ppo.md.
        # max_jobs/num_machines/num_resources get filled in once first_env
        # exists below (read off the env, not hardcoded -- mirrors
        # a2c_policy.py's MaskableA2C.__init__ derivation).
        if policy_type == "pointer":
            from Code.policies.pointer_ppo_policy import PointerMaskableActorCriticPolicy

            policy_cls = PointerMaskableActorCriticPolicy
            policy_kwargs = dict(
                embed_dim=params.get("embed_dim", 128),
                hidden=params.get("hidden", 64),
            )
        else:
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

            policy_cls = "MlpPolicy"
            policy_kwargs = dict(
                net_arch=net_arch,
                activation_fn=activation_fn,
            )

        # Curriculum or single-stage training
        if use_curriculum:
            MAX_JOBS = max_jobs_override if max_jobs_override is not None else 100
            curriculum = [
                {"horizon": 20,  "num_jobs": 15,  "timesteps": 50_000},
                {"horizon": 40,  "num_jobs": 30,  "timesteps": 100_000},
                {"horizon": 60,  "num_jobs": 60,  "timesteps": 150_000},
                {"horizon": 100, "num_jobs": 100, "timesteps": stage4_timesteps},
            ]
        else:
            MAX_JOBS = max_jobs_override if max_jobs_override is not None else 100
            curriculum = [
                {"horizon": 100, "num_jobs": 100, "timesteps": TOTAL_TIMESTEPS},
            ]

        # Rescale the Optuna-tuned n_steps/batch_size for n_envs-way parallel
        # rollout collection -- see resolve_ppo_rollout_params()'s docstring
        # (Schulman et al. 2017, PPO, arXiv:1707.06347, Algorithm 1: N actors
        # x T steps, not T alone, is the tuned quantity).
        tuned_n_steps = params.get("n_steps", 2048)
        tuned_batch_size = params.get("batch_size", 256)
        effective_n_steps, resolved_batch_size, buffer_size = resolve_ppo_rollout_params(
            tuned_n_steps, tuned_batch_size, n_envs
        )
        print(
            f"Parallel rollout: n_envs={n_envs} (backend={vec_backend}), "
            f"n_steps {tuned_n_steps} -> {effective_n_steps} per env "
            f"(buffer size {buffer_size}), batch_size {tuned_batch_size} -> {resolved_batch_size}\n"
        )

        # Create first (vectorized) environment. save_config=False unless
        # this is a single-stage run (--no-curriculum) -- see build_vec_env's
        # BUG FIX docstring: only the FINAL curriculum stage should ever
        # write the shared ENV_CONFIG_PATH file.
        first_env = build_vec_env(
            n_envs,
            vec_backend,
            FIXED_INSTANCE_SEED,
            save_config=(len(curriculum) == 1),
            horizon=curriculum[0]["horizon"],
            num_jobs=curriculum[0]["num_jobs"],
            max_jobs=MAX_JOBS,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            idle_penalty=idle_penalty,
            invalid_penalty=invalid_penalty,
            use_potential_shaping=use_potential_shaping,
            shaping_gamma=shaping_gamma,
            randomize_instances=randomize_instances,
            is_online=is_online,
            arrival_rate=arrival_rate,
            reward_mode=reward_mode,
            job_size_distribution=job_size_distribution,
            job_weight_range=job_weight_range,
            action_mode=action_mode,
        )

        if policy_type == "pointer":
            policy_kwargs["max_jobs"] = first_env.get_attr("max_jobs")[0]
            policy_kwargs["num_machines"] = first_env.get_attr("num_machines")[0]
            policy_kwargs["num_resources"] = first_env.get_attr("num_resources")[0]

        # Create PPO model with optimized hyperparameters
        model = MaskablePPO(
            policy_cls,
            first_env,
            verbose=1,
            tensorboard_log=str(LOG_DIR),
            learning_rate=params.get("learning_rate", 3e-4),
            n_steps=effective_n_steps,
            batch_size=resolved_batch_size,
            n_epochs=params.get("n_epochs", 4),
            gamma=params.get("gamma", 0.99),
            gae_lambda=params.get("gae_lambda", 0.95),
            ent_coef=params.get("ent_coef", 0.05),
            clip_range=params.get("clip_range", 0.2),
            vf_coef=params.get("vf_coef", 0.5),
            max_grad_norm=params.get("max_grad_norm", 0.5),
            seed=seed,
            policy_kwargs=policy_kwargs,
        )

        # PPO-Lagrangian (see Code/policies/ppo_lagrangian.py) -- warm-started
        # from this run's own tuned lambda_2, same convention as A2C's RCPO.
        # Built once, reused across every curriculum stage below (not
        # recreated per stage) so the adapted multiplier survives each
        # stage's env swap -- see PPOLagrangianCallback's docstring.
        learn_callbacks = [plotter]
        lagrangian_callback = None
        if use_rcpo:
            lagrangian_callback = PPOLagrangianCallback(
                alpha=rcpo_alpha,
                lambda_init=lambda_2,
                lambda_lr=rcpo_lambda_lr,
                lambda_max=rcpo_lambda_max,
                update_every_episodes=rcpo_update_every,
            )
            learn_callbacks.append(lagrangian_callback)
            print(
                f"PPO-Lagrangian enabled: lambda_2 warm-started at {lambda_2} (this run's "
                f"tuned value), alpha={rcpo_alpha}, lambda_lr={rcpo_lambda_lr}, "
                f"lambda_max={rcpo_lambda_max}, update_every={rcpo_update_every} episodes\n"
            )

        print(f"\n{'='*80}")
        print(f"Training PPO with optimized hyperparameters")
        print(f"Curriculum stages: {len(curriculum)}")
        print(f"Total timesteps: {sum(stage['timesteps'] for stage in curriculum)}")
        print(f"{'='*80}\n")

        # Curriculum training loop
        stage_ckpts = []
        for i, stage in enumerate(curriculum):
            print(f"\n--- Stage {i+1}/{len(curriculum)} ---")
            print(f"  Horizon: {stage['horizon']}, Jobs: {stage['num_jobs']}, Timesteps: {stage['timesteps']}")

            # Curriculum stages change num_jobs/horizon, which changes each
            # sub-env's observation layout -- close the previous stage's
            # VecEnv (terminating its SubprocVecEnv worker processes, if any)
            # before spawning the next-shape one.
            model.get_env().close()
            env = build_vec_env(
                n_envs,
                vec_backend,
                FIXED_INSTANCE_SEED,
                save_config=(i == len(curriculum) - 1),
                horizon=stage["horizon"],
                num_jobs=stage["num_jobs"],
                max_jobs=MAX_JOBS,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                lambda_3=lambda_3,
                idle_penalty=idle_penalty,
                invalid_penalty=invalid_penalty,
                use_potential_shaping=use_potential_shaping,
                shaping_gamma=shaping_gamma,
                randomize_instances=randomize_instances,
                is_online=is_online,
                arrival_rate=arrival_rate,
                reward_mode=reward_mode,
                job_size_distribution=job_size_distribution,
                job_weight_range=job_weight_range,
                action_mode=action_mode,
            )

            model.set_env(env)

            model.learn(
                total_timesteps=stage["timesteps"],
                # Keyed by run_tag (not a shared constant) -- SB3 auto-numbers
                # tb_log_name dirs (_1, _2, ...) by listing what already
                # exists at learn()-call time, which races when multiple seed
                # processes start concurrently and could compute the same
                # "next" number before either has created its directory.
                tb_log_name=f"ppo_scheduling_optimized_{run_tag}",
                progress_bar=True,
                callback=learn_callbacks,
                reset_num_timesteps=False,
            )

            # Per-stage checkpoint -- see the matching comment in
            # train_rl_agent.py for rationale (diagnosability without rerunning
            # the whole curriculum).
            stage_ckpt = MODELS_DIR / f"ppo_optimized_stage{i}_h{stage['horizon']}{ppo_policy_suffix}{path_suffix}"
            model.save(stage_ckpt)
            print(f"  Saved stage checkpoint: {stage_ckpt}")
            stage_ckpts.append(Path(str(stage_ckpt) + ".zip"))  # SB3 appends .zip itself

        # Save PPO model
        model.save(PPO_MODEL_PATH)
        print(f"\nPPO training complete. Model saved to: {PPO_MODEL_PATH}\n")

        # Close the final stage's VecEnv (terminates SubprocVecEnv worker
        # processes cleanly rather than leaving them to the interpreter exit).
        model.get_env().close()

        # Record this run's exact provenance (resolved hyperparameters, seed,
        # parallelism settings, git commit) alongside the checkpoints so a
        # later paper figure/table can be traced back to what produced it.
        metadata_path = MODELS_DIR / f"run_metadata{path_suffix}.json"
        save_run_metadata(
            metadata_path,
            algorithm=algorithm,
            policy_type=policy_type,
            run_tag=run_tag,
            seed=seed,
            instance_seed=FIXED_INSTANCE_SEED,
            n_envs=n_envs,
            vec_backend=vec_backend,
            torch_threads=torch_threads,
            tuned_n_steps=tuned_n_steps,
            effective_n_steps=effective_n_steps,
            tuned_batch_size=tuned_batch_size,
            resolved_batch_size=resolved_batch_size,
            optuna_params=params,
        )

        extra_archive_files = []
        if use_rcpo:
            # lambda_history: (timestep, lambda_value, mean_episode_cost)
            # tuples logged at every slow-timescale multiplier update -- same
            # shape/purpose as MaskableA2C's, kept alongside the checkpoint so
            # the multiplier's convergence trajectory can be inspected/
            # plotted without rerunning training.
            lambda_history_path = MODELS_DIR / f"ppo_lambda_history{path_suffix}.json"
            with open(lambda_history_path, "w") as f:
                json.dump(lagrangian_callback.lambda_history, f, indent=2)
            print(f"  Saved PPO-Lagrangian lambda history: {lambda_history_path}")
            extra_archive_files.append(lambda_history_path)

        # Archive this run's checkpoints under a dated/tagged folder so the
        # next run overwriting PPO_MODEL_PATH doesn't destroy this one -- see
        # Code/utils/results_log.py.
        archive_dir = archive_checkpoint_files(
            [Path(str(PPO_MODEL_PATH) + ".zip"), *stage_ckpts, ENV_CONFIG_PATH, metadata_path, *extra_archive_files],
            tag=run_tag,
        )
        print(f"Archived checkpoints to: {archive_dir}\n")

    # ==================== A2C TRAINING ====================
    else:
        from Code.policies.a2c_policy import MaskableA2C

        # For A2C, we need to modify the class to accept hyperparameters
        # This is similar to what we did in optuna_tune.py

        if use_curriculum:
            MAX_JOBS = max_jobs_override if max_jobs_override is not None else 100
            curriculum = [
                {"horizon": 20,  "num_jobs": 15,  "timesteps": 50_000},
                {"horizon": 40,  "num_jobs": 30,  "timesteps": 100_000},
                {"horizon": 60,  "num_jobs": 60,  "timesteps": 150_000},
                {"horizon": 100, "num_jobs": 100, "timesteps": stage4_timesteps},
            ]
        else:
            MAX_JOBS = max_jobs_override if max_jobs_override is not None else 100
            curriculum = [
                {"horizon": 100, "num_jobs": 100, "timesteps": TOTAL_TIMESTEPS},
            ]

        # Create first environment. NOTE: MaskableA2C is a hand-rolled agent
        # (Code/policies/a2c_policy.py), not an SB3 model -- it has no VecEnv
        # rollout support, so A2C training stays single-env regardless of
        # n_envs/vec_backend (those only apply to the PPO branch above).
        first_env = make_env(
            seed=FIXED_INSTANCE_SEED,
            save_config=(len(curriculum) == 1),
            horizon=curriculum[0]["horizon"],
            num_jobs=curriculum[0]["num_jobs"],
            max_jobs=MAX_JOBS,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            idle_penalty=idle_penalty,
            invalid_penalty=invalid_penalty,
            use_potential_shaping=use_potential_shaping,
            shaping_gamma=shaping_gamma,
            randomize_instances=randomize_instances,
            is_online=is_online,
            arrival_rate=arrival_rate,
            reward_mode=reward_mode,
            job_size_distribution=job_size_distribution,
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
        rcpo_kwargs = (
            dict(
                use_rcpo=True,
                rcpo_alpha=rcpo_alpha,
                rcpo_lambda_init=lambda_2,
                rcpo_lambda_lr=rcpo_lambda_lr,
                rcpo_lambda_max=rcpo_lambda_max,
                rcpo_update_every_episodes=rcpo_update_every,
            )
            if use_rcpo else {}
        )
        agent = MaskableA2C(
            first_env, device="cpu", policy_type=policy_type, policy_kwargs=policy_kwargs, **rcpo_kwargs
        )
        if use_rcpo:
            print(
                f"RCPO enabled: lambda_2 warm-started at {lambda_2} (this run's tuned "
                f"value), alpha={rcpo_alpha}, lambda_lr={rcpo_lambda_lr}, "
                f"lambda_max={rcpo_lambda_max}, update_every={rcpo_update_every} episodes\n"
            )

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
        stage_ckpts = []
        for i, stage in enumerate(curriculum):
            print(f"\n--- Stage {i+1}/{len(curriculum)} ---")
            print(f"  Horizon: {stage['horizon']}, Jobs: {stage['num_jobs']}, Timesteps: {stage['timesteps']}")

            env = make_env(
                seed=FIXED_INSTANCE_SEED,
                save_config=(i == len(curriculum) - 1),
                horizon=stage["horizon"],
                num_jobs=stage["num_jobs"],
                max_jobs=MAX_JOBS,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                lambda_3=lambda_3,
                idle_penalty=idle_penalty,
                invalid_penalty=invalid_penalty,
                use_potential_shaping=use_potential_shaping,
                shaping_gamma=shaping_gamma,
                randomize_instances=randomize_instances,
                is_online=is_online,
                arrival_rate=arrival_rate,
                reward_mode=reward_mode,
                job_size_distribution=job_size_distribution,
            )

            agent.env = env
            agent.train(total_timesteps=stage["timesteps"], plotter=plotter)

            # Per-stage checkpoint -- see the matching comment in
            # train_rl_agent.py for rationale.
            stage_ckpt = MODELS_DIR / f"a2c_{policy_type}_optimized_stage{i}_h{stage['horizon']}{path_suffix}.pt"
            torch.save(agent.model.state_dict(), stage_ckpt)
            print(f"  Saved stage checkpoint: {stage_ckpt}")
            stage_ckpts.append(stage_ckpt)

        # Save A2C model
        torch.save(agent.model.state_dict(), A2C_MODEL_PATH)
        print(f"\nA2C training complete. Model saved to: {A2C_MODEL_PATH}\n")

        extra_archive_files = []
        if use_rcpo:
            # lambda_history: (timestep, lambda_value, mean_episode_cost) tuples
            # logged at every slow-timescale multiplier update -- see
            # MaskableA2C.train() -- kept alongside the checkpoint so the
            # multiplier's convergence trajectory can be inspected/plotted
            # without rerunning training.
            lambda_history_path = MODELS_DIR / f"a2c_{policy_type}_lambda_history{path_suffix}.json"
            with open(lambda_history_path, "w") as f:
                json.dump(agent.lambda_history, f, indent=2)
            print(f"  Saved RCPO lambda history: {lambda_history_path}")
            extra_archive_files.append(lambda_history_path)

        # Record this run's exact provenance alongside the checkpoints, same
        # as the PPO branch -- see save_run_metadata()'s docstring.
        metadata_path = MODELS_DIR / f"run_metadata{path_suffix}.json"
        save_run_metadata(
            metadata_path,
            algorithm=algorithm,
            policy_type=policy_type,
            run_tag=run_tag,
            seed=seed,
            optuna_params=params,
        )
        extra_archive_files.append(metadata_path)

        # Archive this run's checkpoints under a dated/tagged folder so the
        # next run overwriting A2C_MODEL_PATH doesn't destroy this one -- see
        # Code/utils/results_log.py.
        archive_dir = archive_checkpoint_files(
            [A2C_MODEL_PATH, *stage_ckpts, ENV_CONFIG_PATH, *extra_archive_files],
            tag=run_tag,
        )
        print(f"Archived checkpoints to: {archive_dir}\n")

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
        default=None,
        choices=["pointer", "flat"],
        help="Which architecture's best-params file to load and build. "
             "Defaults to 'pointer' for --algo a2c (unchanged) and 'flat' for "
             "--algo ppo (unchanged, sb3_contrib's stock MlpPolicy) if not "
             "given -- pass --policy-type pointer explicitly to use "
             "PointerMaskableActorCriticPolicy for PPO instead (see "
             "Future/research/2026-09-16-pointer-network-ppo.md).",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning (single-stage training)"
    )
    parser.add_argument(
        "--stage4-timesteps",
        type=int,
        default=200_000,
        help="Timestep budget for the final curriculum stage (100 jobs, horizon=100)."
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Label for this run's archived checkpoints (rl_training/models/archive/). "
             "Defaults to '<algo>_<policy_type>_s4-<stage4_timesteps>[_<params-tag>]'."
    )
    parser.add_argument(
        "--params-tag",
        type=str,
        default=None,
        help="Load an alternative Optuna best-params file (e.g. 'tardiness', produced by "
             "optuna_tune.py --optimize-for tardiness) instead of the default reward-tuned "
             "one, and save checkpoints to a correspondingly-suffixed path so they don't "
             "overwrite the default model."
    )
    parser.add_argument(
        "--use-potential-shaping",
        action="store_true",
        help="Experiment 4: enable Solution 3 potential-based reward shaping (Ng, Harada, "
             "Russell 1999 -- see SchedulingEnv._compute_potential()). Off by default so "
             "it's A/B-able against the unshaped baseline. shaping_gamma is set to this "
             "run's own tuned 'gamma' hyperparameter."
    )
    parser.add_argument(
        "--randomize-instances",
        action="store_true",
        help="Experiment 2: draw a fresh random job set every episode instead of reusing "
             "the single fixed seed=0 instance every prior run trained/evaluated on -- see "
             "make_random_instance_resampler(). Optuna hyperparameters are NOT re-tuned "
             "for this distribution by this flag alone."
    )
    parser.add_argument(
        "--use-rcpo",
        action="store_true",
        help="Experiment 5: replace the fixed lambda_2 tardiness weight with a Lagrange "
             "multiplier adapted during training toward a tardiness constraint (Tessler, "
             "Mankowitz, Mannor, ICLR 2019, arXiv:1805.11074 for A2C's RCPO; Ray, Achiam, "
             "Amodei 2019, arXiv:1910.01708 for PPO-Lagrangian). Supported for both --algo "
             "ppo and --algo a2c. See Future/research/2026-08-21-rcpo-constrained-tardiness.md "
             "and 2026-09-14-ppo-lagrangian-and-reward-structure.md."
    )
    parser.add_argument(
        "--rcpo-alpha",
        type=float,
        default=0.0,
        help="RCPO constraint threshold on E[C(tau)] (weighted normalised tardiness "
             "per episode). Default 0.0 -- see the dated doc's Section 2 for why this "
             "doesn't require literally-zero tardiness to be achieved."
    )
    parser.add_argument(
        "--rcpo-lambda-lr",
        type=float,
        default=0.01,
        help="Step size for the multiplier's projected-ascent update."
    )
    parser.add_argument(
        "--rcpo-lambda-max",
        type=float,
        default=50.0,
        help="Upper projection bound for the multiplier (lower bound is always 0)."
    )
    parser.add_argument(
        "--rcpo-update-every",
        type=int,
        default=5,
        help="Number of completed episodes averaged into one multiplier update."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for this run (numpy/torch global seed, first training "
             "instance's generation, and PPO's own MaskablePPO seed). Run the "
             "same config at several seeds and report mean +/- std for the "
             "final results -- see Henderson et al. 2018, 'Deep Reinforcement "
             "Learning that Matters,' AAAI. Non-zero seeds are folded into the "
             "default --run-tag/checkpoint filenames so concurrent seed runs "
             "don't collide; pass an explicit --run-tag too for clarity."
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="PPO only: number of parallel environment copies for rollout "
             "collection. Defaults to max(1, os.cpu_count() - 1). Ignored for "
             "A2C (MaskableA2C has no vectorized-rollout support)."
    )
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="subproc",
        choices=["subproc", "dummy"],
        help="PPO only: 'subproc' for true multi-process rollout (default), "
             "'dummy' for single-process vectorization (no IPC overhead, no "
             "multi-core speedup). Time both at your target --n-envs before "
             "the final run -- see build_vec_env()'s docstring. On this "
             "project's CPU-only SchedulingEnv, benchmarking found 'dummy' "
             "at least as fast as 'subproc' -- the env is cheap enough that "
             "SubprocVecEnv's per-step IPC offsets its own parallelism gain."
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Explicit torch.set_num_threads() override -- set this when "
             "launching multiple --seed runs concurrently (e.g. cpu_count // "
             "num_concurrent_seeds) to avoid every process independently "
             "defaulting to all cores and oversubscribing/contending. Leave "
             "unset for a single solo run."
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Train the online (dynamic-arrival) case instead of the offline "
             "one -- Poisson job arrivals (Code/env/arrival_process.py) rather "
             "than every job known at t=0. Requires --arrival-rate. See "
             "mathformulation.tex's online MDP subsection."
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="--online only: mean Poisson arrival rate (jobs/tick). Required "
             "when --online is set. See mathformulation.tex's arrival-rate "
             "selection derivation (rho = arrival_rate/12 under this "
             "project's default machine/resource parameters) for how to "
             "choose a value."
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Override the job-slot array capacity (offline and online "
             "alike, default 100). Matters most for --online: at a given "
             "arrival_rate and horizon, expected realized arrivals can be "
             "far more than 100 (roughly arrival_rate*horizon on average) -- "
             "a too-small value silently truncates the arrival process "
             "instead of testing the intended load."
    )
    parser.add_argument(
        "--job-size-distribution",
        type=str,
        choices=["uniform", "lognormal"],
        default="uniform",
        help="--online only: 'uniform' (default, unchanged) or 'lognormal' "
             "(2026-09-17, S2W10 -- heavy-tailed job duration/resource "
             "demand, mean-matched to the uniform baseline so rho stays "
             "calibrated; see Code/env/arrival_process.py's docstring for "
             "the Google/Azure cluster-trace grounding). Makes 'no idea "
             "what's coming next' matter without inflating rho into "
             "disguised-offline overload."
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        choices=["placement", "rule_selection"],
        default="placement",
        help="'placement' (default, unchanged) picks a raw (job, machine) pair. "
             "'rule_selection' (2026-09-18, S2W10 -- Option 1, "
             "Future/research/2026-09-17-action-space-reduction.md) wraps the env "
             "with RuleSelectionGymSchedulingEnv so the policy instead picks among "
             "7 classical priority rules + idle (Discrete(8)) -- lets Option 1 use "
             "this file's real curriculum/Optuna-hyperparameter machinery instead "
             "of only train_action_space_variant.py's standalone flat-timestep "
             "trainer. Only wired through the PPO (non-RCPO) path."
    )
    parser.add_argument(
        "--job-weight-min", type=int, default=None,
        help="2026-09-18, S2W10: with --job-weight-max, draws each job's weight "
             "i.i.d. Uniform{min,...,max-1} instead of the historic constant 1.0 "
             "(previously dead code for WSPT/ATC's w_j/p_j term). Omit both for "
             "unchanged legacy behaviour."
    )
    parser.add_argument("--job-weight-max", type=int, default=None,
                         help="See --job-weight-min (numpy.integers convention: exclusive).")
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="legacy",
        choices=["legacy", "dense_tardiness"],
        help="'legacy' (default, unchanged): tardiness charged once as a "
             "lump sum at scheduling time, plus flat +3.0/+50 completion "
             "bonuses. 'dense_tardiness': exact per-tick decomposition of "
             "weighted tardiness (DeepRM/Decima-style dense reward, see "
             "SchedulingEnv.__init__'s reward_mode docstring and "
             "Future/research/2026-09-17-dense-tardiness-reward.md), no flat "
             "completion bonuses. Gets its own checkpoint path (see "
             "path_suffix) so it never overwrites a 'legacy' run at the same "
             "config."
    )

    args = parser.parse_args()

    # Resolve the algorithm-conditional default (see --policy-type's help):
    # a2c keeps its original "pointer" default, ppo keeps its original "flat"
    # (MlpPolicy) default -- neither changes unless --policy-type is passed
    # explicitly.
    if args.policy_type is None:
        args.policy_type = "pointer" if args.algo == "a2c" else "flat"

    if args.online and args.arrival_rate is None:
        parser.error("--online requires --arrival-rate")
    if (args.job_weight_min is None) != (args.job_weight_max is None):
        parser.error("--job-weight-min and --job-weight-max must be given together")
    if args.action_mode == "rule_selection" and args.policy_type == "pointer":
        parser.error("--action-mode rule_selection needs --policy-type flat -- "
                      "PointerActorCritic is sized for the placement action space, "
                      "not Option 1's Discrete(8) rule-selection space.")
    job_weight_range = (
        (args.job_weight_min, args.job_weight_max) if args.job_weight_min is not None else None
    )

    train_with_optimized_params(
        algorithm=args.algo,
        policy_type=args.policy_type,
        use_curriculum=not args.no_curriculum,
        stage4_timesteps=args.stage4_timesteps,
        run_tag=args.run_tag,
        params_tag=args.params_tag,
        use_potential_shaping=args.use_potential_shaping,
        randomize_instances=args.randomize_instances,
        reward_mode=args.reward_mode,
        use_rcpo=args.use_rcpo,
        rcpo_alpha=args.rcpo_alpha,
        rcpo_lambda_lr=args.rcpo_lambda_lr,
        rcpo_lambda_max=args.rcpo_lambda_max,
        rcpo_update_every=args.rcpo_update_every,
        seed=args.seed,
        n_envs=args.n_envs,
        vec_backend=args.vec_backend,
        torch_threads=args.torch_threads,
        is_online=args.online,
        arrival_rate=args.arrival_rate,
        max_jobs_override=args.max_jobs,
        job_size_distribution=args.job_size_distribution,
        job_weight_range=job_weight_range,
        action_mode=args.action_mode,
    )
