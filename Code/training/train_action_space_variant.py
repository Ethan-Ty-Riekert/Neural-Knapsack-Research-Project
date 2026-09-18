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

    python -m Code.training.train_action_space_variant --option 1
    python -m Code.training.train_action_space_variant --option 2
    python -m Code.training.train_action_space_variant --option 3 --timesteps 300000

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

    python -m Code.training.train_action_space_variant --option 1 --online \\
        --arrival-rate 3 --online-horizon 100 --online-max-jobs 400 \\
        --job-size-distribution lognormal
"""
import argparse
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from Code.env.scheduling_env import SchedulingEnv
from Code.env.env_config import generate_env_config
from Code.env.online_scheduling_env import OnlineSchedulingEnv
from Code.env.arrival_process import generate_poisson_arrivals
from Code.env.gym_scheduling_wrapper import GymSchedulingEnv
from Code.env.online_gym_wrapper import OnlineGymSchedulingEnv
from Code.env.rule_selection_gym_wrapper import RuleSelectionGymSchedulingEnv
from Code.env.priority_only_gym_wrapper import PriorityOnlyGymSchedulingEnv
from Code.policies.priority_pointer_ppo_policy import PriorityPointerMaskableActorCriticPolicy
from Code.training.train_optimized import make_online_resampler, make_random_instance_resampler
from Code.utils.paths import MODELS_DIR, ensure_rl_training_dirs


def mask_fn(env):
    return env.get_action_mask()


def make_base_gym_env(seed=0, reward_mode="legacy", use_potential_shaping=False, shaping_gamma=0.99,
                       randomize_instances=False, job_weight_range=None):
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
    Code.training.train_optimized.make_random_instance_resampler() (the
    exact same resampler every prior randomized-instance PPO/A2C run in
    this project used) as GymSchedulingEnv's job_resampler, so every episode
    draws a fresh instance instead of reusing the one built here for
    construction.
    """
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
    )
    job_resampler = (
        make_random_instance_resampler(config["num_jobs"], config["num_machines"], config["horizon"],
                                        job_weight_range=job_weight_range)
        if randomize_instances else None
    )
    return GymSchedulingEnv(base_env, max_jobs=100, job_resampler=job_resampler)


def make_online_base_gym_env(arrival_rate, horizon, max_jobs, job_size_distribution,
                              seed=0, num_machines=10, num_resources=4, use_resampler=True,
                              reward_mode="legacy", use_potential_shaping=False, shaping_gamma=0.99,
                              job_weight_range=None):
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
    )
    job_resampler = (
        make_online_resampler(arrival_rate, horizon, max_jobs, num_machines, num_resources,
                               job_size_distribution=job_size_distribution, job_weight_range=job_weight_range)
        if use_resampler else None
    )
    return OnlineGymSchedulingEnv(base_env, max_jobs=max_jobs, job_resampler=job_resampler)


def build_env_and_policy(option: str, full_gym_env=None):
    if full_gym_env is None:
        full_gym_env = make_base_gym_env()

    if option == "1":
        env = RuleSelectionGymSchedulingEnv(full_gym_env)
        policy, policy_kwargs = "MlpPolicy", {}
    elif option in ("2", "3"):
        use_atc = option == "3"
        env = PriorityOnlyGymSchedulingEnv(full_gym_env, use_atc=use_atc)
        policy = PriorityPointerMaskableActorCriticPolicy
        policy_kwargs = dict(
            max_jobs=env.max_jobs,
            num_machines=env.num_machines,
            num_resources=env.num_resources,
            use_atc=use_atc,
        )
    else:
        raise ValueError(f"Unknown option {option!r} (expected '1', '2', or '3')")

    monitored = Monitor(ActionMasker(env, mask_fn))
    return monitored, policy, policy_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", choices=["1", "2", "3"], required=True,
                         help="1=hyper-heuristic rule selection, 2=raw-feature priority "
                              "learning, 3=ATC-primed priority learning.")
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
    parser.add_argument("--reward-mode", choices=["legacy", "dense_tardiness"], default="legacy",
                         help="2026-09-17 follow-up: dense_tardiness was only tested against the "
                              "old (huge) action space and ruled out there -- untested against "
                              "the winning action-space design until now.")
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
    args = parser.parse_args()

    if args.online and args.arrival_rate is None:
        parser.error("--online requires --arrival-rate")
    if (args.job_weight_min is None) != (args.job_weight_max is None):
        parser.error("--job-weight-min and --job-weight-max must be given together")
    job_weight_range = (
        (args.job_weight_min, args.job_weight_max) if args.job_weight_min is not None else None
    )

    ensure_rl_training_dirs()

    if args.online:
        full_gym_env = make_online_base_gym_env(
            args.arrival_rate, args.online_horizon, args.online_max_jobs,
            args.job_size_distribution,
            reward_mode=args.reward_mode, use_potential_shaping=args.use_potential_shaping,
            job_weight_range=job_weight_range,
        )
    else:
        full_gym_env = make_base_gym_env(
            reward_mode=args.reward_mode, use_potential_shaping=args.use_potential_shaping,
            randomize_instances=args.randomize_instances, job_weight_range=job_weight_range,
        )

    env, policy, policy_kwargs = build_env_and_policy(args.option, full_gym_env=full_gym_env)

    model = MaskablePPO(
        policy, env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(MODELS_DIR / "tb_action_space"),
        ent_coef=args.ent_coef,
    )

    print(f"Option {args.option}: online={args.online}, action_space={env.action_space}, "
          f"obs_dim={env.observation_space.shape[0]}, timesteps={args.timesteps}")

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, tb_log_name=f"option{args.option}{'_' + args.save_tag if args.save_tag else ''}")
    elapsed_min = (time.time() - t0) / 60.0

    tag_suffix = f"_{args.save_tag}" if args.save_tag else ""
    save_path = MODELS_DIR / f"action_space_option{args.option}_ppo{tag_suffix}.zip"
    model.save(str(save_path))
    print(f"Option {args.option}: trained {args.timesteps} timesteps in {elapsed_min:.1f} min, "
          f"saved to {save_path}")


if __name__ == "__main__":
    main()
