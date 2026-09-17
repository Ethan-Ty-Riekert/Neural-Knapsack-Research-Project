"""pso.py - Particle Swarm Optimization metaheuristic baseline.

Grounded in:
- Kennedy, J., & Eberhart, R. (1995). "Particle Swarm Optimization."
  Proceedings of ICNN'95 -- base velocity/position update rule.
- Shi, Y., & Eberhart, R. (1998). "A modified particle swarm optimizer."
  Proceedings of IEEE ICEC -- the inertia-weight term used below, the
  standard convergence-stabilising extension of the 1995 rule.
- Tasgetiren, M. F., Liang, Y.-C., Sevkli, M., & Gencyilmaz, G. (2004).
  "Particle swarm optimization algorithm for makespan and total flowtime
  minimization in the permutation flowshop sequencing problem." Smallest-
  Position-Value (SPV) rule for decoding a continuous particle position
  into a discrete priority order -- see _decode_priorities().
- Rodriguez, M. A., & Buyya, R. (2014), cited via this project's own
  literature review (NotesForAI/ResearchProjectAsOf_23-07-2026.pdf, p.7-8):
  PSO applied to deadline-constrained workflow scheduling -- the
  project-literature grounding for choosing PSO (over e.g. a genetic
  algorithm) as this project's metaheuristic baseline.

Unlike Code/baselines/priority_rules.py + placement_rules.py (fixed,
universal rules), PSO must be re-optimized per problem instance -- it
searches for a good priority ordering for THIS specific job set rather than
applying a rule that works for any instance. So this module exposes
optimize_and_run(config, ...) rather than a registry.HEURISTICS-style entry,
returning the exact same result schema as eval_rl_agent.py's run_model()/
run_heuristic() for direct comparability, and is a genuine search cost (many
env episodes per instance), not an instant-inference method -- report its
wall-clock cost honestly wherever it's compared against anything else.
"""
import time

import numpy as np

from Code.evaluation.eval_rl_agent import make_env


def _decode_priorities(position, num_jobs, num_machines):
    """Smallest-Position-Value (SPV) decoding (Tasgetiren et al. 2004):
    argsort a continuous vector to get a discrete priority rank per
    job/machine. Lower rank = higher priority, matching
    Code.baselines.registry's "min key wins" priority-rule convention."""
    job_pos = position[:num_jobs]
    machine_pos = position[num_jobs:num_jobs + num_machines]
    job_rank = np.argsort(np.argsort(job_pos))
    machine_rank = np.argsort(np.argsort(machine_pos))
    return job_rank, machine_rank


def _simulate(position, config, num_jobs, num_machines):
    """Replay one episode under the priority order `position` decodes to,
    via the same SchedulingEnv/GymSchedulingEnv every other baseline uses.
    Action selection follows exactly the same "pick min-priority feasible
    job, then min-priority feasible machine" structure as
    Code.baselines.registry._make_priority_placement -- the only difference
    is the priority values come from PSO's learned position instead of a
    hand-picked formula, so an illegal/infeasible action is structurally
    impossible here too (only ever chosen among mask-feasible actions)."""
    job_rank, machine_rank = _decode_priorities(position, num_jobs, num_machines)

    env = make_env(config)
    obs, info = env.reset()
    base_env = env.env.env
    idle_action = env.env.max_jobs * base_env.num_machines

    def decode(a):
        return a // base_env.num_machines, a % base_env.num_machines

    done = truncated = False
    rewards = []
    utilisation_over_time = []
    initial_capacity = base_env.capacity[:, :, 0].copy()
    horizon = base_env.horizon

    while not (done or truncated):
        mask = env.env.get_action_mask()
        valid = [a for a, ok in enumerate(mask) if ok]
        job_actions = [a for a in valid if a != idle_action]
        if not job_actions:
            action = idle_action
        else:
            unique_jobs = {decode(a)[0] for a in job_actions}
            job = min(unique_jobs, key=lambda j: (job_rank[j], j))
            feasible_machines = sorted({decode(a)[1] for a in job_actions if decode(a)[0] == job})
            machine = min(feasible_machines, key=lambda m: (machine_rank[m], m))
            action = job * base_env.num_machines + machine

        obs, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))

        t_idx = min(base_env.time - 1, horizon - 1)
        used = initial_capacity - base_env.capacity[:, :, t_idx]
        utilisation_over_time.append(np.mean(used / (initial_capacity + 1e-8), axis=1))

    return {
        "total_reward": float(np.sum(rewards)),
        "tardiness": base_env.tardiness.copy(),
        "late_jobs": int((base_env.tardiness > 0).sum()),
        "jobs_scheduled": int((base_env.start_times != -1).sum()),
        "utilisation_over_time": np.array(utilisation_over_time),
    }


def optimize_and_run(config, num_jobs, num_machines, swarm_size=15, iterations=30,
                      w=0.7, c1=1.5, c2=1.5, v_max=4.0, seed=None, fitness="reward"):
    """Run PSO to find a good (job-priority, machine-priority) encoding for
    one problem instance, then return the best solution found's episode
    stats (same schema as run_model()/run_heuristic()) plus search
    diagnostics (wall-clock seconds, fitness curve) for honest reporting --
    see this module's docstring for why PSO is a search cost, not an
    instant-inference baseline.

    fitness: "reward" (default, unchanged from the original S2W6 baseline --
    total episode reward, the same objective every other baseline/RL method
    in this comparison was scored on) or "tardiness" (added 2026-09-15,
    S2W9: PSO maximizes fitness by convention here, so this minimizes total
    tardiness by using its negation as the fitness value). Added because the
    "reward" variant was already shown (2026-08-28 PSO baseline doc) to find
    the same high-reward/high-tardiness reward-hacking exploit RL does --
    this variant asks the different, complementary question "what's the best
    achievable tardiness at this instance scale, using a method with no
    reward-shaping/hyperparameter-tuning surface to get wrong," a full-scale
    (not small-instance-limited, unlike CP-SAT) complement to the proven
    optimum in 2026-09-14-ppo-lagrangian-and-reward-structure.md Section 7.

    swarm_size=15, iterations=30 (450 episode evaluations per instance) is a
    deliberately modest budget: unlike every other baseline in this
    comparison, PSO must re-run an entire episode per fitness evaluation, so
    evaluating it on the standard 50 held-out instances this project uses
    for everything else would cost ~50x this budget. See the dated doc for
    the specific instance count actually run and why.
    """
    if fitness not in ("reward", "tardiness"):
        raise ValueError(f"fitness must be 'reward' or 'tardiness', got {fitness!r}")

    rng = np.random.default_rng(seed)
    dim = num_jobs + num_machines

    positions = rng.uniform(-4.0, 4.0, size=(swarm_size, dim))
    velocities = rng.uniform(-1.0, 1.0, size=(swarm_size, dim))

    pbest_pos = positions.copy()
    pbest_fitness = np.full(swarm_size, -np.inf)
    gbest_pos = None
    gbest_fitness = -np.inf
    fitness_curve = []

    start = time.time()
    for _ in range(iterations):
        for i in range(swarm_size):
            sim_result = _simulate(positions[i], config, num_jobs, num_machines)
            fitness_value = (
                sim_result["total_reward"] if fitness == "reward"
                else -float(sim_result["tardiness"].sum())
            )
            if fitness_value > pbest_fitness[i]:
                pbest_fitness[i] = fitness_value
                pbest_pos[i] = positions[i].copy()
            if fitness_value > gbest_fitness:
                gbest_fitness = fitness_value
                gbest_pos = positions[i].copy()

        r1 = rng.uniform(0.0, 1.0, size=(swarm_size, dim))
        r2 = rng.uniform(0.0, 1.0, size=(swarm_size, dim))
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)
        positions = positions + velocities
        fitness_curve.append(float(gbest_fitness))

    wall_clock_seconds = time.time() - start
    result = _simulate(gbest_pos, config, num_jobs, num_machines)
    result["wall_clock_seconds"] = wall_clock_seconds
    result["fitness_curve"] = fitness_curve
    result["swarm_size"] = swarm_size
    result["iterations"] = iterations
    return result


def _main():
    """Run PSO on the fixed instance plus a subset of held-out instances,
    print + persist results the same way eval_rl_agent.py does, so PSO's
    numbers land in the same rl_training/results/eval_results.csv every
    other method is compared through. Only a subset (not the standard 50)
    of held-out instances is run -- see this module's docstring and the
    dated doc for why PSO's per-instance cost (hundreds of full episode
    replays, vs. one for everything else) makes 50 disproportionately
    expensive here.
    """
    import argparse
    import numpy as np
    from Code.env.env_config import generate_env_config
    from Code.evaluation.eval_rl_agent import run_heuristic
    from Code.utils.paths import ENV_CONFIG_PATH
    from Code.utils.results_log import append_eval_result
    from Code.training.train_optimized import RANDOM_INSTANCE_SEED_CEILING

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heldout", type=int, default=15,
                         help="Number of held-out instances to run PSO on (default 15, not "
                              "the standard 50 -- see module docstring for why).")
    parser.add_argument("--swarm-size", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fitness", type=str, default="reward", choices=["reward", "tardiness"],
                         help="PSO's search objective -- see optimize_and_run()'s docstring.")
    parser.add_argument("--num-jobs", type=int, default=None,
                         help="Override the instance dimension instead of reading ENV_CONFIG_PATH -- "
                              "use this (with --num-machines/--horizon/--max-jobs) to run safely while "
                              "a training run is concurrently active, since ENV_CONFIG_PATH transiently "
                              "holds whatever curriculum stage that run is currently on (see "
                              "Future/research/training-log.md's 2026-08-28 hazard entry). All four of "
                              "--num-jobs/--num-machines/--horizon/--max-jobs must be given together.")
    parser.add_argument("--num-machines", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

    overrides_given = [args.num_jobs, args.num_machines, args.horizon, args.max_jobs]
    if any(v is not None for v in overrides_given) and not all(v is not None for v in overrides_given):
        raise ValueError(
            "--num-jobs/--num-machines/--horizon/--max-jobs must all be given together "
            "(or none of them, to read dimensions from ENV_CONFIG_PATH as before)."
        )
    if args.num_jobs is not None:
        num_jobs, num_machines, horizon, max_jobs = args.num_jobs, args.num_machines, args.horizon, args.max_jobs
        print(f"Using explicit dimension overrides (num_jobs={num_jobs}, num_machines={num_machines}, "
              f"horizon={horizon}, max_jobs={max_jobs}) -- NOT reading ENV_CONFIG_PATH.")
    else:
        dims = np.load(ENV_CONFIG_PATH)
        num_jobs, num_machines, horizon = int(dims["num_jobs"]), int(dims["num_machines"]), int(dims["horizon"])
        max_jobs = int(dims["max_jobs"]) if "max_jobs" in dims else None

    def run_one(config, tag):
        t0 = time.time()
        result = optimize_and_run(config, num_jobs, num_machines,
                                   swarm_size=args.swarm_size, iterations=args.iterations, seed=args.seed,
                                   fitness=args.fitness)
        edf = run_heuristic("EDF", config=config)
        print(f"[{tag}] PSO reward={result['total_reward']:.2f} tardiness={result['tardiness'].sum():.2f} "
              f"late={result['late_jobs']} wall_clock={result['wall_clock_seconds']:.1f}s "
              f"| EDF reward={edf['total_reward']:.2f} tardiness={edf['tardiness'].sum():.2f} late={edf['late_jobs']}")
        append_eval_result({
            "algo": f"pso_{args.fitness}fit" if args.fitness != "reward" else "pso",
            "policy_type": "", "tag": tag, "model_path": "",
            "reward_mean": result["total_reward"], "reward_std": 0.0,
            "tardiness_mean": result["tardiness"].sum(), "tardiness_std": 0.0,
            "late_jobs_mean": result["late_jobs"], "late_jobs_std": 0.0,
            "jobs_scheduled_mean": result["jobs_scheduled"], "jobs_scheduled_std": 0.0,
            "heuristic_name": "EDF",
            "heuristic_reward_mean": edf["total_reward"], "heuristic_reward_std": 0.0,
            "heuristic_tardiness_mean": edf["tardiness"].sum(), "heuristic_tardiness_std": 0.0,
            "heuristic_late_jobs_mean": edf["late_jobs"], "heuristic_late_jobs_std": 0.0,
            "heuristic_jobs_scheduled_mean": edf["jobs_scheduled"], "heuristic_jobs_scheduled_std": 0.0,
            "n_episodes": 1,
        })
        return result

    # BUG FIX (2026-09-15, S2W9): passing config=None here made run_one()'s
    # make_env(None)/run_heuristic(config=None) read ENV_CONFIG_PATH directly
    # for the ACTUAL job data (not just dimensions) -- unsafe to run
    # concurrently with any active training run (see the 2026-08-28 hazard
    # entry in training-log.md; the --num-jobs et al. overrides above only
    # protected the held-out configs below, not this fixed-instance one).
    # When overrides are given, build the canonical fixed instance directly
    # (seed=0, matching train_optimized.py's FIXED_INSTANCE_SEED convention)
    # instead of relying on the shared file's live content.
    fixed_config = None
    if args.num_jobs is not None:
        fixed_config = generate_env_config(seed=0, num_jobs=num_jobs, num_machines=num_machines, horizon=horizon)
        fixed_config["max_jobs"] = max_jobs

    print(f"=== PSO on fixed instance (swarm={args.swarm_size}, iterations={args.iterations}) ===")
    run_one(fixed_config, "pso_baseline_S2W9" if args.fitness != "reward" else "pso_baseline_S2W6")

    print(f"\n=== PSO on {args.num_heldout} held-out instances ===")
    for i in range(args.num_heldout):
        cfg = generate_env_config(seed=RANDOM_INSTANCE_SEED_CEILING + i,
                                   num_jobs=num_jobs, num_machines=num_machines, horizon=horizon)
        if max_jobs is not None:
            cfg["max_jobs"] = max_jobs
        heldout_tag = f"pso_baseline_heldout_S2W9_{i}" if args.fitness != "reward" else f"pso_baseline_heldout_S2W6_{i}"
        run_one(cfg, heldout_tag)


if __name__ == "__main__":
    _main()
