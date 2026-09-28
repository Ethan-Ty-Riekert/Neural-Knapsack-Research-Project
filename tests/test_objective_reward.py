"""test_objective_reward.py - Numerical checks of the v2 objective reward (reward_mode="objective",
Code/core/objectives.py) against the claims in Future/research/2026-09-28-v2-objective-formal-
definition.md (2026-09-29, S2W11):

  1. Exactness (secs. 2, 3, 5): over any episode, the tardiness total equals sum_{finished} w_j T_j,
     the drop total equals sum_{dropped} K_j, the late total equals sum_j w_j U_j -- checked on
     random policies (random valid actions incl. voluntary idles), offline and online, with
     random lambdas.
  2. Shaping (sec. 6): with gamma = 1 the shaping telescopes, so the episode reward equals
     -J/c - lambda_D * Phi(s_0)/c exactly.
  3. Drop dominance (sec. 3): K_j > w_j max(0, H - d_j) (dropping costs more than the latest
     possible completion) and rho_j = K_j - accrued_j >= 0.
  4. Without shaping, reward = -J/c exactly for heuristics (so reward ordering == objective
     ordering); legacy reward_mode untouched (EDF off_c_15 seed 500000 = 286.00).

Follows the tests/ convention (runnable script with asserts). Run from the repo root:
    python -m tests.test_objective_reward
"""
import numpy as np

from Code.core.env_config import generate_env_config
from Code.core.arrival_process import generate_poisson_arrivals
from Code.core.objectives import ObjectiveConfig
from Code.methods.rl.evaluation.eval_rl_agent import make_env, run_heuristic

TOL = 1e-6


def random_episode(config, cfg, rng, idle_prob=0.15):
    env = make_env(config, {"reward_mode": "objective", "objective": cfg})
    gym_env = env.env
    base = gym_env.env
    obs, info = env.reset()
    phi0 = base.objective.phi
    idle = gym_env.max_jobs * base.num_machines
    total, done, truncated = 0.0, False, False
    while not (done or truncated):
        valid = np.flatnonzero(info["action_mask"])
        jobs = valid[valid != idle]
        a = idle if (len(jobs) == 0 or rng.random() < idle_prob) else rng.choice(jobs)
        obs, r, done, truncated, info = env.step(a)
        total += r
    return base, total, phi0


def check_episode(base, total, phi0, cfg, label):
    obj = base.objective
    start = np.asarray(base.start_times)
    real = obj.real
    finished = real & (start != -1)
    dropped = real & (start == -1)
    w, P, d = obj.w, obj.P, obj.d
    T = np.maximum(0.0, start + P - d)
    U = ~((start != -1) & (start + P <= d))

    exp_T = float((w * T)[finished].sum())
    exp_D = float(obj.K[dropped].sum())
    exp_U = float(w[real & U].sum())
    assert abs(obj.totals["weighted_tardiness"] - exp_T) < TOL, (label, obj.totals, exp_T)
    assert abs(obj.totals["drop_cost"] - exp_D) < TOL, (label, obj.totals, exp_D)
    assert abs(obj.totals["weighted_late"] - exp_U) < TOL, (label, obj.totals, exp_U)
    assert np.array_equal(obj.dropped, dropped), label

    J = cfg.tardiness * exp_T + cfg.drops * exp_D + cfg.late_count * exp_U
    shaping = cfg.drops * (-phi0) / obj.c if cfg.drop_shaping else 0.0  # gamma = 1 telescoping
    assert abs(total - (-J / obj.c + shaping)) < 1e-6, (label, total, -J / obj.c + shaping)

    H = base.horizon
    assert np.all(obj.K[real] > w[real] * np.maximum(0.0, H - d[real])), label   # dominance
    assert np.all(obj.K[dropped] - obj.accrued[dropped] >= -TOL), label           # rho_j >= 0
    return int(finished.sum()), int(dropped.sum())


rng = np.random.default_rng(0)
n_eps = 0
for seed in range(6):
    lam_T, lam_U = rng.uniform(0.5, 2.0), rng.choice([0.0, rng.uniform(0.5, 3.0)])
    cfg = ObjectiveConfig(tardiness=lam_T, drops=lam_T + rng.uniform(0, 1.0), late_count=lam_U,
                          drop_surcharge=rng.choice([None, 5.0]), drop_shaping=bool(seed % 2),
                          shaping_gamma=1.0)
    off = generate_env_config(seed=900 + seed, num_jobs=30, num_machines=3, horizon=40,
                              job_weight_range=(1, 6))
    on = generate_poisson_arrivals(seed=900 + seed, arrival_rate=2.0, horizon=30, max_jobs=120,
                                   num_machines=3, job_weight_range=(1, 6))
    for label, config in (("offline", off), ("online", on)):
        f, dr = check_episode(*random_episode(config, cfg, rng), cfg, f"{label} seed {seed}")
        n_eps += 1
    print(f"  seed {seed}: offline+online exact (last: {f} finished, {dr} dropped)")
print(f"  exactness, shaping telescoping, drop dominance: {n_eps} random episodes OK")

# 4a. heuristics without shaping: reward == -J/c, on the real off_c_15 instances
nos = {"reward_mode": "objective", "objective": ObjectiveConfig(drop_shaping=False)}
for seed in (500000, 500001):
    cfg15 = generate_env_config(seed=seed, num_jobs=100, num_machines=10, horizon=100)
    for h in ("EDF", "ATC", "SPT"):
        r = run_heuristic(h, config=cfg15, env_kwargs=nos)
        J = r["metrics"]["objective_J"]
        assert abs(r["total_reward"] - (-J / 100)) < 1e-9, (h, seed, r["total_reward"], J)
print("  heuristics on off_c_15: reward == -J/c exactly")

# 4b. legacy untouched
r = run_heuristic("EDF", config=generate_env_config(seed=500000, num_jobs=100, num_machines=10, horizon=100))
assert abs(r["total_reward"] - 286.00) < 1e-6, r["total_reward"]
print("  legacy reward unchanged (EDF seed 500000 = 286.00)")
print("PASS test_objective_reward")
