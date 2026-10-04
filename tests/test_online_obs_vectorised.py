"""test_online_obs_vectorised.py - the vectorised OnlineGymSchedulingEnv._get_obs (2026-09-29,
S2W11) must return BIT-IDENTICAL observations to the original per-element loop, kept verbatim
below as the reference, at every step of random online episodes (legacy and v2 reward modes).

Run from the repo root: python -m tests.test_online_obs_vectorised
"""
import numpy as np

from Code.core.arrival_process import generate_poisson_arrivals
from Code.core.objectives import ObjectiveConfig
from Code.methods.rl.evaluation.eval_rl_agent import make_env


def reference_obs(self):
    """The original (pre-2026-09-29) OnlineGymSchedulingEnv._get_obs, verbatim."""
    obs = []
    t = min(self.env.time, self.horizon)
    obs.append(t / self.horizon)
    t_idx = min(self.env.time, self.horizon - 1)
    for m in range(self.num_machines):
        for r in range(self.num_resources):
            cap = self.env.capacity[m, r, t_idx] / (self.initial_capacity[m, r] + 1e-8)
            obs.append(cap)
    max_dur = max(1.0, float(np.max(self.env.job_durations)))
    max_wgt = max(1.0, float(np.max(self.env.job_weights)))
    max_res = np.maximum(1.0, np.max(self.env.job_resources, axis=0))
    for j in range(self.max_jobs):
        if j in self.env.revealed_jobs:
            obs.append(self.env.job_durations[j] / max_dur)
            obs.append(self.env.job_deadlines[j] / self.horizon)
            obs.append(self.env.job_weights[j] / max_wgt)
            for r in range(self.num_resources):
                obs.append(self.env.job_resources[j, r] / max_res[r])
            scheduled = 0.0 if j in self.env.remaining_jobs else 1.0
            obs.append(scheduled)
        else:
            obs.extend([0.0] * (3 + self.num_resources))
            obs.append(1.0)
    return np.array(obs, dtype=np.float32)


rng = np.random.default_rng(1)
steps = 0
for seed, kw in [(3, None), (4, {"reward_mode": "objective", "objective": ObjectiveConfig()})]:
    cfg = generate_poisson_arrivals(seed=seed, arrival_rate=3.0, horizon=40, max_jobs=200,
                                    job_size_distribution="lognormal", job_weight_range=(1, 6))
    env = make_env(cfg, kw)
    gym_env = env.env
    obs, info = env.reset()
    done = truncated = False
    while True:
        ref = reference_obs(gym_env)
        assert obs.dtype == ref.dtype and obs.shape == ref.shape
        assert np.array_equal(obs, ref), f"mismatch at step {steps}"
        steps += 1
        if done or truncated:
            break
        valid = np.flatnonzero(info["action_mask"])
        obs, _, done, truncated, info = env.step(int(rng.choice(valid)))
print(f"  {steps} observations bit-identical to the reference loop")

# Option 3 (ATC-primed) wrapper: hoisting mean_p out of the per-slot loop must not change the obs.
from Code.methods.heuristics.priority_rules import atc_priority
from Code.methods.rl.action_spaces.priority_only_gym_wrapper import PriorityOnlyGymSchedulingEnv

cfg = generate_poisson_arrivals(seed=5, arrival_rate=3.0, horizon=40, max_jobs=200, job_weight_range=(1, 6))
wrapper = PriorityOnlyGymSchedulingEnv(make_env(cfg).env, use_atc=True)
obs, info = wrapper.reset()
checked, done = 0, False
while not done:
    base = wrapper.env
    atc_col = obs.reshape(-1)[-wrapper.max_jobs * (wrapper._job_slot_width + 1):].reshape(
        wrapper.max_jobs, wrapper._job_slot_width + 1)[:, -1]
    ref = np.array([float(np.clip(atc_priority(base, j), 0.0, 1.0)) if j in base.revealed_jobs else 0.0
                    for j in range(wrapper.max_jobs)], dtype=np.float32)
    assert np.array_equal(atc_col, ref), f"ATC column mismatch at step {checked}"
    checked += 1
    valid = np.flatnonzero(info["action_mask"])
    obs, _, done, truncated, info = wrapper.step(int(rng.choice(valid)))
    done = done or truncated
print(f"  Option 3 ATC feature identical to per-slot recomputation over {checked} steps")
print("PASS test_online_obs_vectorised")
