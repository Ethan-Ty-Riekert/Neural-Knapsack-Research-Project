"""critic_input.py - Critic-only summary of the future job arrivals (input-dependent baseline, 2026-10-06).

Why: online, an episode's return depends mostly on the exogenous arrival sequence z (which jobs arrive,
when, and how large), which the policy cannot see and the critic cannot predict from the state F_t. The
critic's value estimates, and so the advantages, are then dominated by arrival noise. Mao et al. (2019,
ICLR, "Variance Reduction for Reinforcement Learning in Input-Driven Environments", arXiv:1807.02264)
show that a baseline may depend on the future input sequence without biasing the policy gradient.

Derivation (the property relied on). The policy is pi_theta(a | s_t): it sees the state only. The
arrivals z are exogenous: generated with the instance, independent of every action. For any baseline
b(s_t, z),
    E[ grad log pi(a_t | s_t) b(s_t, z) ] = E_{s_t, z}[ b(s_t, z) sum_a pi(a | s_t) grad log pi(a | s_t) ]
                                          = E_{s_t, z}[ b(s_t, z) grad sum_a pi(a | s_t) ] = E[ b * grad 1 ] = 0,
because, given s_t, the action is drawn from pi(. | s_t) whatever z is (the policy never observes z).
So subtracting b(s_t, z) leaves the policy gradient unchanged in expectation (Mao et al. 2019, Thm 1).
Scope: this is exact for Monte Carlo advantages (GAE lambda = 1, e.g. our A2C). With lambda < 1, GAE
advantages are biased by any imperfect critic (Schulman et al. 2016); conditioning the critic on z adds
no new source of bias, it changes which function the critic approximates.

What the critic sees (this module): a fixed-size summary of the jobs that have NOT yet arrived (any
function of z and t is allowed by the derivation above), in BINS bins of BIN_WIDTH ticks after the
current tick t, plus one bin for everything later: per bin, the number of arrivals, their total weight,
their total work per resource sum p_j a_jr, and their mean slack (d_j - arrival_j - p_j), each divided
by a fixed constant. Offline every job is known at t = 0, so the summary is all zeros.

The policy never reads it: CriticInputWrapper appends it as the LAST block of the observation; every
policy network reads its inputs at fixed offsets before that block and only the value head reads the
block (Code/methods/rl/policies/pointer_policy.build_value_head); Option 1's MLP policy has the block
zeroed on its actor path (Code/methods/rl/policies/asymmetric_mlp_policy.py).
"""
import gymnasium as gym
import numpy as np

BINS, BIN_WIDTH = 20, 5  # 100 ticks ahead = the arrival horizon H of every online preset, + 1 "later" bin


def critic_input_dim(num_resources: int) -> int:
    return (BINS + 1) * (3 + num_resources)


def future_arrival_summary(gym_env) -> np.ndarray:
    """(BINS + 1) x (count, weight, work_r for every r, mean slack), flattened. gym_env: a
    GymSchedulingEnv / OnlineGymSchedulingEnv; its base env must expose arrival_times (online)."""
    base = gym_env.env
    R, M = gym_env.num_resources, gym_env.num_machines
    out = np.zeros((BINS + 1, 3 + R), dtype=np.float32)
    arrivals = getattr(base, "arrival_times", None)
    if arrivals is None:
        return out.ravel()
    t, H = base.time, base.preferred_horizon
    future = np.flatnonzero((arrivals > t) & (arrivals <= H))  # phantom padding jobs arrive at H + 1
    if future.size == 0:
        return out.ravel()
    from .difficulty import JOB_WEIGHT_MAX
    rel = arrivals[future] - t - 1
    b = np.minimum(rel // BIN_WIDTH, BINS).astype(int)
    p = base.job_durations[future].astype(float)
    cap = gym_env.initial_capacity.max(axis=0)  # C_r
    slot = M * BIN_WIDTH  # machine-ticks in one bin
    np.add.at(out[:, 0], b, 1.0 / slot)
    np.add.at(out[:, 1], b, base.job_weights[future] / (JOB_WEIGHT_MAX * slot))
    np.add.at(out[:, 2:2 + R], b, p[:, None] * base.job_resources[future] / (cap * slot))
    slack = (base.job_deadlines[future] - arrivals[future] - p) / H
    counts = np.bincount(b, minlength=BINS + 1)
    np.add.at(out[:, 2 + R], b, slack)
    out[:, 2 + R] /= np.maximum(counts, 1)
    return out.ravel()


class CriticInputWrapper(gym.ObservationWrapper):
    """Appends future_arrival_summary() to the (action-space wrapper's) observation as its last block."""

    def __init__(self, env, full_gym_env):
        super().__init__(env)
        self._full = full_gym_env
        self.critic_dim = critic_input_dim(full_gym_env.num_resources)
        inner = env.observation_space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(inner.shape[0] + self.critic_dim,),
                                                dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs, future_arrival_summary(self._full)]).astype(np.float32)

    def get_action_mask(self):  # ActionMasker's mask_fn calls this on the wrapped env
        return self.env.get_action_mask()


def wrap_critic_input(env, full_gym_env):
    """The action-space env, with the critic-only block appended iff the model uses it."""
    return CriticInputWrapper(env, full_gym_env) if getattr(full_gym_env, "critic_arrivals", False) else env


def critic_dim_of(full_gym_env) -> int:
    return critic_input_dim(full_gym_env.num_resources) if getattr(full_gym_env, "critic_arrivals", False) else 0
