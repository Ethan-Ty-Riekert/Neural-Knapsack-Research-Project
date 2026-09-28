"""objectives.py - the v2 reward: reward = exactly the selected objectives.

Implements Future/research/2026-09-28-v2-objective-formal-definition.md (read it for the
proofs; section numbers below refer to it). Used by SchedulingEnv / OnlineSchedulingEnv when
reward_mode="objective". The objective minimised is

    J = lambda_T * sum_{finished j} w_j T_j                       (weighted tardiness, sec. 2)
      + lambda_D * sum_{dropped j}  w_j (max(0, H - d_j) + B)     (dropped jobs, sec. 3)
      + lambda_U * sum_j w_j U_j                                  (weighted late count, sec. 5)

and the per-step reward is r_t = -(1/c) * (this step's share of J) + F_t / c, with c a single
global scale constant (sec. 6) and F_t optional potential-based drop-risk shaping (sec. 6).

Charges are dense (paid when the cost is incurred, not at episode end):
  - tardiness: w_j for every elapsing tick during which j is past its deadline and unfinished;
  - drops: when the clock passes job j's latest start LS_j = H - P_j (it can never start after
    that), charge rho_j = K_j - (tardiness already accrued by j), so a dropped job's total is
    exactly K_j = w_j (max(0, H - d_j) + B); its per-tick charge then stops;
  - late count: w_j once, when the clock reaches d_j and j is not finished by d_j (jobs with
    d_j beyond the episode end are checked at the end).

Energy (sec. 4) is build step 3 and not implemented yet.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ObjectiveConfig:
    """Exchange rates (lambda) and constants of the v2 objective. Defaults = the decided
    default configuration (2026-09-29): tardiness + drops, B = H, c = number of jobs."""
    tardiness: float = 1.0                  # lambda_T, per weighted late job-tick
    drops: float = 1.0                      # lambda_D; must stay > 0 whenever energy is used (sec. 4)
    late_count: float = 0.0                 # lambda_U, per weighted late job
    energy: float = 0.0                     # lambda_E -- build step 3, must be 0 for now
    drop_surcharge: Optional[float] = None  # B in ticks; None -> horizon H (decided 2026-09-29)
    scale: Optional[float] = None           # c; None -> number of real jobs in the instance
    drop_shaping: bool = True               # potential-based shaping on slack to latest start
    shaping_gamma: float = 0.99             # must equal the RL algorithm's discount factor

    def __post_init__(self):
        if self.energy != 0.0:
            raise NotImplementedError("energy objective is build step 3 -- not implemented yet")
        if min(self.tardiness, self.drops, self.late_count) < 0:
            raise ValueError("objective weights must be >= 0")


class ObjectiveReward:
    """Stateful per-episode reward calculator. The env calls reset() at every episode start,
    transition() after every successful placement or tick, and finalize() once at episode end."""

    def __init__(self, config: Optional[ObjectiveConfig] = None):
        self.cfg = config or ObjectiveConfig()

    # ------------------------------------------------------------------ episode lifecycle
    def reset(self, env):
        H = int(env.horizon)
        self.P = np.asarray(env.job_durations, dtype=float)
        self.d = np.asarray(env.job_deadlines, dtype=float)
        self.w = np.asarray(env.job_weights, dtype=float)
        n = len(self.P)
        self.arrival = np.asarray(getattr(env, "arrival_times", np.zeros(n)), dtype=float)
        self.real = self.arrival <= H                      # online padding jobs arrive at H+1
        self.B = float(H if self.cfg.drop_surcharge is None else self.cfg.drop_surcharge)
        self.LS = H - self.P                               # latest feasible start
        self.K = self.w * (np.maximum(0.0, H - self.d) + self.B)   # total cost of dropping j
        self.c = float(self.cfg.scale if self.cfg.scale is not None else max(1, int(self.real.sum())))

        self.dropped = np.zeros(n, dtype=bool)
        self.infeasible_on_arrival = np.zeros(n, dtype=bool)
        self.u_checked = np.zeros(n, dtype=bool)
        self.accrued = np.zeros(n)                          # weighted tardiness ticks charged per job
        # raw (un-lambda'd, unscaled) component totals, for verification and reporting. At episode
        # end: weighted_tardiness = sum over FINISHED jobs of w_j T_j, drop_cost = sum over dropped
        # jobs of K_j, weighted_late = sum_j w_j U_j.
        self.totals = {"weighted_tardiness": 0.0, "drop_cost": 0.0, "weighted_late": 0.0}
        self.phi = self._potential(env)

    def transition(self, env, elapsed_tick: Optional[int]) -> float:
        """Reward for one env transition. elapsed_tick = the tick that just elapsed (env.time has
        already been advanced and, online, arrivals revealed), or None for a placement that did
        not advance time (online case)."""
        cost = 0.0
        if elapsed_tick is not None:
            cost += self._tardiness_tick(env, elapsed_tick)
            cost += self._late_checks(env, upto=env.time)
            cost += self._drops(env)
        return self._reward(env, cost)

    def finalize(self, env) -> float:
        """Episode-end charges: remaining late ticks of jobs still running when the episode ends
        early (offline: all jobs placed), deadline checks not yet reached, and a safety drop of
        anything still unresolved. Keeps every component's episode sum exact."""
        start = np.asarray(env.start_times)
        C = start + self.P
        late_ticks = np.where(start != -1, np.maximum(0.0, C - np.maximum(self.d, env.time)), 0.0)
        self._add_tardiness(self.w * late_ticks)
        cost = self.cfg.tardiness * float((self.w * late_ticks).sum())
        cost += self._late_checks(env, upto=np.inf)
        cost += self._drops(env, force=True)
        return self._reward(env, cost, terminal=True)

    # ------------------------------------------------------------------ components
    def _unfinished_at(self, env, t):
        """Real, revealed, not-dropped jobs unfinished during tick t."""
        start = np.asarray(env.start_times)
        waiting = np.zeros(len(start), dtype=bool)
        if env.remaining_jobs:
            waiting[list(env.remaining_jobs)] = True
        running = (start != -1) & (start + self.P > t)
        return (waiting | running) & self.real & ~self.dropped

    def _add_tardiness(self, per_job):
        self.accrued += per_job
        self.totals["weighted_tardiness"] += float(per_job.sum())

    def _tardiness_tick(self, env, t):
        """lambda_T * (sum of w_j over jobs past deadline and unfinished during tick t)."""
        charged = self._unfinished_at(env, t) & (self.d <= t)
        self._add_tardiness(np.where(charged, self.w, 0.0))
        return self.cfg.tardiness * float(self.w[charged].sum())

    def _late_checks(self, env, upto):
        """lambda_U * w_j once for each real job whose deadline <= upto (not yet checked) and that
        is not finished by its deadline (a dropped or unstarted job counts as not finished)."""
        due = self.real & ~self.u_checked & (self.d <= upto)
        if not due.any():
            return 0.0
        start = np.asarray(env.start_times)
        on_time = (start != -1) & (start + self.P <= self.d)
        late = due & ~on_time
        self.u_checked |= due
        self.totals["weighted_late"] += float(self.w[late].sum())
        return self.cfg.late_count * float(self.w[late].sum())

    def _drops(self, env, force=False):
        """Mark waiting jobs that can no longer start (env.time > LS_j) as dropped and charge
        rho_j = lambda_D K_j - lambda_T accrued_j: the job's total charge becomes exactly
        lambda_D K_j (its pre-drop tardiness is re-attributed from the tardiness total to the
        drop total). rho_j >= 0 whenever lambda_D >= lambda_T (formal doc, Claim 3)."""
        waiting = np.zeros(len(self.P), dtype=bool)
        if env.remaining_jobs:
            waiting[list(env.remaining_jobs)] = True
        new = waiting & self.real & ~self.dropped & ((env.time > self.LS) | force)
        if not new.any():
            return 0.0
        self.infeasible_on_arrival |= new & (self.arrival > self.LS)
        self.dropped |= new
        pre = float(self.accrued[new].sum())
        self.totals["weighted_tardiness"] -= pre
        self.totals["drop_cost"] += float(self.K[new].sum())
        return self.cfg.drops * float(self.K[new].sum()) - self.cfg.tardiness * pre

    def _potential(self, env):
        """Phi(s) = -sum over alive jobs (real, waiting, not dropped) of K_j * g(LS_j - t),
        g(sigma) = 1 / (1 + max(sigma, 0)). Zero whenever no job is alive (every terminal state)."""
        if not self.cfg.drop_shaping or not env.remaining_jobs:
            return 0.0
        idx = np.fromiter(env.remaining_jobs, dtype=int)
        idx = idx[self.real[idx] & ~self.dropped[idx]]
        slack = np.maximum(self.LS[idx] - env.time, 0.0)
        return float(-(self.K[idx] / (1.0 + slack)).sum())

    def _reward(self, env, cost, terminal=False):
        """cost is already lambda-weighted; returns -(cost)/c plus scaled shaping."""
        reward = -cost / self.c
        if self.cfg.drop_shaping:
            phi_new = 0.0 if terminal else self._potential(env)
            reward += self.cfg.drops * (self.cfg.shaping_gamma * phi_new - self.phi) / self.c
            self.phi = phi_new
        return reward

    # ------------------------------------------------------------------ reporting
    def objective_value(self) -> float:
        """J for the episode so far (unscaled, lambda-weighted, shaping excluded)."""
        return (self.cfg.tardiness * self.totals["weighted_tardiness"]
                + self.cfg.drops * self.totals["drop_cost"]
                + self.cfg.late_count * self.totals["weighted_late"])
