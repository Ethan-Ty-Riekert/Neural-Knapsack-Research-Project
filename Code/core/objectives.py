"""objectives.py - the v2 reward: reward = exactly the selected objectives.

Implements Future/research/2026-09-28-v2-objective-formal-definition.md (read it for the
proofs; section numbers below refer to it). Used by SchedulingEnv / OnlineSchedulingEnv when
reward_mode="objective". The objective minimised is

    J = lambda_T * sum_{finished j} w_j T_j                       (weighted tardiness, sec. 2)
      + lambda_S * sum_{finished j} w_j T_j^2                     (weighted SQUARED tardiness)
      + lambda_D * sum_{dropped j}  w_j (max(0, H - d_j) + B)     (dropped jobs, sec. 3)
      + lambda_U * sum_j w_j U_j                                  (weighted late count, sec. 5)
      + lambda_E * sum_t sum_m a_m(t) P(u_m(t)) / P_max            (energy, sec. 4)

and the per-step reward is r_t = -(1/c) * (this step's share of J) + F_t / c, with c a single
global scale constant (sec. 6) and F_t optional potential-based shaping F_t = gamma Phi(s') - Phi(s):
drop-risk shaping (sec. 6, fixed window) and/or lateness shaping (extended horizon, 2026-10-06).

Lateness shaping (lateness_shaping=True; user-approved 2026-10-06). Problem: a job that is already late
costs the same per tick whether it waits or runs, so delaying it is only penalised p_j ticks later, when
it would have finished. Potential (lambda-weighted, in cost units):
    Phi(s_t) = - sum_{j unfinished} w_j [ lambda_S (max(0, C^_j - d_j)^2 - max(0, t - d_j)^2)
                                         + lambda_T (max(0, C^_j - d_j)   - max(0, t - d_j)) ],
    C^_j = s_j + p_j for a running job (its completion is fixed), t + p_j for a waiting job (the earliest
    it can finish) -- minus the least lateness each unfinished job will still be charged.
Effects: idling while a waiting job's earliest completion is past its deadline gives an immediate penalty
equal to the extra lateness the one-tick delay causes; starting the job fixes C^_j (no further penalty);
idling while every waiting job can still finish on time changes nothing (strategic idling stays free);
running jobs net to zero (their charge is offset by Phi). Phi = 0 at every terminal state (no unfinished
job), so with gamma = 1 the shaped return is -J/c - Phi(s_0)/c for every policy, and for any gamma the
optimal policy is unchanged (Ng, Harada & Russell 1999, ICML, Theorem 1; shaping_gamma must equal the
RL discount). The reported J (objective_value) never includes shaping.

Charges are dense (paid when the cost is incurred, not at episode end):
  - tardiness: w_j for every elapsing tick during which j is past its deadline and unfinished;
  - squared tardiness (added 2026-09-30, user decision -- a few very late jobs are worse than many
    slightly late ones): w_j (2k + 1) on job j's k-th overdue tick (k = 0, 1, ...). Exact, because
    1 + 3 + ... + (2T - 1) = T^2, so the charges sum to w_j T_j^2 and still arrive as lateness grows;
  - drops: when the clock passes job j's latest start LS_j = H - P_j (it can never start after
    that), charge rho_j = K_j - (tardiness already accrued by j), so a dropped job's total is
    exactly K_j = w_j (max(0, H - d_j) + B); its per-tick charge then stops;
  - late count: w_j once, when the clock reaches d_j and j is not finished by d_j (jobs with
    d_j beyond the episode end are checked at the end).

  - energy: charged at PLACEMENT -- a placement fixes the job's occupancy of [t, t+P_j) on its
    machine (no preemption), so the exact increase in the machine's energy over that window is
    known immediately (power models in Code/core/power.py; "linear" = active machine-ticks).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .power import POWER_MODELS, relative_power


@dataclass
class ObjectiveConfig:
    """Exchange rates (lambda) and constants of the v2 objective. Defaults = the decided
    default configuration (2026-09-29): tardiness + drops, B = H, c = number of jobs."""
    tardiness: float = 1.0                  # lambda_T, per weighted late job-tick
    tardiness_sq: float = 0.0               # lambda_S, per weighted squared late job-tick (v2 default: 1)
    drops: float = 1.0                      # lambda_D; must stay > 0 whenever energy is used (sec. 4)
    late_count: float = 0.0                 # lambda_U, per weighted late job
    energy: float = 0.0                     # lambda_E, per normalised energy unit (sec. 4)
    power_model: str = "linear"             # "linear" (active machine-ticks) or "specpower_ml110g5"
    drop_surcharge: Optional[float] = None  # B in ticks; None -> horizon H (decided 2026-09-29)
    scale: Optional[float] = None           # c; None -> number of real jobs in the instance
    drop_shaping: bool = True               # potential-based shaping on slack to latest start
    lateness_shaping: bool = False          # potential-based shaping on projected lateness (extended)
    shaping_gamma: float = 0.99             # must equal the RL algorithm's discount factor

    def __post_init__(self):
        if min(self.tardiness, self.tardiness_sq, self.drops, self.late_count, self.energy) < 0:
            raise ValueError("objective weights must be >= 0")
        if self.energy > 0 and self.drops <= 0 and self.tardiness_sq <= 0:
            raise ValueError("energy rewards dropping work unless a drop cost is on (formal doc sec. 4)")
        if self.power_model not in POWER_MODELS:
            raise ValueError(f"unknown power model {self.power_model!r}; choose from {POWER_MODELS}")


class ObjectiveReward:
    """Stateful per-episode reward calculator. The env calls reset() at every episode start,
    transition() after every successful placement or tick, and finalize() once at episode end."""

    def __init__(self, config: Optional[ObjectiveConfig] = None):
        self.cfg = config or ObjectiveConfig()

    # ------------------------------------------------------------------ episode lifecycle
    def reset(self, env):
        H = int(env.horizon)                                        # physical window
        H_pref = int(getattr(env, "preferred_horizon", H))          # deadlines / arrivals horizon
        self.extended = bool(getattr(env, "extend_horizon", False))
        self.P = np.asarray(env.job_durations, dtype=float)
        self.d = np.asarray(env.job_deadlines, dtype=float)
        self.w = np.asarray(env.job_weights, dtype=float)
        n = len(self.P)
        self.arrival = np.asarray(getattr(env, "arrival_times", np.zeros(n)), dtype=float)
        self.real = self.arrival <= H_pref                 # online padding jobs arrive at H+1
        # Drop surcharge B. Extended horizon: every job can be finished, so a "drop" only happens if
        # a policy idles past the whole extended window; it is then charged its lateness lower bound
        # (finishing at the end of the physical window), B = 0 -- no arbitrary constant. Standard
        # window: B = H (decided 2026-09-29, before the extended horizon replaced it).
        default_B = 0.0 if self.extended else float(H_pref)
        self.B = default_B if self.cfg.drop_surcharge is None else float(self.cfg.drop_surcharge)
        self.LS = H - self.P                               # latest feasible start
        drop_lateness = np.maximum(0.0, H - self.d) + self.B        # a drop counts as this late
        self.K = self.w * drop_lateness                            # total cost of dropping j (linear)
        self.K_sq = self.w * drop_lateness ** 2                    # ... and in squared-lateness units
        self.c = float(self.cfg.scale if self.cfg.scale is not None else max(1, int(self.real.sum())))

        self.dropped = np.zeros(n, dtype=bool)
        self.infeasible_on_arrival = np.zeros(n, dtype=bool)
        self.u_checked = np.zeros(n, dtype=bool)
        self.accrued = np.zeros(n)                          # weighted tardiness ticks charged per job
        self.accrued_sq = np.zeros(n)                       # weighted squared-tardiness charged per job
        # raw (un-lambda'd, unscaled) component totals, for verification and reporting. At episode
        # end: weighted_tardiness = sum over FINISHED jobs of w_j T_j, drop_cost = sum over dropped
        # jobs of K_j, weighted_late = sum_j w_j U_j.
        self.totals = {"weighted_tardiness": 0.0, "drop_cost": 0.0, "weighted_late": 0.0, "energy": 0.0,
                       "weighted_sq_tardiness": 0.0, "drop_cost_sq": 0.0}
        # energy bookkeeping: machine-activity and CPU-utilisation grids (always tracked, so
        # energy is reported even when lambda_E = 0)
        self.active = np.zeros((env.num_machines, H), dtype=bool)
        self.cpu = np.zeros((env.num_machines, H))
        self.cpu_cap = float(np.asarray(env.machine_capacity)[0])
        env._last_placement = None
        self.phi = self._potential(env)

    def transition(self, env, elapsed_tick: Optional[int]) -> float:
        """Reward for one env transition. elapsed_tick = the tick that just elapsed (env.time has
        already been advanced and, online, arrivals revealed), or None for a placement that did
        not advance time (online case)."""
        cost = self._energy_placement(env)
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
        # squared: the remaining overdue ticks k0 .. T-1 (k0 = ticks already overdue) add T^2 - k0^2
        k0 = np.maximum(0.0, env.time - self.d)
        sq = np.where(late_ticks > 0, self.w * ((C - self.d) ** 2 - k0 ** 2), 0.0)
        self._add_tardiness(self.w * late_ticks, sq)
        cost = (self.cfg.tardiness * float((self.w * late_ticks).sum())
                + self.cfg.tardiness_sq * float(sq.sum()))
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

    def _energy_placement(self, env):
        """lambda_E * exact energy increase caused by the placement just made (if any)."""
        placement, env._last_placement = getattr(env, "_last_placement", None), None
        if placement is None:
            return 0.0
        j, m, t = placement
        window = slice(t, t + int(self.P[j]))
        model = self.cfg.power_model
        before = (relative_power(self.cpu[m, window], model) * self.active[m, window]).sum()
        self.cpu[m, window] += float(env.job_resources[j][0]) / self.cpu_cap
        self.active[m, window] = True
        delta = float(relative_power(self.cpu[m, window], model).sum() - before)
        self.totals["energy"] += delta
        return self.cfg.energy * delta

    def _add_tardiness(self, per_job, per_job_sq):
        self.accrued += per_job
        self.accrued_sq += per_job_sq
        self.totals["weighted_tardiness"] += float(per_job.sum())
        self.totals["weighted_sq_tardiness"] += float(per_job_sq.sum())

    def _tardiness_tick(self, env, t):
        """lambda_T * sum w_j + lambda_S * sum w_j (2k_j + 1) over jobs past deadline and unfinished
        during tick t, where k_j = t - d_j is how many ticks j has already been overdue."""
        charged = self._unfinished_at(env, t) & (self.d <= t)
        lin = np.where(charged, self.w, 0.0)
        sq = np.where(charged, self.w * (2.0 * (t - self.d) + 1.0), 0.0)
        self._add_tardiness(lin, sq)
        return self.cfg.tardiness * float(lin.sum()) + self.cfg.tardiness_sq * float(sq.sum())

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
        pre_sq = float(self.accrued_sq[new].sum())
        self.totals["weighted_tardiness"] -= pre
        self.totals["weighted_sq_tardiness"] -= pre_sq
        self.totals["drop_cost"] += float(self.K[new].sum())
        self.totals["drop_cost_sq"] += float(self.K_sq[new].sum())
        # squared part: a dropped job counts as finishing drop_lateness late in squared units too
        return (self.cfg.drops * float(self.K[new].sum()) - self.cfg.tardiness * pre
                + self.cfg.tardiness_sq * (float(self.K_sq[new].sum()) - pre_sq))

    def _shaping_on(self):
        return (self.cfg.drop_shaping and not self.extended) or (self.cfg.lateness_shaping and self.extended)

    def _potential(self, env):
        """The shaping potential, lambda-weighted (cost units): drop-risk (fixed window) and/or lateness
        (extended horizon) parts. Zero whenever no job is unfinished (every terminal state)."""
        phi = 0.0
        if self.cfg.drop_shaping and not self.extended:
            phi += self.cfg.drops * self._drop_potential(env)
        if self.cfg.lateness_shaping and self.extended:
            phi += self._lateness_potential(env)
        return phi

    def _drop_potential(self, env):
        """-sum over alive jobs (real, waiting, not dropped) of K_j * g(LS_j - t),
        g(sigma) = 1 / (1 + max(sigma, 0))."""
        if not env.remaining_jobs:
            return 0.0
        idx = np.fromiter(env.remaining_jobs, dtype=int)
        idx = idx[self.real[idx] & ~self.dropped[idx]]
        slack = np.maximum(self.LS[idx] - env.time, 0.0)
        return float(-(self.K[idx] / (1.0 + slack)).sum())

    def _lateness_potential(self, env):
        """-(least lateness every unfinished job will still be charged), see the module docstring."""
        t = float(env.time)
        start = np.asarray(env.start_times, dtype=float)
        waiting = np.zeros(len(start), dtype=bool)
        if env.remaining_jobs:
            waiting[list(env.remaining_jobs)] = True
        running = (start >= 0) & (start + self.P > t)
        alive = (waiting | running) & self.real & ~self.dropped
        if not alive.any():
            return 0.0
        completion = np.where(start >= 0, start + self.P, t + self.P)[alive]
        d, w = self.d[alive], self.w[alive]
        late, k0 = np.maximum(0.0, completion - d), np.maximum(0.0, t - d)
        return float(-(self.cfg.tardiness_sq * w * (late ** 2 - k0 ** 2)
                       + self.cfg.tardiness * w * (late - k0)).sum())

    def _reward(self, env, cost, terminal=False):
        """cost is already lambda-weighted; returns -(cost)/c plus scaled shaping."""
        reward = -cost / self.c
        if self._shaping_on():
            phi_new = 0.0 if terminal else self._potential(env)
            reward += (self.cfg.shaping_gamma * phi_new - self.phi) / self.c
            self.phi = phi_new
        return reward

    # ------------------------------------------------------------------ reporting
    def objective_value(self) -> float:
        """J for the episode so far (unscaled, lambda-weighted, shaping excluded)."""
        return (self.cfg.tardiness * self.totals["weighted_tardiness"]
                + self.cfg.tardiness_sq * (self.totals["weighted_sq_tardiness"] + self.totals["drop_cost_sq"])
                + self.cfg.drops * self.totals["drop_cost"]
                + self.cfg.late_count * self.totals["weighted_late"]
                + self.cfg.energy * self.totals["energy"])
