# Formal Definition: v2 Objective Reward (for review before implementation)

**Date:** 2026-09-28 (S2W11)
**Status:** DRAFT for user review. Nothing here is implemented yet. Decisions and reasoning are in
`2026-09-28-objective-redesign-discussion.md`. Items marked **OPEN** need a decision.

Goal: each reward component is defined so that its **episode sum equals a stated objective exactly**, with a
proof. Nothing is included that isn't part of a selected objective.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| J | jobs; offline: all known at t = 0; online: those with arrival r_j <= H |
| r_j, P_j, d_j, w_j | arrival (0 offline), duration, deadline, weight |
| M, H | machines, horizon; ticks t = 0 .. H |
| S_j, C_j = S_j + P_j | start and completion of a scheduled job |
| LS_j = H − P_j | latest start: by the feasibility rule (t + P_j <= H), j can never start after LS_j |
| t^drop_j = LS_j + 1 | the tick at which an unstarted job becomes a certain drop |
| T_j = max(0, C_j − d_j) | tardiness (scheduled jobs) |
| a_m(t) ∈ {0,1} | machine m runs >= 1 job during tick t |
| u_m(t) ∈ [0,1] | utilisation of machine m at tick t (**OPEN**, section 4: which resource) |

Reward at each step: r_t = −(1/c) · Σ_k λ_k · c_k(t) + F_t, where the c_k are the component charges
below, λ_k are the exchange rates chosen by the user, c > 0 is one global scale constant, and F_t is
optional potential-based shaping (section 6).

**Removed from v2 entirely:** +3 per job, +50 completion, λ₁ activation, λ₃ hotspot, flat idle penalty
and invalid-action penalty. (Masking makes invalid actions impossible. Idle remains a legal action.)

## 2. Component: weighted tardiness (ΣwT)

Charge each tick that elapses, for every arrived job that is past its deadline and not yet finished:

c_T(t) = Σ_j w_j · 1[ d_j <= t < C_j ]          (for unscheduled j, read C_j = ∞ until t^drop_j; section 3)

**Claim.** For a scheduled job, the charges over the episode sum to w_j · T_j.
**Proof.** T_j = max(0, C_j − d_j) = |{ t ∈ ℤ : d_j <= t < C_j }|. Each such tick contributes w_j once. ∎

This is the existing `dense_tardiness` charge **without** the /H normalisation. In v2 the unit is
"weighted job-ticks late", and scale comes from λ and c. Jobs still running when the episode ends early
are finalised in one lump (the existing `_finalize_dense_tardiness_running_jobs`), which keeps the sum
exact.

## 3. Component: dropped jobs

A job that hasn't started by LS_j can never be scheduled. At tick t^drop_j, charge once:

ρ_j = w_j · [ max(0, H − d_j) + B ] − w_j · max(0, t^drop_j − d_j)

and stop charging c_T for j after that tick.

**Claim 1 (total cost of a drop).** A dropped job's total charge (its c_T accrual up to t^drop_j, plus
ρ_j) is exactly w_j · [ max(0, H − d_j) + B ].
**Proof.** Before t^drop_j, j is unfinished, so c_T charges w_j for each tick in [d_j, t^drop_j), which is
w_j · max(0, t^drop_j − d_j) in total. Adding ρ_j cancels that term. ∎

**Claim 2 (per-job dominance).** Any completion of j costs strictly less than dropping it, by at least
w_j · B.
**Proof.** A scheduled job has C_j <= H (feasibility), so w_j · T_j <= w_j · max(0, H − d_j). ∎

**Claim 3 (ρ_j >= 0, so the charge is never a bonus).** t^drop_j − d_j = H − P_j + 1 − d_j <= H − d_j
because P_j >= 1. So the subtracted term is at most w_j · max(0, H − d_j), and B >= 0. ∎

**Limit of Claim 2, stated plainly.** It compares job j's *own* cost only. Fitting j in might delay other
jobs, so a policy could still rationally drop j if keeping it would cost the others more than
w_j · (max(0, H − d_j) + B). The surcharge B decides which regime applies. **OPEN, user decision:**
- **(a) Rejection price.** B is a moderate number of ticks, for example B = H / 10. A drop is allowed when
  it saves more total lateness than it costs. This is the "scheduling with rejection" model (Bartal et al.
  2000) and matches admission control.
- **(b) Completion first (lexicographic).** Choose B · w_min > H · Σ_k w_k, which is larger than any
  possible total tardiness. Then no drop can ever pay off while some completion is feasible, so the
  completion rate is maximised first and tardiness second. It matches "every job should be finished". The
  cost is that the very large numbers make the lateness signal small by comparison during learning.

**Online edge case.** A job arriving with r_j > LS_j is unschedulable the moment it arrives. It is
charged its full drop cost at arrival. No policy could have avoided this, so by the action-independence
argument in section 6 it doesn't bias learning. It's reported separately as "infeasible on arrival", so it
isn't blamed on the scheduler.

**Late-count interaction.** If ΣwU is selected (section 5), a dropped job with d_j <= H is already
counted as late, so no extra rule is needed.

**DAG extension (later).** LS_j becomes H − (longest path of durations from j through its descendants),
so a job is dropped as soon as it can no longer finish *with* its descendants. Claims 1–3 are unchanged.

## 4. Component: energy

**Linear power model (default).** P_m(t) = a_m(t) · [ P_idle + (P_max − P_idle) · u_m(t) ]
(Fan, Weber & Barroso 2007).

**Claim.** Total energy is E = P_idle · A + (P_max − P_idle) · W, where A = Σ_t Σ_m a_m(t) is active
machine-ticks and W = Σ_t Σ_m u_m(t) is total work. If u measures a single resource r* as a share of
capacity, then W = Σ_{scheduled j} P_j · a_{j,r*} / Cap_{r*}.
**Proof.** Sum P_m(t) over m and t. The u-term only contributes where a_m(t) = 1, because u_m(t) > 0
implies a_m(t) = 1. ∎

**Consequence.** For a fixed set of scheduled jobs, W is fixed, so minimising energy is the same as
minimising A. The component is:

c_E(t) = Σ_m a_m(t)        (unit: machine-ticks)

**Caveat, which is why the drop penalty is needed.** W falls when jobs are dropped, so a pure energy
objective rewards dropping work. The drop component (section 3) must always be active whenever energy is
selected, and it is on by default.

**When to charge.** A placement at tick t fixes that job's occupancy for [t, t + P_j), because there is no
preemption. So the increase in A (the number of (m, t') ticks that go from idle to active) is known
exactly at placement. v2 charges it **at placement**: the same total, with immediate credit to the
decision that caused it.

**SPECpower option.** Replace the bracket with a piecewise-linear P(u) interpolated from a published
SPECpower result (11 points, 0–100%) and normalised by P_max. Then c_E(t) = Σ_m a_m(t) · P(u_m(t)) / P_max,
charged per tick. The "efficiency peak at intermediate load" (Wong 2016) comes out of P(u) with no extra
term.

**OPEN, user decision:**
- **Which resource defines u.** Options: resource 0 treated as CPU (the usual server-power convention), or
  the bottleneck, max_r(used/cap).
- **Which SPECpower server to use**, if that option is chosen.

## 5. Component: weighted late count (ΣwU, SLA violations)

Charge w_j once, at tick min(d_j, H + 1), if j isn't finished by then:

c_U(t) = Σ_j w_j · 1[ t = min(d_j, H+1) and C_j > d_j (or j unfinished) ]

**Claim.** The charges sum to Σ_j w_j · U_j, where U_j = 1[job j is not finished by d_j], and a dropped
job counts as not finished.
**Proof.** Each job's status is checked exactly once, at its deadline. Deadlines beyond H are checked at
the end of the episode, when an unfinished job is certainly dropped. ∎

## 6. Global scale and shaping

**Global constant c.** Multiplying every reward by 1/c multiplies every return by 1/c, for any discount γ.
So argmax_π J(π) is unchanged, and so is the ordering of all policies. c is a numerical-conditioning choice
only. Proposed default: c = |J| (number of jobs), so rewards are per-job averages.

**Drop-risk shaping (optional, on by default).**
Φ(s) = − Σ_{j arrived, unstarted, not dropped} K_j · g(LS_j − t), with K_j = w_j · (max(0, H − d_j) + B)
and g(σ) = 1 / (1 + max(σ, 0)), which runs from about 0 (lots of slack) to 1 (no slack left).
Add F_t = γ · Φ(s_{t+1}) − Φ(s_t).

**Claim.** The optimal policy is unchanged (Ng, Harada & Russell 1999, Theorem 1). In episodic form this
needs Φ = 0 at terminal states. That holds here: at termination every job is started or dropped, so the sum
is empty. The shaping γ must equal the learning algorithm's γ.

**Effect.** Every tick in which a job's slack shrinks without it being started pays part of K_j straight
away. Starting the job refunds the potential. This answers the credit-assignment question in the decision
record (section 4).

**Action-independent charges don't bias the choice.** If a charge x is paid at state s whatever action is
taken, then Q(s, a) = −x + Q̃(s, a) for every a, so A(s, a) = Q(s, a) − V(s) and argmax_a Q(s, a) are
unaffected. This covers ρ_j at t^drop_j, since at that tick the drop is certain.

## 7. Observation additions

Add a per-job feature for normalised slack to the latest start, max(0, LS_j − t) / H, in
`Code/core/gym_scheduling_wrapper.py::_get_obs`. It can be derived from features already present, but
giving it directly makes the shaping potential visible to the policy.

## 8. Default configuration proposal (for discussion)

| Setting | Default |
|---|---|
| Objectives selected | ΣwT + drops (energy off) |
| λ_T | 1 |
| λ_E (if energy on) | user-chosen exchange rate; report a sweep and the Pareto front |
| λ_U (if late count on) | user-chosen |
| B | **OPEN** (section 3: regime a or b) |
| c | number of jobs |
| Shaping | on, γ = the algorithm's γ |

## 9. Verification planned for implementation (Step 5)

- Algebraic: on random instances and random schedules, each component's episode sum equals its closed form
  (ΣwT, ΣwU, A, and w_j · (max(0, H − d_j) + B) per drop).
- Dominance: for each job, the maximum cost of completing it is less than the cost of dropping it.
- Shaping: the telescoped sum of F_t equals −Φ(s_0) exactly, since Φ(terminal) = 0 (γ = 1 check).
- Ordering: on off_c_15 under ΣwT + drops, the reward ranking of EDF, ATC and PSO must agree with the
  ranking by tardiness with the drop cost included.

## References

1. Fan, X., Weber, W.-D. & Barroso, L.A. (2007). Power provisioning for a warehouse-sized computer. *ISCA '07*. doi:10.1145/1250662.1250665
2. Wong, D. (2016). Peak Efficiency Aware Scheduling for Highly Energy Proportional Servers. *ISCA '16*.
3. Bartal, Y. et al. (2000). Multiprocessor scheduling with rejection. *SIAM J. Discrete Math.* 13(1):64–78. (DOI unverified)
4. Ng, A.Y., Harada, D. & Russell, S. (1999). Policy Invariance Under Reward Transformations. *ICML '99*.
5. Mao, H. et al. (2016). DeepRM. *HotNets '16*. doi:10.1145/3005745.3005750
6. Pinedo, M.L. (2022). *Scheduling: Theory, Algorithms, and Systems*, 6th ed. Springer. (tardiness / U_j notation)
