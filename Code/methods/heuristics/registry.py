"""registry.py - Named baseline heuristics for Code/methods/rl/evaluation/eval_rl_agent.py.

Each entry maps a name to choose(base_env, job_actions, decode) -> action_id,
matching the exact call shape eval_rl_agent.py's run_heuristic() already
uses: job_actions is the list of currently-feasible non-idle action ids,
decode(a) -> (job, machine).

Composable priority+placement combos are generated automatically from
priority_rules.PRIORITY_RULES x placement_rules.PLACEMENT_RULES (e.g.
"EDF+BestFit", "LPT+WorstFit"). Tetris is registered separately since it
scores (job, machine) pairs jointly rather than picking a job first.
"""
import numpy as np

from Code.methods.heuristics.priority_rules import PRIORITY_RULES, ALL_PRIORITY_RULES, atc_key, _atc_mean_p
from Code.methods.heuristics.placement_rules import PLACEMENT_RULES, tetris_score


def _make_priority_placement(priority_name, placement_name):
    priority_key = ALL_PRIORITY_RULES[priority_name]
    placement_rule = PLACEMENT_RULES[placement_name]
    # PERF (2026-09-21, S2W9, from Future/research/2026-09-20-optimisation-
    # and-efficiency-critique.md Section 1.5): ATC's priority_key recomputes
    # an O(num_jobs) mean_p sum from scratch on every call, and sorted()
    # below calls priority_key once per candidate job -- O(J*num_jobs) for
    # one ranking otherwise. mean_p depends only on base_env's state (not on
    # which job is being scored), and nothing mutates base_env's state
    # between the J calls within one sorted() call, so it's safe to compute
    # once here and pass it through -- unlike a naive "cache the whole
    # mask/decision," which would be wrong (see rule_selection_gym_wrapper.py's
    # module docstring for why that specific class of "fix" was investigated
    # and rejected as unsafe there).
    is_atc = priority_name == "ATC"

    def choose(base_env, job_actions, decode):
        t = base_env.time
        unique_jobs = {decode(a)[0] for a in job_actions}
        if is_atc:
            mean_p = _atc_mean_p(base_env)
            key_fn = lambda j: (atc_key(base_env, j, mean_p=mean_p), j)
        else:
            key_fn = lambda j: (priority_key(base_env, j), j)
        # Secondary sort key on job index guarantees a deterministic,
        # reproducible tie-break (ascending job index) matching the
        # original inline dispatch's behaviour exactly for EDF/SPT/LST.
        job = sorted(unique_jobs, key=key_fn)[0]
        feasible_machines = sorted({decode(a)[1] for a in job_actions if decode(a)[0] == job})
        machine = placement_rule(base_env, job, feasible_machines, t)
        return job * base_env.num_machines + machine

    choose.__doc__ = f"{priority_name} job selection + {placement_name} machine placement."
    return choose


def _tetris(base_env, job_actions, decode):
    t = base_env.time
    return max(job_actions, key=lambda a: (tetris_score(base_env, *decode(a), t), -a))


def _random(base_env, job_actions, decode):
    return int(np.random.choice(job_actions))


HEURISTICS = {"Random": _random, "Tetris": _tetris}

for _priority_name in ALL_PRIORITY_RULES:  # includes the weight-aware WMDD / COVERT (2026-10-05)
    for _placement_name in PLACEMENT_RULES:
        HEURISTICS[f"{_priority_name}+{_placement_name}"] = _make_priority_placement(_priority_name, _placement_name)

# Back-compat: the original eval_rl_agent.py dispatch for "EDF"/"SPT"/"LST"
# always took the first feasible machine in ascending action-id order, i.e.
# First-Fit by construction -- so every existing eval_results.csv row and
# training-log reference to these names keeps meaning exactly the same
# thing after this refactor.
HEURISTICS["EDF"] = HEURISTICS["EDF+FirstFit"]
HEURISTICS["SPT"] = HEURISTICS["SPT+FirstFit"]
HEURISTICS["LST"] = HEURISTICS["LST+FirstFit"]
HEURISTICS["ATC"] = HEURISTICS["ATC+FirstFit"]


def _make_random_rule_selector(placements):
    """Random selection hyper-heuristic (2026-10-05): at every decision, apply one of the
    priority-rule x placement-rule heuristics chosen uniformly at random -- the same menu RL
    Option 1 chooses from (Code/methods/rl/action_spaces/rule_selection_gym_wrapper.py). It is the
    standard control for a learned selection hyper-heuristic (Burke et al. 2013, J. Oper. Res.
    Soc. 64(12)): a learned selector is only adding value where it beats this. The draw is seeded
    from the current scheduling state, so a given instance always gets the same schedule."""
    keys = [f"{r}+{p}" for p in placements for r in PRIORITY_RULES]

    def choose(base_env, job_actions, decode):
        state = (int(base_env.time), len(job_actions), int(np.asarray(base_env.start_times).sum()))
        rng = np.random.default_rng(abs(hash(state)) % (2 ** 32))
        return HEURISTICS[keys[int(rng.integers(len(keys)))]](base_env, job_actions, decode)
    choose.__doc__ = f"Uniformly random choice among {len(keys)} rule heuristics ({'/'.join(placements)})."
    return choose


HEURISTICS["RandomRule+FirstFit"] = _make_random_rule_selector(("FirstFit",))
HEURISTICS["RandomRule+FirstFitConsolidate"] = _make_random_rule_selector(("FirstFit", "Consolidate"))

# Curated default set for eval_rl_agent.py's --heuristics (all 18
# priority+placement combos x 6 priority rules x 3 placements = 18, plus
# Tetris and Random, is too many bars for one comparison run/plot set --
# this subset covers each priority rule and each placement rule at least
# once, plus the joint Tetris scorer).
DEFAULT_HEURISTICS = [
    "EDF", "SPT", "LST", "ATC",
    "FCFS+FirstFit", "LPT+WorstFit", "WSPT+BestFit", "EDF+BestFit",
    "Tetris",
]

ALL_HEURISTICS = sorted(HEURISTICS.keys())
