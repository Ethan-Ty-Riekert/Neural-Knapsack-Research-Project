"""registry.py - Named baseline heuristics for Code/evaluation/eval_rl_agent.py.

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

from Code.baselines.priority_rules import PRIORITY_RULES
from Code.baselines.placement_rules import PLACEMENT_RULES, tetris_score


def _make_priority_placement(priority_name, placement_name):
    priority_key = PRIORITY_RULES[priority_name]
    placement_rule = PLACEMENT_RULES[placement_name]

    def choose(base_env, job_actions, decode):
        t = base_env.time
        unique_jobs = {decode(a)[0] for a in job_actions}
        # Secondary sort key on job index guarantees a deterministic,
        # reproducible tie-break (ascending job index) matching the
        # original inline dispatch's behaviour exactly for EDF/SPT/LST.
        job = sorted(unique_jobs, key=lambda j: (priority_key(base_env, j), j))[0]
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

for _priority_name in PRIORITY_RULES:
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

# Curated default set for eval_rl_agent.py's --heuristics (all 18
# priority+placement combos x 6 priority rules x 3 placements = 18, plus
# Tetris and Random, is too many bars for one comparison run/plot set --
# this subset covers each priority rule and each placement rule at least
# once, plus the joint Tetris scorer).
DEFAULT_HEURISTICS = [
    "EDF", "SPT", "LST",
    "FCFS+FirstFit", "LPT+WorstFit", "WSPT+BestFit", "EDF+BestFit",
    "Tetris",
]

ALL_HEURISTICS = sorted(HEURISTICS.keys())
