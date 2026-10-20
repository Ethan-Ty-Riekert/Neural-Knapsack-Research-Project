"""Headroom bound: one-step rollout over the EDF+Consolidate base policy (clairvoyant), no learning.

At every decision with >= 2 schedulable candidates in the window, each of the top-K candidates (by EDF key) is tried:
apply it, then let EDF+Consolidate finish the episode on a COPY of the environment, and keep the candidate with the lowest
final objective J (ties go to EDF's own choice). Bertsekas' rollout / policy improvement: with the base policy simulated
exactly, the result cannot be worse than the base policy (up to the one-step greedy argument; stated by Bertsekas, to be
verified before citing).

CAVEAT (important): the environment copy contains the FUTURE arrivals of the instance, so this rollout is clairvoyant
about arrivals. It is an UPPER BOUND on what an online policy can gain over EDF+Consolidate at this look-ahead depth, not a
realisable policy. If the gain is small (a few %), there is little headroom for any online policy; if it is large, the bound
alone says nothing about the online case.

    python tools/analysis/rollout_headroom.py --n-instances 5 --k 3 --out Results/v2_objectives/tuning/rollout_headroom.csv

Uses the sidecar env spec of a finished windowed run (--spec-tag) so the environment options match the windowed models.
"""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def run_edf_to_end(env, wrapper, expert_action):
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(expert_action(wrapper, wrapper.get_action_mask(), "EDF"))
        done = terminated or truncated


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="on_rho095")
    ap.add_argument("--spec-tag", default="v2_on_rho095_o2cwimlubrzex_s100")
    ap.add_argument("--n-instances", type=int, default=5)
    ap.add_argument("--k", type=int, default=3, help="top-K EDF candidates tried at each decision")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import run
    from Code.core.difficulty import DIFFICULTIES, generate
    from Code.core.metrics import schedule_metrics
    from Code.methods.rl.action_spaces.wrapper_chain import selection_wrapper
    from Code.methods.rl.training.imitation import expert_action
    from Code.methods.rl.training.tune_v2 import VALIDATION_SEEDS
    from Code.utils.paths import MODELS_DIR
    from Code.variants import get_variant

    ns = run.build_parser().parse_args(["--variant", "v2_objectives", "--preset", a.preset, "--method", "heuristics"])
    env_kwargs = get_variant("v2_objectives").env_kwargs(ns)
    spec = json.load(open(MODELS_DIR / f"action_space_option2_ppo_{a.spec_tag}.json"))
    configs = [generate(DIFFICULTIES[a.preset], s) for s in list(VALIDATION_SEEDS)[:a.n_instances]]
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, cfg in enumerate(configs):
        t0 = time.time()
        # reference: plain EDF+Consolidate through the same windowed wrapper
        env = run.build_rl_eval_env("2", spec, cfg, env_kwargs)
        w = selection_wrapper(env)
        env.reset()
        run_edf_to_end(env, w, expert_action)
        j_edf = schedule_metrics(run._base_env(env))["objective_J"]
        # rollout
        env = run.build_rl_eval_env("2", spec, cfg, env_kwargs)
        w = selection_wrapper(env)
        env.reset()
        done, tried = False, 0
        while not done:
            mask = w.get_action_mask()
            cands = [i for i, job in enumerate(w._window_jobs) if job is not None and mask[i]]
            cands.sort(key=lambda i: (w.env.job_deadlines[w._window_jobs[i]], w._window_jobs[i]))  # EDF order
            action = expert_action(w, mask, "EDF")
            if len(cands) >= 2:
                best_j = None
                for c in cands[:a.k]:
                    e2 = copy.deepcopy(env)
                    w2 = selection_wrapper(e2)
                    _, _, terminated, truncated, _ = e2.step(c)
                    if not (terminated or truncated):
                        run_edf_to_end(e2, w2, expert_action)
                    j = schedule_metrics(run._base_env(e2))["objective_J"]
                    if best_j is None or j < best_j:  # strict: ties keep the earlier (EDF-first) candidate
                        best_j, action = j, c
                tried += 1
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        j_roll = schedule_metrics(run._base_env(env))["objective_J"]
        rows.append((idx, j_edf, j_roll, tried, time.time() - t0))
        print(f"instance {idx}: EDF+Consolidate J {j_edf:,.0f}  rollout J {j_roll:,.0f}  "
              f"({(j_roll - j_edf) / j_edf * 100:+.1f}%)  decisions with a choice: {tried}  {time.time() - t0:.0f}s", flush=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["instance", "J_edf_consolidate", "J_rollout", "decisions_with_choice", "seconds"])
        w_.writerows(rows)
    base, roll = np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows])
    print(f"mean J: EDF+Consolidate {base:,.0f}  rollout {roll:,.0f}  ({(roll - base) / base * 100:+.1f}%)")


if __name__ == "__main__":
    main()
