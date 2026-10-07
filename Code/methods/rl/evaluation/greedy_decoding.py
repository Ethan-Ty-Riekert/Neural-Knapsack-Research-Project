"""greedy_decoding.py - the deterministic (greedy) policy used for evaluation and tuning (2026-10-07).

Problem (diagnosed 2026-10-07, Future/research/training-log.md). The usual greedy policy takes the single most
probable action, argmax_a pi(a | s). In every design the idle action is ONE action, while the probability of
"start a job" is spread over many actions (every waiting job, and in Option 0 every (job, machine) pair). An
undecided policy can then have idle as its single most probable action in every state although it starts a job
with probability 1 - pi(idle | s) close to 1: a checkpoint with pi(idle) = 0.003 against a largest single action
of 0.002 idled at every step under argmax (J = 1.3e8) and only 2% of steps when sampled (J = 2.0e5). The argmax
of a joint distribution over composite actions depends on how finely the non-idle mass is split, not only on
what the policy prefers.

Rule. Decide in two stages, each by its own most probable outcome: first idle or start a job, comparing
pi(idle | s) with the total probability of every job action; then, if starting a job, the most probable job
action. The two stages mirror the decision's structure, as in sequential/branched action decoding
(Mao et al. 2019, Decima: stage then parallelism; Tavakoli et al. 2018, action branching). The policy itself is
unchanged; only the read-out of one action from it is. When the policy is confident (pi(idle) near 0 or 1), the
rule gives exactly the argmax action.
"""
import numpy as np
import torch


def two_stage_choice(probs):
    """probs: masked action probabilities with idle LAST -> index of the greedy action."""
    probs = np.asarray(probs, dtype=float)
    idle = len(probs) - 1
    if idle == 0 or probs[idle] >= probs[:idle].sum():
        return idle
    return int(np.argmax(probs[:idle]))


def greedy_action(model, obs, action_mask):
    """Two-stage greedy action of a MaskablePPO model (Discrete: idle is the last action; MultiDiscrete, Option 4:
    idle is the last entry of the job branch, the machine branch is decoded by its own argmax)."""
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_t, action_masks=np.asarray(action_mask)[None])
    if hasattr(dist, "distributions"):  # MultiDiscrete: (job + idle, machine)
        job_p, machine_p = (d.probs[0].cpu().numpy() for d in dist.distributions)
        return np.array([two_stage_choice(job_p), int(np.argmax(machine_p))])
    return two_stage_choice(dist.distribution.probs[0].cpu().numpy())
