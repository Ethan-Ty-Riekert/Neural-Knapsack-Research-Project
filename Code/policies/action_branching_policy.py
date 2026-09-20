"""action_branching_policy.py - Network for Option 4 (action-branching,
learned placement), added 2026-09-20 (S2W9) -- see
Code/env/action_branching_gym_wrapper.py's module docstring for the full
motivation and the parallel-independent-branches design decision this
network implements.

Reuses JobEncoder/MachineEncoder/GlobalContextHead from
Code/policies/pointer_policy.py and JobScoreHead from
Code/policies/priority_pointer_policy.py UNCHANGED -- no duplication. Adds
one new head, MachineScoreHead, producing a per-machine logit from that
machine's own embedding plus a shared pooled context (NOT conditioned on
which job the job branch actually sampled -- see the wrapper's docstring
for why this remains a deliberate, logged approximation, not an oversight).

FIRST RESULT AND FIX (2026-09-20, S2W9): the first real training run at this
design (weighted_tardiness=355.64 at 300k, worse than both fixed-FirstFit
Options 1/3 at the same budget -- training-log.md's matching entry) came
with a plausible mechanism: the job branch's entropy never left 1.0 (fully
uniform) for the whole run. Research into why (user-prompted -- "maybe we
need to do some more research into the decoupling") found Tavakoli et al.
(2018, arXiv:1711.08946)'s own paper explicitly warns that fully-independent
branches without a shared coordinating representation "lead to convergence
problems," empirically demonstrated via their own IDQ (independent-network)
baseline deteriorating with action dimensionality -- verified via direct
quote, not inferred. Their fix is a SHARED TRUNK whose common latent lets
branches coordinate implicitly. This network already had that in spirit
(both heads read a shared `context`) -- but the gap found was that
`job_context` was a FLAT, UNWEIGHTED average over every job embedding,
discarding exactly the one thing the machine branch needs to place well:
which job is actually likely to be chosen. Job-shop-scheduling literature's
own two-stage designs condition the machine/operation stage on the selected
job's own representation, not a pooled average across all candidates.

Fix: `job_context` is now a job-choice-WEIGHTED average, using the job
branch's own (mask-respecting) softmax distribution as attention weights --
still one shared forward pass, still no bypass of SB3's parallel-branch
masking API (see the "honest limitation" in the wrapper's docstring, which
this does NOT resolve -- the machine branch still can't condition on the
job branch's actually-SAMPLED value, only on its predicted distribution).
This is a genuine attempt at a fix, not yet validated -- see
training-log.md for whichever result follows this change.
"""
from typing import Tuple

import torch
import torch.nn as nn

from Code.policies.pointer_policy import JobEncoder, MachineEncoder, GlobalContextHead
from Code.policies.priority_pointer_policy import JobScoreHead


class MachineScoreHead(nn.Module):
    """Per-machine scalar score from its own embedding + the shared pooled
    context (job_context, machine_context, time) -- NOT conditioned on which
    job the job branch sampled, per the parallel-independent-branches design
    (see action_branching_gym_wrapper.py's module docstring)."""

    def __init__(self, embed_dim: int, context_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, machine_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """machine_emb: (B, M, E), context: (B, context_dim) -> (B, M)."""
        M = machine_emb.shape[1]
        ctx = context.unsqueeze(1).expand(-1, M, -1)  # (B, M, context_dim)
        return self.net(torch.cat([machine_emb, ctx], dim=-1)).squeeze(-1)


class ActionBranchingActorCritic(nn.Module):
    """Drop-in replacement for MaskableActorCritic on the
    MultiDiscrete([max_jobs+1, num_machines]) action-branching action space:
    forward(obs) -> (logits, value), logits shape (B, (max_jobs+1)+num_machines)
    -- job branch first (max_jobs+1 entries, idle last), then machine branch
    (num_machines entries), matching MultiDiscrete's declared branch order
    exactly (MaskableMultiCategoricalDistribution splits by this order).
    Consumes the SAME flat obs vector GymSchedulingEnv._get_obs() produces
    (ActionBranchingGymSchedulingEnv does not extend the observation -- no
    ATC feature in this first pass, see the wrapper's module docstring)."""

    def __init__(
        self,
        max_jobs: int,
        num_machines: int,
        num_resources: int,
        embed_dim: int = 128,
        hidden: int = 64,
    ):
        super().__init__()
        self.max_jobs = max_jobs
        self.num_machines = num_machines
        self.num_resources = num_resources

        # Must match GymSchedulingEnv._get_obs()'s per-job-slot layout
        # exactly: [duration, deadline, weight, resource_0..R-1, scheduled]
        # -- same as pointer_policy.py's PointerActorCritic (no ATC offset
        # here, unlike priority_pointer_policy.py).
        job_feat_dim = num_resources + 4
        machine_feat_dim = num_resources

        self.job_encoder = JobEncoder(job_feat_dim, embed_dim)
        self.machine_encoder = MachineEncoder(machine_feat_dim, embed_dim)
        self.job_score_head = JobScoreHead(embed_dim, hidden)

        context_dim = 2 * embed_dim + 1  # job_context + machine_context + time
        self.machine_score_head = MachineScoreHead(embed_dim, context_dim, hidden)
        self.idle_head = GlobalContextHead(context_dim, hidden)
        self.value_head = GlobalContextHead(context_dim, hidden)

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same layout as pointer_policy.py::PointerActorCritic._split_obs --
        "scheduled" is the LAST job-slot feature (no ATC offset here)."""
        B = obs.shape[0]
        M, R, J = self.num_machines, self.num_resources, self.max_jobs

        time_feat = obs[:, 0:1]
        machine_end = 1 + M * R
        machine_feats = obs[:, 1:machine_end].reshape(B, M, R)
        job_feats = obs[:, machine_end:machine_end + J * (R + 4)].reshape(B, J, R + 4)
        return time_feat, machine_feats, job_feats

    def _pool_job_context(self, job_emb: torch.Tensor, job_logits: torch.Tensor,
                           active_mask: torch.Tensor) -> torch.Tensor:
        """Job-choice-WEIGHTED context (fix, 2026-09-20 -- see module
        docstring): attention-weight each job's embedding by the job
        branch's own mask-respecting softmax distribution, instead of a flat
        unweighted mean over every active job. Gives the shared context (and
        hence the machine branch) a representation biased toward "the job
        likely to be picked," not diluted by ~100 slots almost certainly not
        chosen. job_emb: (B,J,E), job_logits: (B,J), active_mask: (B,J,1)
        -> (B,E). Extracted as its own method (not inlined in forward()) so
        it's directly unit-testable -- see
        tests/test_action_branching_wrapper.py."""
        denom = active_mask.sum(dim=1).clamp(min=1.0)
        masked_job_logits = job_logits.masked_fill(active_mask.squeeze(-1) < 0.5, float("-inf"))
        any_active = (active_mask.sum(dim=1).squeeze(-1) > 0).view(-1, 1, 1)  # (B, 1, 1)
        job_weights = torch.softmax(masked_job_logits, dim=-1).unsqueeze(-1)  # (B, J, 1)
        # Guard: softmax over an all "-inf" row (no active jobs this step,
        # e.g. right at episode end) is NaN, not just uniform -- fall back to
        # the original flat-mean weighting in that (rare, edge-case) scenario.
        flat_weights = active_mask / denom.unsqueeze(-1)  # (B, J, 1)
        job_weights = torch.where(any_active, job_weights, flat_weights)
        return (job_emb * job_weights).sum(dim=1)  # (B, E)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_feat, machine_feats, job_feats = self._split_obs(obs)

        job_emb = self.job_encoder(job_feats)              # (B, J, E)
        machine_emb = self.machine_encoder(machine_feats)  # (B, M, E)

        job_logits = self.job_score_head(job_emb)  # (B, J), unmasked

        active_mask = (job_feats[..., -1] < 0.5).float().unsqueeze(-1)  # (B, J, 1)
        job_context = self._pool_job_context(job_emb, job_logits, active_mask)  # (B, E)
        machine_context = machine_emb.mean(dim=1)                 # (B, E)

        context = torch.cat([job_context, machine_context, time_feat], dim=-1)  # (B, 2E+1)
        idle_logit = self.idle_head(context)  # (B, 1)
        value = self.value_head(context)      # (B, 1)

        machine_logits = self.machine_score_head(machine_emb, context)  # (B, M), unmasked

        job_branch_logits = torch.cat([job_logits, idle_logit], dim=-1)  # (B, J+1)
        logits = torch.cat([job_branch_logits, machine_logits], dim=-1)  # (B, (J+1)+M)
        return logits, value
