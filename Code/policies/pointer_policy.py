"""pointer_policy.py

Pointer/attention-style action head for the scheduling problem, replacing
MaskableActorCritic's flat one-weight-per-action-index policy_head.

Motivation and literature basis: see
Future/research/2026-08-09-pointer-network-action-head.md. In short: the flat head
(Linear(256, max_jobs*num_machines+1)) gives every (job_slot, machine) pair its own
independent weight row, with zero parameter sharing across job/machine identity. This
was implicated in both the original idle-collapse (Future/research/2026-07-24-idle-
action-policy-collapse.md) and, more concretely, in the curriculum-transition collapse
recorded in Future/research/training-log.md's 2026-08-09 entry: job slots that are
masked out during early curriculum stages (padding) never receive a gradient, so when
a later stage unmasks them for the first time, those weight rows are still at their
random initialisation.

This module instead encodes each job slot and each machine slot through a
SHARED-WEIGHT encoder (same MLP applied to every slot), then scores every
(job, machine) pair with a shared compatibility function (Kool et al. 2019's
scaled-dot-product + tanh-clipping pattern). A job's score is therefore a function of
its *features*, learned from every job seen so far regardless of slot index -- a
newly-unmasked slot is scored correctly immediately, with no separate fix needed for
curriculum transitions.
"""

from typing import Tuple

import torch
import torch.nn as nn


class JobEncoder(nn.Module):
    """Shared-weight 2-layer MLP applied independently to every job slot."""

    def __init__(self, job_feat_dim: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(job_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, job_feats: torch.Tensor) -> torch.Tensor:
        """job_feats: (B, max_jobs, job_feat_dim) -> (B, max_jobs, embed_dim).
        nn.Linear broadcasts over all leading dims, so this applies identical
        weights to every job slot -- no explicit per-slot loop needed."""
        return self.net(job_feats)


class MachineEncoder(nn.Module):
    """Shared-weight 2-layer MLP applied independently to every machine slot."""

    def __init__(self, machine_feat_dim: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(machine_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, machine_feats: torch.Tensor) -> torch.Tensor:
        """machine_feats: (B, num_machines, machine_feat_dim) -> (B, num_machines, embed_dim)."""
        return self.net(machine_feats)


class CompatibilityScorer(nn.Module):
    """Scaled dot-product compatibility score per (job, machine) pair, with
    Kool et al. (2019)-style tanh clipping. Deliberately UNMASKED -- masking stays
    entirely in a2c_policy.py's masked_softmax, so this is a drop-in logits source."""

    def __init__(self, embed_dim: int, clip_c: float = 10.0):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.clip_c = clip_c

    def forward(self, job_emb: torch.Tensor, machine_emb: torch.Tensor) -> torch.Tensor:
        """job_emb: (B, J, E), machine_emb: (B, M, E) -> (B, J, M) unmasked, clipped logits."""
        q = self.query_proj(job_emb)      # (B, J, E)
        k = self.key_proj(machine_emb)    # (B, M, E)
        scores = torch.einsum("bje,bme->bjm", q, k) / self.scale
        return self.clip_c * torch.tanh(scores)


class GlobalContextHead(nn.Module):
    """Small MLP mapping a pooled global-context vector to a single scalar. Used for
    both the idle logit and the value estimate (separate instances, not shared)."""

    def __init__(self, context_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, context_dim) -> (B, 1)."""
        return self.net(context)


class PointerActorCritic(nn.Module):
    """Drop-in replacement for MaskableActorCritic: forward(obs) -> (logits, value)
    with identical shapes ((B, act_dim), (B, 1)), consuming the SAME flat obs vector
    GymSchedulingEnv._get_obs() produces -- no env/wrapper changes required.

    Unlike MaskableActorCritic(obs_dim, act_dim), this takes the *structured* shape
    (max_jobs, num_machines, num_resources) directly, since it needs to reshape the
    flat observation back into per-slot job/machine feature tensors.
    """

    def __init__(
        self,
        max_jobs: int,
        num_machines: int,
        num_resources: int,
        embed_dim: int = 128,
        hidden: int = 64,
        clip_c: float = 10.0,
    ):
        super().__init__()
        self.max_jobs = max_jobs
        self.num_machines = num_machines
        self.num_resources = num_resources

        # Must match GymSchedulingEnv._get_obs()'s per-job-slot feature layout
        # exactly: [duration, deadline, weight, resource_0..resource_{R-1}, scheduled].
        job_feat_dim = num_resources + 4
        machine_feat_dim = num_resources

        self.job_encoder = JobEncoder(job_feat_dim, embed_dim)
        self.machine_encoder = MachineEncoder(machine_feat_dim, embed_dim)
        self.scorer = CompatibilityScorer(embed_dim, clip_c)

        context_dim = 2 * embed_dim + 1  # job_context + machine_context + time
        self.idle_head = GlobalContextHead(context_dim, hidden)
        self.value_head = GlobalContextHead(context_dim, hidden)

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice the flat (B, obs_dim) vector back into (time, machine_feats,
        job_feats) using GymSchedulingEnv._get_obs()'s exact layout:
        [time(1)] [machine-block: machine-major, resource-minor]
        [job-block: job-major, (duration,deadline,weight,R resources,scheduled)-minor]."""
        B = obs.shape[0]
        M, R, J = self.num_machines, self.num_resources, self.max_jobs

        time_feat = obs[:, 0:1]
        machine_end = 1 + M * R
        machine_feats = obs[:, 1:machine_end].reshape(B, M, R)
        job_feats = obs[:, machine_end:machine_end + J * (R + 4)].reshape(B, J, R + 4)
        return time_feat, machine_feats, job_feats

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        time_feat, machine_feats, job_feats = self._split_obs(obs)

        job_emb = self.job_encoder(job_feats)          # (B, J, E)
        machine_emb = self.machine_encoder(machine_feats)  # (B, M, E)

        pair_logits = self.scorer(job_emb, machine_emb)  # (B, J, M), unmasked
        # Row-major (job-major, machine-minor) reshape matches GymSchedulingEnv's
        # action_id = job * num_machines + machine exactly.
        flat_pair_logits = pair_logits.reshape(B, self.max_jobs * self.num_machines)

        # Masked mean-pool for global context: exclude padded AND already-scheduled
        # job slots (both marked scheduled_flag==1, the last job feature) so neither
        # dilutes "what's actually available right now". Machines have no padding.
        active_mask = (job_feats[..., -1] < 0.5).float().unsqueeze(-1)  # (B, J, 1)
        denom = active_mask.sum(dim=1).clamp(min=1.0)                   # (B, 1)
        job_context = (job_emb * active_mask).sum(dim=1) / denom        # (B, E)
        machine_context = machine_emb.mean(dim=1)                       # (B, E)

        context = torch.cat([job_context, machine_context, time_feat], dim=-1)  # (B, 2E+1)
        idle_logit = self.idle_head(context)  # (B, 1)
        value = self.value_head(context)      # (B, 1)

        logits = torch.cat([flat_pair_logits, idle_logit], dim=-1)  # (B, J*M+1)
        return logits, value
