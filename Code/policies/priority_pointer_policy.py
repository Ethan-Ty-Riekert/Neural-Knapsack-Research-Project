"""priority_pointer_policy.py - Adapted pointer network for Options 2/3
(priority-only action space, Discrete(max_jobs+1)) of the 2026-09-17
action-space-reduction work. Reuses JobEncoder/MachineEncoder/
GlobalContextHead from Code/policies/pointer_policy.py unchanged;
CompatibilityScorer (which scores every job x machine PAIR, sized for the
now-bypassed max_jobs*num_machines+1 action space) is replaced with
JobScoreHead, producing one logit per job SLOT directly -- placement is no
longer a policy decision (FirstFit, see
Code/env/priority_only_gym_wrapper.py), so only job-selection needs an
output.
"""
from typing import Tuple

import torch
import torch.nn as nn

from Code.policies.pointer_policy import JobEncoder, MachineEncoder, GlobalContextHead


class JobScoreHead(nn.Module):
    """Per-job-slot scalar score from its own embedding -- unlike
    CompatibilityScorer, no cross term with any machine embedding, since
    which machine to use is no longer a policy decision."""

    def __init__(self, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, job_emb: torch.Tensor) -> torch.Tensor:
        """job_emb: (B, J, E) -> (B, J)."""
        return self.net(job_emb).squeeze(-1)


class PriorityPointerActorCritic(nn.Module):
    """Drop-in replacement for MaskableActorCritic on the Discrete(max_jobs+1)
    priority-only action space: forward(obs) -> (logits, value) with shapes
    ((B, max_jobs+1), (B, 1)), consuming the flat obs vector
    PriorityOnlyGymSchedulingEnv._get_obs() produces."""

    def __init__(
        self,
        max_jobs: int,
        num_machines: int,
        num_resources: int,
        use_atc: bool = False,
        embed_dim: int = 128,
        hidden: int = 64,
    ):
        super().__init__()
        self.max_jobs = max_jobs
        self.num_machines = num_machines
        self.num_resources = num_resources
        self.use_atc = use_atc

        # Must match PriorityOnlyGymSchedulingEnv._get_obs()'s per-job-slot
        # layout: [duration, deadline, weight, resource_0..R-1, scheduled]
        # (+ atc appended last, only when use_atc).
        self._slot_width = num_resources + 4 + (1 if use_atc else 0)
        machine_feat_dim = num_resources

        self.job_encoder = JobEncoder(self._slot_width, embed_dim)
        self.machine_encoder = MachineEncoder(machine_feat_dim, embed_dim)
        self.job_score_head = JobScoreHead(embed_dim, hidden)

        context_dim = 2 * embed_dim + 1  # job_context + machine_context + time
        self.idle_head = GlobalContextHead(context_dim, hidden)
        self.value_head = GlobalContextHead(context_dim, hidden)

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        M, R, J = self.num_machines, self.num_resources, self.max_jobs

        time_feat = obs[:, 0:1]
        machine_end = 1 + M * R
        machine_feats = obs[:, 1:machine_end].reshape(B, M, R)
        job_feats = obs[:, machine_end:machine_end + J * self._slot_width].reshape(B, J, self._slot_width)
        return time_feat, machine_feats, job_feats

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_feat, machine_feats, job_feats = self._split_obs(obs)

        job_emb = self.job_encoder(job_feats)              # (B, J, E)
        machine_emb = self.machine_encoder(machine_feats)  # (B, M, E)

        job_logits = self.job_score_head(job_emb)  # (B, J), unmasked

        # "scheduled" is always feature index R+3 regardless of whether the
        # ATC feature is appended after it (use_atc puts ATC at the very
        # end, one slot further out) -- must reference it by this fixed
        # index, not job_feats[..., -1], unlike pointer_policy.py's
        # PointerActorCritic (where "scheduled" IS always the last feature).
        scheduled_idx = 3 + self.num_resources
        active_mask = (job_feats[..., scheduled_idx] < 0.5).float().unsqueeze(-1)  # (B, J, 1)
        denom = active_mask.sum(dim=1).clamp(min=1.0)
        job_context = (job_emb * active_mask).sum(dim=1) / denom  # (B, E)
        machine_context = machine_emb.mean(dim=1)                 # (B, E)

        context = torch.cat([job_context, machine_context, time_feat], dim=-1)  # (B, 2E+1)
        idle_logit = self.idle_head(context)  # (B, 1)
        value = self.value_head(context)      # (B, 1)

        logits = torch.cat([job_logits, idle_logit], dim=-1)  # (B, J+1)
        return logits, value
