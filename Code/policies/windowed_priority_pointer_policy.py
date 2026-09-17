"""windowed_priority_pointer_policy.py - Adapted pointer network for the
DeepRM-style bounded-window action space
(Code/env/windowed_priority_gym_wrapper.py, 2026-09-18, S2W10 -- PREPARED
FOR REVIEW, NOT YET WIRED INTO ANY TRAINING RUN, see that module's
docstring). Identical to Code/policies/priority_pointer_policy.py's
PriorityPointerActorCritic except: (a) the job-encoder only ever sees
`window_size` slots, not the full max_jobs, and (b) one extra scalar
(the backlog count, appended at the very end of the observation by
WindowedPriorityGymSchedulingEnv._get_obs()) is folded into the context
vector alongside job/machine context and time, so the value/idle heads can
condition on "how much unaddressed backlog exists" even though no
individual backlogged job is itself visible.
"""
from typing import Tuple

import torch
import torch.nn as nn

from Code.policies.pointer_policy import JobEncoder, MachineEncoder, GlobalContextHead
from Code.policies.priority_pointer_policy import JobScoreHead


class WindowedPriorityPointerActorCritic(nn.Module):
    """forward(obs) -> (logits, value), shapes ((B, window_size+1), (B, 1))."""

    def __init__(
        self,
        window_size: int,
        num_machines: int,
        num_resources: int,
        use_atc: bool = False,
        embed_dim: int = 128,
        hidden: int = 64,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_machines = num_machines
        self.num_resources = num_resources
        self.use_atc = use_atc

        # Must match WindowedPriorityGymSchedulingEnv._get_obs()'s per-slot
        # layout: [duration, deadline, weight, resource_0..R-1, scheduled]
        # (+ atc appended last, only when use_atc).
        self._slot_width = num_resources + 4 + (1 if use_atc else 0)
        machine_feat_dim = num_resources

        self.job_encoder = JobEncoder(self._slot_width, embed_dim)
        self.machine_encoder = MachineEncoder(machine_feat_dim, embed_dim)
        self.job_score_head = JobScoreHead(embed_dim, hidden)

        # +1 extra for the backlog scalar, beyond PriorityPointerActorCritic's
        # job_context + machine_context + time.
        context_dim = 2 * embed_dim + 1 + 1
        self.idle_head = GlobalContextHead(context_dim, hidden)
        self.value_head = GlobalContextHead(context_dim, hidden)

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        M, R, J = self.num_machines, self.num_resources, self.window_size

        time_feat = obs[:, 0:1]
        machine_end = 1 + M * R
        machine_feats = obs[:, 1:machine_end].reshape(B, M, R)
        job_end = machine_end + J * self._slot_width
        job_feats = obs[:, machine_end:job_end].reshape(B, J, self._slot_width)
        backlog = obs[:, job_end:job_end + 1]
        return time_feat, machine_feats, job_feats, backlog

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_feat, machine_feats, job_feats, backlog = self._split_obs(obs)

        job_emb = self.job_encoder(job_feats)              # (B, J, E)
        machine_emb = self.machine_encoder(machine_feats)  # (B, M, E)

        job_logits = self.job_score_head(job_emb)  # (B, J), unmasked

        scheduled_idx = 3 + self.num_resources
        active_mask = (job_feats[..., scheduled_idx] < 0.5).float().unsqueeze(-1)  # (B, J, 1)
        denom = active_mask.sum(dim=1).clamp(min=1.0)
        job_context = (job_emb * active_mask).sum(dim=1) / denom  # (B, E)
        machine_context = machine_emb.mean(dim=1)                 # (B, E)

        context = torch.cat([job_context, machine_context, time_feat, backlog], dim=-1)  # (B, 2E+2)
        idle_logit = self.idle_head(context)  # (B, 1)
        value = self.value_head(context)      # (B, 1)

        logits = torch.cat([job_logits, idle_logit], dim=-1)  # (B, J+1)
        return logits, value
