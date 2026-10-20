"""test_compact_policy.py - the compact (active-slot) PriorityPointerActorCritic must give the same outputs and gradients
as the full network on every selectable action.

Run from the repo root: python -m tests.test_compact_policy
"""
import copy

import torch

from Code.core.obs_layout import ObsLayout
from Code.methods.rl.policies.priority_pointer_policy import PriorityPointerActorCritic

torch.manual_seed(0)
M, R, J = 3, 2, 60
full = PriorityPointerActorCritic(max_jobs=J, num_machines=M, num_resources=R, embed_dim=16, hidden=8)
compact = copy.deepcopy(full)
compact.compact = True

layout = ObsLayout(M, R, J, markov=False, lookahead=0)
B = 12
obs = torch.randn(B, layout.dim)
sched = 3 + R
slots = obs[:, layout.machine_block_end:layout.machine_block_end + J * layout.job_slot_width]
slots = slots.reshape(B, J, layout.job_slot_width)
active = torch.rand(B, J) < torch.linspace(0.0, 0.5, B).unsqueeze(1)  # row 0 has no active slot at all
slots[..., sched] = torch.where(active, torch.zeros(B, J), torch.ones(B, J))
obs[:, layout.machine_block_end:layout.machine_block_end + J * layout.job_slot_width] = slots.reshape(B, -1)

lf, vf = full(obs)
lc, vc = compact(obs)
sel = torch.cat([active, torch.ones(B, 1, dtype=torch.bool)], dim=1)  # active job slots + the idle action
assert torch.allclose(lf[sel], lc[sel], atol=1e-6), "logits differ on a selectable action"
assert torch.allclose(vf, vc, atol=1e-6), "values differ"

(lf[sel].square().sum() + vf.sum()).backward()
(lc[sel].square().sum() + vc.sum()).backward()
for (name, pf), (_, pc) in zip(full.named_parameters(), compact.named_parameters()):
    assert torch.allclose(pf.grad, pc.grad, atol=1e-5), f"gradient differs for {name}"
print("test_compact_policy: outputs and gradients identical on selectable actions "
      f"(batch {B}, {int(active.sum())} active of {B * J} slots)")
