"""imitation.py - behaviour-cloning warm start for the job-selection options (Options 2/3), 2026-10-20 (S2W14).

Why: the online heuristics' best entry (EDF + Consolidate) is a fixed, simple policy that PPO from scratch matches
only on some seeds (J 23k-32k across seeds vs 22.0k). Starting PPO at the heuristic's policy makes "RL >= heuristic"
the starting point and lets PPO's improvement show as a gain over it. Grounding (to be verified before citing in a
report): imitation of an expert followed by RL fine-tuning -- Hester et al. 2018 (DQfD, arXiv:1704.03732);
Rajeswaran et al. 2018 (DAPG, arXiv:1709.10087); expert labels on the learner's own states -- Ross, Gordon & Bagnell
2011 (DAgger, arXiv:1011.0686). Observed only, not proven, to help here.

Three stages (all inside one training run, the action space and free idling untouched):
  1. warm_start: DAgger-style behaviour cloning. Iteration 0 rolls out the expert; later iterations roll out the
     current (stochastic) policy; the label is always the expert's choice on the visited state. Loss = negative
     log-likelihood of the expert action under the masked policy distribution.
  2. critic_warmup: with every parameter the cloning step trained frozen, PPO collects rollouts from the cloned
     policy and trains only the value side (a freshly initialised critic would otherwise feed PPO garbage
     advantages that destroy the cloned policy in the first updates -- an engineering precaution, not a cited result;
     Andrychowicz et al. 2021, arXiv:2006.05990, show PPO implementation details matter). Also warms the reward-scaling statistics.
  3. the caller then unfreezes and continues ordinary PPO; the warm-up steps count towards the training budget
     (the expert rollouts used for cloning do not, and are reported).
"""
import numpy as np
import torch

from Code.methods.heuristics.priority_rules import ALL_PRIORITY_RULES


def expert_action(wrapper, mask, rule):
    """The priority rule's choice among the currently valid job slots (idle only when no job is valid)."""
    valid = np.flatnonzero(mask[:wrapper.max_jobs])
    if valid.size == 0:
        return wrapper.max_jobs
    key = ALL_PRIORITY_RULES[rule]
    return int(min(valid, key=lambda j: (key(wrapper.env, j), j)))


def job_selection_wrapper(env):
    """The innermost job-selection wrapper (has max_jobs and get_action_mask) under Monitor / ActionMasker / the
    critic-input wrapper."""
    while not (hasattr(env, "max_jobs") and hasattr(env, "get_action_mask")):
        env = env.env
    return env


def collect(env, wrapper, n_episodes, rule, model=None, stride=4):
    """Roll out n_episodes (the expert's moves, or the model's stochastic moves when model is given) and return
    (obs, masks, expert_actions) arrays, labelled by the expert on every visited state. Only every stride-th
    state is stored: consecutive states are near-duplicates, and an online observation is ~14k floats (stored as
    float16 to bound memory)."""
    obs_l, mask_l, act_l = [], [], []
    step = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = wrapper.get_action_mask()
            expert = expert_action(wrapper, mask, rule)
            if step % stride == 0:
                obs_l.append(obs.astype(np.float16))
                mask_l.append(mask.astype(bool))
                act_l.append(expert)
            step += 1
            if model is None:
                action = expert
            else:
                action = int(model.predict(obs, action_masks=mask, deterministic=False)[0])
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.array(obs_l), np.array(mask_l), np.array(act_l, dtype=np.int64)


def clone(model, obs, masks, actions, epochs, batch_size=128, lr=1e-3, seed=0):
    """Supervised fit of the masked policy to the expert labels. Returns (final loss, agreement with the expert,
    ids of the parameters that received gradients, i.e. the actor side incl. shared encoders)."""
    policy = model.policy
    device = model.device
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_t = torch.as_tensor(obs)  # float16 on the CPU; each batch is moved to the device as float32
    act_t = torch.as_tensor(actions, device=device)
    n = len(obs_t)
    trained, loss_val = set(), float("nan")
    policy.set_training_mode(True)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            dist = policy.get_distribution(obs_t[idx].to(device, torch.float32), action_masks=masks[idx.numpy()])
            loss = -dist.log_prob(act_t[idx.to(device)]).mean()
            opt.zero_grad()
            loss.backward()
            trained.update(id(p) for p in policy.parameters() if p.grad is not None)
            opt.step()
            loss_val = float(loss.item())
    with torch.no_grad():
        agree = 0
        for start in range(0, n, 2048):
            dist = policy.get_distribution(obs_t[start:start + 2048].to(device, torch.float32),
                                           action_masks=masks[start:start + 2048])
            agree += int((dist.get_actions(deterministic=True) == act_t[start:start + 2048]).sum().item())
    return loss_val, agree / n, trained


def warm_start(model, env, rule="EDF", episodes=30, iterations=2, epochs=8, seed=0):  # epochs per iteration
    """Stage 1 (see the module docstring). env: the single (non-vectorised) training env, i.e. template_env.
    Returns a stats dict and the set of actor-side parameter ids."""
    wrapper = job_selection_wrapper(env)
    all_obs, all_masks, all_actions = [], [], []
    stats = {"rule": rule, "episodes_per_iteration": episodes, "iterations": iterations}
    trained = set()
    for it in range(iterations):
        o, m, a = collect(env, wrapper, episodes, rule, model=None if it == 0 else model)
        all_obs.append(o)
        all_masks.append(m)
        all_actions.append(a)
        loss, agree, trained = clone(model, np.concatenate(all_obs), np.concatenate(all_masks),
                                     np.concatenate(all_actions), epochs, seed=seed + it)
        stats[f"iter{it}"] = dict(samples=int(sum(len(x) for x in all_obs)), loss=round(loss, 4),
                                  agreement=round(agree, 4))
        print(f"behaviour cloning iteration {it}: {stats[f'iter{it}']}")
    return stats, trained


def critic_warmup(model, actor_param_ids, steps):
    """Stage 2: train only the parameters the cloning step did not touch (the value side) for `steps` env steps
    of PPO rollouts, then unfreeze everything."""
    frozen = [p for p in model.policy.parameters() if id(p) in actor_param_ids]
    for p in frozen:
        p.requires_grad_(False)
    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=True)
    finally:
        for p in frozen:
            p.requires_grad_(True)
