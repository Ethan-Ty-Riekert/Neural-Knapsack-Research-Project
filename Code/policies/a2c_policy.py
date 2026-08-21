"""a2c_policy.py
Advantage Actor Critic policy implementation. Stable baselines does not provide a masked
A2C method, requiring us to create our own"""

import os
from typing import Tuple, List, Dict, Any

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from .pointer_policy import PointerActorCritic




### Actor Critic Network ###

class MaskableActorCritic(nn.Module):
    """Shared-body Actor-Critic network.
    - shared MLP trunk
    - policy_head: outputs logits over actions
    - value_head: outputs scalar value estimate"""

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Shared feature extractor (2-layer MLP)
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Policy head: produces logits for each action
        self.policy_head = nn.Linear(256, act_dim)

        # Value head: produces a single scalar value
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Inputs
        obs : torch.Tensor - Shape (batch_size, obs_dim)

        Outputs
        logits : torch.Tensor - Unmasked action logits, shape (batch_size, act_dim)

        value : torch.Tensor - Value estimates, shape (batch_size, 1)
        """
        x = self.shared(obs)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value
    

### Masked softmax and action selection ###

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply action mask to logits and compute softmax. Invalid actions (mask==0) get a large negative logit.
    Valid actions (mask==1) keep their original logits"""

    # Negative number for invalid actions
    invalid = (mask == 0)
    masked_logits = logits.clone()
    masked_logits[invalid] = -1e10

    probs = torch.softmax(masked_logits, dim=-1)
    return probs

def select_action(model, obs, mask, device, deterministic: bool = False):
    """Select an action using masked softmax.
    This is used both during training and evaluation.

    deterministic: if True, take the argmax action instead of sampling from the
    masked Categorical distribution. BUG FIX (this session): this previously had
    no deterministic/greedy mode at all, so `MaskableA2C.act()` always sampled --
    even during evaluation -- while PPO's evaluation uses
    `model.predict(..., deterministic=True)`. That asymmetry made any A2C-vs-PPO
    or A2C-vs-EDF comparison noisier/unfair for A2C specifically, since a
    partially-converged stochastic policy can select a materially worse action
    than its own argmax on any given step. log_prob/value are still computed from
    the same masked distribution either way, for callers that need them.
    """

    # Convert obs + mask to tensors
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

    # Forward pass through actor-critic
    logits, value = model(obs_t)

    # Apply mask to logits → get valid action probabilities
    probs = masked_softmax(logits, mask_t)

    # Create categorical distribution over valid actions
    dist = torch.distributions.Categorical(probs)

    # Sample (training / stochastic eval) or take the argmax (greedy eval)
    action = probs.argmax(dim=-1) if deterministic else dist.sample()

    # Log-probability of chosen action
    log_prob = dist.log_prob(action)

    return int(action.item()), float(log_prob.item()), float(value.item())


### Reward normalisation ###

class RunningMeanStd:
    """Tracks a running mean/variance of raw rewards (Welford-style streaming
    stats), used to keep the value function's regression target on a roughly
    stationary scale as the curriculum's problem size grows.

    Rationale (see Future/research/training-log.md, 2026-08-09 entry): with no
    normalisation, per-episode reward scale grows with num_jobs -- stage 1-2 lives
    in roughly [0, 130], stage 3-4 jumps to [-1500, +300] -- so the value function's
    bootstrapped TD targets shift abruptly at every curriculum transition, right
    when the network can least afford instability. Following the convention used by
    SB3's VecNormalize(norm_reward=True), only the SCALE is normalised (divide by
    running std); the mean is deliberately not subtracted, since centering would
    distort the sign of terminal completion/tardiness rewards.
    """

    def __init__(self, eps: float = 1e-4):
        self._sum = 0.0
        self._sumsq = 0.0
        self._count = eps

    def update(self, x: float):
        self._sum += x
        self._sumsq += x * x
        self._count += 1

    @property
    def mean(self) -> float:
        return self._sum / self._count

    @property
    def var(self) -> float:
        return max(self._sumsq / self._count - self.mean ** 2, 1e-8)

    @property
    def std(self) -> float:
        return self.var ** 0.5


### Rollout Buffer ###

class RolloutBuffer:
    """Stores n-step transitions for A2C.
    A2C does NOT use replay buffers, it uses short rollouts.
    """

    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.masks = []

    def add(self, obs, action, reward, done, value, log_prob, mask):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)

    def compute_returns_and_advantages(self, last_value, gamma, lam):
        """Compute A2C advantages using GAE with lambda=1.0.
        This is the standard A2C advantage formula.
        """

        values = np.array(self.values + [last_value])
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        gae = 0.0

        # Reverse-time GAE computation
        for t in reversed(range(len(rewards))):
            # TD residual
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]

            # Accumulate GAE
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return returns, advantages


### Maskable A2C Agent ###

class MaskableA2C:
    """
    Full A2C implementation with action masking.

    This class handles:
      - rollout collection
      - masked action selection
      - A2C loss computation
      - gradient updates
    """

    def __init__(
        self,
        env,
        device="cpu",
        policy_type: str = "pointer",
        policy_kwargs: dict = None,
        normalize_rewards: bool = True,
        use_rcpo: bool = False,
        rcpo_alpha: float = 0.0,
        rcpo_lambda_init: float = 1.0,
        rcpo_lambda_lr: float = 0.01,
        rcpo_lambda_max: float = 50.0,
        rcpo_update_every_episodes: int = 5,
    ):
        """
        policy_type: "pointer" (default) uses PointerActorCritic -- shared job/machine
        encoders + pairwise compatibility scoring (see pointer_policy.py). "flat" uses
        the original MaskableActorCritic (one weight row per action index), kept as
        the A/B baseline. policy_kwargs are forwarded to whichever network is built.

        normalize_rewards: if True, rewards used for the returns/advantage
        computation are divided by a running estimate of their std (see
        RunningMeanStd above) before being stored in the buffer. Episode-reward
        logging/plotting still uses the raw, un-normalised reward.

        use_rcpo: if True, replace the environment's fixed lambda_2 (tardiness
        weight) with a Lagrange multiplier updated on a slower timescale than
        the policy, per Tessler, Mankowitz, Mannor (ICLR 2019, arXiv:1805.11074)
        -- see Future/research/2026-08-21-rcpo-constrained-tardiness.md for the
        full CMDP formulation and grounding of the defaults below. Off by
        default (lambda_2 stays whatever the caller set on env construction).

        rcpo_alpha: constraint threshold on the per-episode cost
        C(tau) = sum w_j*(T_j/H); the multiplier increases while the sampled
        mean cost exceeds this and decays back toward 0 while below it.

        rcpo_lambda_init/rcpo_lambda_lr/rcpo_lambda_max: multiplier starting
        value, step size, and projection upper bound (lower bound is always 0,
        since lambda is a dual variable for an inequality constraint).

        rcpo_update_every_episodes: number of completed episodes averaged into
        one multiplier update (variance reduction on the cost estimate) -- see
        the dated doc's Section 3/4 for why this, combined with the small
        constant rcpo_lambda_lr, gives the required timescale separation from
        the policy's per-rollout (n_steps) update.
        """
        self.env = env
        self.device = torch.device(device)

        self.use_rcpo = use_rcpo
        if use_rcpo:
            self.rcpo_alpha = rcpo_alpha
            self.rcpo_lambda_lr = rcpo_lambda_lr
            self.rcpo_lambda_max = rcpo_lambda_max
            self.rcpo_update_every_episodes = rcpo_update_every_episodes
            # Held on the agent (not just the env) because train_optimized.py's
            # curriculum loop reassigns self.env to a freshly-constructed
            # SchedulingEnv every stage -- the adapted multiplier must survive
            # that swap instead of resetting to rcpo_lambda_init each stage.
            # _resolve_sched_env() re-derives the raw-env reference and pushes
            # this value into it at the start of every train() call.
            self._lambda_value = float(rcpo_lambda_init)
            self._rcpo_episode_costs = []
            self.lambda_history = [(0, float(rcpo_lambda_init), None)]
            self._resolve_sched_env()  # validates env shape early, at construction time

        # Extract dimensions from environment
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        # Hyperparameters (tweak these later)
        self.n_steps = 5
        self.gamma = 0.99
        self.lam = 1.0
        # Raised from 0.0 (previously: no exploration bonus at all, ever). At a
        # curriculum-stage transition that unmasks new job slots for the first
        # time, zero entropy bonus means nothing forces the policy to explore that
        # newly-unmasked region -- see training-log.md's 2026-08-09 entry.
        self.ent_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.lr = 7e-4

        self.policy_type = policy_type
        self.normalize_rewards = normalize_rewards
        self.reward_rms = RunningMeanStd() if normalize_rewards else None

        # Actor-Critic network
        policy_kwargs = policy_kwargs or {}
        if policy_type == "pointer":
            # env may be wrapped (ActionMasker/Monitor/...); gymnasium>=1.x removed
            # automatic attribute forwarding through Wrapper.__getattr__ (the same
            # issue that broke self.env.get_action_mask() -- see the comment on
            # that fix in train() below), so reach GymSchedulingEnv explicitly via
            # .unwrapped rather than relying on getattr(env, ...) directly.
            base_env = getattr(env, "unwrapped", env)
            max_jobs = getattr(base_env, "max_jobs", None)
            num_machines = getattr(base_env, "num_machines", None)
            num_resources = getattr(base_env, "num_resources", None)
            if None in (max_jobs, num_machines, num_resources):
                raise ValueError(
                    "policy_type='pointer' requires max_jobs/num_machines/"
                    "num_resources on env.unwrapped (GymSchedulingEnv)."
                )
            self.model = PointerActorCritic(
                max_jobs, num_machines, num_resources, **policy_kwargs
            ).to(self.device)
        elif policy_type == "flat":
            self.model = MaskableActorCritic(self.obs_dim, self.act_dim).to(self.device)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type!r}")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Rollout buffer
        self.buffer = RolloutBuffer(self.n_steps)

    def _resolve_sched_env(self):
        """Reach the raw SchedulingEnv through self.env's wrapper stack
        (Monitor/ActionMasker/GymSchedulingEnv/...) and push the agent's
        current RCPO multiplier value into it. Called at __init__ (to validate
        early) and again at the top of every train() call, since
        train_optimized.py's curriculum loop reassigns self.env to a new env
        per stage -- see the comment on self._lambda_value above."""
        base_env = getattr(self.env, "unwrapped", self.env)
        sched_env = getattr(base_env, "env", None)
        if sched_env is None or not hasattr(sched_env, "lambda2") or not hasattr(sched_env, "episode_cost"):
            raise ValueError(
                "use_rcpo=True requires self.env.unwrapped.env to be a SchedulingEnv "
                "instance exposing lambda2/episode_cost (see GymSchedulingEnv)."
            )
        sched_env.lambda2 = self._lambda_value
        return sched_env

    def act(self, obs, mask, deterministic: bool = False):
        """
        Action selection for evaluation (no gradients).

        deterministic: passed straight through to select_action() -- set True for
        a fair, greedy comparison against PPO's model.predict(deterministic=True)
        and against heuristics like EDF (see select_action()'s docstring).
        """
        self.model.eval()
        with torch.no_grad():
            action, _, _ = select_action(self.model, obs, mask, self.device, deterministic=deterministic)
        return action

    def train(self, total_timesteps=200_000, plotter=None):
        """
        Main A2C training loop.

        plotter: optional LiveTrainingPlotter (see Code/plotting_utils.py). If given,
        notify_episode(episode_reward, timestep) is called each time an episode ends,
        which live-plots and periodically saves the reward curve.
        """

        sched_env = self._resolve_sched_env() if self.use_rcpo else None

        obs, info = self.env.reset()
        done = False
        truncated = False
        t = 0
        episode_reward = 0.0

        while t < total_timesteps:

            # Clear rollout buffer
            self.buffer.reset()

            # Collect n-step rollout
            for _ in range(self.n_steps):

                # Get action mask from env. GymSchedulingEnv.reset()/step() always
                # populate info["action_mask"], so this is always present -- the
                # previous `info.get("action_mask", self.env.get_action_mask())`
                # looked like a safe fallback but dict.get() evaluates its default
                # argument eagerly, so it called self.env.get_action_mask() on every
                # step regardless. That crashed under gymnasium>=1.x, which removed
                # automatic attribute forwarding through Wrapper.__getattr__ (Monitor
                # has no get_action_mask of its own; use env.get_wrapper_attr(...) or
                # env.unwrapped if a genuine fallback is ever needed here).
                mask = info["action_mask"]

                # Select masked action
                action, log_prob, value = select_action(self.model, obs, mask, self.device)

                # Step environment
                next_obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward  # raw reward, for logging/plotting

                # Reward used for the returns/advantage computation: optionally
                # scale-normalised (see RunningMeanStd), independent of the raw
                # reward tracked above for episode_reward/plotter logging.
                if self.normalize_rewards:
                    self.reward_rms.update(reward)
                    train_reward = float(np.clip(reward / (self.reward_rms.std + 1e-8), -10.0, 10.0))
                else:
                    train_reward = reward

                # Store transition
                self.buffer.add(obs, action, train_reward, float(done or truncated), value, log_prob, mask)

                obs = next_obs
                t += 1

                # If episode ends, reset environment
                if done or truncated:
                    if plotter is not None:
                        plotter.notify_episode(episode_reward, t)
                    episode_reward = 0.0

                    # RCPO slow-timescale multiplier update -- must read
                    # episode_cost from `info` (this step's return value)
                    # BEFORE env.reset() zeroes it out. See
                    # Future/research/2026-08-21-rcpo-constrained-tardiness.md
                    # Section 3 for the projected-ascent update rule and why it
                    # only fires every rcpo_update_every_episodes episodes.
                    if self.use_rcpo:
                        self._rcpo_episode_costs.append(info.get("episode_cost", 0.0))
                        if len(self._rcpo_episode_costs) >= self.rcpo_update_every_episodes:
                            mean_cost = float(np.mean(self._rcpo_episode_costs))
                            new_lambda = self._lambda_value + self.rcpo_lambda_lr * (mean_cost - self.rcpo_alpha)
                            new_lambda = float(np.clip(new_lambda, 0.0, self.rcpo_lambda_max))
                            self._lambda_value = new_lambda
                            sched_env.lambda2 = new_lambda
                            self.lambda_history.append((t, new_lambda, mean_cost))
                            self._rcpo_episode_costs = []

                    obs, info = self.env.reset()
                    done = False
                    truncated = False
                    break

            # Bootstrap value for last state
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, last_value = self.model(obs_t)
            last_value = float(last_value.item())

            # Compute returns + advantages
            returns, advantages = self.buffer.compute_returns_and_advantages(
                last_value, self.gamma, self.lam
            )

            # Convert rollout to tensors
            obs_batch = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32, device=self.device)
            actions_batch = torch.tensor(self.buffer.actions, dtype=torch.int64, device=self.device)
            returns_batch = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advantages_batch = torch.tensor(advantages, dtype=torch.float32, device=self.device)

            # Forward pass
            logits, values = self.model(obs_batch)
            values = values.squeeze(-1)

            # Compute policy distribution using the SAME mask each action was
            # actually sampled under during rollout collection (select_action, via
            # masked_softmax). Previously this recomputed an unmasked softmax here,
            # which meant log_probs/entropy were both computed under a different
            # distribution than the one really used to pick actions -- diluting
            # gradient signal across masked/infeasible actions instead of
            # concentrating it on the legal action space.
            mask_batch = torch.tensor(np.array(self.buffer.masks), dtype=torch.float32, device=self.device)
            probs = masked_softmax(logits, mask_batch)
            dist = torch.distributions.Categorical(probs)

            # Log-probs of chosen actions
            log_probs = dist.log_prob(actions_batch)

            # Entropy bonus (encourages exploration)
            entropy = dist.entropy().mean()

            # A2C losses
            policy_loss = -(log_probs * advantages_batch.detach()).mean()
            value_loss = (returns_batch - values).pow(2).mean()
            entropy_loss = -entropy * self.ent_coef

            loss = policy_loss + self.value_coef * value_loss + entropy_loss

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()


def make_maskable_a2c(
    env,
    device="cpu",
    policy_type: str = "pointer",
    policy_kwargs: dict = None,
    normalize_rewards: bool = True,
    **rcpo_kwargs,
):
    """
    Create a true maskable A2C agent.

    policy_type: "pointer" (default, PointerActorCritic) or "flat"
    (MaskableActorCritic, the original A/B baseline).

    rcpo_kwargs: forwarded to MaskableA2C (use_rcpo, rcpo_alpha,
    rcpo_lambda_init, rcpo_lambda_lr, rcpo_lambda_max,
    rcpo_update_every_episodes) -- see its docstring.
    """
    return MaskableA2C(
        env,
        device=device,
        policy_type=policy_type,
        policy_kwargs=policy_kwargs,
        normalize_rewards=normalize_rewards,
        **rcpo_kwargs,
    )


def train_a2c(agent, total_timesteps=200_000, plotter=None):
    """
    Train the agent.
    """
    agent.train(total_timesteps, plotter=plotter)
    return agent