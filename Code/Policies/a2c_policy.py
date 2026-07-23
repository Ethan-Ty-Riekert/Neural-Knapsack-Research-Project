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

def select_action(model, obs, mask, device):
    """Select an action using masked softmax.
    This is used both during training and evaluation.
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

    # Sample an action
    action = dist.sample()

    # Log-probability of chosen action
    log_prob = dist.log_prob(action)

    return int(action.item()), float(log_prob.item()), float(value.item())


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

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

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

    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = torch.device(device)

        # Extract dimensions from environment
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        # Hyperparameters (tweak these later)
        self.n_steps = 5
        self.gamma = 0.99
        self.lam = 1.0
        self.ent_coef = 0.0
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.lr = 7e-4

        # Actor-Critic network
        self.model = MaskableActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Rollout buffer
        self.buffer = RolloutBuffer(self.n_steps)

    def act(self, obs, mask):
        """
        Action selection for evaluation (no gradients).
        """
        self.model.eval()
        with torch.no_grad():
            action, _, _ = select_action(self.model, obs, mask, self.device)
        return action

    def train(self, total_timesteps=200_000):
        """
        Main A2C training loop.
        """

        obs, info = self.env.reset()
        done = False
        truncated = False
        t = 0

        while t < total_timesteps:

            # Clear rollout buffer
            self.buffer.reset()

            # Collect n-step rollout
            for _ in range(self.n_steps):

                # Get action mask from env
                mask = info.get("action_mask", self.env.get_action_mask())

                # Select masked action
                action, log_prob, value = select_action(self.model, obs, mask, self.device)

                # Step environment
                next_obs, reward, done, truncated, info = self.env.step(action)

                # Store transition
                self.buffer.add(obs, action, reward, float(done or truncated), value, log_prob)

                obs = next_obs
                t += 1

                # If episode ends, reset environment
                if done or truncated:
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

            # Compute policy distribution (unmasked entropy for simplicity)
            probs = torch.softmax(logits, dim=-1)
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


def make_maskable_a2c(env, device="cpu"):
    """
    Create a true maskable A2C agent.
    """
    return MaskableA2C(env, device=device)


def train_a2c(agent, total_timesteps=200_000):
    """
    Train the agent.
    """
    agent.train(total_timesteps)
    return agent