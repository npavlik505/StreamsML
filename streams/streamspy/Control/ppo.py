# Proximal Policy Optimization algorithm implementation for jet actuation
# Similar interface to dqn.py and ddpg.py

import os
from pathlib import Path
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from streamspy.base_agent import BaseAgent


class ActorCritic(nn.Module):
    """Combined actor-critic network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_width: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        # actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, action_dim),
        )
        # critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )
        # log standard deviation for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor):
        action_mean = self.actor(state)
        value = self.critic(state)
        # scale the mean to allowed range
        action_mean = self.max_action * torch.tanh(action_mean)
        return action_mean, value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        mean, value = self.forward(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action)
        return log_prob.sum(1, keepdim=True), value


class agent(BaseAgent):
    """PPO agent used for learning-based jet actuation."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_width: int, batch_size: int, lr: float, GAMMA: float, eps_clip: float, K_epochs: int):
        self.batch_size = batch_size
        self.gamma = GAMMA
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, hidden_width, max_action)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = []

        # self.run_timestamp = time.strftime("%Y%m%d.%H%M%S")
        # self.run_name = self.run_timestamp
        self.checkpoint = checkpoint_dir
        
        self.initialize_networks()

    def initialize_networks(self) -> None:
        # save_dir = f"{self.run_name}/Initial_Parameters"
        save_dir = f"{self.checkpoint}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(save_dir, "policy_initial.pt"))

    def choose_action(self, s: torch.Tensor, step):
        state = torch.unsqueeze(torch.clone(s), 0)
        with torch.no_grad():
            mean, _ = self.policy(state)
            std = self.policy.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(1)
        self.last_state = state.squeeze(0)
        self.last_action = action.squeeze(0)
        self.last_logprob = log_prob.squeeze(0)
        action = torch.clamp(action, -self.policy.max_action, self.policy.max_action)
        return action.squeeze(0).cpu().numpy()

    def learn(self, obs, action, reward, next_obs):
        state = self.last_state
        action_tensor = self.last_action
        logprob = self.last_logprob
        next_state = torch.tensor(next_obs, dtype=torch.float32)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        self.memory.append((state, action_tensor, logprob, reward_tensor, next_state))

        if len(self.memory) < self.batch_size:
            return

        # prepare batch tensors
        states = torch.stack([m[0] for m in self.memory])
        actions = torch.stack([m[1] for m in self.memory])
        old_logprobs = torch.stack([m[2] for m in self.memory]).unsqueeze(1)
        rewards = torch.stack([m[3] for m in self.memory])
        next_states = torch.stack([m[4] for m in self.memory])
        self.memory.clear()

        with torch.no_grad():
            _, state_values = self.policy(states)
            _, next_values = self.policy(next_states)
            advantages = rewards + self.gamma * next_values - state_values
            returns = rewards + self.gamma * next_values

        for _ in range(self.K_epochs):
            logprobs, values = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_checkpoint(self, directory: Path, tag: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), directory / f"policy_{tag}.pt")

    def load_checkpoint(self, checkpoint: Path) -> None:
        directory = checkpoint.parent
        tag = checkpoint.name
        self.policy.load_state_dict(torch.load(directory / f"policy_{tag}.pt"))


