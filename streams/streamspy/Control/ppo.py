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
from collections import deque

from streamspy.base_LearningBased import BaseAgent
import libstreams as streams


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

    def __init__(self, env):
        params = env.config.jet.jet_params
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        hidden_width = params.get("hidden_width")
        lr = params.get("learning_rate")
        self.sensor_start = params.get("obs_xstart")
        self.sensor_end = params.get("obs_xend")
        self.slot_start = params.get("slot_start") 
        self.slot_end = params.get("slot_end")

        # grid parameters
        self.nx = env.config.grid.nx
        self.ny = env.config.grid.ny

        #length parameters
        self.lx = env.config.length.lx
        self.ly = env.config.length.ly
        
        self.batch_size = params.get("batch_size")
        self.gamma = params.get("gamma")
        self.eps_clip = params.get("eps_clip")
        self.K_epochs = params.get("K_epochs")

        self.policy = ActorCritic(state_dim, action_dim, hidden_width, max_action)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = []

        # self.run_timestamp = time.strftime("%Y%m%d.%H%M%S")
        # self.run_name = self.run_timestamp
        self.checkpoint = params.get("checkpoint_dir")
        
        self.initialize_networks()
        
        self.actuation_queue = deque()
        self.observation_queue = deque()
        self._skip_delay = False
        self.env = env

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
    
    def delay_action(self, action, observation):
        convection_complete = False

        def _zero_like(obs):
            if isinstance(obs, torch.Tensor):
                return torch.zeros_like(obs)
            if isinstance(obs, np.ndarray):
                return np.zeros_like(obs)
            if isinstance(obs, (list, tuple)):
                zeros = [0 for _ in obs]
                return type(obs)(zeros)
            return 0.0

        def _as_float(value):
            if isinstance(value, torch.Tensor):
                flat = value.detach().reshape(-1)
                if flat.numel() == 0:
                    return 0.0
                return float(flat[0].item())
            arr = np.asarray(value)
            if arr.size == 0:
                return 0.0
            return float(arr.reshape(-1)[0])

        default_prev_obs = _zero_like(observation)
        default_next_obs = _zero_like(observation)
        
        if self.env.step_count == 0:
            self.actuation_queue.clear()
            self.observation_queue.clear()
            self._skip_delay = False
            # Acceptable sensor actuator setup
            ok_upstream = (self.env._obs_xend < self.slot_start)
            # Unacceptable sensor actuator setup (No delay)
            contiguous = (self.env._obs_xend == self.slot_start)
            overlaps = (self.env._obs_xend > self.slot_start)
            self._skip_delay = contiguous or overlaps
            if self._skip_delay:
                print(f"observation window x: {self.env._obs_xstart}-{self.env._obs_xend} must be upstream from actuator x: {self.slot_start}-{self.slot_end}")
                print(f"delay will not be applied")
                return _as_float(action), default_prev_obs, default_next_obs, False
            
        if self._skip_delay:
            return _as_float(action), default_prev_obs, default_next_obs, False
            
        # enqueue the control action with zero accumulated convection
        self.actuation_queue.append({"actuation": action, "convection": 0.0})
        self.observation_queue.append({"observation": observation})

        # compute local convective velocity at the sensing region
        rho_slice = streams.wrap_get_w_avzg_slice(
            self.env._obs_xstart,
            self.env._obs_xend,
            self.env._obs_ystart,
            self.env._obs_yend,
            1,
        )
        rhou_slice = streams.wrap_get_w_avzg_slice(
            self.env._obs_xstart,
            self.env._obs_xend,
            self.env._obs_ystart,
            self.env._obs_yend,
            2,
        )
        u_slice = rhou_slice[0] / rho_slice[0]
        Uc = float(np.mean(u_slice))

        dt = float(streams.wrap_get_dtglobal())
        dx = self.lx / self.nx

        # distance between sensor and actuator centroids in index units
        sensor_centroid = 0.5 * (self.env._obs_xstart + self.env._obs_xend)
        slot_centroid = 0.5 * (self.slot_start + self.slot_end)
        distance_index = slot_centroid - sensor_centroid

        # update convection progress for all queued actions
        for entry in self.actuation_queue:
            entry["convection"] += (Uc * dt) / dx

        step_actuation = 0.0
        step_observation = _zero_like(observation)
        next_step_observation = _zero_like(observation)
        if self.actuation_queue and self.actuation_queue[0]["convection"] >= distance_index:
            step_actuation = self.actuation_queue.popleft()["actuation"]
            step_observation = self.observation_queue.popleft()["observation"]
            if self.observation_queue:
                next_step_observation = self.observation_queue[0]["observation"]
                convection_complete = True
            
        return _as_float(step_actuation), step_observation, next_step_observation, convection_complete   

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


