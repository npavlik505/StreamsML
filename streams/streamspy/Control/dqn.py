# DQN reinforcement learning algorithm, adapted from ddpg.py

import os
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import deque

from streamspy.base_LearningBased import BaseAgent
import libstreams as streams


class QNetwork(nn.Module):
    # ``discretized_action_dim`` gives the number of discrete actions available
    def __init__(self, state_dim: int, discretized_action_dim: int, hidden_width: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, discretized_action_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        return self.l3(x)


class ReplayBuffer:
    def __init__(self, state_dim: int, max_size: int):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        self.s = torch.zeros((max_size, state_dim))
        self.a = torch.zeros((max_size, 1), dtype=torch.int64)
        self.r = torch.zeros((max_size, 1))
        self.s_ = torch.zeros((max_size, state_dim))

    def store(self, s, a, r, s_):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)
        batch_s = torch.clone(self.s[index])
        batch_a = torch.clone(self.a[index])
        batch_r = torch.clone(self.r[index])
        batch_s_ = torch.clone(self.s_[index])
        return batch_s, batch_a, batch_r, batch_s_


class agent(BaseAgent):
    def __init__(self, env):
        params = env.config.jet.jet_params
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        buffer_size = params.get("buffer_size")
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

        self.env = env
        self.actuation_queue = deque()
        self.observation_queue = deque()
        self._skip_delay = False
        
        self.hidden_width = params.get("hidden_width")
        self.batch_size = params.get("batch_size")
        self.GAMMA = params.get("gamma")
        self.lr = params.get("learning_rate")
        self.TAU = params.get("tau")
        self.epsilon = params.get("epsilon")
        self.target_update = params.get("target_update")
        self.learn_step = 0
        self.max_amplitude = max_action

        # Discretization step 1: Three discrete actions are provided as network outputs
        self.discretized_action_dim = torch.tensor([-self.max_amplitude, 0.0, self.max_amplitude], dtype=torch.float32)
        self.q = QNetwork(state_dim, len(self.discretized_action_dim), self.hidden_width)      
        
        self.q_target = copy.deepcopy(self.q)

        self.replay_buffer = ReplayBuffer(state_dim, buffer_size)

        # self.run_timestamp = time.strftime("%Y%m%d.%H%M%S")
        # self.run_name = self.run_timestamp
        self.checkpoint = params.get("checkpoint_dir")
        
        self.initialize_networks()

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def initialize_networks(self) -> None:
        # save_dir = f"{self.run_name}/Initial_Parameters"
        save_dir = f"{self.checkpoint}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.q.state_dict(), os.path.join(save_dir, "q_initial.pt"))
        torch.save(self.q_target.state_dict(), os.path.join(save_dir, "targetq_initial.pt"))

    def choose_action(self, s, step):
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(len(self.discretized_action_dim))
        else:
            with torch.no_grad():
                qs = self.q(torch.unsqueeze(torch.clone(s), 0))
                # Discretization step 3: argmax determines the output neuron with the highest value
                action_index = int(torch.argmax(qs, dim=1).item())
        self.last_action_index = action_index
        amplitude = float(self.discretized_action_dim[action_index].item())
        return np.array([amplitude], dtype=np.float32)

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
        self.replay_buffer.store(
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor([self.last_action_index], dtype=torch.int64),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
        )
        if self.replay_buffer.size < self.batch_size:
            return

        batch_s, batch_a, batch_r, batch_s_ = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            target_q = batch_r + self.GAMMA * torch.max(self.q_target(batch_s_), dim=1, keepdim=True)[0]
        current_q = self.q(batch_s).gather(1, batch_a)
        loss = self.loss_fn(target_q, current_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0: 
            # hard copy main network -> target network
            self.q_target.load_state_dict(self.q.state_dict())         

    def save_checkpoint(self, directory: Path, tag: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.q.state_dict(), directory / f"q_{tag}.pt")

    def load_checkpoint(self, checkpoint: Path) -> None:
        directory = checkpoint.parent
        tag = checkpoint.name
        self.q.load_state_dict(torch.load(directory / f"q_{tag}.pt"))
        self.q_target.load_state_dict(self.q.state_dict())
