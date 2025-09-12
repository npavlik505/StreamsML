# DQN reinforcement learning algorithm, adapted from ddpg.py

import os
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from streamspy.base_LearningBased import BaseAgent


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
    def __init__(self, state_dim, action_dim, max_action, hidden_width, buffer_size, batch_size, lr, target_update, GAMMA, TAU, epsilon, checkpoint_dir):
        self.hidden_width = hidden_width
        self.batch_size = batch_size
        self.GAMMA = GAMMA
        self.lr = lr
        self.TAU = TAU
        self.epsilon = epsilon
        self.target_update = target_update
        self.learn_step = 0
        self.max_amplitude = max_action

        # Discretization step 1: Three discrete actions are provided as network outputs
        self.discretized_action_dim = torch.tensor([-self.max_amplitude, 0.0, self.max_amplitude,], dtype=torch.float32)
        self.q = QNetwork(state_dim, len(self.discretized_action_dim), hidden_width)        
        
        self.q_target = copy.deepcopy(self.q)

        self.replay_buffer = ReplayBuffer(state_dim, buffer_size)

        # self.run_timestamp = time.strftime("%Y%m%d.%H%M%S")
        # self.run_name = self.run_timestamp
        self.checkpoint = checkpoint_dir
        
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
