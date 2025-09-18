##DDPG REINFORCEMENT LEARNING ALGORITHM

#Add to python path
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)

#Imports for DDPG
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import inspect
from collections import deque

from streamspy.base_LearningBased import BaseAgent
import libstreams as streams

#Define actor and critic NN.
# Actor produces single action; The state is inputed, the action is out made continuous by multiplying max acion with tanh(NN output)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s): #Changed l1 and l2 from F.relu() to tanh
        s = torch.tanh(self.l1(s))
        s = torch.tanh(self.l2(s))
        a = self.max_action * torch.tanh(self.l3(s))  # [-max,max]
        return a

# Critic produces single value (Q value); The state AND action is inputed, the output represents the value of taking the action-state pair
class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

# Replay buffer stores 1000 state, action, reward, next state, change in weights (S,A,R,S_,dw) data sets
class ReplayBuffer(object):
    # Specifies max number of SARSA tuples collected (max_size) and creates matrices to store the collected data
    def __init__(self, state_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        self.s = torch.zeros((self.max_size, state_dim))
        self.a = torch.zeros((self.max_size, action_dim))
        self.r = torch.zeros((self.max_size, 1))
        self.s_ = torch.zeros((self.max_size, state_dim))
    #The method to store the data
    def store(self, s, a, r, s_):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    #Out of the stored data, which is on length self.size, a batch_size number of SARS_dw samples are randomly collected
    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.clone(self.s[index])
        batch_a = torch.clone(self.a[index]) 
        batch_r = torch.clone(self.r[index]) 
        batch_s_ = torch.clone(self.s_[index]) 

        return batch_s, batch_a, batch_r, batch_s_
        
class OUNoise:
    """Ornstein-Uhlenbeck (OU) method of adding noise for exploration."""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state

#This the policy gradient algorithm, notice this uses the actor and critic classes made earlier
#The hyperparameters are defined, the actor & critc NN are defined as attributes and their Target NN are created
#Lastly, the optimizer, Adam, is selected to adjust the NN weights and the MSELoss is selected for use in the backprop calc
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
    
        self.hidden_width = params.get("hidden_width")  # The number of neurons in hidden layers of the neural network
        self.batch_size = params.get("batch_size")  # batch size
        self.GAMMA = params.get("gamma")  # discount factor
        self.TAU = params.get("tau")  # Softly update the target network
        self.lr = params.get("learning_rate")  # learning rate

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        self.ou_noise = OUNoise(action_dim)

        # self.run_timestamp = time.strftime("%Y%m%d.%H%M%S")
        # self.run_name = self.run_timestamp
        self.checkpoint = params.get("checkpoint_dir")  

        self.initialize_networks()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.MseLoss = nn.MSELoss()
        
        self.add_noise = True 
        
        self.mach = env.config.physics.mach
        
        self.env = env
        self.actuation_queue = deque()
        self.observation_queue = deque()
        self._skip_delay = False
        self.sensor_actuator_delay = params.get("sensor_actuator_delay")

    def initialize_networks(self) -> None:
        # save_dir = f"{self.run_name}/Initial_Parameters"
        save_dir = f"{self.checkpoint}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor_initial.pt"))
        torch.save(self.actor_target.state_dict(), os.path.join(save_dir, "actor_target_initial.pt"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic_initial.pt"))
        torch.save(self.critic_target.state_dict(), os.path.join(save_dir, "critic_target_initial.pt"))

    # An action is chosen by feeding the state into the actor NN which outputs the action a
    def choose_action(self, s, step):
        s = torch.unsqueeze(torch.clone(s), 0)
        a = self.actor(s).data.numpy().flatten()
        if self.add_noise:
            if step == 0:
                self.ou_noise.reset()
            a = a + self.ou_noise.sample()
        return np.clip(a, -self.actor.max_action, self.actor.max_action)

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
            
        if self._skip_delay or not self.sensor_actuator_delay:
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
            torch.tensor(action, dtype=torch.float32),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
        )
        if self.replay_buffer.size < self.batch_size:
            return
            
        batch_s, batch_a, batch_r, batch_s_= self.replay_buffer.sample(self.batch_size) # Sample a batch

        # Compute the target Q. This is done with no_grad so the target Q NN weights won't be adjusted every learning
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * Q_

        # Compute the current Q and then the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        actor_loss, critic_loss = actor_loss.item(), critic_loss.item()
        # LOGGER.debug("actor_loss=%f critic_loss=%f", actor_loss, critic_loss)
            
    def save_checkpoint(self, directory: Path, tag: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), directory / f"actor_{tag}.pt")
        torch.save(self.critic.state_dict(), directory / f"critic_{tag}.pt")

    def load_checkpoint(self, checkpoint: Path) -> None:
        self.add_noise = False
        directory = checkpoint.parent
        tag = checkpoint.name
        self.actor.load_state_dict(torch.load(directory / f"actor_{tag}.pt"))
        self.critic.load_state_dict(torch.load(directory / f"critic_{tag}.pt"))
