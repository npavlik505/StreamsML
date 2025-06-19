##DDPG REINFORCEMENT LEARNING ALGORITHM

#Add to python path
import os
import sys
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

#Define actor and critic NN.
# Actor produces single action; The state is inputed, the action is out made continuous by multiplying max acion with tanh(NN output)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        #self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s): #Changed l1 and l2 from F.relu() to tanh
        s = torch.tanh(self.l1(s))
        #s = torch.tanh(self.l2(s))
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
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6)):
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

#This the policy gradient algorithm, notice this uses the actor and critic classes made earlier
#The hyperparameters are defined, the actor & critc NN are defined as attributes and their Target NN are created
#Lastly, the optimizer, Adam, is selected to adjust the NN weights and the MSELoss is selected for use in the backprop calc
class ddpg(object):
    def __init__(self, state_dim, action_dim, max_action, verbose: bool):
        self.verbose = verbose
        self.hidden_width = 8  # The number of neurons in hidden layers of the neural network
        self.batch_size = 50 #100  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        if self.verbose:
            print("Actor network:\n, self.actor")
            actor_params = sum(p.numel() for p in self.actor.parameters())
            print(f"Total actor parameters: {actor_params}")
            print("Critic network:\n", self.critic)
            critic_params = sum(p.numel() for p in self.critic.parameters())
            print(f"Total critic parameters: {critic_params}")
        self.run_timestamp = time.strftime("%Y%m%d.%H%M%S")
        self.run_name = self.run_timestamp

        #This call saves the initial network params (for verification) unless called from the Analysis folder
        if not self.is_called_from_analysis_folder():
            self.initialize_networks()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.MseLoss = nn.MSELoss()

    def initialize_networks(self):
        save_dir = f"{self.run_name}/Initial_Parameters"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'InitialActorParameters.pt'))
        torch.save(self.actor_target.state_dict(), os.path.join(save_dir, 'InitialActorTargetParameters.pt'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'InitialCriticParameters.pt'))
        torch.save(self.critic_target.state_dict(), os.path.join(save_dir, 'InitialCriticTargetParameters.pt'))        

    def is_called_from_analysis_folder(self):
        frame = inspect.currentframe().f_back.f_back
        module = inspect.getmodule(frame)
        if module and module.__file__:
            return 'Analysis' in module.__file__
        return False

    # An action is chosen by feeding the state into the actor NN which outputs the action a
    def choose_action(self, s):
        s = torch.unsqueeze(torch.clone(s), 0)
        a = self.actor(s).data.numpy().flatten()
        return a

    # We use our sample method, previously defined, to select the SARS_dw samples
    def learn(self, replay_buffer):
        batch_s, batch_a, batch_r, batch_s_= replay_buffer.sample(self.batch_size) # Sample a batch

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

        return actor_loss.item(), critic_loss.item()
