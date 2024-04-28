import sys
sys.path.append("./ML/memory.py")
sys.path.append("../ML/DQN.py")

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import DQN
import ReplayMemory
from collections import namedtuple




# defines the transition tuple that will be stored in the replay memory buffer.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Agent:
    def __init__(self, input_size, output_size, lr=0.001):
        self.dqn = DQN(input_size, output_size)
        self.memory = ReplayMemory(capacity=10000)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.gamma = 0.99
        self.batch_size = 64

    def select_action(self, state):
        # Implement epsilon-greedy action selection
        pass

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute the Q(s, a) values
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.dqn(state_batch).gather(1, action_batch)

        # Compute the expected Q values
        next_state_values = self.dqn(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
