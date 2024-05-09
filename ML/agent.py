from .dqn_model import DQN
from .memory import ReplayMemory

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
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
        self.num_actions = output_size

    def select_action(self, state, epsilon):
        if random.random() > epsilon:  # Exploitation
            with torch.no_grad():
                # Use the DQN to select the action with the highest Q-value
                return self.dqn(state).max(0)[1].view(1, 1)
        else:  # Exploration
            # Select a random action
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.dqn(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.dqn(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1).float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
