import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple
from .dqn_model import DQN
from .memory import ReplayMemory

# Defining the transition tuple that will be stored in the replay memory buffer.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Agent:
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.99, batch_size=64, capacity=10000,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=0.999):
        self.dqn = DQN(input_size, output_size)
        self.memory = ReplayMemory(capacity)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = output_size
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.dqn(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)

    def optimize_model(self, experiences):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Forward pass through the DQN model
        state_action_values = self.dqn(state_batch)

        # Select the Q-values for the actions taken
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute expected state-action values
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.dqn(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss using smooth L1 loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1).float())

        # Perform backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
