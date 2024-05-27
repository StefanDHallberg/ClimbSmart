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
        # print(f"State shape: {state.shape}")  # Debugging statement
        if random.random() > self.epsilon:
            with torch.no_grad():
                action = self.dqn(state).max(1)[1].view(1, 1)
                # print(f"Selected action (exploitation): {action}, shape: {action.shape}")  # Debugging statement
                return action
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)
            # print(f"Selected action (exploration): {action}, shape: {action.shape}")  # Debugging statement
            return action


    def update_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(-1, self.dqn.fc1.in_features)  # Ensure correct shape
        # print(f"Non-final mask shape: {non_final_mask.shape}")  # Debugging statement
        # print(f"Non-final next states shape: {non_final_next_states.shape}")  # Debugging statement

        state_batch = torch.cat(batch.state).view(-1, self.dqn.fc1.in_features)  # Ensure correct shape
        action_batch = torch.cat(batch.action).view(-1, 1)  # Ensure actions have shape [batch_size, 1]
        reward_batch = torch.cat(batch.reward).view(-1, 1)  # Ensure rewards have shape [batch_size, 1]

        # print(f"State batch shape: {state_batch.shape}")  # Debugging statement
        # print(f"Action batch shape: {action_batch.shape}")  # Debugging statement
        # print(f"Reward batch shape: {reward_batch.shape}")  # Debugging statement

        # Forward pass through the DQN model
        state_action_values = self.dqn(state_batch).gather(1, action_batch)
        # print(f"State action values shape: {state_action_values.shape}")  # Debugging statement

        # Compute expected state-action values
        next_state_values = torch.zeros(self.batch_size, device=self.dqn.fc1.weight.device)
        if non_final_next_states.size(0) > 0:  # Check if there are any non-final next states
            next_q_values = self.dqn(non_final_next_states).max(1)[0].detach()
            # print(f"Next Q values shape: {next_q_values.shape}")  # Debugging statement
            next_state_values[non_final_mask] = next_q_values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.view(-1)
        # print(f"Next state values shape: {next_state_values.shape}")  # Debugging statement
        expected_state_action_values = expected_state_action_values.view(self.batch_size, 1)  # Ensure shape is [batch_size, 1]
        # print(f"Expected state action values shape: {expected_state_action_values.shape}")  # Debugging statement

        # Compute loss using smooth L1 loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # print(f"Loss shape: {loss.shape}")  # Debugging statement

        # Perform backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


