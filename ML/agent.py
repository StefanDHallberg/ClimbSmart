import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import config
from collections import namedtuple
from .dqn_model import DQN
from .memory import ReplayMemory
from torch.amp import autocast, GradScaler

# Defining the transition tuple that will be stored in the replay memory buffer.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, input_channels, num_actions, input_width, input_height, lr, gamma, batch_size, epsilon_start, epsilon_final, epsilon_decay, verbose, target_update_frequency):
        self.policy_net = DQN(input_channels, num_actions, input_width, input_height).to(device)
        self.target_net = DQN(input_channels, num_actions, input_width, input_height).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(config.memory_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.num_actions = config.num_actions
        self.epsilon_start = config.epsilon_start
        self.epsilon_final = config.epsilon_final
        self.epsilon_decay = config.epsilon_decay
        self.verbose = config.verbose
        self.target_update_frequency = target_update_frequency
        self.steps_done = 0

        self.reset()  # Initialize internal states

    def reset(self):
        self.epsilon = self.epsilon_start
        self.steps_done = 0
        self.policy_net.load_state_dict(self.target_net.state_dict())  # Sync the networks
        if self.verbose:
            print("Agent reset: epsilon reset, steps_done reset, networks synchronized")

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        if self.verbose:
            print(f"State shape before action selection: {state.shape}")

        with torch.no_grad():
            # Epsilon-greedy action selection
            if random.random() > self.epsilon:
                q_values = self.policy_net(state) #forward pass the state through the network
                if self.verbose:
                    print(f"Q-values shape: {q_values.shape}")
                    print(f"Q-values: {q_values}")

                # Select the action with the highest Q-value
                action = q_values.max(1)[1].view(1, 1)
                if self.verbose:
                    print(f"Action selected (exploitation): {action}")
                return action
            else:
                # Select a random action
                action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long).to(device)
                if self.verbose:
                    print(f"Selected action (exploration): {action}")
                return action

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
        if self.verbose:
            print(f"Updated epsilon: {self.epsilon}")

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size) # Sample a batch of transitions from memory
        batch = Transition(*zip(*transitions)) # Transpose the batch (see the Transition tuple definition)

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        with autocast():
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(self.batch_size, device=device)
            if non_final_next_states.size(0) > 0:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

            expected_state_action_values = (next_state_values * self.gamma) + reward_batch.view(-1)
            expected_state_action_values = expected_state_action_values.unsqueeze(1)

            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Scale the loss and backward pass
        scaler = GradScaler()
        scaler.scale(loss).backward()

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        scaler.step(self.optimizer)
        scaler.update()

        self.update_epsilon()

        # Update target network periodically
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.verbose:
                print("Updated target network")

        # Increment step counter
        self.update_step_counter()


    def update_step_counter(self):
        self.steps_done += 1
