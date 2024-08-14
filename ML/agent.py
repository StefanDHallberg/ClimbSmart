import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import config
from collections import namedtuple
from .dqn_model import DQN

# Defining the transition tuple that will be stored in the replay memory buffer.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, input_channels, num_actions, input_width, input_height, lr, gamma, batch_size, epsilon_start, epsilon_final, epsilon_decay, verbose, target_update_frequency, memory):
        self.policy_net = DQN(input_channels, num_actions, input_width, input_height).to(device)
        self.target_net = DQN(input_channels, num_actions, input_width, input_height).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # Correctly use the passed lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.verbose = verbose
        self.target_update_frequency = target_update_frequency
        self.steps_done = 0

        self.reset()  # Initialize internal states

    def reset(self):
        self.epsilon = self.epsilon_start
        self.steps_done = 0
        self.policy_net.load_state_dict(self.target_net.state_dict())  # Sync the networks
        if self.verbose:
            print("Agent reset: epsilon reset, steps_done reset, networks synchronized")

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory"""
        self.memory.push(state, action, reward, next_state, done)

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        if self.verbose:
            print(f"State shape before action selection: {state.shape}")

        with torch.no_grad():
            if random.random() > self.epsilon:
                q_values = self.policy_net(state)
                if self.verbose:
                    print(f"Q-values shape: {q_values.shape}")

                if q_values.size(0) == 1 and q_values.size(1) == self.num_actions:
                    action = q_values.max(1)[1].view(1, 1)
                    if self.verbose:
                        print(f"Action selected (exploitation): {action}")
                else:
                    raise ValueError(f"Unexpected Q-values shape: {q_values.shape}")

                return action
            else:
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

        # Sample a batch of transitions with priority sampling
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size, beta=0.4)

        # Convert to tensors
        state_batch = torch.tensor(states, dtype=torch.float32).to(device)
        action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32).to(device)
        done_batch = torch.tensor(dones, dtype=torch.float32).to(device)

        # Ensure there are no NaNs in the state batches
        if torch.isnan(state_batch).any() or torch.isnan(next_state_batch).any():
            print("NaN detected in state or next state batch")
            return

        # Compute Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute the next Q values for non-terminal states
        non_final_mask = (done_batch == 0)
        non_final_next_states = next_state_batch[non_final_mask]
        next_state_values = torch.zeros(self.batch_size, device=device)
        if non_final_next_states.size(0) > 0:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Compute the loss using the importance sampling weights
        loss = (weights * F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values, reduction='none')).mean()

        # Check for NaN in loss
        if torch.isnan(loss).any():
            print("NaN detected in loss")
            return

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update priorities in the replay memory
        td_errors = (state_action_values.squeeze() - expected_state_action_values).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)

        self.update_epsilon()

        # Update the target network if necessary
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.verbose:
                print("Target network updated")

    def update_step_counter(self):
        self.steps_done += 1
