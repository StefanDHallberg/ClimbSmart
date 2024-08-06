import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import config
from collections import namedtuple
from .dqn_model import DQN
from .memory import ReplayMemory
from torch.cuda.amp import autocast, GradScaler

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
        self.epsilon = config.epsilon_start
        self.epsilon_final = config.epsilon_final
        self.epsilon_decay = config.epsilon_decay
        self.verbose = config.verbose
        self.target_update_frequency = target_update_frequency
        self.steps_done = 0

    def select_action(self, state):
        # Ensure state is a tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        if self.verbose:
            # Print state shape for debugging
            print(f"State shape before action selection: {state.shape}")  # Debugging statement

        with torch.no_grad():
            if random.random() > self.epsilon:
                # Get Q-values for all actions
                q_values = self.policy_net(state)
                # Print Q-values shape for debugging
                print(f"Q-values shape: {q_values.shape}")  # Debugging statement

                # Check if q_values has the correct shape
                if q_values.size(0) == 1 and q_values.size(1) == self.num_actions:
                    # Select the action with the maximum Q-value
                    action = q_values.max(1)[1].view(1, 1)  # Ensure correct indexing
                    if self.verbose:
                        print(f"Action selected (exploitation): {action}")
                else:
                    raise ValueError(f"Unexpected Q-values shape: {q_values.shape}")

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

        # Sample batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        chunk_size = self.batch_size // 16
        device = torch.device("cuda")
        accumulation_steps = 4

        scaler = GradScaler()

        self.optimizer.zero_grad()

        # Iterate over the batch in chunks to save memory and speed up training
        for i in range(0, self.batch_size, chunk_size):
            chunk_transitions = Transition(
                state=batch.state[i:i + chunk_size],
                action=batch.action[i:i + chunk_size],
                next_state=batch.next_state[i:i + chunk_size],
                reward=batch.reward[i:i + chunk_size]
            )

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, chunk_transitions.next_state)), dtype=torch.bool, device=device)
            non_final_next_states = torch.cat([s for s in chunk_transitions.next_state if s is not None], dim=0).to(device)

            state_batch = torch.cat(chunk_transitions.state, dim=0).to(device)
            action_batch = torch.cat(chunk_transitions.action, dim=0).to(device)
            reward_batch = torch.cat(chunk_transitions.reward, dim=0).to(device)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                available_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
                if available_memory < state_batch.element_size() * state_batch.nelement() * 2:
                    print("Insufficient CUDA memory, skipping this chunk")
                    continue

            with autocast():
                state_action_values = self.policy_net(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(chunk_size, device=device)
                if non_final_next_states.size(0) > 0:
                    next_q_values = self.target_net(non_final_next_states).max(1)[0]
                    next_state_values[non_final_mask] = next_q_values

                expected_state_action_values = (next_state_values * self.gamma) + reward_batch.view(-1)
                expected_state_action_values = expected_state_action_values.unsqueeze(1)

                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
                if self.verbose:
                    print(f"Loss: {loss.item()}")

            scaler.scale(loss).backward()
            if (i // chunk_size + 1) % accumulation_steps == 0:
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            del state_batch, action_batch, reward_batch, non_final_next_states, next_state_values, state_action_values, expected_state_action_values, loss

            torch.cuda.empty_cache()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad()

        self.update_epsilon()

        # Update the target network if necessary
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.verbose:
                print("Updated target network")

    def update_step_counter(self):
        self.steps_done += 1