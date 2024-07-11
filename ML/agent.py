import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import gc
from collections import namedtuple
from .dqn_model import DQN
from .memory import ReplayMemory
from torch.cuda.amp import autocast, GradScaler

from memory_profiler import profile

# Function to apply @profile to all methods in a class
# def apply_profile_to_methods(cls):
#     for attr_name in dir(cls):
#         attr = getattr(cls, attr_name)
#         if callable(attr) and not attr_name.startswith("__"):
#             setattr(cls, attr_name, profile(attr))
#     return cls

# Defining the transition tuple that will be stored in the replay memory buffer.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# @apply_profile_to_methods
class Agent:
    def __init__(self, input_channels, num_actions, lr=0.001, gamma=0.99, batch_size=64, capacity=10000,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=0.999, verbose=False):
        self.dqn = DQN(input_channels, num_actions)
        self.memory = ReplayMemory(capacity)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.verbose = verbose
    
    def select_action(self, state):
        with torch.no_grad():  # Ensure no gradients are tracked
            if random.random() > self.epsilon:
                action = self.dqn(state).max(1)[1].view(1, 1)
                if self.verbose:
                    print(f"Selected action (exploitation): {action}")
                return action
            else:
                action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)
                if self.verbose:
                    print(f"Selected action (exploration): {action}")
                return action

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
        if self.verbose:
            print(f"Updated epsilon: {self.epsilon}")

    def check_cuda_memory(self, device):
        if torch.cuda.is_available():
            print(f"Total memory: {torch.cuda.get_device_properties(device).total_memory}")
            print(f"Allocated memory: {torch.cuda.memory_allocated(device)}")
            print(f"Cached memory: {torch.cuda.memory_reserved(device)}")

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        device = self.dqn.fc1.weight.device
        scaler = GradScaler()

        # Separate batch data
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        self.optimizer.zero_grad()

        # Compute Q(s_t, a)
        with autocast():
            state_action_values = self.dqn(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            next_state_values = torch.zeros(self.batch_size, device=device)
            if non_final_next_states.size(0) > 0:
                next_q_values = self.dqn(non_final_next_states).max(1)[0].detach()
                next_state_values[non_final_mask] = next_q_values

            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            expected_state_action_values = expected_state_action_values.unsqueeze(1)

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            if self.verbose:
                print(f"Loss: {loss.item()}")

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        scaler.step(self.optimizer)
        scaler.update()

        # Update epsilon after optimization
        self.update_epsilon()

        # Print CUDA memory summary
        if self.verbose and torch.cuda.is_available():
            print(torch.cuda.memory_summary(device=device))
