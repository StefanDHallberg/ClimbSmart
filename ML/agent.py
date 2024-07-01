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

        # Break down the batch processing into smaller chunks
        chunk_size = self.batch_size // 16  # Smaller chunk size
        device = self.dqn.fc1.weight.device
        accumulation_steps = 4  # Adjust as needed for your model and GPU memory

        scaler = GradScaler()

        self.optimizer.zero_grad()

        with torch.no_grad():
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

                # Log tensor sizes and devices
                if self.verbose:
                    print(f"state_batch size: {state_batch.size()}, device: {state_batch.device}")
                    print(f"action_batch size: {action_batch.size()}, device: {action_batch.device}")
                    print(f"reward_batch size: {reward_batch.size()}, device: {reward_batch.device}")
                    print(f"non_final_next_states size: {non_final_next_states.size()}, device: {non_final_next_states.device}")

                # Check CUDA memory allocation before processing
                if self.verbose:
                    print("Before processing chunk:")
                    self.check_cuda_memory(device)

                # Check CUDA memory allocation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    available_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
                    if available_memory < state_batch.element_size() * state_batch.nelement() * 2:  # Adjust multiplier as needed
                        print("Insufficient CUDA memory, skipping this chunk")
                        continue

                # Compute Q(s_t, a) - the model computes Q(s_t) and we select the columns of actions taken
                with autocast():
                    state_action_values = self.dqn(state_batch).gather(1, action_batch)

                    # Compute V(s_{t+1}) for all next states.
                    next_state_values = torch.zeros(chunk_size, device=device)
                    if non_final_next_states.size(0) > 0:
                        next_q_values = self.dqn(non_final_next_states).max(1)[0]
                        next_state_values[non_final_mask] = next_q_values

                    # Compute the expected Q values
                    expected_state_action_values = (next_state_values * self.gamma) + reward_batch.view(-1)
                    expected_state_action_values = expected_state_action_values.unsqueeze(1)  # Ensure the same shape

                    # Compute Huber loss
                    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
                    if self.verbose:
                        print(f"Loss: {loss.item()}")

                # Backward pass with gradient accumulation
                scaler.scale(loss).backward()
                if (i // chunk_size + 1) % accumulation_steps == 0:
                    for param in self.dqn.parameters():
                        param.grad.data.clamp_(-1, 1)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                # Clean up to free memory
                del state_batch
                del action_batch
                del reward_batch
                del non_final_next_states
                del next_state_values
                del state_action_values
                del expected_state_action_values
                del loss

                # Explicit garbage collection
                gc.collect()

                # Ensure no gradients are tracked
                torch.cuda.empty_cache()

                # Check CUDA memory allocation after processing
                if self.verbose:
                    print("After processing chunk:")
                    self.check_cuda_memory(device)

        # Final optimizer step in case the last accumulation didn't reach the accumulation_steps
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad()

        # Update epsilon after optimization
        self.update_epsilon()

        # Print CUDA memory summary
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
