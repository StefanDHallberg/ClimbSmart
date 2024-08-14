import os
import pickle
import numpy as np
import torch
from collections import deque

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, save_dir="replay_memory"):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.save_dir = save_dir
        self.save_file = os.path.join(save_dir, "memory.pkl")

        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def push(self, state, action, reward, next_state, done):
        """Save a transition."""
        max_priority = self.priorities.max() if len(self.memory) > 0 else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of transitions with probability proportional to their priority."""
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Add a check for NaN values in probabilities
        if np.isnan(probabilities).any():
            raise ValueError("Probabilities contain NaN values.")

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        # Compute importance sampling weights
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool),  
            indices,
            weights,
        )

    def update_priorities(self, batch_indices, batch_priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def save_memory(self):
        """Save replay memory to disk."""
        try:
            with open(self.save_file, 'wb') as f:
                pickle.dump((self.memory, self.priorities, self.position), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Replay memory saved to {self.save_file}")
        except Exception as e:
            print(f"Failed to save replay memory: {e}")

    def load_memory(self):
        """Load replay memory from disk."""
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'rb') as f:
                    self.memory, self.priorities, self.position = pickle.load(f)
                print(f"Replay memory loaded from {self.save_file}")
        except Exception as e:
            print(f"Failed to load replay memory: {e}")

    def __len__(self):
        return len(self.memory)
