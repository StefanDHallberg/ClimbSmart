from collections import namedtuple, deque
import os
import random
import pickle
import numpy as np
import torch

# Defining the Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        # Convert incoming tensor data to NumPy arrays if necessary
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.numpy()

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a random batch of transitions."""
        batch = random.sample(self.memory, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.memory)

    def save_memory(self, filename):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.memory, f)
            print(f"Saved replay memory to '{filename}'")
        except Exception as e:
            print(f"Error saving replay memory to '{filename}': {e}")

    def load_memory(self, filename):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            try:
                with open(filename, 'rb') as f:
                    self.memory = pickle.load(f)
                print(f"Loaded replay memory from '{filename}'")
            except EOFError:
                print(f"Error: End of file reached while loading '{filename}'")
            except Exception as e:
                print(f"Error loading replay memory from '{filename}': {e}")
        else:
            print(f"File '{filename}' does not exist or is empty.")

    def clear(self):
        self.memory.clear()
        print("Cleared replay memory")
