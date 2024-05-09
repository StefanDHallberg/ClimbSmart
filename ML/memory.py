import collections
import os
import random
import pickle

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)


    def push(self, transition):
        # If the memory is at capacity, remove the oldest transition
        if len(self.memory) == self.capacity:
            self.memory.popleft()

        # Add the new transition to the memory
        self.memory.append(transition)


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save_memory(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)
            f.close()
        print(f"Saved replay memory to '{filename}'")

    def load_memory(self, filename):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'rb') as f:
                self.memory = pickle.load(f)
        else:
            print(f"File '{filename}' does not exist or is empty.")

# Define the replay memory
replay_memory = ReplayMemory(capacity=10000)

try:
    replay_memory.load_memory('replay_memory.pkl')
except (FileNotFoundError, pickle.UnpicklingError):
    print("Could not load replay memory from 'replay_memory.pkl'")


