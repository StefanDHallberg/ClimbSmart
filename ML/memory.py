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
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.memory, f)
            print(f"Saved replay memory to '{filename}'")
        except Exception as e:
            print(f"Error saving replay memory to '{filename}': {e}")

    def load_memory(self, filename):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'rb') as f:
                try:
                    self.memory = pickle.load(f)
                    print(f"Loaded replay memory from '{filename}'")
                except EOFError:
                    print(f"Error: End of file reached while loading '{filename}'")
        else:
            print(f"File '{filename}' does not exist or is empty.")
    
    def clear(self):
        self.memory.clear()
        print("Cleared replay memory")



# Define the replay memory
replay_memory = ReplayMemory(capacity=10000)

# try:
#     replay_memory.load_memory('replay_memory.pkl')
# except (FileNotFoundError, pickle.UnpicklingError):
#     print("Could not load replay memory from 'replay_memory.pkl'")


