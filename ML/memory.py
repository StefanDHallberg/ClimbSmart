import collections
import os
import random
import pickle
import threading

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, transition):
        with self.lock:
            self.memory.append(transition)

    def sample(self, batch_size):
        with self.lock:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        with self.lock:
            return len(self.memory)

    def save_memory(self, filename):
        with self.lock:
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(self.memory, f)
                print(f"Saved replay memory to '{filename}'")
            except Exception as e:
                print(f"Error saving replay memory to '{filename}': {e}")

    def load_memory(self, filename):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with self.lock:
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
        with self.lock:
            self.memory.clear()
            print("Cleared replay memory")
