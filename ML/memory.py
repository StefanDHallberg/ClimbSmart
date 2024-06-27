from collections import namedtuple, deque
import os
import random
import pickle

# Defining the Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    # def push(self, transition):
    #         self.memory.append(transition)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

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
