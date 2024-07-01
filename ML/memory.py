from collections import namedtuple, deque
import os
import random
import pickle
import asyncio

# Defining the Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

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

    async def async_push(self, states, actions, rewards, next_states, dones):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.push, states, actions, rewards, next_states, dones)
