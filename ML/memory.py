import os
import pickle
import random
import numpy as np
import torch
import blosc
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

class ReplayMemory:
    def __init__(self, capacity, save_batch_size=1000, max_chunk_size=1000, save_dir="replay_memory"):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.save_batch_size = save_batch_size
        self.max_chunk_size = max_chunk_size  # Max number of transitions per chunk
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()  # Lock for synchronizing file access
        self.current_file_index = 0
        self.save_dir = save_dir
        self.index_filename = os.path.join(save_dir, "memory_agent_0_index.pkl")

        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Load the index file if it exists
        if os.path.exists(self.index_filename):
            with open(self.index_filename, 'rb') as f:
                self.current_file_index = pickle.load(f)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy().astype(np.float32)
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy().astype(np.float32)

        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self, batch_size):
        """Samples a random batch of transitions."""
        batch = random.sample(self.memory, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.memory)

    def _compress_and_save(self):
        try:
            with self.lock:
                # Prepare the filename where data will be saved
                filename = os.path.join(self.save_dir, f"memory_agent_0_{self.current_file_index}.pkl")

                # Split memory into chunks and save each chunk
                for i in range(0, len(self.memory), self.max_chunk_size):
                    chunk = list(self.memory)[i:i + self.max_chunk_size]
                    
                    # Serialize and compress the chunk
                    serialized_data = pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL)
                    compressed_data = blosc.compress(serialized_data, typesize=8, clevel=5, cname='lz4')
                    
                    # Save the compressed chunk to the file
                    with open(filename, 'ab') as f:  # Append to the file
                        f.write(compressed_data)
                    
                    # Clean up memory used by this chunk
                    del serialized_data, compressed_data, chunk
                    torch.cuda.empty_cache()  # Clear GPU memory if applicable

                    # Increment the file index for the next chunk
                    self.current_file_index += 1

                # Update the index file after saving all chunks
                with open(self.index_filename, 'wb') as f:
                    pickle.dump(self.current_file_index, f)

                print(f"Saved {len(self.memory)} transitions to '{filename}'")
        except Exception as e:
            print(f"Error during saving to file: {e}")
        finally:
            torch.cuda.empty_cache()  # Ensure GPU memory is cleared at the end


    def save_memory_async(self):
        """Asynchronously saves the replay memory."""
        self.executor.submit(self._compress_and_save)

    def load_memory(self):
        """Loads and decompresses the replay memory, limiting to the most recent experiences."""
        all_transitions = []
        filename = os.path.join(self.save_dir, "memory_agent_0.pkl")
        try:
            with open(filename, 'rb') as f:
                while len(all_transitions) < self.capacity:
                    compressed_data = f.read(self.max_chunk_size)
                    if not compressed_data:
                        break
                    decompressed_data = blosc.decompress(compressed_data)
                    loaded_memory = pickle.loads(decompressed_data)
                    all_transitions.extend(loaded_memory[:self.capacity - len(all_transitions)])
        except Exception as e:
            print(f"Error loading replay memory: {e}")

        self.memory = deque(all_transitions, maxlen=self.capacity)
        print(f"Loaded replay memory with {len(self.memory)} transitions.")


