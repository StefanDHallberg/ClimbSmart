# General Game Settings
screen_width = 800
screen_height = 900
max_episode_duration = 30  # in seconds

# Agent Settings
num_agents = 1
input_channels = 1 # Grayscale image 
num_actions = 3
learning_rate = 0.001
gamma = 0.99  # Discount factor
batch_size = 32
target_update_frequency = 10  # Update target network every X episodes

# Replay Memory Settings
memory_capacity = 50000  # Capacity of the replay memory
save_increment_size = 1000  # Save X entries to memory

# Epsilon-Greedy Strategy Settings
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 0.999

# File Paths
replay_memory_save_interval = 3  # Save memory every X episodes
replay_memory_file_template = "memory_agent_{i}.pkl"  # Template for saving replay memory

# Debugging and Logging
verbose = False
