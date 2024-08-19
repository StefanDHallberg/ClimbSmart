# General Game Settings
screen_width = 800
screen_height = 900
max_episode_duration = 30  # in seconds

# Agent Settings
num_agents = 1
input_channels = 1 # Grayscale image 
num_actions = 3 # Move left, right, or jump
learning_rate = 0.001 # Learning rate for the optimizer (Adam)
gamma = 0.99  # Discount factor, lower value favors immediate rewards over future rewards (0 to 1)
batch_size = 32 # Number of samples to train on in a batch
target_update_frequency = 10000  # Update target network every X steps
# Replay Memory Settings
memory_capacity = 50000  # Capacity of the replay memory
save_increment_size = 1000  # Save X entries to memory

# Epsilon-Greedy Strategy Settings
epsilon_start = 1.0 # exploration rate at the start of training (1 = 100% random actions) 
epsilon_final = 0.01
epsilon_decay = 0.999

# File Paths
replay_memory_save_interval = 5  # Save memory every X episodes
replay_memory_file_template = "memory_agent_{i}.pkl"  # Template for saving replay memory

# Debugging and Logging
verbose = False
