import time
import numpy as np
import pygame
import torch
from ML.memory import ReplayMemory
from ML.agent import Agent
from Game.platforms import PlatformManager
from Game.player import Player
from Integration.game_ai_integrations import GameAIIntegrations
import config

class TrainingGame:
    def __init__(self, renderer, queues, stop_event):
        self.renderer = renderer  # Use the shared renderer instance
        self.queues = queues
        self.num_agents = config.num_agents
        self.verbose = config.verbose
        self.max_episode_duration = config.max_episode_duration
        self.episode = 1
        self.stop_event = stop_event
        self.save_interval = config.replay_memory_save_interval
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height

        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        
        self.players = [
            Player(
                self.screen_width // 2,
                self.screen_height - 20,
                self.screen_width,
                self.screen_height,
                self.platform_manager
            ) for _ in range(self.num_agents)
        ]
        
        # Initialize agents with parameters from config
        self.agents = [
            Agent(
                input_channels=config.input_channels,
                num_actions=config.num_actions,
                input_width=self.screen_width,
                input_height=self.screen_height,
                lr=config.learning_rate,
                gamma=config.gamma,
                batch_size=config.batch_size,
                epsilon_start=config.epsilon_start,
                epsilon_final=config.epsilon_final,
                epsilon_decay=config.epsilon_decay,
                verbose=self.verbose
            ) for _ in range(self.num_agents)
        ]

        self.ai_integrations = [
            GameAIIntegrations(
                agent,
                ReplayMemory(config.memory_capacity)
            ) for agent in self.agents
        ]

        self.state_tensor = torch.zeros((self.num_agents, config.input_channels, self.screen_width, self.screen_height), dtype=torch.float32)
        self.clock = pygame.time.Clock()

        if self.verbose:
            print("Initializing TrainingGame")
            print(f"Initialized {self.ai_integrations}")
            print(f"State tensor shape: {self.state_tensor.shape}") #This creates a tensor that can hold the state for each agent separately.

        # Load memory if exists
        for i, ai_integration in enumerate(self.ai_integrations):
            ai_integration.replay_memory.load_memory(f"memory_agent_{i}.pkl")

    def get_states(self, preprocessed_screen):
        with torch.no_grad():
            # Ensure the input is a NumPy array before converting it to a Tensor
            if isinstance(preprocessed_screen, np.ndarray):
                # Here, modify the function to handle multiple agents
                state_tensors = []
                for _ in range(self.num_agents):
                    # Convert the image to a PyTorch tensor and add a batch dimension
                    state_tensor = torch.from_numpy(preprocessed_screen).permute(2, 0, 1).unsqueeze(0).float()
                    state_tensors.append(state_tensor)
                # Stack tensors to create a batch
                batch_state_tensor = torch.cat(state_tensors, dim=0)
                if self.verbose:
                    print(f"Batch state tensor shape: {batch_state_tensor.shape}")  # Should show: [num_agents, 3, height, width]
                return batch_state_tensor
            else:
                raise TypeError("Expected preprocessed_screen to be a NumPy array.")



    def run_game(self):
        try:
            while not self.stop_event.is_set():
                if self.verbose:
                    print(f"Starting episode {self.episode}")
                total_reward = 0
                self.start_time = time.time()
                self.is_running = True

                while self.is_running and not self.stop_event.is_set() and time.time() - self.start_time <= self.max_episode_duration:
                    self.handle_events()

                    # Capture and preprocess screen
                    raw_screen = self.renderer.capture_screen()
                    preprocessed_screen = self.renderer.preprocess_image(raw_screen, self.screen_width, self.screen_height)

                    # Convert preprocessed image to tensor and check its shape
                    states = self.get_states(preprocessed_screen)
                   
                    # Use states as input for your neural network
                    total_rewards = self.update_agents(self.episode, states, preprocessed_screen)
                    total_reward += sum(total_rewards)
                    self.update_platforms()
                    self.update_display(self.episode, total_reward)

                    self.clock.tick(60)  # Limit frame rate to 60 FPS

                if not self.stop_event.is_set():
                    for ai_integration in self.ai_integrations:
                        if ai_integration:
                            ai_integration.agent.optimize_model()

                    if self.verbose:
                        print(f"Episode {self.episode} completed with total reward: {total_reward}")

                self.cleanup()
                self.episode += 1
                self.reset_game_state()

                # Save replay memory at intervals
                if self.episode % self.save_interval == 0:
                    for i, ai_integration in enumerate(self.ai_integrations):
                        ai_integration.replay_memory.save_memory(config.replay_memory_file_template.format(i=i))

        except KeyboardInterrupt:
            print("Training loop interrupted by user")
            self.stop_event.set()

        finally:
            self.cleanup()

    def handle_events(self):
        """ Handle basic pygame events like quitting the game. """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False


    def calculate_reward(self, agent_id, action, on_platform):
        reward = 0
        player = self.players[agent_id]

        if on_platform:
            reward += 1
        if action == 2 and player.is_jumping:
            reward += 0.5
        elif action in [0, 1]:  # Moving left or right
            reward += 0.1

        # Additional reward for reaching higher scores
        milestones = [50, 100, 150]
        reward_increment = 50
        for milestone in milestones:
            if player.score >= milestone and not player.reached_milestones.get(milestone, False):
                reward += reward_increment
                player.reached_milestones[milestone] = True
                print(f"Agent {agent_id} reached score {milestone}, additional reward: {reward_increment}")

    def update_agents(self, episode, states, preprocessed_screen):
        total_rewards = []
        for agent_id, ai_integration in enumerate(self.ai_integrations):
            if self.verbose:
                print(f"Agent {agent_id} state: {states[agent_id].shape}")

            # Get the action from the AI integration
            state = states[agent_id].unsqueeze(0)  # Add batch dimension if necessary
            action = ai_integration.select_action_and_update(state)
            if isinstance(action, torch.Tensor):
                action = action.item()  # Convert torch.Tensor to a Python int if necessary

            if self.verbose:
                print(f"Selected action for agent {agent_id}: {action}")

            # Apply the selected action to update player states
            self.update_players(agent_id, action)

            # Handle collisions after the player has moved
            self.players[agent_id].handle_collision(self.platform_manager.platforms)

            # Check if the player is on a platform after moving and handling collisions
            on_platform = self.check_on_platform(agent_id)

            # Calculate the reward based on the new state after collision handling
            reward = self.calculate_reward(agent_id, action, on_platform)

            # Get the next state after all updates
            next_state = self.get_states(preprocessed_screen)[agent_id]

            # Add the transition to the replay memory
            done = False  # Update this based on your game's end condition

            # Ensure the transition is stored as NumPy arrays
            ai_integration.replay_memory.push(
                states[agent_id].numpy(),   # Convert tensor to NumPy array
                action,
                reward,
                next_state.numpy(),  # Convert tensor to NumPy array
                done
            )

            # Log the reward
            total_rewards.append(reward if reward is not None else 0)  # Ensure reward is numeric
        return total_rewards






    def cleanup(self):
        try:
            if self.is_running and not self.stop_event.is_set():
                self.flush_queues()
            self.is_running = False
        except Exception as e:
            print(f"Exception during cleanup: {e}")
        finally:
            print("Clean up in TrainingGame")

    def initialize_platforms(self):
        self.platform_manager.generate_bottom_platform()
        self.platform_manager.generate_additional_platforms()

    def initialize_players(self):
        for player in self.players:
            player.rect.x = self.screen_width // 2
            player.rect.y = self.screen_height - 100
            player.vel_y = 0
            player.is_jumping = False

    def reset_game(self):
        self.reset_players()
        self.reset_platform_manager()
        if self.verbose:
            print("Game reset complete.")

    def reset_players(self):
        for player in self.players:
            player.reset()
        if self.verbose:
            print("Players reset")

    def reset_platform_manager(self):
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        if self.verbose:
            print("Platform manager reset")

    def update_players(self, agent_id, action):
        keys = {pygame.K_a: False, pygame.K_d: False, pygame.K_w: False, pygame.K_UP: False}
        action_map = {0: pygame.K_a, 1: pygame.K_d, 2: pygame.K_w}

        action = action.item() if isinstance(action, torch.Tensor) else action
        if action in action_map:
            keys[action_map[action]] = True

        if self.verbose:
            print(f"Action in update_players: {action}, type: {type(action)}")
            print(f"Keys before update: {keys}")

        self.players[agent_id].update(keys, self.platform_manager.platforms)

        if self.verbose:
            print(f"Keys after update: {keys}")

    def check_on_platform(self, agent_id):
        player = self.players[agent_id]
        for platform in self.platform_manager.platforms:
            if platform.on_platform and player.rect.colliderect(platform.rect):
                # print(f"Player {player.rect} on platform {platform.rect}")  # Debugging print
                return True
        return False

    def update_platforms(self):
        for player in self.players:
            self.platform_manager.update(player)

    def get_render_data(self, episode, total_reward):
        players_data = []
        platforms_data = []

        for p in self.players:
            players_data.append({'rect': p.rect, 'image': pygame.image.tostring(p.image, 'RGBA')})

        for p in self.platform_manager.platforms:
            platforms_data.append({'rect': p.rect, 'image': pygame.image.tostring(p.image, 'RGBA')})

        data = {
            'players': players_data,
            'platforms': platforms_data,
            'score': sum(player.score for player in self.players),
            'episode': episode,
            'total_reward': total_reward
        }
        return data

    def terminate_game_loop(self):
        self.stop_event.set()

    def update_display(self, episode, total_reward):
        if self.stop_event.is_set():
            return
        data = self.get_render_data(episode, total_reward)
        if not data:
            return
        for queue in self.queues:
            if not queue.full():
                queue.put_nowait(data)

    def flush_queues(self):
        for queue in self.queues:
            while not queue.empty():
                queue.get_nowait()

    def reset_game_state(self):
        if self.verbose:
            print("Resetting game state...")
        self.update_display(self.episode, 0)
        self.reset_game()
        self.ai_integrations = [GameAIIntegrations(agent, ReplayMemory(config.memory_capacity)) for agent in self.agents]
        self.update_display(self.episode, 0)
        if self.verbose:
            print("Game state reset complete.")
