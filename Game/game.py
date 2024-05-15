import os
import sys

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)
import pygame
import torch
import time
from ML.agent import Agent
from ML.memory import ReplayMemory
from graphics import GraphicsHandler
from player import Player
from platforms import PlatformManager

from torch.utils.tensorboard import SummaryWriter # Terminal command is ###    tensorboard --logdir=runs   ### to view tensorboard - http://localhost:6006/  to see the plots and histograms of the training data.

class Game:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ClimbSmart")
        self.clock = pygame.time.Clock()
        self.is_running = True
        self.camera_offset_y = 0  # Initialize camera offset

        self.max_episode_duration = 120  # Maximum duration in seconds
        self.start_time = None

        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        self.player = Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height)
        self.agent = Agent(input_size=10, output_size=3)  # output = three discrete actions (move left, move right, jump)
        self.replay_memory = ReplayMemory(10000)

        self.writer = SummaryWriter('runs/ClimbSmart') # Tensorboard writer for logging metrics and visualizations.

    def main(self):
        episode = 1
        while self.is_running:
            self.run_episode(episode)
            episode += 1

    def run_episode(self, episode):
        self.restart_game()
        total_reward = 0
        self.start_time = time.time()

        while self.is_running:
            self.handle_events()
            state = self.get_state()
            action = self.agent.select_action(state)  # Use agent's internal method to choose action
            next_state, reward, done = self.step(action)
            total_reward += reward
            self.replay_memory.push((state, action, reward, next_state, done))

            if len(self.replay_memory) > self.agent.batch_size:
                experiences = self.replay_memory.sample(self.agent.batch_size)
                self.agent.optimize_model(experiences)

            self.agent.update_epsilon()  # Update epsilon managed within Agent

            # Log total reward per episode using TensorBoard
            self.writer.add_scalar('Total Reward', total_reward, episode)

            self.update_display(episode, total_reward)

            if done:
                print(f"Episode: {episode} ended. Player game over.")
                break

            if time.time() - self.start_time >= self.max_episode_duration:
                print(f"Episode {episode} terminated due to maximum duration ({self.max_episode_duration} seconds).")
                break

            if total_reward <= -50:
                print(f"Episode {episode} terminated due to low total reward: {total_reward}")
                break

        # Ensure all events are written to disk
        self.writer.flush()
        # Save replay memory
        self.replay_memory.save_memory('replay_memory.pkl')
        print(f"Episode: {episode}, Score: {self.player.score}, High Score: {self.player.high_score}, Total Reward: {total_reward}")



    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False

    def get_state(self):
        state = [self.player.rect.centerx, self.player.rect.centery, self.player.vel_y, self.player.is_jumping]
        platforms = self.platform_manager.platforms.sprites()

        # Ensure we get exactly 3 platforms or fill with dummy values
        for i in range(3):
            if i < len(platforms):
                platform = platforms[i]
                state.extend([platform.rect.centerx, platform.rect.centery])
            else:
                state.extend([-1, -1])  # Dummy values for missing platforms

        # Ensure state has exactly 10 elements
        if len(state) != 10:
            raise ValueError(f"State does not have 10 elements: {state}")

        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        # print(f"State tensor shape: {state_tensor.shape}")  # Debug print
        return state_tensor

    def step(self, action):
        keys = {pygame.K_a: 0, pygame.K_d: 0, pygame.K_w: 0, pygame.K_UP: 0}
        if action == 0:
            keys[pygame.K_a] = 1
        elif action == 1:
            keys[pygame.K_d] = 1
        elif action == 2:
            keys[pygame.K_w] = 1
            keys[pygame.K_UP] = 1

        self.player.update(keys, self.platform_manager.platforms)
        on_platform = self.player.is_on_platform(self.platform_manager.platforms)
        next_state = self.get_state()
        reward = self.calculate_reward(action, on_platform)
        done = self.player.is_game_over()
        
        return next_state, reward, done

    def calculate_reward(self, action, on_platform):
        reward = 0
        if action == 2:  # Jump action
            if on_platform:  # Player is on a platform
                reward += 1  # Reward for successful platform landing
                # Check if this action leads to a new high score
                if self.player.score > self.player.high_score:
                    reward += 100  # Add a significant reward for achieving a new high score
                    self.player.high_score = self.player.score  # Update the high score
                    print(f"New high score achieved: {self.player.high_score}")
            else:
                reward -= 1  # Penalty for jumping without being on a platform
        elif action == 0 or action == 1:  # Move left or move right actions
            reward += 0.1  # Slight reward for moving to encourage exploration
        else:  # No action taken
            reward -= 0.05  # Penalty for staying still to discourage inaction

        return reward
    


    def update_display(self, episode, total_reward):
        self.platform_manager.update(self.player)
        self.player.update_score()
        # Calculate camera offset
        camera_offset_y = self.screen_height // 2 - self.player.rect.centery
        # pygame.display.flip()
        self.clock.tick(60)
        GraphicsHandler.render(self.screen, self.player, self.platform_manager.platforms, camera_offset_y, episode, total_reward)

    def restart_game(self):
        self.player.reset()
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)  # Initialize new platform manager
        self.platform_manager.generate_bottom_platform()
        self.player.highest_y = self.player.rect.bottom 

if __name__ == "__main__":
    game = Game()
    game.main()
    game.writer.close()  # Close the TensorBoard writer
