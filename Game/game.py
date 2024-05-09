import os
import sys

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)
import torch
import time
import pygame

from ML.agent import Agent
from ML.memory import ReplayMemory

from graphics import GraphicsHandler
from player import Player
from platforms import PlatformManager

class Game:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 900
        self.camera_offset_y = 0
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ClimbSmart")
        self.clock = pygame.time.Clock()
        self.is_running = True

        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        self.player = Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height)
        self.score = 0  # Initialize score
        self.game_state = "running"  # Initialize game state as running
        self.restart_requested = False  # Flag to indicate if a restart is requested
        self.prev_x = None
        self.prev_y = None
        self.prev_action = None
        self.step_counter = 0
        self.max_steps = 1000  # Maximum number of steps before the game is reset


        # Initialize the agent
        self.agent = Agent(input_size=10, output_size=3) # output = three discrete actions (move left, move right, jump)

        # Initialize exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01

        # Initialize replay memory
        self.replay_memory = ReplayMemory(capacity=10000)
        self.replay_memory.load_memory('replay_memory.pkl')

    def end_episode(self):
        self.restart_game()
    
    def step(self, action):
        done = False  # Initialize done to False
        # Perform the action in the game environment
        self.step_counter += 1


        on_platform = False
        if action == 0:  # Move left
            self.player.rect.centerx -= self.player.vel
        elif action == 1:  # Move right
            self.player.rect.centerx += self.player.vel
        elif action == 2:  # Jump
            for platform in self.platform_manager.platforms:
                if self.player.rect.colliderect(platform.rect) and self.player.vel_y >= 0:
                    on_platform = True
                    break

            if on_platform:
                self.player.vel_y = self.player.jump_vel
                self.player.is_jumping = True

        if self.step_counter >= self.max_steps:
            self.end_episode()
            self.step_counter = 0
            done = True

        self.update()

        next_state = self.get_state()

        done = done or (self.game_state == "game_over")

        # Check if prev_x and prev_y are None and, if so, set them to the current position
        if self.prev_x is None and self.prev_y is None:
            self.prev_x = self.player.rect.centerx
            self.prev_y = self.player.rect.centery

        # Modify the reward function to penalize the agent for choosing the same action repeatedly
        reward = self.player.score
        if action == 2 and on_platform:  # If the action was a successful jump
            reward += 10  # Give additional reward for successful jumps
        elif self.player.rect.centerx == self.prev_x and self.player.rect.centery == self.prev_y:
            reward -= 0.1  # Small negative reward for staying in the same state
        if action == self.prev_action:
            reward -= 0.1  # Small negative reward for choosing the same action repeatedly

        # Update the previous position and action
        self.prev_x = self.player.rect.centerx
        self.prev_y = self.player.rect.centery
        self.prev_action = action

        return next_state, reward, done
    
    def get_state(self):
        # Get the platforms as a list
        platforms = self.platform_manager.platforms.sprites()

        # Check if there are at least three platforms
        if len(platforms) < 3:
            # If not, add None values to the platforms list until it has three elements
            platforms += [None] * (3 - len(platforms))

        # Get the state of the game
        state = [self.player.rect.centerx, self.player.rect.centery, self.player.vel_y, self.player.is_jumping]

        # Add the platforms to the state
        for platform in platforms:
            if platform is not None:
                state.extend([platform.rect.centerx, platform.rect.centery])
            else:
                state.extend([None, None])

        # Ensure that the state list has exactly 10 elements
        state = state[:10]

        # Convert the state to a PyTorch Tensor, ignoring None values
        state = torch.tensor([value for value in state if value is not None], dtype=torch.float32)

        return state


    def update(self):
        keys = pygame.key.get_pressed()
        # Update player movement and gravity
        self.player.update(keys, self.platform_manager.platforms)
        # Update the platform manager
        self.platform_manager.update(self.player)

        # Monitor the player's vertical position
        player_y = self.player.rect.bottom
        # Calculate the camera offset to always follow the player upwards
        self.camera_offset_y = self.screen_height // 2 - player_y

        # Update the score
        self.player.update_score()
        
        # Check if the player falls off the platforms
        if self.player.rect.top > self.screen_height:
            self.game_state = "game_over"

    def run(self, agent, epsilon, replay_memory, training=False, episode=None, total_reward=None):  # Add 'training' parameter
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                    break  # Break out of the event loop
            
            if self.game_state == "running":
                # Select an action using epsilon-greedy policy
                state = self.get_state()
                action = self.agent.select_action(state, self.epsilon)
                next_state, reward, done = self.step(action)

                # Store the transition in the replay memory
                self.replay_memory.push((state, action, next_state, reward))

                # Optimize the model
                self.agent.optimize_model()

                # Decay exploration rate
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

                # Update the display
                pygame.display.flip()
                self.clock.tick(60)
                # Render the graphics
                GraphicsHandler.render(self.screen, self.player, self.platform_manager.platforms, self.camera_offset_y, episode, total_reward)


                #print(f"Selected action: {action}")  # Debug print

            elif self.game_state == "game_over":
                self.show_game_over_screen()
                time.sleep(1.5)  # Wait for a few seconds
                self.restart_requested = True
                self.game_state = "waiting"

            elif self.game_state == "waiting":
                if self.restart_requested:
                    self.restart_game()
                    self.restart_requested = False

            # Save the replay memory at the end of the episode
        if not self.is_running:
            self.replay_memory.save_memory('replay_memory.pkl')


    def restart_game(self):
        # Reset game state
        self.game_state = "running"  # Set game state to running to start a new game
        self.player = Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height)
        self.player.score = 0
        self.step_counter = 0
        # self.player.highest_y = self.screen_height
        self.prev_x = None
        self.prev_y = None
        self.prev_action = None
        self.camera_offset_y = 0
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)  # Initialize new platform manager
        self.platform_manager.generate_bottom_platform()  # Generate the initial bottom platform
        

    def show_game_over_screen(self):
        # Render the game over screen with the final score
        font = pygame.font.Font(None, 36)
        game_over_text = font.render("Game Over", True, (255, 0, 0))
        score_text = font.render(f"Score: {self.player.score}", True, (0, 0, 0))  # Use player's score
        
        # Calculate the positions for rendering text
        game_over_rect = game_over_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        score_rect = score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Blit the text onto the screen
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)

        # Update the display
        pygame.display.flip()

# Adds a list to store the rewards for each episode
episode_rewards = []


if __name__ == "__main__":
    # Initialize the game and agent
    game = Game()
    agent = Agent(input_size=10, output_size=3) # output = three discrete actions (move left, move right, jump)

    # Initialize exploration parameters
    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.01

    # Load replay memory if available
    replay_memory = ReplayMemory(capacity=10000)
    replay_memory.load_memory('replay_memory.pkl')

    # Training loop
    for episode in range(3000):
        game.end_episode()
        total_reward = 0  # Initialize the total reward for this episode

        done = False
        while not done:
            state = game.get_state()  # Get the current state of the game
            action = agent.select_action(state, epsilon)  # Let the agent choose an action based on the current state and exploration rate
            next_state, reward, done = game.step(action)  # Perform the action in the game environment
            total_reward += reward  # Add the reward to the total reward

            # Store the experience in the replay memory
            replay_memory.push((state, action, reward, next_state, done))

            # Call the run method to update the game screen
            game.run(agent, epsilon, replay_memory, training=False, episode=episode, total_reward=total_reward)

            # Train the agent with a batch of experiences from the replay memory
            if len(replay_memory) > agent.batch_size:
                experiences = replay_memory.sample(agent.batch_size)
                agent.optimize_model()

        # Add the total reward for this episode to the list of episode rewards
        episode_rewards.append(total_reward)

        # Print the episode number and the total reward
        print(f"Episode: {episode + 1}, Total reward: {total_reward}")

        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Save replay memory at the end of training
        replay_memory.save_memory('replay_memory.pkl')