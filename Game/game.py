import os
import sys

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

import time
import pygame

from ML.agent import Agent

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

    def run(self):
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False

            if self.game_state == "running":
                self.update()
            elif self.game_state == "game_over":
                self.show_game_over_screen()
                time.sleep(1.5)  # Wait for a few seconds
                self.restart_requested = True
                self.game_state = "waiting"

            elif self.game_state == "waiting":
                if self.restart_requested:
                    self.restart_game()
                    self.restart_requested = False

            # Render the graphics
            GraphicsHandler.render(self.screen, self.player, self.platform_manager.platforms, self.camera_offset_y)

            # Update the display
            pygame.display.flip()
            self.clock.tick(60)

    def restart_game(self):
        # Reset game state
        self.player = Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height)
        self.player.score = 0
        # self.player.highest_y = self.screen_height
        self.camera_offset_y = 0
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)  # Initialize new platform manager
        self.platform_manager.generate_bottom_platform()  # Generate the initial bottom platform
        self.game_state = "running"  # Set game state to running to start a new game

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


if __name__ == "__main__":
    game = Game()
    game.run()
    agent = Agent(input_size=10, output_size=3) # output = three discrete actions (move left, move right, jump)

    epsilon = 1.0  # Initial exploration rate
    epsilon_decay = 0.999  # Decay rate for epsilon
    min_epsilon = 0.01  # Minimum exploration rate

    for episode in range(3000):
        game.reset()  # Reset the game environment for each episode
        state = game.get_state()  # Get the initial state

        while not game.is_over():
            # Select an action using epsilon-greedy policy
            action = agent.select_action(state, epsilon)

            # Execute the action in the game environment
            next_state, reward, done = game.step(action)

            # Store the transition in the replay memory
            agent.memory.push((state, action, next_state, reward))

            # Move to the next state
            state = next_state

            # Optimize the model
            agent.optimize_model()

            # Decay exploration rate
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
