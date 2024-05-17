import os
import sys
import pygame

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)
from Integration import GameSetup, GameAIIntegrations, TrainingLoop

def main():
    pygame.init()  # Initialize all Pygame modules
    pygame.font.init()  # Specifically ensure the font module is initialized
    # clock = pygame.time.Clock()
    # clock.tick(60)  # Maintain the frame rate

    game_setup = GameSetup()  # Instantiate GameSetup
    ai_integrations = GameAIIntegrations(game_setup.agent, game_setup.replay_memory)  # Assume this is already defined somewhere
    training_loop = TrainingLoop(game_setup, ai_integrations)

    training_loop.run_game()

if __name__ == "__main__":
    main()
