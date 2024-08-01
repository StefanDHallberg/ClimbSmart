import sys
import os
import pygame
import threading

# Set up paths
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from Game.rendering import GameRenderer
from Integration.training_game import TrainingGame
import config

def run_game_instance_thread(renderer, queues, stop_event, verbose=config.verbose):
    game = None
    try:
        print("Initializing game instance...")
        game = TrainingGame(renderer, queues, stop_event)
        print("Game setup initialized.")
        game.run_game()
    except Exception as e:
        print(f"Exception in game instance: {e}")
    finally:
        print("Exiting game instance thread.")

def main():
    pygame.init()  # Initialize Pygame

    # Use config values
    renderer = GameRenderer(config.screen_width, config.screen_height, config.num_agents)
    stop_event = threading.Event()
    queues = renderer.get_queues()

    # Start game thread
    game_thread = threading.Thread(
        target=run_game_instance_thread,
        args=(renderer, queues, stop_event, config.verbose)
    )
    game_thread.start()

    try:
        renderer.render()  # Continue rendering in the main thread
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Terminating game process...")
        stop_event.set()  # Signal the game thread to stop
        game_thread.join()  # Wait for the game thread to finish
        pygame.quit()
        print("Pygame quit in main")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Exception in main: {e}")
