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

def run_game_instance_thread(renderer, queue, num_agents, screen_width, screen_height, stop_event, verbose=False):
    game = None
    try:
        print("Initializing game instance...")
        game = TrainingGame(renderer, queue, num_agents, screen_width, screen_height, stop_event, verbose)
        print("Game setup initialized.")
        game.run_game()
    except Exception as e:
        print(f"Exception in game instance: {e}")
    

def main():
    pygame.init()  # Initialize Pygame
    screen_width, screen_height = 800, 900
    num_agents = 2

    # Initialize GameRenderer once and share across threads
    renderer = GameRenderer(screen_width, screen_height, num_agents)
    stop_event = threading.Event()
    queues = renderer.get_queues()



    # Pass renderer to the thread
    game_thread = threading.Thread(target=run_game_instance_thread, args=(renderer, queues, num_agents, screen_width, screen_height, stop_event, False))
    game_thread.start()

    try:
        renderer.render()  # Continue rendering in the main thread
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Terminating game process...")
        stop_event.set()
        game_thread.join()
        pygame.quit()
        print("Pygame quit in main")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Exception in main: {e}")
