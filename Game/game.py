import sys
import os
import pygame
import threading

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from Game.rendering import GameRenderer
from Integration.training_game import TrainingGame

def run_game_instance_thread(queue, num_agents, screen_width, screen_height, stop_event, verbose=False):
    try:
        print("Initializing game instance...")
        game = TrainingGame(num_agents, screen_width, screen_height, queue, stop_event, verbose)
        print("Game setup initialized.")
        game.run_game()
    except Exception as e:
        print(f"Exception in game instance: {e}")
    finally:
        game.cleanup()
        print("Game instance terminated")


def main():
    pygame.init()
    screen_width, screen_height = 800, 900
    num_agents = 2

    renderer = GameRenderer(screen_width, screen_height, num_agents)
    queues = renderer.get_queues()

    stop_event = threading.Event()

    game_thread = threading.Thread(target=run_game_instance_thread, args=(queues, num_agents, screen_width, screen_height, stop_event, False))
    game_thread.start()

    try:
        renderer.render()
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
