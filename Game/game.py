import sys
import os
import pygame
import multiprocessing
import asyncio

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from Game.rendering import GameRenderer
from Integration.training_game import TrainingGame




def run_game_instance_process(queue, num_agents, screen_width, screen_height, verbose=False):
    async def run_game_instance():
        try:
            print("Initializing game instance...")
            game = TrainingGame(num_agents, screen_width, screen_height, queue, verbose)
            print("Game setup initialized.")
            await game.run_game()
        except Exception as e:
            print(f"Exception in game instance: {e}")
        finally:
            print("Game instance terminated")

    asyncio.run(run_game_instance())

def main():
    pygame.init()  # Initialize Pygame in the main process
    screen_width, screen_height = 800, 900
    num_agents = 16

    renderer = GameRenderer(screen_width, screen_height, num_agents)
    queues = renderer.get_queues()

    game_process = multiprocessing.Process(target=run_game_instance_process, args=(queues, num_agents, screen_width, screen_height, False))
    game_process.start()

    try:
        renderer.render()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Terminating game process...")
        game_process.terminate()
        game_process.join()
        pygame.quit()
        print("Pygame quit in main")

if __name__ == "__main__":
    main()

