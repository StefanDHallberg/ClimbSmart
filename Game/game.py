import sys
import os
import pygame
import asyncio

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from Game.rendering import GameRenderer
from Integration.training_game import TrainingGame

async def run_game_instance(queue, num_agents, screen_width, screen_height, verbose=False):
    try:
        print("Initializing game instance...")
        game = TrainingGame(num_agents, screen_width, screen_height, queue, verbose)
        print("Game setup initialized.")
        await game.run_game()
    except asyncio.CancelledError:
        print("Game task cancelled")
    except Exception as e:
        print(f"Exception in game instance: {e}")
    finally:
        print("Game instance terminated")

async def main():
    pygame.init()  # Initialize Pygame in the main process
    screen_width, screen_height = 800, 900
    num_agents = 16

    renderer = GameRenderer(screen_width, screen_height, num_agents)
    queue = renderer.get_queue()

    game_task = asyncio.create_task(run_game_instance(queue, num_agents, screen_width, screen_height, False))

    try:
        await renderer.render()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        game_task.cancel()  # Cancel the game task
        await game_task  # Wait for the game task to properly handle cancellation
    finally:
        pygame.quit()
        print("Pygame quit in main")

if __name__ == "__main__":
    asyncio.run(main())
