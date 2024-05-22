import sys
import os
import pygame
import multiprocessing

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from Integration import GameSetup, GameAIIntegrations, TrainingLoop
from Game.graphics import GraphicsHandler

def run_game_instance(queues, num_agents):
    try:
        pygame.init()
        pygame.font.init()
        print("Pygame initialized in run_game_instance")

        training_loop = TrainingLoop(num_agents, queues, verbose=False)
        training_loop.run_game()
    except Exception as e:
        print(f"Exception in game instance: {e}")
    finally:
        pygame.quit()
        print("Pygame quit in run_game_instance")

def main():
    pygame.init()
    pygame.font.init()
    print("Pygame initialized in main")

    num_agents = 8  # Number of agents
    screen_width, screen_height = 800, 900

    # Initialize the main screen
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("ClimbSmart Multi-Agent")

    queues = [multiprocessing.Queue() for _ in range(num_agents)]
    clock = pygame.time.Clock()

    # Run a single game instance
    game_process = multiprocessing.Process(target=run_game_instance, args=(queues, num_agents))
    game_process.start()

    try:
        while game_process.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_process.terminate()
                    game_process.join()
                    pygame.quit()
                    print("Pygame quit in main")
                    sys.exit()

            for queue in queues:
                if not queue.empty():
                    render_data = queue.get()
                    GraphicsHandler.render(screen, render_data)

            pygame.display.flip()  # Ensure display is updated in the main process
            clock.tick(60)  # Limit the frame rate to 60 FPS
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Terminating game process...")
        game_process.terminate()
        game_process.join()
        pygame.quit()
        print("Pygame quit in main")


if __name__ == "__main__":  # python -m tensorboard.main --logdir=runs in terminal to see the tensorboard graph on http://localhost:6006/
    main()

# venv\Scripts\Activate.ps1 for activating the virtual environment in powershell