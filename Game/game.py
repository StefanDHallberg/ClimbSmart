import multiprocessing
import sys
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from Integration import GameSetup, GameAIIntegrations, TrainingLoop

def run_game_instance():
    # Avoid initializing pygame in worker processes
    game_setup = GameSetup()
    ai_integrations = GameAIIntegrations(game_setup.agent, game_setup.replay_memory)
    training_loop = TrainingLoop(game_setup, ai_integrations)
    training_loop.run_game()

def main():
    num_processes = 4  # Number of game instances to run in parallel

    processes = []
    
    # Create and start processes for each game instance
    for _ in range(num_processes):
        p = multiprocessing.Process(target=run_game_instance)
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
