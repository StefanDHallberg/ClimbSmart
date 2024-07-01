import sys
import os
import pygame
import threading
import time
import asyncio
import psutil
from memory_profiler import profile

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from Game.rendering import GameRenderer
from Integration.training_game import TrainingGame

def monitor_resources():
    process = psutil.Process(os.getpid())
    while True:
        cpu_usage = process.cpu_percent(interval=1)
        memory_info = process.memory_info()
        print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
        time.sleep(1)

def run_game_instance_thread(queue, num_agents, screen_width, screen_height, verbose=False):
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
    pygame.init()
    screen_width, screen_height = 800, 900
    num_agents = 1  # Adjust number of agents as needed

    renderer = GameRenderer(screen_width, screen_height, num_agents)
    queues = renderer.get_queues()

    game_thread = threading.Thread(target=run_game_instance_thread, args=(queues, num_agents, screen_width, screen_height, False))
    game_thread.start()

    resource_monitor_thread = threading.Thread(target=monitor_resources)
    resource_monitor_thread.start()

    try:
        renderer.render()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Terminating game process...")
        game_thread.join()
        resource_monitor_thread.join()
        pygame.quit()
        print("Pygame quit in main")

if __name__ == "__main__":
    main()



# import cProfile
# import io
# import pstats
# import sys
# import os
# import pygame
# import threading
# import asyncio
# # from memory_profiler import profile  # Import memory profiler

# # Get the directory of the script
# script_dir = os.path.dirname(os.path.realpath(__file__))
# # Append the parent directory of the script directory to the system path
# parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
# sys.path.append(parent_dir)

# from Game.rendering import GameRenderer
# from Integration.training_game import TrainingGame

# def run_game_instance_thread(queue, num_agents, screen_width, screen_height, verbose=False):
#     async def run_game_instance():
#         try:
#             print("Initializing game instance...")
#             game = TrainingGame(num_agents, screen_width, screen_height, queue, verbose)
#             print("Game setup initialized.")
#             await game.run_game()
#         except Exception as e:
#             print(f"Exception in game instance: {e}")
#         finally:
#             print("Game instance terminated")

#     asyncio.run(run_game_instance())

# #@profile  # Add the profiler decorator
# def main():
#     profiler = cProfile.Profile()
#     profiler.enable()
    
#     # Your existing main function logic here
#     pygame.init()  # Initialize Pygame in the main process
#     screen_width, screen_height = 800, 900
#     num_agents = 1  # Testing with a single agent

#     renderer = GameRenderer(screen_width, screen_height, num_agents)
#     queues = renderer.get_queues()

#     game_thread = threading.Thread(target=run_game_instance_thread, args=(queues, num_agents, screen_width, screen_height, True))
#     game_thread.start()

#     try:
#         renderer.render()
#     except KeyboardInterrupt:
#         print("Interrupted by user.")
#     finally:
#         print("Terminating game process...")
#         game_thread.join()
#         pygame.quit()
#         print("Pygame quit in main")
    
#     profiler.disable()
#     s = io.StringIO()
#     sortby = 'cumulative'
#     ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
#     ps.print_stats()
#     with open("profile_output.txt", "w") as f:
#         f.write(s.getvalue())

# if __name__ == "__main__":
#     main()