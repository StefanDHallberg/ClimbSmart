import sys
import os
import pygame
import threading
import time
import psutil
import cProfile
import pstats
from io import StringIO

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Append the parent directory of the script directory to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from Game.rendering import GameRenderer
from Integration.training_game import TrainingGame

def monitor_resources(stop_event):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_usage = process.cpu_percent(interval=1)
        memory_info = process.memory_info()
        print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
        time.sleep(1)

def run_game_instance_thread(queue, num_agents, screen_width, screen_height, stop_event, verbose=False):
    try:
        print("Initializing game instance...")
        game = TrainingGame(num_agents, screen_width, screen_height, queue, stop_event, verbose)
        print("Game setup initialized.")
        game.run_game()
        pygame.display.flip()
    except Exception as e:
        print(f"Exception in game instance: {e}")
    finally:
        print("Game instance terminated")

def main():
    pygame.init()
    screen_width, screen_height = 800, 900
    num_agents = 2

    stop_event = threading.Event()

    renderer = GameRenderer(screen_width, screen_height, num_agents)
    queues = renderer.get_queues()

    game_thread = threading.Thread(target=run_game_instance_thread, args=(queues, num_agents, screen_width, screen_height, stop_event, False))
    game_thread.start()

    resource_monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event,))
    resource_monitor_thread.start()

    try:
        renderer.render()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Terminating game process...")
        stop_event.set()
        renderer.stop()
        game_thread.join()
        resource_monitor_thread.join()
        pygame.quit()
        print("Pygame quit in main")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        main()
    except Exception as e:
        print(f"Exception in main: {e}")
    finally:
        profiler.disable()
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open('profile_stats.txt', 'w') as f:
            f.write(s.getvalue())
        print("Profiling results saved to profile_stats.txt")
