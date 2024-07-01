import threading
import pygame
import queue
from Game.graphics import GraphicsHandler
from Integration.utilities import pygame_lock

class GameRenderer:
    def __init__(self, screen_width, screen_height, num_agents):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("ClimbSmart Multi-Agent")
        self.clock = pygame.time.Clock()
        self.queues = [queue.Queue() for _ in range(num_agents)]
        self.stop_event = threading.Event()

    def render(self):
        running = True
        while running and not self.stop_event.is_set():
            with pygame_lock:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                for q in self.queues:
                    if not q.empty():
                        render_data = q.get()
                        self._render_frame(render_data)

                pygame.display.flip()
                self.clock.tick(60)  # Cap the frame rate to 60 FPS

        pygame.quit()

    def _render_frame(self, render_data):
        self.screen.fill((0, 0, 0))
        GraphicsHandler.render(self.screen, render_data)

    def get_queues(self):
        return self.queues

    def stop(self):
        self.stop_event.set()
