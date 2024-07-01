import pygame
import queue
from Game.graphics import GraphicsHandler
from Integration.utilities import pygame_lock  # Import the lock from utilities

class GameRenderer:
    def __init__(self, screen_width, screen_height, num_agents):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("ClimbSmart Multi-Agent")
        self.clock = pygame.time.Clock()
        self.queues = [queue.Queue() for _ in range(num_agents)]

    def render(self):
        running = True
        while running:
            with pygame_lock:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                for q in self.queues:
                    if not q.empty():
                        render_data = q.get()
                        self._render_frame(render_data)

                pygame.display.flip()
                self.clock.tick(60)

        pygame.quit()  # Ensure pygame quits properly

    def _render_frame(self, render_data):
        self.screen.fill((0, 0, 0))  # Clear screen to black before rendering
        GraphicsHandler.render(self.screen, render_data)

    def get_queues(self):
        return self.queues
