import pygame
import multiprocessing
from Game.graphics import GraphicsHandler

class GameRenderer:
    def __init__(self, screen_width, screen_height, num_agents):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("ClimbSmart Multi-Agent")
        self.clock = pygame.time.Clock()
        self.queues = [multiprocessing.Queue() for _ in range(num_agents)]

    def render(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for queue in self.queues:
                if not queue.empty():
                    render_data = queue.get()
                    self._render_frame(render_data)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def _render_frame(self, render_data):
        GraphicsHandler.render(self.screen, render_data)

    def get_queues(self):
        return self.queues



