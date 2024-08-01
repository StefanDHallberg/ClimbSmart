import cv2
import numpy as np
import pygame
import queue
from Game.graphics import GraphicsHandler

class GameRenderer:
    def __init__(self, screen_width, screen_height, num_agents):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("ClimbSmart Multi-Agent")
        self.clock = pygame.time.Clock()
        self.queues = [queue.Queue() for _ in range(num_agents)]

    def render(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for q in self.queues:
                if not q.empty():
                    render_data = q.get()
                    self._render_frame(render_data)

                pygame.display.flip()
                self.clock.tick(60)  # Cap the frame rate at 60 FPS

        pygame.quit()

    def _render_frame(self, render_data):
        self.screen.fill((0, 0, 0))
        GraphicsHandler.render(self.screen, render_data)

    def get_queues(self):
        return self.queues
    
    def capture_screen(screen):
        """Capture the current game screen as a numpy array."""
        # Capture the screen image
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        # Transpose the image to have the color channel as the first dimension
        screen_image = np.transpose(screen_image, (2, 0, 1))
        return screen_image

    def preprocess_image(image, width, height):
        """Resize and normalize the image."""
        # Resize the image to match the network's input size
        resized_image = cv2.resize(image, (width, height))
        # Normalize the pixel values to [0, 1]
        normalized_image = resized_image / 255.0
        return normalized_image