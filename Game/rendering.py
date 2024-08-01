import threading
import cv2
import numpy as np
import pygame
import queue
from Game.graphics import GraphicsHandler
# # Shared lock for pygame access
pygame_lock = threading.Lock()

class GameRenderer:
    def __init__(self, screen_width, screen_height, num_agents):
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
                self.clock.tick(60)  # Cap the frame rate at 60 FPS

        pygame.quit()

    def _render_frame(self, render_data):
        self.screen.fill((0, 0, 0))
        GraphicsHandler.render(self.screen, render_data)

    def get_queues(self):
        return self.queues
    
    def capture_screen(self):
        """Capture the current game screen as a numpy array."""
        # Capture the screen image from self.screen
        screen_image = pygame.surfarray.array3d(self.screen)  # Use self.screen
        # Convert the shape from (width, height, channels) to (height, width, channels) for OpenCV
        screen_image = np.transpose(screen_image, (1, 0, 2))  # Transpose to (height, width, channels)
        # print(f"Captured screen image shape: {screen_image.shape}")  # debug
        return screen_image

    def preprocess_image(self, image, width, height):
        """Resize and normalize the image."""
        if image.size == 0:
            raise ValueError("Captured image is empty. Ensure the screen is being captured correctly.")

        # Resize the image to match the network's input size
        # print(f"Resizing image from shape {image.shape} to ({height}, {width})")  # Debugging statement
        resized_image = cv2.resize(image, (width, height))  # OpenCV expects width, height in this order
        # Normalize the pixel values to [0, 1]
        normalized_image = resized_image / 255.0
        return normalized_image
