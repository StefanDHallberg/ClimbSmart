import pygame
import asyncio
from Game.graphics import GraphicsHandler

class GameRenderer:
    def __init__(self, screen_width, screen_height, num_agents):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("ClimbSmart Multi-Agent")
        self.clock = pygame.time.Clock()
        self.queue = asyncio.Queue()

    async def render(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not self.queue.empty():
                render_data = await self.queue.get()
                print("Render data received from queue")  # Debugging
                self._render_frame(render_data)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    #def _render_frame(self, render_data):
    #    self.screen.fill((0, 0, 0))  # Clear the screen with black

#        for player_data in render_data['players']:
 #           rect = pygame.Rect(player_data['rect'])
  #          image = pygame.image.fromstring(player_data['image'], rect.size, 'RGBA')
   #         self.screen.blit(image, rect.topleft)
    #        print(f"Rendered player at {rect.topleft}")  # Debugging

     #   for platform_data in render_data['platforms']:
      #      rect = pygame.Rect(platform_data['rect'])
       #     image = pygame.image.fromstring(platform_data['image'], rect.size, 'RGBA')
        #    self.screen.blit(image, rect.topleft)
         #   print(f"Rendered platform at {rect.topleft}")  # Debugging

    def get_queue(self):
        return self.queue


    def _render_frame(self, render_data):
        GraphicsHandler.render(self.screen, render_data)