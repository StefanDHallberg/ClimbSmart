import pygame
from player import Player
from platform import Platform

class Game:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ClimbSmart")
        self.clock = pygame.time.Clock()
        self.is_running = True

        # Load player and platform images
        self.player_image = pygame.image.load("./Assets/Tiles/Characters/tile_0000.png")
        self.platform_image = pygame.image.load("./Assets/Tiles/tile_0000.png")

        # Opret spiller og platform objekter
        self.player = Player(100, 400, self.screen_width, self.screen_height, self.player_image)
        self.platform = Platform(200, 500, 100, 20, self.platform_image)

        # Opret en gruppe til platforme og tilføj den ene platform
        self.platforms = pygame.sprite.Group()
        self.platforms.add(self.platform)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False

    
    def render(self):
        self.screen.fill((255, 255, 255))
        # Tegn spilleren og platformen på skærmen
        self.screen.blit(self.player.image, self.player.rect)
        self.platforms.draw(self.screen)
        pygame.display.flip()


    def update(self):
        keys = pygame.key.get_pressed()
        self.player.move(keys, self.platforms)

    def run(self):
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
