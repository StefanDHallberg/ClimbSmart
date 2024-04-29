import pygame
from platforms import PlatformManager, Platform
from graphics import GraphicsHandler
from player import Player

class Game:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 900

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ClimbSmart")
        self.clock = pygame.time.Clock()
        self.is_running = True

        # Create platform manager
        self.platform_manager = PlatformManager()

        # Call the create_bottom_line function from the PlatformManager class
        self.platforms = self.platform_manager.create_bottom_line(self.screen_width, self.screen_height)

        # Create the player object
        self.player = Player(self.screen_width // 2, self.screen_height - 50, self.screen_width, self.screen_height)

        # Generate initial platforms
        num_platforms = 10
        # amount of platforms generated so far and the number of platforms to generate per screen
        self.generated_platforms = 0
        self.platforms_per_screen = 5
        self.platforms.add(self.platform_manager.generate_platforms(self.screen_width, self.screen_height, num_platforms))


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player.jump(self.platforms)  # Pass the platforms group to the jump method

    def update(self):
        keys = pygame.key.get_pressed()
        self.player.move(keys, self.platforms)
        self.player.apply_gravity(self.platforms)  # Gravity application moved inside the apply_gravity method

        # Monitor the player's vertical position
        player_y = self.player.rect.bottom

        # Check if player's vertical position is below a certain threshold
        if player_y < self.screen_height / 3:
            print("Player is near the top of the screen. Generating new platforms.")

            # Generate new platforms
            new_platforms = self.platform_manager.generate_new_platforms(self.screen_width, self.screen_height, self.platforms_per_screen)
            print("New platforms generated:", len(new_platforms))  # Verify if new platforms are generated

            # Add new platforms to the existing platforms group
            self.platforms.add(new_platforms)




    def run(self):
        while self.is_running:
            self.handle_events()
            self.update()

            # Render the game objects
            GraphicsHandler.render(self.screen, self.player, self.platforms)

            # Update the display
            pygame.display.flip()

            # fps.
            self.clock.tick(60)

if __name__ == "__main__":
    game = Game()
    game.run()
