import pygame
import random

class Platform(pygame.sprite.Sprite):
    def __init__(self, centerx, centery, width, height):
        super().__init__()
        self.image = pygame.image.load("./Game/Assets/Tiles/tile_0000.png")
        self.image = pygame.transform.scale(self.image, (width, height))  # Scale the image to the desired width
        self.rect = self.image.get_rect()

        # Update the rect attribute with custom hitbox size and position
        self.rect.centerx = centerx
        self.rect.centery = centery

class PlatformManager:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.platforms = pygame.sprite.Group()
        self.generate_bottom_platform()

    def generate_bottom_platform(self):
        platform_width = self.screen_width
        platform_height = 18  # Define the height of the platform
        platform_centerx = self.screen_width // 2
        platform_centery = self.screen_height - platform_height // 2
        bottom_platform = Platform(platform_centerx, platform_centery, platform_width, platform_height)
        self.platforms.add(bottom_platform)

    def update(self, player):
        platform_height = player.height  # Get the height from the player object
        # Generate platforms continuously
        while len(self.platforms) < 20:
            last_platform = self.platforms.sprites()[-1]
            platform_width = random.randint(24, 50)
            platform_centerx = random.randint(platform_width // 2, self.screen_width - platform_width // 2)
            
            # Ensure the next platform is not too far from the last one
            min_distance = 100
            while abs(platform_centerx - last_platform.rect.centerx) < min_distance:
                platform_centerx = random.randint(platform_width // 2, self.screen_width - platform_width // 2)

            # Adjust platform_centery to make platforms closer
            platform_centery = last_platform.rect.centery - random.randint(65, 85)  # Adjust these values as needed
            new_platform = Platform(platform_centerx, platform_centery, platform_width, platform_height)
            self.platforms.add(new_platform)

        # Remove platforms that are out of view
        for platform in self.platforms:
            if platform.rect.top > player.rect.bottom + self.screen_height:
                platform.kill()



    
    # def update(self, player, platform_height):
    #     # Generate platforms continuously
    #     while len(self.platforms) < 50:
    #         last_platform = self.platforms.sprites()[-1]
    #         platform_width = random.randint(40, 50)
    #         platform_centerx = random.randint(platform_width // 2, self.screen_width - platform_width // 2)
    #         # Adjust platform_centery to make platforms closer
    #         platform_centery = last_platform.rect.centery - random.randint(65, 85)  # Adjust these values as needed
    #         new_platform = Platform(platform_centerx, platform_centery, platform_width, platform_height)
    #         self.platforms.add(new_platform)

    #     # Remove platforms that are out of view
    #     for platform in self.platforms:
    #         if platform.rect.top > player.rect.bottom + self.screen_height:
    #             platform.kill()
