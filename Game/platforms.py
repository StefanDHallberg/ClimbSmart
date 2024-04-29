import pygame
import random

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, image):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class PlatformManager:
    def __init__(self):
        # Load player and platform images
        self.platform_image = pygame.image.load("./Game/Assets/Tiles/tile_0000.png")
        self.platform_width = 18
        self.min_platform_height = 18
    
    def create_platform(self, x, y):
        platform = Platform(x, y, self.platform_image)
        return platform

    def create_bottom_line(self, screen_width, screen_height):
        platforms = pygame.sprite.Group()  # Create a group for platforms
        num_platforms = screen_width // self.platform_width  # Calculate the number of platforms needed
        print("Number of platforms:", num_platforms)  # Print the number of platforms
        for i in range(num_platforms):
            platform = Platform(i * self.platform_width, screen_height - 20, self.platform_image)
            platforms.add(platform)  # Add the platform instance to the group
        return platforms

    def generate_platforms(self, screen_width, screen_height, num_platforms):
        platforms = pygame.sprite.Group()  # Create a group for platforms
        for _ in range(num_platforms):
            x = random.randint(0, screen_width - self.platform_width)
            y = random.randint(screen_height - 150, screen_height - 100)
            platform = Platform(x, y, self.platform_image)
            platforms.add(platform)
        return platforms

    def generate_new_platforms(self, screen_width, screen_height, num_platforms):
        new_platforms = pygame.sprite.Group()
        for _ in range(num_platforms):
            x = random.randint(0, screen_width - self.platform_width)
            y = random.randint(screen_height - 370, screen_height - 200)
            platform = Platform(x, y, self.platform_image)
            new_platforms.add(platform)
        return new_platforms
