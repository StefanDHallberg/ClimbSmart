import pygame

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, screen_width, screen_height):
        super().__init__()
        self.image_path = "./Game/Assets/Tiles/Characters/tile_0000.png"  # Image path
        self.image = pygame.image.load(self.image_path)  # Load the image from the provided path
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        self.width = 50
        self.height = 50
        self.vel = 5
        self.jump_vel = -10
        self.gravity = 0.5
        self.is_jumping = False
        self.vel_y = 0  # Define vel_y attribute for vertical velocity

        # Save screen dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def move(self, keys, platforms):
        # Save the current position in case of collision
        old_x = self.rect.x
        old_y = self.rect.y

        if keys[pygame.K_LEFT] and self.rect.x > 0:
            self.rect.x -= self.vel
        if keys[pygame.K_RIGHT] and self.rect.x < self.screen_width - self.width:
            self.rect.x += self.vel
        if keys[pygame.K_UP] and self.rect.y > 0:
            self.rect.y -= self.vel
        if keys[pygame.K_DOWN] and self.rect.y < self.screen_height - self.height:
            self.rect.y += self.vel

        # Check for collisions with platforms
        collisions = pygame.sprite.spritecollide(self, platforms, False)
        for platform in collisions:
            # If the player hits a platform, revert to the old position
            self.rect.x = old_x
            self.rect.y = old_y

    def jump(self, platforms):
        if not self.is_jumping:
            self.is_jumping = True
            self.vel_y = self.jump_vel

    def apply_gravity(self, platforms):
        # Check if the player is on a platform
        on_platform = pygame.sprite.spritecollideany(self, platforms)

        if not on_platform:
            # Apply gravity only if the player is not on a platform
            self.vel_y += self.gravity
            self.rect.y += self.vel_y

            # Check for collisions with platforms while falling
            collisions = pygame.sprite.spritecollide(self, platforms, False)
            for platform in collisions:
                if self.vel_y < 0:
                    # If jumping and hitting a platform from below, stop jumping
                    self.rect.top = platform.rect.bottom
                    self.vel_y = 0
                else:
                    # If falling down onto a platform, stop falling and reset jumping state
                    self.rect.bottom = platform.rect.top
                    self.is_jumping = False
                    self.vel_y = 0
        else:
            # Reset vertical velocity when on a platform
            self.vel_y = 0