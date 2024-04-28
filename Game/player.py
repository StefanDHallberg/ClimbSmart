import pygame

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, screen_width, screen_height, image):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        self.width = 50
        self.height = 50
        self.vel = 5

        # Gem skærmstørrelsen
        self.screen_width = screen_width
        self.screen_height = screen_height

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def move(self, keys, platforms):
        # Gem den nuværende position, hvis der er kollision
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

        # Check for kollisioner med platforme
        collisions = pygame.sprite.spritecollide(self, platforms, False)
        for self.platform in collisions:
            # Hvis spilleren rammer en platform, skal den vende tilbage til den gamle position
            self.rect.x = old_x
            self.rect.y = old_y
