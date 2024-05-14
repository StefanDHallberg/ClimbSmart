import pygame

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, screen_width, screen_height):
        super().__init__()
        self.image_path = "./Game/Assets/Tiles/Characters/tile_0000.png"  # Image path
        self.image = pygame.image.load(self.image_path)  # Load the image from the provided path
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.width = 24
        self.height = 24
        self.vel = 10  # Initialize horizontal velocity
        self.vel_y = 0  # Initialize vertical velocity
        self.jump_vel = -20  # Initialize jump velocity
        self.gravity = 0.5
        self.is_jumping = False
        self.flip = False  # For flipping the player image

        self.highest_y = self.rect.bottom  # Initialize highest position reached
        self.score = 0  # Initialize score

        # Store initial position
        self.initial_x = x
        self.initial_y = y

    def handle_collision(self, platforms):
        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self.vel_y > 0 and self.rect.bottom <= platform.rect.top:
                    self.rect.bottom = platform.rect.top
                    self.rect.centery = platform.rect.centery
                    self.vel_y = 0
                    self.is_jumping = False
                    self.update_score(platform.rect.centery)

    def update_score(self, platform_y):
        if platform_y < self.highest_y:  # Check if climbed upwards
            self.score += self.highest_y - platform_y  # Update score with climbed distance
            self.highest_y = platform_y  # Update highest position reached
    
    def reset(self):
        # Reset player position and attributes to initial state
        self.rect.centerx = self.initial_x
        self.rect.centery = self.initial_y
        self.vel_y = 0
        self.is_jumping = False

    def update(self, keys, platforms):
        self.handle_movement(keys)
        self.jump(keys)
        self.handle_collision(platforms)
        self.apply_gravity(platforms)
        self.update_score()  # Update score based on player's movement

        # Ensure the player stays within the horizontal screen boundaries
        self.rect.left = max(self.rect.left, 0)
        self.rect.right = min(self.rect.right, self.screen_width)
    
    def is_on_platform(self, platforms):
        for platform in platforms:
            if self.rect.colliderect(platform.rect) and self.rect.bottom == platform.rect.top:
                return True
        return False

    def update_score(self):
        current_y = self.rect.bottom
        if current_y < self.highest_y:  # Check if climbed upwards
            climbed_distance = self.highest_y - current_y
            # Scale down the climbed distance to adjust the scoring
            scaled_score = climbed_distance // 10  # Adjust the scaling factor as needed
            self.score += scaled_score
            self.highest_y = current_y  # Update highest position reached
    
    def is_game_over(self):
        # Define conditions for game over
        # if the player falls below a certain y-coordinate
        return self.rect.top > self.screen_height
    
    def staying_still(self):
        return self.initial_x == 0 and self.initial_y == 0

    def handle_movement(self, keys):
        dx = 0
        if keys[pygame.K_a]:
            dx = -self.vel
            self.flip = True
        if keys[pygame.K_d]:
            dx = self.vel
            self.flip = False

        # Update position horizontally
        self.rect.centerx += dx

    def apply_gravity(self, platforms):
        self.rect.centery += self.vel_y  # Update self.rect.centery
        self.vel_y += self.gravity

        on_platform = False
        for platform in platforms:
            if self.rect.colliderect(platform.rect) and self.vel_y >= 0:
                on_platform = True
                self.rect.bottom = platform.rect.top
                self.vel_y = 0
                self.is_jumping = False
                break

        if not on_platform:
            self.vel_y += self.gravity

    def jump(self, keys):
        if (keys[pygame.K_w] or keys[pygame.K_UP]) and not self.is_jumping:
            self.vel_y = self.jump_vel
            self.is_jumping = True

