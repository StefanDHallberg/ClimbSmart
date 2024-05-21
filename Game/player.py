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
        self.high_score = 0
        self.reached_high_score = False

        # Store initial position
        self.initial_x = x
        self.initial_y = y

    def handle_collision(self, platforms):
        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self.vel_y > 0 and self.rect.bottom <= platform.rect.top + self.vel_y:
                    self.rect.bottom = platform.rect.top
                    self.vel_y = 0
                    self.is_jumping = False
                    self.update_score(platform.rect.centery)
                    platform.on_platform = True

    def update_score(self, platform_y=None):
        current_y = self.rect.bottom if platform_y is None else platform_y
        climbed_distance = self.highest_y - current_y
        if climbed_distance > 0:
            scaled_score = climbed_distance // 10
            self.score += scaled_score
            self.highest_y = current_y
        if self.score > self.high_score:
            self.high_score = self.score
    
    def staying_still(self):
        return self.initial_x == 0 and self.initial_y == 0

    def reset(self):
        self.score = 0
        self.rect.centerx = self.initial_x
        self.rect.centery = self.initial_y
        self.vel_y = 0
        self.is_jumping = False
        self.reached_high_score = False
        # self.camera_offset_y = 0

    def update(self, keys, platforms):
        self.handle_movement(keys)
        self.jump(keys)
        self.handle_collision(platforms)
        self.apply_gravity(platforms)
        self.update_score()

        self.rect.left = max(self.rect.left, 0)
        self.rect.right = min(self.rect.right, self.screen_width)

    def is_on_platform(self, platforms):
        for platform in platforms:
            if self.rect.colliderect(platform.rect) and self.rect.bottom == platform.rect.top:
                return True
        return False

    # def is_game_over(self, platforms):
    #     if self.rect.top > self.screen_height:
    #         print(f"Game over because player y-coordinate {self.rect.top} > {self.screen_height}")
    #         self.reset()
    #         return True, self.score
    #     if self.rect.bottom > self.screen_height and not any(platform.rect.colliderect(self.rect) for platform in platforms):
    #         print(f"Game over because player y-coordinate {self.rect.bottom} > {self.screen_height} and no platform beneath.")
    #         self.reset()
    #         return True, self.score
    #     return False, self.score
    

    def handle_movement(self, keys):
        if keys.get(pygame.K_a, False):
            self.rect.centerx -= self.vel
        if keys.get(pygame.K_d, False):
            self.rect.centerx += self.vel
        if (keys.get(pygame.K_w, False) or keys.get(pygame.K_UP, False)) and not self.is_jumping:
            self.vel_y = self.jump_vel
            self.is_jumping = True

    def apply_gravity(self, platforms):
        self.rect.centery += self.vel_y
        self.vel_y += self.gravity

        on_platform = False
        for platform in platforms:
            if self.rect.colliderect(platform.rect) and self.vel_y >= 0:
                on_platform = True
                platform.on_platform = True
                self.rect.bottom = platform.rect.top
                self.vel_y = 0
                self.is_jumping = False
                break

        if not on_platform:
            self.vel_y += self.gravity

    def jump(self, keys):
        if (keys.get(pygame.K_w) or keys.get(pygame.K_UP)) and not self.is_jumping:
            self.vel_y = self.jump_vel
            self.is_jumping = True
