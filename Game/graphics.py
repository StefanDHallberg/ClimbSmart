import pygame

class GraphicsHandler:
    @staticmethod
    def render(screen, player, platforms, camera_offset_y, episode, total_reward):
        screen.fill((0, 0, 0))  # Clear the screen with black

        # Drawing platforms and player
        for platform in platforms:
            adjusted_rect = platform.rect.move(0, camera_offset_y)
            screen.blit(platform.image, adjusted_rect)

        adjusted_player_rect = player.rect.move(0, camera_offset_y)
        screen.blit(player.image, adjusted_player_rect)

        if pygame.font.get_init():  # Check if the font module is initialized
            # Render the score, episode, and total reward
            score_font = pygame.font.Font(None, 36)
            score_text = score_font.render(f"Score: {player.score}", True, (255, 0, 0))
            screen.blit(score_text, (10, 10))

            episode_text = score_font.render(f"Episode: {episode}", True, (255, 0, 0))
            screen.blit(episode_text, (10, 50))

            reward_text = score_font.render(f"Total reward: {total_reward}", True, (255, 0, 0))
            screen.blit(reward_text, (10, 90))

        pygame.display.flip()  # Update the display

class Viewport:
    def __init__(self, screen, x, y, width, height):
        self.screen = screen
        self.rect = pygame.Rect(x, y, width, height)
        self.subsurface = self.screen.subsurface(self.rect)
        self.camera_offset_y = 0

    def update_camera(self, player_y, screen_height):
        if player_y < screen_height // 2:
            self.camera_offset_y = screen_height // 2 - player_y
        else:
            self.camera_offset_y = 0

    def render(self, player, platforms):
        self.subsurface.fill((0, 0, 0))  # Clear the viewport with black

        for platform in platforms:
            adjusted_rect = platform.rect.move(0, self.camera_offset_y)
            self.subsurface.blit(platform.image, adjusted_rect)

        adjusted_player_rect = player.rect.move(0, self.camera_offset_y)
        self.subsurface.blit(player.image, adjusted_player_rect)
        
        # Render the score
        score_font = pygame.font.Font(None, 36)
        score_text = score_font.render(f"Score: {player.score}", True, (255, 0, 0))
        self.subsurface.blit(score_text, (10, 10))

        pygame.display.update(self.rect)

