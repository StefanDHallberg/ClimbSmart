import pygame

class GraphicsHandler:
    @staticmethod
    def render(screen, data):
        screen.fill((0, 0, 0))  # Clear the screen with black

        # Drawing platforms
        for platform in data['platforms']:
            rect = pygame.Rect(platform['rect'])
            image = pygame.image.fromstring(platform['image'], rect.size, 'RGBA')
            screen.blit(image, rect)

        # Drawing players
        for player in data['players']:
            rect = pygame.Rect(player['rect'])
            image = pygame.image.fromstring(player['image'], rect.size, 'RGBA')
            screen.blit(image, rect)

        if pygame.font.get_init():  # Check if the font module is initialized
            score_font = pygame.font.Font(None, 36)
            score_text = score_font.render(f"Score: {data['score']}", True, (255, 0, 0))
            screen.blit(score_text, (10, 10))

            episode_text = score_font.render(f"Episode: {data['episode']}", True, (255, 0, 0))
            screen.blit(episode_text, (10, 50))

        pygame.display.flip()  # Update the display
