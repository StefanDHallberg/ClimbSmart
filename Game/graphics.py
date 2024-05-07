import pygame

class GraphicsHandler:
    @staticmethod
    def render(screen, player, platforms, camera_offset_y):
        screen.fill((255, 255, 255))  # Fill the screen with white

        # Loop over all platforms and draw them on the screen with the adjusted position
        for platform in platforms:
            adjusted_rect = platform.rect.move(0, camera_offset_y)
            screen.blit(platform.image, adjusted_rect)

        # Draw the player on the screen at the adjusted position
        adjusted_player_rect = player.rect.move(0, camera_offset_y)
        screen.blit(player.image, adjusted_player_rect)

        # Render the player's score
        score_font = pygame.font.Font(None, 36)
        score_text = score_font.render("Score: " + str(player.score), True, (0, 0, 0))  # Render score text
        screen.blit(score_text, (10, 10))  # Adjust position as needed

        # Update the display
        pygame.display.flip()
