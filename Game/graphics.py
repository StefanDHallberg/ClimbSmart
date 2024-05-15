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
