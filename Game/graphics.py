import pygame

class GraphicsHandler:
    @staticmethod
    def render(screen, player, platforms):
        screen.fill((255, 255, 255)) # fill the screen with white
        
        # get the player's position
        player_x, player_y = player.rect.center
        
        # adjust the camera position to center the player on the screen
        camera_offset_x = screen.get_width() // 2 - player_x
        camera_offset_y = screen.get_height() // 2 - player_y
        
        # loop over the platforms and draw them on the screen with the adjusted position
        for platform in platforms:
            adjusted_rect = platform.rect.move(camera_offset_x, camera_offset_y)
            screen.blit(platform.image, adjusted_rect)
        
        # draw the player on the screen with the adjusted position
        adjusted_player_rect = player.rect.move(camera_offset_x, camera_offset_y)
        screen.blit(player.image, adjusted_player_rect)
        
        pygame.display.flip()
