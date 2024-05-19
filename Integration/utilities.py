import pygame
from Game.graphics import GraphicsHandler

def handle_events(game_setup):
    """ Handle basic pygame events like quitting the game. """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_setup.is_running = False

def update_display(game_setup, episode, total_reward):
    """ Update game components and render the new game state. """
    game_setup.platform_manager.update(game_setup.player)
    game_setup.player.update_score()
    
    # Render the game state
    GraphicsHandler.render(game_setup.screen, game_setup.player, game_setup.platform_manager.platforms, 
                           game_setup.camera_offset_y, episode, total_reward)
    pygame.display.flip() # Refresh the screen display
    # game_setup.clock.tick(60)  # Maintaining a consistent frame rate
