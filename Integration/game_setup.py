import pygame
import torch
from ML.memory import ReplayMemory
from ML.agent import Agent
from Game.platforms import PlatformManager
from Game.player import Player


class GameSetup:
    def __init__(self):
        self.is_running = True
        self.camera_offset_y = 0
        self.setup_screen()
        self.initialize_game_elements()

    def setup_screen(self):
        self.screen_width = 800
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ClimbSmart")

    def initialize_game_elements(self):
        x_position = self.screen_width // 2
        y_position = self.screen_height - 20
        self.player = Player(x_position, y_position, self.screen_width, self.screen_height)
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        self.agent = Agent(input_size=10, output_size=3)
        self.replay_memory = ReplayMemory(10000)

    def get_state(self):
        state = [self.player.rect.centerx, self.player.rect.centery, self.player.vel_y, self.player.is_jumping]
        platforms = self.platform_manager.platforms.sprites()

        # Ensure we get exactly 3 platforms or fill with dummy values
        for i in range(3):
            if i < len(platforms):
                platform = platforms[i]
                state.extend([platform.rect.centerx, platform.rect.centery])
            else:
                state.extend([-1, -1])  # Dummy values for missing platforms

        # Ensure state has exactly 10 elements
        if len(state) != 10:
            raise ValueError(f"State does not have 10 elements: {state}")

        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        # print(f"State tensor shape: {state_tensor.shape}")  # Debug print
        return state_tensor
    
    def update_camera(self):
        # Adjust the camera to follow the player
        player_center_y = self.player.rect.centery
        screen_center_y = self.screen_height // 2

        if player_center_y < screen_center_y:
            self.camera_offset_y = screen_center_y - player_center_y
        else:
            self.camera_offset_y = 0

