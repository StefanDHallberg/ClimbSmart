import pygame
import torch
from ML.memory import ReplayMemory
from ML.agent import Agent
from Game.platforms import PlatformManager
from Game.player import Player

class GameSetup:
    def __init__(self, num_agents):
        self.is_running = True
        self.screen_width = 800
        self.screen_height = 900
        self.camera_offset_y = 0
        self.num_agents = num_agents
        self.players = [Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height) for _ in range(num_agents)]
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        self.agents = [Agent(input_size=10, output_size=3) for _ in range(num_agents)]
        self.replay_memory = ReplayMemory(10000)

    def reset_players(self):
        self.players = [Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height) for _ in range(self.num_agents)]

    def get_state(self, agent_id):
        state = [self.players[agent_id].rect.centerx, self.players[agent_id].rect.centery, self.players[agent_id].vel_y, self.players[agent_id].is_jumping]
        platforms = self.platform_manager.platforms.sprites()

        for i in range(3):
            if i < len(platforms):
                platform = platforms[i]
                state.extend([platform.rect.centerx, platform.rect.centery])
            else:
                state.extend([-1, -1])

        if len(state) != 10:
            raise ValueError(f"State does not have 10 elements: {state}")

        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        return state_tensor

    def update_players(self, agent_id, keys):
        self.players[agent_id].update(keys, self.platform_manager.platforms)

    def check_on_platform(self, agent_id):
        return self.players[agent_id].is_on_platform(self.platform_manager.platforms)

    def check_game_over(self, agent_id):
        return self.players[agent_id].is_game_over()

    def update_platforms(self):
        for player in self.players:
            self.platform_manager.update(player)
