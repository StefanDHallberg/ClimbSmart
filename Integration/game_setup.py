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
        self.num_agents = num_agents
        self.players = [Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height) for _ in range(num_agents)]
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        # Assuming input_channels is 3 (for RGB images) and num_actions is 3
        self.agents = [Agent(input_channels=3, num_actions=3) for _ in range(num_agents)]
        self.replay_memory = ReplayMemory(10000)

        print(f"Initialized GameSetup with {num_agents} agents")

        # Ensure initial platforms and players are set up correctly
        self.initialize_platforms()
        self.initialize_players()

    def initialize_platforms(self):
        self.platform_manager.generate_bottom_platform()
        self.platform_manager.generate_additional_platforms()  # Ensure additional platforms are generated

    def initialize_players(self):
        for player in self.players:
            player.rect.x = self.screen_width // 2
            player.rect.y = self.screen_height - 100  # Position just above the bottom
            player.vel_y = 0
            player.is_jumping = False

    def reset_game(self):
        self.reset_players()
        self.reset_platform_manager()
        self.is_running = True
        print("Game reset complete.")

    def reset_players(self):
        for player in self.players:
            player.reset()
        print("Players reset")

    def reset_platform_manager(self):
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        print("Platform manager reset")

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
        on_platform = self.players[agent_id].is_on_platform(self.platform_manager.platforms)
        return on_platform

    def update_platforms(self):
        for player in self.players:
            self.platform_manager.update(player)

    def get_render_data(self, episode, total_reward):
        if not self.is_running:
            return {}

        players_data = []
        platforms_data = []

        for p in self.players:
            if not self.is_running:
                return {}
            players_data.append({'rect': p.rect, 'image': pygame.image.tostring(p.image, 'RGBA')})

        for p in self.platform_manager.platforms:
            if not self.is_running:
                return {}
            platforms_data.append({'rect': p.rect, 'image': pygame.image.tostring(p.image, 'RGBA')})

        data = {
            'players': players_data,
            'platforms': platforms_data,
            'score': sum(player.score for player in self.players),  # Sum of all player scores
            'episode': episode,
            'total_reward': total_reward
        }
        return data
