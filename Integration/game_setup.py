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
        print(f"Initialized GameSetup with {num_agents} agents")

    def reset_players(self):
        for player in self.players:
            player.reset()
        print("Players reset")

    def get_highest_player_y(self):
        return min(player.rect.top for player in self.players)  # Use rect.top instead of rect.centery for more accuracy

    def update_camera(self):
        highest_y = self.get_highest_player_y()
        self.camera_offset_y = self.screen_height // 2 - highest_y

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
        self.update_camera()  # Update the camera position
        data = {
            'players': [{'rect': p.rect, 'image': pygame.image.tostring(p.image, 'RGBA')} for p in self.players],
            'platforms': [{'rect': p.rect, 'image': pygame.image.tostring(p.image, 'RGBA')} for p in self.platform_manager.platforms],
            'score': sum(player.score for player in self.players),  # Sum of all player scores
            'episode': episode,
            'total_reward': total_reward,
            'camera_offset_y': self.camera_offset_y  # Include camera offset in render data
        }
        return data
