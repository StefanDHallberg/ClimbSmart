# import pygame
# import torch
# from ML.memory import ReplayMemory
# from ML.agent import Agent
# from Game.platforms import PlatformManager
# from Game.player import Player

# class GameSetup:
#     def __init__(self, num_agents, screen_width, screen_height):
#         self.is_running = True
#         self.screen_width = screen_width
#         self.screen_height = screen_height
#         self.num_agents = num_agents
#         self.players = [Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height) for _ in range(num_agents)]
#         self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
#         self.agents = [Agent(input_channels=3, num_actions=3) for _ in range(num_agents)]
#         self.replay_memory = ReplayMemory(10000)

#         print(f"Initialized GameSetup with {num_agents} agents")

#         self.initialize_platforms()
#         self.initialize_players()

#     def initialize_platforms(self):
#         self.platform_manager.generate_bottom_platform()
#         self.platform_manager.generate_additional_platforms()

#     def initialize_players(self):
#         for player in self.players:
#             player.rect.x = self.screen_width // 2
#             player.rect.y = self.screen_height - 100
#             player.vel_y = 0
#             player.is_jumping = False

#     async def reset_game(self):
#         self.reset_players()
#         self.reset_platform_manager()
#         self.is_running = True
#         print("Game reset complete.")

#     def reset_players(self):
#         for player in self.players:
#             player.reset()
#         print("Players reset")

#     def reset_platform_manager(self):
#         self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
#         print("Platform manager reset")

#     def get_state(self, agent_id):
#         screen_surface = pygame.display.get_surface()
#         if screen_surface is None:
#             pygame.display.set_mode((self.screen_width, self.screen_height))
#             screen_surface = pygame.display.get_surface()
        
#         state = pygame.surfarray.array3d(screen_surface)
#         state = state.transpose((2, 0, 1))
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         return state_tensor

#     def update_players(self, agent_id, keys):
#         self.players[agent_id].update(keys, self.platform_manager.platforms)

#     def check_on_platform(self, agent_id):
#         on_platform = self.players[agent_id].is_on_platform(self.platform_manager.platforms)
#         return on_platform

#     def update_platforms(self):
#         for player in self.players:
#             self.platform_manager.update(player)

#     def get_render_data(self, episode, total_reward):
#         if not self.is_running:
#             return {}

#         players_data = []
#         platforms_data = []

#         for p in self.players:
#             if not self.is_running:
#                 return {}
#             players_data.append({'rect': p.rect, 'image': pygame.image.tostring(p.image, 'RGBA')})

#         for p in self.platform_manager.platforms:
#             if not self.is_running:
#                 return {}
#             platforms_data.append({'rect': p.rect, 'image': pygame.image.tostring(p.image, 'RGBA')})

#         data = {
#             'players': players_data,
#             'platforms': platforms_data,
#             'score': sum(player.score for player in self.players),
#             'episode': episode,
#             'total_reward': total_reward
#         }
#         return data
