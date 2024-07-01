import torch
import time
import asyncio
import pygame
from ML.memory import ReplayMemory
from ML.agent import Agent
from Game.platforms import PlatformManager
from Game.player import Player
from Integration.utilities import handle_events
from Integration.game_ai_integrations import GameAIIntegrations
from memory_profiler import profile

class TrainingGame:
    def __init__(self, num_agents, screen_width, screen_height, queues, verbose=False):
        pygame.init()
        self.num_agents = num_agents
        self.queues = queues
        self.verbose = verbose
        self.max_episode_duration = 15
        self.episode = 1
        self.terminate_immediately = False

        self.is_running = True
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.players = [Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height) for _ in range(num_agents)]
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)

        self.agents = [Agent(input_channels=3, num_actions=3) for _ in range(num_agents)]
        self.ai_integrations = [GameAIIntegrations(agent, ReplayMemory(10000)) for agent in self.agents]

        self.state_tensor = torch.zeros((num_agents, 3, self.screen_width, self.screen_height), dtype=torch.float32)

        if self.verbose:
            print(f"Initialized {self.ai_integrations}")

        self.initialize_platforms()
        self.initialize_players()

    def get_states(self):
        with torch.no_grad():
            self.state_tensor.zero_()
            states = torch.randn_like(self.state_tensor)
            self.state_tensor.copy_(states)
        return self.state_tensor

    async def run_game(self):
        try:
            while not self.terminate_immediately:
                if self.verbose:
                    print(f"Starting episode {self.episode}")
                total_reward = 0
                self.start_time = time.time()
                self.is_running = True

                try:
                    while self.is_running and not self.terminate_immediately:
                        if time.time() - self.start_time > self.max_episode_duration:
                            if self.verbose:
                                print("Episode duration exceeded the maximum limit.")
                            self.is_running = False
                            break

                        handle_events(self)

                        states = self.get_states()
                        total_rewards = await self.update_agents(self.episode, states)
                        total_reward += sum(total_rewards)
                        self.update_platforms()
                        await self.update_display(self.episode, total_reward)

                        await asyncio.sleep(0.016)

                    if not self.terminate_immediately:
                        for ai_integration in self.ai_integrations:
                            if ai_integration:
                                ai_integration.writer.add_scalar('Total Reward', total_reward, self.episode)
                                ai_integration.agent.optimize_model()

                        if self.verbose:
                            print(f"Episode {self.episode} completed with total reward: {total_reward}")

                except Exception as e:
                    print(f"Exception during episode: {e}")
                    self.is_running = False
                    self.terminate_immediately = True

                await self.cleanup()
                self.episode += 1

        except KeyboardInterrupt:
            print("Training loop interrupted by user")

        await self.cleanup()

    async def update_agents(self, episode, states):
        try:
            total_rewards = []
            for agent_id, ai_integration in enumerate(self.ai_integrations):
                if self.verbose:
                    print(f"Agent {agent_id} state: {states[agent_id].shape}")

                start_time = time.time()
                action = ai_integration.select_action_and_update(states[agent_id])
                if self.verbose:
                    print(f"Time taken to select action for agent {agent_id}: {time.time() - start_time}s")

                if self.verbose:
                    print(f"Action before update_players: {action}, type: {type(action)}")
                action = action.item() if isinstance(action, torch.Tensor) else action
                if self.verbose:
                    print(f"Action after conversion: {action}, type: {type(action)}")

                start_time = time.time()
                self.update_players(agent_id, action)
                if self.verbose:
                    print(f"Time taken to update player for agent {agent_id}: {time.time() - start_time}s")

                next_state = self.get_states()[agent_id]
                reward = self.calculate_reward(agent_id, action, self.check_on_platform(agent_id))
                done = False

                start_time = time.time()
                ai_integration.replay_memory.push([states[agent_id]], [action], [reward], [next_state], [done])
                if self.verbose:
                    print(f"Time taken to push to replay memory for agent {agent_id}: {time.time() - start_time}s")

                ai_integration.log_data('Total Reward', reward, episode)
                total_rewards.append(reward)
            return total_rewards
        except Exception as e:
            print(f"Exception in update_agent: {e}")
            raise

    async def cleanup(self):
        if not self.terminate_immediately:
            await self.flush_queues()
        for ai_integration in self.ai_integrations:
            try:
                if ai_integration:
                    ai_integration.writer.close()
            except Exception as e:
                print(f"Exception closing writer: {e}")
        self.is_running = False
        if self.verbose:
            print("Training loop terminated")

        pygame.quit()

    def initialize_platforms(self):
        self.platform_manager.generate_bottom_platform()
        self.platform_manager.generate_additional_platforms()

    def initialize_players(self):
        for player in self.players:
            player.rect.x = self.screen_width // 2
            player.rect.y = self.screen_height - 100
            player.vel_y = 0
            player.is_jumping = False

    async def reset_game(self):
        self.reset_players()
        self.reset_platform_manager()
        self.is_running = True
        if self.verbose:
            print("Game reset complete.")

    def reset_players(self):
        for player in self.players:
            player.reset()
        if self.verbose:
            print("Players reset")

    def reset_platform_manager(self):
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        if self.verbose:
            print("Platform manager reset")

    def update_players(self, agent_id, action):
        keys = {pygame.K_a: False, pygame.K_d: False, pygame.K_w: False, pygame.K_UP: False}
        action_map = {0: pygame.K_a, 1: pygame.K_d, 2: pygame.K_w}

        action = action.item() if isinstance(action, torch.Tensor) else action
        if action in action_map:
            keys[action_map[action]] = True

        if self.verbose:
            print(f"Action in update_players: {action}, type: {type(action)}")
            print(f"Keys before update: {keys}")

        self.players[agent_id].update(keys, self.platform_manager.platforms)

        if self.verbose:
            print(f"Keys after update: {keys}")

    def check_on_platform(self, agent_id):
        return self.players[agent_id].is_on_platform(self.platform_manager.platforms)

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
            'score': sum(player.score for player in self.players),
            'episode': episode,
            'total_reward': total_reward
        }
        return data

    def terminate_game_loop(self):
        self.is_running = False
        self.terminate_immediately = True

    async def update_display(self, episode, total_reward):
        if self.terminate_immediately:
            return
        data = self.get_render_data(episode, total_reward)
        if not data:
            return
        for queue in self.queues:
            if not queue.full():
                queue.put_nowait(data)

    async def flush_queues(self):
        for queue in self.queues:
            while not queue.empty():
                queue.get_nowait()

    async def reset_game_state(self):
        if self.verbose:
            print("Resetting game state...")
        self.update_display(self.episode, 0)
        await self.reset_game()
        self.ai_integrations = [GameAIIntegrations(agent, self.replay_memory) for agent in self.agents]
        await self.update_display(self.episode, 0)
        if self.verbose:
            print("Game state reset complete.")

    def calculate_reward(self, agent_id, action, on_platform):
        reward = 0
        player = self.players[agent_id]

        if player.rect.centery < (self.screen_height // 2 - 50):
            if action == 2 and on_platform:
                reward = 1
                if player.score > player.highest_y:
                    player.high_score = player.score
                    reward += 100
                    if self.verbose:
                        print(f"New high score achieved by agent {agent_id}: {player.highest_y}")
            elif action == 0 or action == 1:
                reward = 0.1
            else:
                reward = -0.05
        return reward