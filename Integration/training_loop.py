import time
import asyncio
import torch
import pygame
from .game_ai_integrations import GameAIIntegrations
from .game_setup import GameSetup
from .utilities import handle_events

class TrainingLoop:
    def __init__(self, num_agents, queues, verbose=False):
        self.num_agents = num_agents
        self.queues = queues
        self.verbose = verbose
        self.max_episode_duration = 20  # Maximum duration in seconds
        self.clock = pygame.time.Clock()  # Define a clock object
        self.episode = 1  # Ensure episode is initialized here
        self.initialize_game()

    def initialize_game(self):
        print("Initializing game setup...")
        self.game_setup = GameSetup(self.num_agents)
        self.ai_integrations = [GameAIIntegrations(agent, self.game_setup.replay_memory) for agent in self.game_setup.agents]
        self.load_replay_memory()
        if self.verbose:
            print(f"Game initialized with {self.num_agents} agents.")
        # Initial display update to ensure the first frame is drawn correctly
        self.update_display(self.episode, 0)

    def load_replay_memory(self):
        self.game_setup.replay_memory.load_memory('replay_memory.pkl')
        print("Replay memory loaded")

    def save_replay_memory(self):
        self.game_setup.replay_memory.save_memory('replay_memory.pkl')
        print("Replay memory saved")

    async def run_game(self):
        try:
            while True:  # Run until an explicit break
                print(f"Starting episode {self.episode}")
                should_continue = await self.run_episode(self.episode)
                self.save_replay_memory()
                if not should_continue:
                    break
                await self.reset_game_state()
                self.update_display(self.episode, 0)  # Update display immediately after reset
                self.episode += 1
        except Exception as e:
            print(f"Exception during game loop: {e}")
        finally:
            for ai_integration in self.ai_integrations:
                try:
                    ai_integration.writer.close()
                except Exception as e:
                    print(f"Exception closing writer: {e}")
            pygame.quit()
            print("Training loop terminated")

    async def run_episode(self, episode):
        if self.verbose:
            print(f"Starting episode {episode}")
        total_reward = 0
        self.start_time = time.time()
        self.game_setup.is_running = True

        try:
            while self.game_setup.is_running:
                handle_events(self.game_setup)
                tasks = [self.update_agent(agent_id, ai_integration, episode) for agent_id, ai_integration in enumerate(self.ai_integrations)]
                await asyncio.gather(*tasks)
                self.game_setup.update_platforms()
                self.update_display(episode, total_reward)

                if time.time() - self.start_time > self.max_episode_duration:
                    if self.verbose:
                        print(f"Episode {episode} ended due to exceeding maximum duration of {self.max_episode_duration} seconds.")
                    self.game_setup.is_running = False  # Ensure it stops the loop
                    break  # Break the loop to trigger reset

                await asyncio.sleep(0.016)  # Ensure this matches the frame update rate

            for ai_integration in self.ai_integrations:
                ai_integration.writer.add_scalar('Total Reward', total_reward, episode)
                ai_integration.agent.optimize_model()  # Optimize the model

            if self.verbose:
                print(f"Episode {episode} completed with total reward: {total_reward}")

            return True
        except Exception as e:
            print(f"Exception during episode: {e}")
            return False

    async def update_agent(self, agent_id, ai_integration, episode):
        state = self.game_setup.get_state(agent_id)
        state_tensor = torch.FloatTensor(state).view(1, -1)  # Ensure state has shape [1, feature_size]
        action = ai_integration.select_action_and_update(state_tensor)
        next_state, reward, done, current_score = self.step(agent_id, action)

        next_state_tensor = torch.FloatTensor(next_state).view(1, -1) if next_state is not None else None
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([done])

        ai_integration.agent.memory.push((state_tensor, action, next_state_tensor, reward_tensor))

        ai_integration.writer.add_scalar('Reward', reward, episode)

        if self.verbose:
            print(f"Step result for agent {agent_id}: next_state={next_state}, reward={reward}, done={done}, score={current_score}")
            print(f"Reward written for agent {agent_id}")

    async def reset_game_state(self):
        print("Resetting game state...")
        await self.game_setup.reset_game()
        self.ai_integrations = [GameAIIntegrations(agent, self.game_setup.replay_memory) for agent in self.game_setup.agents]
        self.update_display(self.episode, 0)
        if self.verbose:
            print("Game state reset complete.")

    def step(self, agent_id, action):
        action = action.item() if isinstance(action, torch.Tensor) else action
        keys = {pygame.K_a: False, pygame.K_d: False, pygame.K_w: False, pygame.K_UP: False}
        action_map = {0: pygame.K_a, 1: pygame.K_d, 2: pygame.K_w}

        if action in action_map:
            keys[action_map[action]] = True

        self.game_setup.update_players(agent_id, keys)
        on_platform = self.game_setup.check_on_platform(agent_id)
        next_state = self.game_setup.get_state(agent_id)
        reward = self.calculate_reward(agent_id, action, on_platform)
        done = False  # Always False
        score = self.game_setup.players[agent_id].score
        if self.verbose:
            print(f"Agent {agent_id}, Action {action}, Reward {reward}, Done {done}, Score {score}")

        return next_state, reward, done, score

    def calculate_reward(self, agent_id, action, on_platform):
        reward = 0
        player = self.game_setup.players[agent_id]

        # Only reward if the player has moved +50 in y-value from the starting point
        if player.rect.centery < (self.game_setup.screen_height // 2 - 50):
            if action == 2 and on_platform:
                reward = 1
                if player.score > player.high_score:
                    player.high_score = player.score
                    reward += 100
                    if self.verbose:
                        print(f"New high score achieved by agent {agent_id}: {player.high_score}")
            elif action == 0 or action == 1:
                reward = 0.1
            else:
                reward = -0.05
        return reward

    def update_display(self, episode, total_reward):
        data = self.game_setup.get_render_data(episode, total_reward)
        for queue in self.queues:
            if not queue.full():  # Check if the queue is not full before putting data
                queue.put_nowait(data)  # Use non-blocking put
