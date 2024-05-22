import torch
import time
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
        self.initialize_game()

    def initialize_game(self):
        self.game_setup = GameSetup(self.num_agents)
        self.ai_integrations = [GameAIIntegrations(agent, self.game_setup.replay_memory) for agent in self.game_setup.agents]
        self.game_setup.reset_players()
        self.load_replay_memory()
        if self.verbose:
            print(f"Game initialized with {self.num_agents} agents.")
        
    def load_replay_memory(self):
        self.game_setup.replay_memory.load_memory('replay_memory.pkl')

    def save_replay_memory(self):
        self.game_setup.replay_memory.save_memory('replay_memory.pkl')

    def run_game(self):
        episode = 1
        try:
            while self.game_setup.is_running:
                print(f"Starting episode {episode}")
                should_continue = self.run_episode(episode)
                self.save_replay_memory()
                if not should_continue:
                    self.game_setup.is_running = False
                else:
                    self.initialize_game()  # Reset the game properly between episodes
                    episode += 1
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

    def run_episode(self, episode):
        if self.verbose:
            print(f"Starting episode {episode}")
        total_reward = 0
        self.start_time = time.time()

        while self.game_setup.is_running:
            handle_events(self.game_setup)
            for agent_id, ai_integration in enumerate(self.ai_integrations):
                state = self.game_setup.get_state(agent_id)
                action = ai_integration.select_action_and_update(state)
                self.next_state, reward, done, current_score = self.step(agent_id, action)

                total_reward += reward

                ai_integration.writer.add_scalar('Reward', reward, episode)

            self.game_setup.update_platforms()
            self.update_display(episode, total_reward)

            if time.time() - self.start_time > self.max_episode_duration:
                if self.verbose:
                    print(f"Episode {episode} ended due to exceeding maximum duration of {self.max_episode_duration} seconds.")
                self.game_setup.is_running = False  # Ensure it stops the loop
                break

            self.clock.tick(60)

        for ai_integration in self.ai_integrations:
            ai_integration.writer.add_scalar('Total Reward', total_reward, episode)
        if self.verbose:
            print(f"Episode {episode} completed with total reward: {total_reward}")

        # Reset the game state to running for the next episode
        self.game_setup.is_running = True
        return self.game_setup.is_running

    def step(self, agent_id, action):
        action = action.item() if isinstance(action, torch.Tensor) else action
        keys = {pygame.K_a: False, pygame.K_d: False, pygame.K_w: False, pygame.K_UP: False}
        action_map = {0: pygame.K_a, 1: pygame.K_d, 2: pygame.K_w}

        mapped_keys = action_map.get(action, [])
        if isinstance(mapped_keys, list):
            for key in mapped_keys:
                keys[key] = True
        else:
            keys[mapped_keys] = True

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
            queue.put(data)