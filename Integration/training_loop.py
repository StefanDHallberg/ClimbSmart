import torch
import pygame
import time
from .game_setup import GameSetup

class TrainingLoop:
    def __init__(self, game_setup, ai_integrations):
        self.game_setup = game_setup
        self.ai_integrations = ai_integrations
        self.max_episode_duration = 10
        self.start_time = None

    def run_game(self):
        episode = 1
        while self.game_setup.is_running:
            should_continue = self.run_episode(episode)
            if not should_continue:
                self.game_setup.is_running = False
            else:
                episode += 1
        self.ai_integrations.writer.close()

    def run_episode(self, episode):
        self.game_setup.player.reset()
        total_reward = 0
        self.start_time = time.time()

        while self.game_setup.is_running:
            state = self.game_setup.get_state()
            action = self.ai_integrations.select_action_and_update(state)
            next_state, reward, done, current_score = self.step(action)

            total_reward += reward

            if done:
                print(f"Game over: Ending episode {episode} with score {current_score}")
                break

            self.ai_integrations.writer.add_scalar('Reward', reward, episode)

            if time.time() - self.start_time > self.max_episode_duration:
                print(f"Episode {episode} ended due to exceeding maximum duration of {self.max_episode_duration} seconds.")
                break

        self.ai_integrations.writer.add_scalar('Total Reward', total_reward, episode, current_score)
        print(f"Episode {episode} completed with total reward: {total_reward}")
        return self.game_setup.is_running

    def step(self, action):
        action = action.item() if isinstance(action, torch.Tensor) else action
        keys = {pygame.K_a: False, pygame.K_d: False, pygame.K_w: False, pygame.K_UP: False}
        action_map = {0: pygame.K_a, 1: pygame.K_d, 2: [pygame.K_w, pygame.K_UP]}

        mapped_keys = action_map.get(action, [])
        if isinstance(mapped_keys, list):
            for key in mapped_keys:
                keys[key] = True
        else:
            keys[mapped_keys] = True

        self.game_setup.player.update(keys, self.game_setup.platform_manager.platforms)
        on_platform = self.game_setup.player.is_on_platform(self.game_setup.platform_manager.platforms)
        next_state = self.game_setup.get_state()
        reward = self.calculate_reward(action, on_platform)
        done, score = self.game_setup.player.is_game_over()

        return next_state, reward, done, score

    def calculate_reward(self, action, on_platform):
        reward = 0
        if action == 2 and on_platform:
            reward = 1
            if self.game_setup.player.score > self.game_setup.player.high_score:
                self.game_setup.player.high_score = self.game_setup.player.score
                reward += 100
                print(f"New high score achieved: {self.game_setup.player.high_score}")
        elif action == 0 or action == 1:
            reward = 0.1
        else:
            reward = -0.05
        return reward
