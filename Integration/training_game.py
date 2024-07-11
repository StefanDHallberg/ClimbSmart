import time
import pygame
import torch
from ML.memory import ReplayMemory
from ML.agent import Agent
from Game.platforms import PlatformManager
from Game.player import Player
from Integration.utilities import handle_events
from Integration.game_ai_integrations import GameAIIntegrations

class TrainingGame:
    def __init__(self, num_agents, screen_width, screen_height, queues, stop_event, verbose=False):
        self.num_agents = num_agents
        self.queues = queues
        self.verbose = verbose
        self.max_episode_duration = 25
        self.episode = 1
        self.stop_event = stop_event
        self.save_interval = 3  # Save memory every X episodes
        self.terminate_immediately = False

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.platform_manager = PlatformManager(self.screen_width, self.screen_height)
        # self.players = [Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height) for _ in range(num_agents)]
        self.players = [Player(self.screen_width // 2, self.screen_height - 20, self.screen_width, self.screen_height, self.platform_manager) for _ in range(num_agents)]

        self.agents = [Agent(input_channels=3, num_actions=3) for _ in range(num_agents)]
        self.ai_integrations = [GameAIIntegrations(agent, ReplayMemory(10000)) for agent in self.agents]

        self.state_tensor = torch.zeros((num_agents, 3, self.screen_width, self.screen_height), dtype=torch.float32)
       
        self.clock = pygame.time.Clock()



        if self.verbose:
            print(f"Initialized {self.ai_integrations}")

        self.initialize_platforms()
        self.initialize_players()

        # Load memory if exists
        for i, ai_integration in enumerate(self.ai_integrations):
            ai_integration.replay_memory.load_memory(f"memory_agent_{i}.pkl")

    def get_states(self):
        with torch.no_grad():
            self.state_tensor.zero_()
            self.state_tensor.normal_()
        return self.state_tensor


    def run_game(self):
        try:
            while not self.stop_event.is_set():
                if self.verbose:
                    print(f"Starting episode {self.episode}")
                total_reward = 0
                self.start_time = time.time()
                self.is_running = True

                while self.is_running and not self.stop_event.is_set() and time.time() - self.start_time <= self.max_episode_duration:
                    handle_events(self)

                    states = self.get_states()
                    total_rewards = self.update_agents(self.episode, states)
                    total_reward += sum(total_rewards)
                    self.update_platforms()
                    self.update_display(self.episode, total_reward)

                    # time.sleep(0.016)
                    self.clock.tick(60)  # Limit frame rate to 60 FPS


                if not self.stop_event.is_set():
                    for ai_integration in self.ai_integrations:
                        if ai_integration:
                            ai_integration.writer.add_scalar('Total Reward', total_reward, self.episode)
                            ai_integration.agent.optimize_model()

                    if self.verbose:
                        print(f"Episode {self.episode} completed with total reward: {total_reward}")

                self.cleanup()
                self.episode += 1
                self.reset_game_state()



                # Save replay memory at intervals
                if self.episode % self.save_interval == 0:
                    for i, ai_integration in enumerate(self.ai_integrations):
                        ai_integration.replay_memory.save_memory(f"memory_agent_{i}.pkl")

        except KeyboardInterrupt:
            print("Training loop interrupted by user")
            self.stop_event.set()

        finally:
            self.cleanup()

    def calculate_reward(self, agent_id, action, on_platform):
        reward = 0
        player = self.players[agent_id]

        if action == 2 and on_platform:
            reward = 1
        elif action == 0 or action == 1:
            reward = 0.1
        else:
            reward = -0.05

        milestones = [50, 100, 150]
        reward_increment = 50
        for milestone in milestones:
            if player.score >= milestone and not player.reached_milestones.get(milestone, False):
                reward += reward_increment
                player.reached_milestones[milestone] = True
                print(f"Agent {agent_id} reached score {milestone}, additional reward: {reward_increment}")
        return reward



    def update_agents(self, episode, states):
        total_rewards = []
        for agent_id, ai_integration in enumerate(self.ai_integrations):
            if self.verbose:
                print(f"Agent {agent_id} state: {states[agent_id].shape}")

            # Get the action from the AI integration
            action = ai_integration.select_action_and_update(states[agent_id])
            if isinstance(action, torch.Tensor):
                action = action.item()  # Convert torch.Tensor to a Python int if necessary
            
            if self.verbose:
                print(f"Selected action for agent {agent_id}: {action}")

            # Apply the selected action to update player states
            self.update_players(agent_id, action)

            # Handle collisions after the player has moved
            self.players[agent_id].handle_collision(self.platform_manager.platforms)

            # Check if the player is on a platform after moving and handling collisions
            on_platform = self.check_on_platform(agent_id)

            # Calculate the reward based on the new state after collision handling
            reward = self.calculate_reward(agent_id, action, on_platform)

            # Get the next state after all updates
            next_state = self.get_states()[agent_id]

            # Add the transition to the replay memory
            done = False  # Update this based on your game's end condition
            ai_integration.replay_memory.push([states[agent_id]], [action], [reward], [next_state], [done])

            # Log the reward
            ai_integration.log_data('Total Reward', reward, episode)
            total_rewards.append(reward)
            # print(f"Total rewards: {total_rewards}")
        return total_rewards


    def cleanup(self):
        try:
            if not self.terminate_immediately:
                self.flush_queues()
            for ai_integration in self.ai_integrations:
                try:
                    if ai_integration:
                        ai_integration.writer.close()
                except Exception as e:
                    print(f"Exception closing writer: {e}")
            self.is_running = False
            if self.verbose:
                print("Training loop terminated")
        except Exception as e:
            print(f"Exception during cleanup: {e}")
        finally:
            print("Clean up in TrainingGame")


    def initialize_platforms(self):
        self.platform_manager.generate_bottom_platform()
        self.platform_manager.generate_additional_platforms()

    def initialize_players(self):
        for player in self.players:
            player.rect.x = self.screen_width // 2
            player.rect.y = self.screen_height - 100
            player.vel_y = 0
            player.is_jumping = False

    def reset_game(self):
        self.reset_players()
        self.reset_platform_manager()
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
        player = self.players[agent_id]
        for platform in self.platform_manager.platforms:
            if platform.on_platform and player.rect.colliderect(platform.rect):
                print(f"Player {player.rect} on platform {platform.rect}")  # Debugging print
                return True
        # print(f"Agent {agent_id} not on platform")
        return False

    def update_platforms(self):
        for player in self.players:
            self.platform_manager.update(player)

    def get_render_data(self, episode, total_reward):
        players_data = []
        platforms_data = []

        for p in self.players:
            players_data.append({'rect': p.rect, 'image': pygame.image.tostring(p.image, 'RGBA')})

        for p in self.platform_manager.platforms:
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
        self.stop_event.set()

    def update_display(self, episode, total_reward):
        if self.stop_event.is_set():
            return
        data = self.get_render_data(episode, total_reward)
        if not data:
            return
        for queue in self.queues:
            if not queue.full():
                queue.put_nowait(data)

    def flush_queues(self):
        for queue in self.queues:
            while not queue.empty():
                queue.get_nowait()

    def reset_game_state(self):
        if self.verbose:
            print("Resetting game state...")
        self.update_display(self.episode, 0)
        self.reset_game()
        self.ai_integrations = [GameAIIntegrations(agent, ReplayMemory(10000)) for agent in self.agents]
        self.update_display(self.episode, 0)
        if self.verbose:
            print("Game state reset complete.")