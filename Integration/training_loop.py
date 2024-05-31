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
        self.max_episode_duration = 15  # Maximum duration in seconds
        self.clock = pygame.time.Clock()  # Define a clock object
        self.episode = 1  # Ensure episode is initialized here
        self.display_event = asyncio.Event()  # Event to signal display update completion
        self.terminate_immediately = False  # Flag to indicate immediate termination

    async def initialize_game(self):
        print("Initializing game setup...")
        self.game_setup = GameSetup(self.num_agents)
        self.ai_integrations = [GameAIIntegrations(agent, self.game_setup.replay_memory) for agent in self.game_setup.agents]
        self.load_replay_memory()
        if self.verbose:
            print(f"Game initialized with {self.num_agents} agents.")
        # Initial display update to ensure the first frame is drawn correctly
        await self.update_display(self.episode, 0)

    def load_replay_memory(self):
        self.game_setup.replay_memory.load_memory('replay_memory.pkl')
        print("Replay memory loaded.")

    def save_replay_memory(self):
        self.game_setup.replay_memory.save_memory('replay_memory.pkl')
        print("Replay memory saved.")

    async def run_game(self):
        await self.initialize_game()

        try:
            while True:  # Run until an explicit break
                print(f"Starting episode {self.episode}")
                total_reward = 0
                self.start_time = time.time()
                self.game_setup.is_running = True
                self.terminate_immediately = False

                try:
                    while self.game_setup.is_running:
                        if self.terminate_immediately:
                            break  # Ensure immediate termination if flagged

                        handle_events(self.game_setup)  # Correctly calling the function
                        tasks = [self.update_agent(agent_id, ai_integration, self.episode) for agent_id, ai_integration in enumerate(self.ai_integrations)]
                        await asyncio.gather(*tasks)
                        self.game_setup.update_platforms()
                        await self.update_display(self.episode, total_reward)

                        if time.time() - self.start_time > self.max_episode_duration:
                            if self.verbose:
                                print(f"Episode {self.episode} ended due to exceeding maximum duration of {self.max_episode_duration} seconds.")
                            self.terminate_game_loop()
                            break

                        await asyncio.sleep(0.016)  # Ensure this matches the frame update rate

                    if not self.terminate_immediately:
                        for ai_integration in self.ai_integrations:
                            if ai_integration:
                                ai_integration.writer.add_scalar('Total Reward', total_reward, self.episode)
                                ai_integration.agent.optimize_model()  # Optimize the model

                        if self.verbose:
                            print(f"Episode {self.episode} completed with total reward: {total_reward}")

                except Exception as e:
                    print(f"Exception during episode: {e}")
                    self.terminate_game_loop()
                    break

                self.save_replay_memory()
                if not self.game_setup.is_running:
                    break
                await self.reset_game_state()
                await self.update_display(self.episode, 0)  # Update display immediately after reset
                self.episode += 1

        except Exception as e:
            print(f"Exception during game loop: {e}")
        finally:
            await self.cleanup()

    def terminate_game_loop(self):
        self.game_setup.is_running = False  # Ensure it stops the loop
        self.terminate_immediately = True

    async def update_display(self, episode, total_reward):
        if self.terminate_immediately:
            return  # Exit immediately if termination flag is set

        data = self.game_setup.get_render_data(episode, total_reward)
        if not data:  # If the game setup was terminated
            return
        for queue in self.queues:
            if not queue.full():  # Check if the queue is not full before putting data
                queue.put_nowait(data)  # Use non-blocking put

    async def flush_queues(self):
        for queue in self.queues:
            while not queue.empty():
                queue.get_nowait()

    async def update_agent(self, agent_id, ai_integration, episode):
        if self.terminate_immediately:
            return  # Exit immediately if termination flag is set

        state = self.game_setup.get_state(agent_id)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension for grayscale
        action = ai_integration.select_action_and_update(state_tensor)
        next_state, reward, done, current_score = self.step(agent_id, action)

        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0) if next_state is not None else None
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([done])

        ai_integration.agent.memory.push((state_tensor, action, next_state_tensor, reward_tensor))

        ai_integration.log_data('Total Reward', reward, episode)

        if self.verbose:
            print(f"Step result for agent {agent_id}: next_state={next_state}, reward={reward}, done={done}, score={current_score}")
            print(f"Reward written for agent {agent_id}")

    async def reset_game_state(self):
        print("Resetting game state...")
        self.update_display(self.episode, 0) # Update display immediately after reset
        await self.game_setup.reset_game() 
        self.ai_integrations = [GameAIIntegrations(agent, self.game_setup.replay_memory) for agent in self.game_setup.agents]
        await self.update_display(self.episode, 0)
        if self.verbose:
            print("Game state reset complete.")

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
        
        # Only reward if the player has moved +50 in y-value from the starting point
        if player.rect.centery < (self.game_setup.screen_height // 2 - 50):
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

    async def cleanup(self):
        if not self.terminate_immediately:
            await self.flush_queues()
        for ai_integration in self.ai_integrations:
            try:
                if ai_integration:
                    ai_integration.writer.close()
            except Exception as e:
                print(f"Exception closing writer: {e}")
        pygame.quit()
        print("Training loop terminated")
