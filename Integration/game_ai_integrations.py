class GameAIIntegrations:
    def __init__(self, agent, replay_memory):
        self.agent = agent
        self.replay_memory = replay_memory
    
    def select_action_and_update(self, state):
        action = self.agent.select_action(state)
        return action