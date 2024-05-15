from torch.utils.tensorboard import SummaryWriter

class GameAIIntegrations:
    def __init__(self, agent, replay_memory):
        self.agent = agent
        self.replay_memory = replay_memory
        self.writer = SummaryWriter('runs/ClimbSmart')

    def select_action_and_update(self, state):
        action = self.agent.select_action(state)
        self.agent.update_epsilon()
        return action
