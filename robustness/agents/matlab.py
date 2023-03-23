from robustness.agents import Agent


class Traditional(Agent):
    def __init__(self):
        self.type = 'T'


class RLAgent(Agent):
    def __init__(self, path):
        self.path = path
        self.type = 'RL'
