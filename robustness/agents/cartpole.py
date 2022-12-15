import numpy as np

from robustness.agents import Agent


class PID(Agent):
    '''
    Credits to: https://gist.github.com/HenryJia/23db12d61546054aa43f8dc587d9dc2c
    '''
    def __init__(self):
        self.desired_state = np.array([0, 0, 0, 0])
        self.desired_mask = np.array([0, 0, 1, 0])
        self.P, self.I, self.D = 0.1, 0.01, 0.5
        
        self.reset()

    def next_action(self, obs):
        error = obs - self.desired_state

        self.integral += error
        self.derivative = error - self.prev_error
        self.prev_error = error

        pid = np.dot(self.P * error + self.I * self.integral + self.D * self.derivative, self.desired_mask)
        action = self.sigmoid(pid)
        return np.round(action).astype(np.int32)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def reset(self):
        self.integral = 0
        self.derivative = 0
        self.prev_error = 0


class DQN(Agent):
    def __init__(self, model_path) -> None:
        from stable_baselines3 import DQN as BaselineDQN

        self.model = BaselineDQN.load(model_path)
    
    def next_action(self, obs):
        return self.model.predict(obs, deterministic=True)[0]
    
    def reset(self):
        pass
