import numpy as np
from scipy import linalg

from robustness.agents import Agent


class PPO(Agent):
    def __init__(self, model_path) -> None:
        from stable_baselines3 import PPO as BaselinePPO

        self.model = BaselinePPO.load(model_path)
    
    def next_action(self, obs):
        return self.model.predict(obs, deterministic=True)[0]
    
    def reset(self):
        pass


class LQR(Agent):
    def __init__(self, FPS, VIEWPORT_H, VIEWPORT_W, SCALE):
        super().__init__()
        self.FPS = FPS
        self.VIEWPORT_H = VIEWPORT_H
        self.VIEWPORT_W = VIEWPORT_W
        self.SCALE = SCALE
    
    def next_action(self, obs):
        # gravity = 9.8/FPS/FPS/SCALE
        gravity = 9.8 / self.FPS / self.FPS # gravity changes depending on SCALE
        m_main_inv = gravity / 0.56    # determined by test
        m_side_inv = gravity * 0.365    # determined by test
        a_sina_i_inv= 0.198 / 100 # determined by test # not depending on SCALE
        cos_alpha = 0.72

        # target point set
        x_target = 0
        y_target = 0   # the landing point is 0
        Vx_target = 0
        Vy_target = 0
        theta_target = 0
        omega_target = 0

        """
        Design of the reference trajectory
        """
        y_target = obs[1] * (self.VIEWPORT_H / self.SCALE / 2) / 1.6 # 1.6 succeeds all the times

        """
        Design of a state space representation
        """
        X = np.array([
            [obs[0] * (self.VIEWPORT_W / self.SCALE / 2) - x_target],
            [obs[1] * (self.VIEWPORT_H / self.SCALE / 2) - y_target],
            [obs[2] / (self.VIEWPORT_W / self.SCALE / 2) - Vx_target],
            [obs[3] / (self.VIEWPORT_H / self.SCALE / 2) - Vy_target],
            [obs[4] - theta_target],
            [obs[5] / 20.0 - omega_target]
        ])

        A = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, -1 * gravity, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ])

        B = np.array([
            [0, 0],
            [0, 0],
            [0, m_side_inv * cos_alpha * cos_alpha],
            [1 * m_main_inv, 0],
            [0, 0],
            [0, -1 * a_sina_i_inv]
        ])
        # the second term of the 4th row of B was igonred for simplification assuming that Fside is smaller than Fmain and negligible while Fmain is used

        sigma = np.array([
            [0],
            [0],
            [0],
            [-1 * gravity],
            [0],
            [0]
        ])

        # gravity compensation
        BTB = np.dot(B.T, B)
        u_sigma = -1 * np.linalg.inv(BTB).dot(B.T).dot(sigma)

        """
        Design of LQR
        Solve Riccati equation to find a optimal control input
        """
        R = np.array([
            [1, 0],
            [0, 1]
        ])

        Q = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 100, 0],
            [0, 0, 0, 0, 0, 100]
        ])

        # Solving Riccati equation
        P = linalg.solve_continuous_are(A, B, Q, R)
        # print("P {}\n".format(P))

        # u = -KX
        # K = R-1*Rt*P
        K = np.linalg.inv(R).dot(B.T).dot(P)
        # print("K {}\n".format(K))
        thrust = -1 * np.dot(K, X) + u_sigma
        # thrust = -1*np.dot(K, X)

        """
        Free fall from the final short distance
        """
        if obs[1] < 0.3 / self.SCALE:
            thrust[0] = 0
            thrust[1] = 0

        # conversion to compensate main thruster's tricky thrusting
        thrust[0] = thrust[0] / 0.5 - 1.0

        a = np.array([thrust[0], thrust[1]])
        a = np.clip(a, -1, +1)  #  if the value is less than 0.5, it's ignored
        return a.flatten()
    
    def reset(self):
        pass
