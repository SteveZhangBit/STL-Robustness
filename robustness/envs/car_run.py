import numpy as np
import signal_tl as stl
from rsrl.evaluator import Evaluator
from rsrl.util.run_util import setup_eval_configs

from robustness.analysis.stl import STLEvaluator, STLEvaluator2
from robustness.analysis.utils import normalize
from robustness.envs import DeviatableEnv


class CarRunWrapper:
    def __init__(self, env):
        self.env = env
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def reset_to(self, state):
        '''
        state[0] and state[1] the x and y position
        state[2] the orientation
        '''
        self.env.reset()
        agent_xyz = self.env.agent.init_xyz
        agent_xyz[0:2] = state[:2]
        self.env.agent.set_position(agent_xyz)

        quaternion = self.env.bc.getQuaternionFromEuler([0, 0, state[2]])
        self.env.agent.set_orientation(quaternion)

        return self.env.get_observation()


class DevCarRun(DeviatableEnv):
    def __init__(self, load_dir, speed_multiplier, steering_multiplier, delta_0=(20.0, 0.5)):
        super().__init__()

        self.load_dir = load_dir
        self.speed_multiplier = speed_multiplier
        self.steering_multiplier = steering_multiplier
        self.x0_bounds = np.array([ [-0.1, 0.1], [-0.1, 0.1], [3*np.pi/4, 5*np.pi/4] ])
        self.delta_0 = np.array(delta_0)
        self.dev_bounds = np.array([self.speed_multiplier, self.steering_multiplier])

    def instantiate(self, delta, render=False):
        _, config = setup_eval_configs(self.load_dir)
        evaluator = Evaluator(**config, config_dict=config)
        evaluator._init_env()
        env = evaluator.env
        if render:
            env.render()
        env.agent.speed_multiplier, env.agent.steering_multiplier = delta[0], delta[1]
        return CarRunWrapper(env), self.x0_bounds
    
    def get_dev_bounds(self):
        return self.dev_bounds
    
    def get_delta_0(self):
        return self.delta_0
    
    def observation_space(self):
        env = self.instantiate(self.delta_0)[0]
        return env.observation_space


class SafetyProp(STLEvaluator):
    def __init__(self, env, pickle_safe=False):
        super().__init__(pickle_safe)

        self.y_range = np.asarray([env.observation_space.low[1], env.observation_space.high[1]])
        self.vel_range = np.asarray([-10.0, 10.0])

    def prop(self):
        y = stl.Predicate('y')
        vel = stl.Predicate('vel')
        y_check = normalize(0.25, self.y_range)
        vel_check = normalize(1.5, self.vel_range)
        return stl.Always( (y < y_check) & (vel < vel_check) )
    
    def build_signal(self, record, time_index):
        return {
            'y': stl.Signal(
                normalize(np.abs(record[:, 1]), self.y_range),
                time_index
            ),
            'vel': stl.Signal(
                normalize(
                    np.clip(np.linalg.norm(record[:, 2:4], axis=1),
                            self.vel_range[0], self.vel_range[1]),
                    self.vel_range
                ),
                time_index
            )
        }


class SafetyProp2(STLEvaluator2):
    '''
    Deprecated.
    '''
    def eval_one_timepoint(self, obs):
        y = np.abs(obs[1])
        vel = np.linalg.norm(obs[2:4])
        return np.min((0.25 - y, 1.5 - vel))
