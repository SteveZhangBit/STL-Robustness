import numpy as np
import signal_tl as stl
from bullet_safety_gym.envs.tasks import angle2pos
from rsrl.evaluator import Evaluator
from rsrl.util.run_util import setup_eval_configs

from robustness.analysis.stl import STLEvaluator
from robustness.envs import DeviatableEnv


class CarCircleWrapper:
    def __init__(self, env):
        self.env = env
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def reset_to(self, state):
        self.env.reset()
        agent_xyz = self.env.agent.init_xyz
        agent_xyz[0:2] = state
        self.env.agent.set_position(agent_xyz)

        # set agent orientation in forward run direction
        # Note: this is important, by default, the orientaion of the car is
        # computed based on the initial position.
        y = angle2pos(self.env.agent.get_position(), np.zeros(3)) + np.pi / 2
        y += self.env.agent.init_rpy[2]
        quaternion = self.env.bc.getQuaternionFromEuler([0, 0, y])
        self.env.agent.set_orientation(quaternion)

        return self.env.get_observation()


class DevCarCircle(DeviatableEnv):
    def __init__(self, load_dir, speed_multiplier, steering_multiplier, delta_0=(20.0, 0.5)):
        super().__init__()

        self.load_dir = load_dir
        self.speed_multiplier = speed_multiplier
        self.steering_multiplier = steering_multiplier
        self.x0_bounds = np.repeat([[-3, 3]], 2, axis=0)
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
        return CarCircleWrapper(env), self.x0_bounds
    
    def get_dev_bounds(self):
        return self.dev_bounds
    
    def get_delta_0(self):
        return self.delta_0
    
    def observation_space(self):
        env = self.instantiate(self.delta_0)[0]
        return env.observation_space


class SafetyProp(STLEvaluator):
    def prop(self):
        x = stl.Predicate('x')
        return stl.Always( x < 0.7 )
    
    def build_signal(self, record, time_index):
        return {
            'x': stl.Signal(np.abs(record[:, 0]), time_index)
        }
