from types import SimpleNamespace

import numpy as np
import signal_tl as stl
from gym.envs.classic_control import CartPoleEnv

from robustness.analysis.stl import STLEvaluator, STLEvaluator2
from robustness.analysis.utils import normalize
from robustness.envs import DeviatableEnv


class ParamCartPoleEnv(CartPoleEnv):
    def __init__(self, masscart = 1.0, masspole = 0.1, length = 0.5, force_mag = 10.0):
        super().__init__()
        
        self.spec = SimpleNamespace()
        self.spec.id = f"ParamCartPole-{masscart:.3f}-{masspole:.3f}-{length:.3f}-{force_mag:.3f}"
        
        self.gravity = 9.8
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = length  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau = 0.02  # seconds between state updates
    
    def reset_to(self, state, seed=None):
        self.seed(seed)
        self.state = state
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)


class DevCartPole(DeviatableEnv):
    def __init__(self, masses, forces, delta_0=(1.0, 10.0)):
        assert masses[0] <= delta_0[0] <= masses[1] and forces[0] <= delta_0[1] <= forces[1], \
            'delta_0 is not in the domain of the deviation'

        self.masses = masses
        self.forces = forces
        self.x0_bounds = np.repeat([[-0.05, 0.05]], 4, axis=0)
        self.delta_0 = np.array(delta_0)
        self.dev_bounds = np.array([self.masses, self.forces])

    def instantiate(self, delta):
        return ParamCartPoleEnv(masscart=delta[0], force_mag=delta[1]), self.x0_bounds
    
    def get_dev_bounds(self):
        return self.dev_bounds
    
    def get_delta_0(self):
        return self.delta_0
    
    def observation_space(self):
        env = self.instantiate(self.delta_0)[0]
        return env.observation_space


class SafetyProp(STLEvaluator):
    def __init__(self, pickle_safe=False):
        super().__init__(pickle_safe)

        obs_space = ParamCartPoleEnv().observation_space
        self.pos_range = np.asarray([obs_space.low[0], obs_space.high[0]])
        self.angle_range = np.asarray([obs_space.low[2], obs_space.high[2]])

    def prop(self):
        pos = stl.Predicate('pos')
        angle = stl.Predicate('angle')

        pos_threshold = normalize(2.4, self.pos_range)
        angle_threshold = normalize(12 * 2 * np.pi / 360, self.angle_range)

        return stl.Always( (pos < pos_threshold) & (angle < angle_threshold) )
    
    def build_signal(self, record, time_index):
        return {
            "pos": stl.Signal(
                normalize(np.abs(record[:, 0]), self.pos_range),
                time_index
            ),
            "angle": stl.Signal(
                normalize(np.abs(record[:, 2]), self.angle_range),
                time_index
            )
        }


class SafetyProp2(STLEvaluator2):
    def __init__(self):
        obs_space = ParamCartPoleEnv().observation_space
        self.pos_range = np.asarray([obs_space.low[0], obs_space.high[0]])
        self.angle_range = np.asarray([obs_space.low[2], obs_space.high[2]])

        self.pos_threshold = normalize(2.4, self.pos_range)
        self.angle_threshold = normalize(12 * 2 * np.pi / 360, self.angle_range)

    def eval_one_timepoint(self, obs):
        pos = normalize(obs[0], self.pos_range)
        angle = normalize(obs[2], self.angle_range)
        return np.min((self.pos_threshold - pos, self.angle_threshold - angle))
