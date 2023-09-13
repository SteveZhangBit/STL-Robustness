import numpy as np

from robustness.agents import Agent
from robustness.envs import DeviatableEnv


class DistanceEvaluator:
    def __init__(self, env: DeviatableEnv):
        self.env = env
    
    def eval_dist(self, delta):
        '''Return a real value of the distance measurement.'''
        raise NotImplementedError()


class TraceEvaluator:
    def eval_trace(self, obs_record, reward_record):
        '''
        Evaluates a given trajectory given the observation record and
        reward record.
        '''
        raise NotImplementedError()


class Problem:
    def __init__(self, env: DeviatableEnv, agent: Agent, phi: TraceEvaluator, dist: DistanceEvaluator):
        self.env = env
        self.agent = agent
        self.phi = phi
        self.dist = dist


class SystemEvaluator:
    def __init__(self, phi: TraceEvaluator, opts=None):
        self.phi = phi
        self._options = {
            # number of times to restart when evaluating (falsifying) a system against a property.
            'restarts': 0,
            # timeout in minutes for evaluating a system against a property.
            'timeout': np.inf,
            # max number of samples when evaluating (falsifying) a system against a property.
            'evals': 100,
            # length of the episode (trajectory) to simulate when evaluating a system at a certain
            # initial state.
            'episode_len': 200,
        }
        if opts is not None:
            self._options.update(opts)
    
    def set_options(self, opts):
        self._options.update(opts)
    
    def options(self):
        return self._options.copy()

    def eval_sys(self, delta, problem: Problem):
        '''Return a tuple of <result, x0?>'''
        raise NotImplementedError()
    
    def _eval_trace(self, x0, env, agent):
        space = env.observation_space
        episode_len = self._options['episode_len']

        obs = env.reset_to(x0)
        obs_record = [obs]
        reward_record = [0]

        agent.reset()
        for _ in range(episode_len):
            action = agent.next_action(obs)
            obs, reward, _, _ = env.step(action)
            obs_record.append(np.clip(obs, space.low, space.high))
            reward_record.append(reward)
        
        return self.phi.eval_trace(np.array(obs_record), np.array(reward_record))


class Solver:
    def __init__(self, sys_evaluator: SystemEvaluator, opts=None):
        self.sys_evaluator = sys_evaluator
        self._options = {
            # number of times to restart when search any or a minimum counterexample.
            'restarts': 2,
            # timeout in minutes for each trial of counterexample search.
            'timeout': np.inf,
            # max number of samples when search a counterexample.
            'evals': 100,
        }
        if opts is not None:
            self._options.update(opts)
    
    def set_options(self, opts):
        self._options.update(opts)
    
    def options(self):
        return self._options.copy()

    def any_unsafe_deviation(self, problem: Problem, boundary=None, constraints=None):
        '''Return a tuple of <delta?, dist?, x0?>'''
        raise NotImplementedError()
    
    def min_unsafe_deviation(self, problem: Problem, boundary=None, sample_logger=None):
        '''Return a tuple of <min_delta?, min_dist?, x0?>'''
        raise NotImplementedError()
