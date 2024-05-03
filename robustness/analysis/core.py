import numpy as np
import csv
import os
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
    
    def save_data(self, delta, obs, action, robustness,env_name, episode_len,eval_id,save_best, type_traj):
        '''this function saves all the traces found during a run of a system evaluation'''
        os.makedirs(f'../../traces/{env_name}', exist_ok=True)
        if save_best is False:
            if type_traj=='violated':
                os.makedirs(f'../../traces/{env_name}/violated', exist_ok=True)
                file_path = f'../../traces/{env_name}/violated/trace_{eval_id}_data.csv' 
            else:
                os.makedirs(f'../../traces/{env_name}/satisfied', exist_ok=True)
                file_path = f'../../traces/{env_name}/satisfied/trace_{eval_id}_data.csv'
        else:
            if type_traj=='violated':
                os.makedirs(f'../../traces/{env_name}/delta0/violated', exist_ok=True)
                file_path = f'../../traces/{env_name}/delta0/violated/trace_{eval_id}_data.csv' 
            else:
                os.makedirs(f'../../traces/{env_name}/delta0/satisfied', exist_ok=True)
                file_path = f'../../traces/{env_name}/delta0/satisfied/trace_{eval_id}_data.csv'
        
        # print(obs.shape)
        # print(action.shape)
        if len(action.shape)==1:
            action = action.reshape((-1,1))
        # print(action.shape)
        fin = np.concatenate((obs, action), axis=1)
        delta = delta.T
        delta_repeat = np.tile(delta, (len(obs), 1))
        rob = np.array([robustness]).reshape(1,1)
        rob_repeat = np.tile(rob, (episode_len+1, 1))
        temp_trace = np.concatenate((delta_repeat, fin), axis=1)
        final_trace = np.concatenate((rob_repeat, temp_trace), axis=1)
        # Open the CSV file in write mode
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['Robustness', ' Delta', ' States', ' Actions'])

            # Write the data rows
            for row in final_trace:
                writer.writerow(row)
    
    def _eval_trace(self, x0, env, agent, delta, save_best):
        # added delta to the function call for logging traces
        space = env.observation_space
        episode_len = self._options['episode_len']
        eval_id = np.random.randint(0,100000)
        obs = env.reset_to(x0)
        env_id = env.spec.id # for creating traces for different envs
        env_name = env_id.split('-')[0]
        obs_record = [obs]
        reward_record = [0]
        action_array = []
        agent.reset()
        for _ in range(episode_len):
            action = agent.next_action(obs)
            action_array.append(action)
            obs, reward, _, _ = env.step(action)
            obs_record.append(np.clip(obs, space.low, space.high))
            reward_record.append(reward)
            # import time
            # time.sleep(0.05)
        action_array.append(env.action_space.sample())
        score = self.phi.eval_trace(np.array(obs_record), np.array(reward_record))
        # print(score)
        # print('\n\n')
        # commenting this part out because now i am not trying to save the trajs now
        # print(np.array(action_array).shape)
        # if score < 0:
        #     self.save_data(delta, np.array(obs_record),np.array(action_array), score,env_name, episode_len,eval_id, save_best, type_traj='violated')
        # else:
        #     self.save_data(delta, np.array(obs_record),np.array(action_array), score,env_name, episode_len,eval_id, save_best, type_traj='safe')
        # if score < 0:
        #     self.save_data(delta, np.array(obs_record),np.array(action_array), score,env_name, episode_len,eval_id, save_best, type_traj='violated')
        # # new code (returns only score)
        if save_best is False:
            return score
        else:
            return score, np.array(obs_record)
        
        # # old code
        # return score

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
