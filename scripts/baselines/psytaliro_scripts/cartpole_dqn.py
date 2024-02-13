import logging
import math
from typing import Any
import rtamt
import numpy as np
import os
import logging
import math
from typing import List, Sequence
import plotly.graph_objects as go
from staliro.core import BasicResult, ModelResult, Trace, best_eval, best_run, worst_eval, worst_run
from staliro.models import SignalTimes, SignalValues, blackbox
from staliro.optimizers import DualAnnealing
from staliro.options import Options
from staliro.specifications import TaliroPredicate, TpTaliro
from staliro.staliro import simulate_model, staliro
from staliro.specifications import RTAMTDense
from stable_baselines3 import PPO
import os
from staliro.core import best_eval, best_run
from PIL import Image
import gym
from robustness.agents.cartpole import DQN
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.cartpole import DevCartPole, SafetyProp, SafetyProp2
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

CartpoleDataT = ModelResult[List[float], None]



@blackbox()
def cartmodel(static: Sequence[float], times: SignalTimes, signals: SignalValues) -> CartpoleDataT:
   masses = [0.1, 2.0]
   forces = [1.0, 20.0]
   env = DevCartPole(masses, forces, (1.0, 10.0))
   agent = DQN('/usr0/home/parvk/cj_project/STL-Robustness/models/cartpole/best_dqn')
   phi = SafetyProp()
   episode_len = 200
   # set the deviation params first
   env, x0bounds = env.instantiate(static[0:2])
   # set the initial state after
   obs = env.reset_to(static[2:6])  
   simulation_state = []
   posit_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
   angle_signal = normalize(np.abs(obs[2]), np.asarray([env.observation_space.low[2], env.observation_space.high[2]]))
   simulation_state.append(np.array([posit_signal, angle_signal]))
   done = False
   for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action)
        posit_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
        angle_signal = normalize(np.abs(obs[2]), np.asarray([env.observation_space.low[2], env.observation_space.high[2]]))
        simulation_state.append(np.array([posit_signal, angle_signal]))
   simulation_state = np.array(simulation_state)
   states = simulation_state.T
   
   simulation_trace = Trace(states=states.tolist(), times= list(range(episode_len+1)))

   return BasicResult(simulation_trace)




if __name__ == "__main__":
    phi = "(always(x <= 0.75 and y <= 0.75))"
    specification = RTAMTDense(phi, {"x": 0, "y": 1})
    optimizer = DualAnnealing()
    options = Options(runs=1, iterations=50, interval=(0, 1), static_parameters=[(0.0,2.0),(0.0,20.0),(-0.05,0.05),(-0.05,0.05),(-0.05,0.05), (-0.05,0.05)])
    result = staliro(cartmodel, specification, optimizer, options)
    for run in result.runs:
        for evaluation in run.history:
           print(f"Sample: {evaluation.sample} -> Cost: {evaluation.cost}")
    best_sample = worst_eval(best_run(result)).sample
    print('here')
    print(best_sample)
    masses = [0.1, 2.0]
    forces = [1.0, 20.0]
    env = DevCartPole(masses, forces, (1.0, 10.0))
    agent = DQN('/usr0/home/parvk/cj_project/STL-Robustness/models/cartpole/best_dqn')
    phi = SafetyProp()
    episode_len = 200
    # set the deviation params first
    env, x0bounds = env.instantiate(best_sample[0:2])
    # set the initial state after
    obs = env.reset_to(best_sample[2:6]) 
    for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action) 
        env.render()
    # evalidk = []
    # for run in result.runs:
    #     evalidk.append(best_run(run))
    # best_run_ = best_run(evalidk)
    # best_run_ = best_run(result)
    # best_sample = best_eval(best_run_).sample
    # # e = MutatedCartPoleEnv()
    # en = MutatedCartPoleEnv(masscart = best_sample[0], force_mag=best_sample[1])
    # #en.render_mode = "human"
    # en.render_mode = "rgb_array"
    # #st = en.reset_to(best_sample)   # Prepare simulator given static parameters
    # sta = en.reset()
    # sta = np.array(sta)
    # #print(sta[0])
    # fa = sta[0]
    # st = en.reset_to((fa[0], fa[1], fa[2], fa[3]))
    # if os.path.exists('ppocart1.zip'):  
    #     mod = PPO.load('ppocart1')
    # simulator = en
    # done = False
    # i=0
    # top_reward = 0
    # frame = 0
    # while not done and i!=500:
    #     i+=1
    #     a,idk = mod.predict(st)
    #     st,r,done,_, _ = simulator.step(a)
    #     rgb = simulator.render()
    #     (Image.fromarray(rgb, 'RGB')).save(os.path.join("trial/frame_"+str(frame)+".png"))
    #     frame += 1
    #     # print(st)
    #     top_reward+=r
    #     simulator.render()
    # print(top_reward)
    #best_result = simulate_model(top_layer, options, best_sample)

    # figure = go.Figure()
    # fas = go.Figure()
    # figure.add_trace(
    #     go.Scatter(
    #         x=best_result.trace.times,
    #         y=best_result.trace.states[0],
    #         mode="lines",
    #         line_color="green",
    #         name="position",
    #     )
    # )
    # fas.add_trace(
    #     go.Scatter(
    #         x=best_result.trace.times,
    #         y=best_result.trace.states[1],
    #         mode="lines",
    #         line_color="blue",
    #         name="theta",
    #     )
    # )
    # figure.update_layout(xaxis_title="time (s)", yaxis_title="value (m)")
    # figure.add_hline(y=0.75, line_color="red")
    # figure.write_image("pos.jpeg")
    # fas.update_layout(xaxis_title="time (s)", yaxis_title="value (m)")
    # fas.add_hline(y=0.75, line_color="red")
    # fas.write_image("theta.jpeg")