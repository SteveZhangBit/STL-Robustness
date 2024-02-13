import logging
import math
from typing import Any
import rtamt
import numpy as np

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
#import staliro

CartpoleDataT = ModelResult[List[float], None]
# Create a mutated environment
from gym.envs.classic_control import CartPoleEnv
from types import SimpleNamespace


class MutatedCartPoleEnv(CartPoleEnv):
    def __init__(self, masscart = 1.0, masspole = 0.1, length = 0.5, force_mag = 10.0):
        super().__init__()
        
        self.spec = SimpleNamespace()
        self.spec.id = f"MutatedCartPole-{masscart:.3f}-{masspole:.3f}-{length:.3f}-{force_mag:.3f}"
        
        self.gravity = 9.8
        self.masscart = masscart #trying by 2 
        self.masspole = masspole
        self.total_mass = (self.masspole + self.masscart)
        self.length = length  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag #trying by 2)
        self.tau = 0.02  # seconds between state updates
        #self.render_mode = "human"    
    def reset_to(self, state, seed=None):
        self.state = state
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

obs_space = MutatedCartPoleEnv().observation_space
pos_range = np.asarray([obs_space.low[0], obs_space.high[0]])
angle_range = np.asarray([obs_space.low[2], obs_space.high[2]])


def normalize(x_scaled, bounds):
    if bounds.ndim == 1:
        return (x_scaled - bounds[0]) / (bounds[1] - bounds[0])
    else:
        return (x_scaled - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

@blackbox()
def cartmodel(static: Sequence[float], times: SignalTimes, signals: SignalValues) -> CartpoleDataT:
   global e 
   #e = MutatedCartPoleEnv()
#    print(e.force_mag)
#    print(e.masscart)
   st = e.reset_to(static)   # Prepare simulator given static parameters
   if os.path.exists('ppocart1.zip'):  
    mod = PPO.load('ppocart1')
   simulator = e
   simulation_state = []
   simulation_data = {"trajectories": None, "timestamps": None}
   i = 0 
   tim = []
   posit_signal = normalize(np.abs(st[0]), pos_range)
   angle_signal = normalize(np.abs(st[2]), angle_range)
   #simulation_state.append(st)
   simulation_state.append(np.array([posit_signal, angle_signal]))
   tim.append(i)
   done = False
   top_r=0
   while not done and i!=500:
       i+=1
       a,idk = mod.predict(st)
       st,r,done,info, d= simulator.step(a)
       #simulator.render()
       #s = simulator.step(a)
       #print(s)
       top_r+=r
       posit_signal = normalize(np.abs(st[0]), pos_range)
       angle_signal = normalize(np.abs(st[2]), angle_range)
       tim.append(i)
       #simulation_state.append(st)
       simulation_state.append(np.array([posit_signal, angle_signal]))
   #print(f'ran till{i}, {info}, {d}, {top_r}')
#    simulation_data["trajectories"] = simulation_state   # Simulate system given signal times and values
#    simulation_data["timestamps"] = list(enumerate(simulation_state)) 
    
   simulation_state = np.array(simulation_state)
   states = simulation_state.T
#    states = np.vstack(
#         (
#             simulation_state[:, 0],  # roll
#             simulation_state[:, 1],  # pitch
#             simulation_state[:, 2],  # yaw
#             simulation_state[:, 3],  # altitude
#         )
#     )
   
   simula = Trace(states=states.tolist(), times=tim)
   #print(tim)
   #print(simula)
   return BasicResult(simula)



pos_threshold = normalize(2.4, pos_range)
angle_threshold = normalize(12 * 2 * np.pi / 360, angle_range)
# print(pos_threshold)
# print(angle_threshold)
phi = "(always(x <= 0.75 and y <= 0.75))"
specification = RTAMTDense(phi, {"x": 0, "y": 1})

optimizer = DualAnnealing()
np.repeat([[-0.05, 0.05]], 4, axis=0)
options = Options(runs=1, iterations=50, interval=(0, 1), static_parameters=[(-0.05,0.05),(-0.05,0.05),(-0.05,0.05),(-0.05,0.05)])

@blackbox()
def top_layer(static: Sequence[float], times: SignalTimes, signals: SignalValues):
    global e 
    e = MutatedCartPoleEnv(masscart = static[0], force_mag=static[1])
    # print(static)
    top_result = staliro(cartmodel, specification, optimizer, options)
    for run in top_result.runs:
        for evaluation in run.history:
           print(f"lower Sample: {evaluation.sample} -> Cost: {evaluation.cost}")
    print('\n')
    top_best_run_ = worst_run(top_result)
    # top_best_sample = best_eval(top_best_run_).sample
    top_best_sample = worst_eval(top_best_run_).cost
    fin = np.array([[top_best_sample]])
    print(f'best cost is {top_best_sample} \n')
    top_tim =[]
    top_tim.append(0)
    top_simula = Trace(states=fin.tolist(), times=top_tim)
    #print(top_tim)
    #print(simula)
    return BasicResult(top_simula)

phi_new = "always (c <=0.0)"
top_specification = RTAMTDense(phi_new, {"c": 0})
top_optimizer = DualAnnealing()
top_options = Options(runs=1, iterations=50, interval=(0, 1), static_parameters=[(0.0,2.0),(0.0,20.0)])

if __name__ == "__main__":
    #result = staliro(cartmodel, specification, optimizer, options)
    result = staliro(top_layer, top_specification, top_optimizer, top_options)
    for run in result.runs:
        for evaluation in run.history:
           print(f"Sample: {evaluation.sample} -> Cost: {evaluation.cost}")

    # logging.basicConfig(level=logging.DEBUG)

    # result = staliro(nonlinear_model, specification, optimizer, options)
    # evalidk = []
    # for run in result.runs:
    #     evalidk.append(best_run(run))
    # best_run_ = best_run(evalidk)
    best_run_ = best_run(result)
    best_sample = best_eval(best_run_).sample
    # e = MutatedCartPoleEnv()
    en = MutatedCartPoleEnv(masscart = best_sample[0], force_mag=best_sample[1])
    #en.render_mode = "human"
    en.render_mode = "rgb_array"
    #st = en.reset_to(best_sample)   # Prepare simulator given static parameters
    sta = en.reset()
    sta = np.array(sta)
    #print(sta[0])
    fa = sta[0]
    st = en.reset_to((fa[0], fa[1], fa[2], fa[3]))
    if os.path.exists('ppocart1.zip'):  
        mod = PPO.load('ppocart1')
    simulator = en
    done = False
    i=0
    top_reward = 0
    frame = 0
    while not done and i!=500:
        i+=1
        a,idk = mod.predict(st)
        st,r,done,_, _ = simulator.step(a)
        rgb = simulator.render()
        (Image.fromarray(rgb, 'RGB')).save(os.path.join("trial/frame_"+str(frame)+".png"))
        frame += 1
        # print(st)
        top_reward+=r
        simulator.render()
    print(top_reward)
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