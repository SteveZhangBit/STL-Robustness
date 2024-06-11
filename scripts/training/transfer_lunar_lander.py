from robustness.envs.lunar_lander import DevLunarLander, SafetyProp, SafetyProp2
import time
import imageio

winds = [0.0, 10.0]
turbulences = [0.0, 1.0]
env = DevLunarLander(winds, turbulences, (5.0, 0.5))
from stable_baselines3 import PPO as BaselinePPO


episode_len = 300
env, x0bounds = env.instantiate([7.843638754739874,0.505699763079408])
#env, x0bounds = env.instantiate([6.8851463628585545,0.6669314213708963])
#env, x0bounds = env.instantiate([5.879106789841222, 0.5553259436583973])
agent = BaselinePPO.load('/usr0/home/parvk/cj_project/STL-Robustness/models/lunar-lander/ppo2.zip', env)  

# # set the initial state after
obs = env.reset_to([-0.00034799575223587453,1.40477454662323])  
#obs = env.reset()


# Train the agent
#agent.learn(total_timesteps=50000) 
frames = []
# # Save the trained model
# model.save("ppo_lunarlander")
# obs = env.reset_to([-1.842157832288649,-1.4106696024537086])  
# # Load the trained model (optional)
for _ in range(episode_len):
    action = agent.predict(obs, deterministic=True)[0]
    obs, reward, _, _ = env.step(action)
    #time.sleep(0.01) 
    env.render()
    frames.append(env.render(mode="rgb_array"))

imageio.mimsave('lunar_lander_sat2new.gif', frames, fps=30)