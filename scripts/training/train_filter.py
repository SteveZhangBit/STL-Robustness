import os
from hyperpol import HyperPolicy
import torch
# from robustness.envs.lunar_lander import DevLunarLander, SafetyProp, SafetyProp2
from robustness.envs.car_run import DevCarRun, SafetyProp, SafetyProp2
import time
import numpy as np
import imageio
from stable_baselines3 import PPO as BaselinePPO
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import random 
from robustness.agents.rsrl import PPOVanilla

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class transfer_env():
    
    def __init__(self, dev, env, episode_len, agent) -> None:
        self.episode_len = episode_len
        self.dev = dev
        self.observation_space = env.observation_space()
        self.env, x0bounds = env.instantiate(dev)
        self.parent_env = env
        self.action_space = self.env.action_space
        # self.base_policy = BaselinePPO.load('/usr0/home/parvk/cj_project/STL-Robustness/models/lunar-lander/ppo2.zip', self.env) 
        self.base_policy = agent 
        self.phi = SafetyProp()


    def set_to_new(self, dev, render=False):
        if render is True:
            self.env, x0bounds = self.parent_env.instantiate(dev, render=True)
        else:
            self.env, x0bounds = self.parent_env.instantiate(dev)
        #self.env, x0bounds = self.parent_env.instantiate([7.843638754739874,0.505699763079408])
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        return self.env




    # # Train the agent
    # agent.learn(total_timesteps=50000) 
    # frames = []
    # # # Save the trained model
    # # model.save("ppo_lunarlander")
    # obs = env.reset_to([-1.842157832288649,-1.4106696024537086])  
    # # # Load the trained model (optional)
    # for _ in range(episode_len):
    #     action = agent.predict(obs, deterministic=True)[0]
    #     obs, reward, _, _ = env.step(action)
    #     time.sleep(0.01) 
    #     #env.render()
    #     frames.append(env.render(mode="rgb_array"))

    # imageio.mimsave('lunar_lander_sat2.gif', frames, fps=30)

def train(filter, fin, dev_file_path, opt, env_name):
    '''
    here take the base policy, the hypernetwork, the deviation file, the env
    then 
    instantiate the env, 
    reset it, 
    get the base policy action, 
    pass it through the hypernetwork
    take a step in the environment
    compute the stl satisfaction score
    compute the l2 norm between the nominal action and the safe action
    run the loss.backward
    however, how do you do it for batch? is one step okay or should i accumulate over an episode?
    '''
    # load violating devs from a csv
    # currently just repeating values
    de = pd.read_csv(dev_file_path).to_numpy()
    devs = de[:,:2]
    num_episodes = 10
    num_epochs = 100
    # data = []
    # episode_data = []
    # obs_record= []
    # rew_record =[ ]
    writer = SummaryWriter('runs/experiment_car_run_final')
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        running_score = 0.0
        for i in tqdm(range(0, len(devs))):
            dev = devs[i]
            # set to this dev
            env = fin.set_to_new(dev)
            final_loss = 0.0
            opt.zero_grad()
            for episode in range(0,num_episodes):
                obs = env.reset()
                for j in range(0, fin.episode_len):
                    if isinstance(fin.base_policy, BaselinePPO):
                        action = fin.base_policy.predict(obs, deterministic=True)[0]
                    else:
                        action = fin.base_policy.next_action(obs)
                    dev_input = torch.tensor(dev, dtype=torch.float32)
                    nom_ac =  torch.tensor([action], dtype=torch.float32)
                    safe_ac = filter(dev_input, nom_ac)
                    
                    with torch.no_grad():
                        f = safe_ac.numpy()
                        # take the safe action, get the obs
                        obs, reward, _, _ = env.step(f[0])
                        # compute stl score
                        score = fin.phi.eval_trace(np.array([obs]), np.array([reward]))
                    squared_diff = (safe_ac - nom_ac) ** 2
                    # Compute the mean of the squared differences
                    action_penalty = torch.mean(squared_diff)
                    final_loss += -10*score + action_penalty

            final_loss.backward()
            opt.step()
            running_loss += final_loss
            running_score += score

            if i % 50 == 0:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 50}')
                writer.add_scalar('training loss', running_loss / 50, epoch * len(devs) + i)
                writer.add_scalar('score', running_score / 50, epoch * len(devs) + i)
                running_loss = 0.0
                running_score =0.0
                checkpoint_prefix = 'filter_checkpoint_'
                checkpoint_filename = checkpoint_prefix + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pth'
                checkpoint_path = os.path.join(f'model_weights/{env_name}/', checkpoint_filename)
                torch.save(filter.state_dict(), checkpoint_path)
                # torch.save(filter.state_dict(), f'model_weights/{}filter_weights.pth')

    print('finished training')
    writer.close()
    # torch.save(filter.state_dict(), 'model_weights/filter_weights.pth')


def eval(filter, fin, use_filter, dev_file_path):
    '''
    checks the policy performance on the deviations
    '''
    de = pd.read_csv(dev_file_path).to_numpy()
    devs = de[:,:2]
    num_episodes = 2
    data_store = []
    for i in tqdm(range(0, len(devs))):
            dev = devs[i]
            # print(dev)
            # set to this dev
            env = fin.set_to_new(dev, render=False)
            final_loss =0.0
            for episode in range(0,num_episodes):
                rewards = 0
                obs_record = []
                obs = env.reset()
                for j in range(0, fin.episode_len):
                    if isinstance(fin.base_policy, BaselinePPO):
                        action = fin.base_policy.predict(obs, deterministic=True)[0]
                    else:
                        action = fin.base_policy.next_action(obs)
                    if use_filter is True:
                        dev_input = torch.tensor(dev, dtype=torch.float32)
                        nom_ac =  torch.tensor([action], dtype=torch.float32)
                        safe_ac = filter(dev_input, nom_ac)
                        
                        with torch.no_grad():
                            f = safe_ac.numpy()
                        # take the safe action, get the obs
                        obs, reward, _, _ = env.step(f[0])
                    else:
                        obs, reward, _, _ = env.step(action)

                    obs_record.append(obs)
                    rewards+=reward
                    #env.render()

                       
                    # squared_diff = (safe_ac - nom_ac) ** 2
                    # # Compute the mean of the squared differences
                    # action_penalty = torch.mean(squared_diff)
                    # final_loss += action_penalty
                score = fin.phi.eval_trace(np.array([obs]), np.array([reward]))
                print(f'score is {score}')
                print(f'reward is {rewards}')
                #print(final_loss)
                data_store.append([score, rewards])
    
    print(data_store)

if __name__ == '__main__':
    # winds = [0.0, 10.0]  
    # turbulences = [0.0, 1.0]
    
    # #instantiate transfer env
    # fin = transfer_env(dev = [7.832101630613496,0.4323776925279945],env = DevLunarLander(winds, turbulences, (0.0, 0.0)))

    key ='eval' #use to switch between eval and trace
    load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_run_ppo_vanilla/model_save/model.pt'
    speed = [5.0, 35.0]
    steering = [0.2, 0.8]
    env = DevCarRun(load_dir, speed, steering)
    phi = SafetyProp()
    fin = transfer_env(dev = [0.2,5.0],env = DevCarRun(load_dir, speed, steering),episode_len=200, agent=PPOVanilla(load_dir))


    #instantiate the model correctly based on the environment
    dev_dim = 2
    action_d = fin.action_space.shape[0] #env.action_space
    emb_dim = dev_dim #random and not used anywhere since i am not learning any embeddings anyways
    hid_dim = 256
    filter = HyperPolicy(dev_dim,action_d, action_d, emb_dim, hid_dim) 

    optimizer= torch.optim.Adam
    lr=0.001
    weight_decay=0.001
    opt = optimizer(filter.parameters(), lr=lr, weight_decay=weight_decay)
    dev_file_path = 'violations/car_run.csv'
    # for loading
    # filter.load_state_dict(torch.load('filter_weights.pth'))
    if key == 'train':
        train(filter, fin, dev_file_path, opt, 'car_run')
    else:
        filter.load_state_dict(torch.load('model_weights/car_run/filter_checkpoint_20240517_132838.pth'))
        eval(filter, fin, False, dev_file_path)

