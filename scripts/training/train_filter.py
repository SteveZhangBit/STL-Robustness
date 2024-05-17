import os
from hyperpol import HyperPolicy
import torch
from robustness.envs.lunar_lander import DevLunarLander, SafetyProp, SafetyProp2
import time
import numpy as np
import imageio
from stable_baselines3 import PPO as BaselinePPO
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class transfer_env():
    
    def __init__(self, dev, env) -> None:
        self.episode_len = 300
        self.dev = dev
        self.observation_space = env.observation_space()
        #env, x0bounds = env.instantiate([7.832101630613496,0.4323776925279945])
        self.env, x0bounds = env.instantiate(dev)
        self.parent_env = env
        self.action_space = self.env.action_space
        #env, x0bounds = env.instantiate([5.879106789841222, 0.5553259436583973])
        self.base_policy = BaselinePPO.load('/usr0/home/parvk/cj_project/STL-Robustness/models/lunar-lander/ppo.zip', self.env)  
        self.phi = SafetyProp()
        # # set the initial state after
        # obs = env.reset()

    def set_to_new(self, dev):
        self.env, x0bounds = self.parent_env.instantiate(dev)
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

def train(filter, fin, dev_file_path, opt):
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
    devs = np.tile(np.array([7.832101630613496,0.4323776925279945]), (10, 1))
    num_episodes = 10
    num_epochs = 10
    data = []
    episode_data = []
    obs_record= []
    rew_record =[ ]
    writer = SummaryWriter('runs/experiment_2')
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
                    action = fin.base_policy.predict(obs, deterministic=True)[0]
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

    print('finished training')
    writer.close()
    torch.save(filter.state_dict(), 'filter_weights.pth')


    

if __name__ == '__main__':
    winds = [0.0, 10.0]  
    turbulences = [0.0, 1.0]
    
    #instantiate transfer env
    fin = transfer_env(dev = [7.832101630613496,0.4323776925279945],env = DevLunarLander(winds, turbulences, (0.0, 0.0)))

    # load dev from the csv
    dev = np.array([7.832101630613496,0.4323776925279945])





    #instantiate the model correctly based on the environment
    dev_dim = dev.shape[0] 
    action_d = fin.action_space.shape[0] #env.action_space
    emb_dim = dev_dim #random and not used anywhere since i am not learning any embeddings anyways
    hid_dim = 256
    filter = HyperPolicy(dev_dim,action_d, action_d, emb_dim, hid_dim) 

    optimizer= torch.optim.Adam
    lr=0.001
    weight_decay=0.001
    opt = optimizer(filter.parameters(), lr=lr, weight_decay=weight_decay)
    dev_file_path =''
    # for loading
    # filter.load_state_dict(torch.load('filter_weights.pth'))
    train(filter, fin, dev_file_path, opt)

