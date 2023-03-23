from robustness.envs import DeviatableEnv
import numpy as np


class DevACC(DeviatableEnv):
    def __init__(self, eng, x0_lead_bounds, v0_lead_bounds, delta_0=(50, 25)):
        self.eng = eng
        self.x0_lead_bounds = x0_lead_bounds
        self.v0_lead_bounds = v0_lead_bounds
        self.dev_bounds = np.array([x0_lead_bounds, v0_lead_bounds])
        self.delta_0 = np.array(delta_0)
    
    def instantiate(self, delta, agent):
        self.eng.workspace['name'] = f'ACC_{agent.type}_breach'
        if agent.type == 'RL':
            self.eng.workspace['agent'] = self.eng.load(agent.path)['agent']
        # the velocity and the position of lead car
        self.eng.workspace['x0_lead'] = delta[0]
        self.eng.workspace['v0_lead'] = delta[1]
        self.eng.InitACC(nargout=0)
        return self.eng.workspace['model']
    
    def get_dev_bounds(self):
        return self.dev_bounds
    
    def get_delta_0(self):
        return self.delta_0
    
    def observation_space(self):
        raise NotImplementedError()
