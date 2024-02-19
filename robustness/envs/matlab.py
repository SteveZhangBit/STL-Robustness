from robustness.envs import DeviatableEnv
import numpy as np
import matlab


class DevACC(DeviatableEnv):
    def __init__(self, eng, amax_lead_bounds, amin_lead_bounds, delta_0):
        self.eng = eng
        self.amax_lead_bounds = amax_lead_bounds
        self.amin_lead_bounds = amin_lead_bounds
        self.dev_bounds = np.array([amax_lead_bounds, amin_lead_bounds])
        self.delta_0 = np.array(delta_0)
    
    def instantiate(self, delta, agent):
        self.eng.workspace['name'] = f'ACC_{agent.type}_breach'
        if agent.type == 'RL':
            self.eng.workspace['agent'] = self.eng.load(agent.path)['agent']
        # the acceleration is constrained to the range [-1,1] (m/s^2).
        self.eng.workspace['amax_lead'] = delta[0]
        self.eng.workspace['amin_lead'] = delta[1]
        self.eng.InitACC(nargout=0)
        return self.eng.workspace['model']
    
    def get_dev_bounds(self):
        return self.dev_bounds
    
    def get_delta_0(self):
        return self.delta_0


class DevACC2(DeviatableEnv):
    def __init__(self, eng, amax_lead_bounds, amin_lead_bounds, mass_bounds, delta_0):
        self.eng = eng
        self.amax_lead_bounds = amax_lead_bounds
        self.amin_lead_bounds = amin_lead_bounds
        self.mass_bounds = mass_bounds
        self.dev_bounds = np.array([amax_lead_bounds, amin_lead_bounds, mass_bounds])
        self.delta_0 = np.array(delta_0)
    
    def instantiate(self, delta, agent):
        self.eng.workspace['name'] = f'ACC_{agent.type}_breach'
        if agent.type == 'RL':
            self.eng.workspace['agent'] = self.eng.load(agent.path)['agent']
        # the acceleration is constrained to the range [-1,1] (m/s^2).
        self.eng.workspace['amax_lead'] = delta[0]
        self.eng.workspace['amin_lead'] = delta[1]
        # set G_ego = tf(1, [M * 0.5, M * 1, 0]);
        numerator, denominator = matlab.double([1.0]), matlab.double([delta[2] * 0.5, delta[2] * 1.0, 0.0])
        self.eng.workspace['G_ego'] = self.eng.tf(numerator, denominator)
        self.eng.InitACC(nargout=0)
        return self.eng.workspace['model']
    
    def get_dev_bounds(self):
        return self.dev_bounds
    
    def get_delta_0(self):
        return self.delta_0


class DevLKA(DeviatableEnv):
    def __init__(self, eng, turn_pos1, turn_pos2, delta_0) -> None:
        self.eng = eng
        self.turn_pos1 = turn_pos1
        self.turn_pos2 = turn_pos2
        self.dev_bounds = np.array([turn_pos1, turn_pos2])
        self.delta_0 = np.array(delta_0)
    
    def instantiate(self, delta, agent):
        self.eng.workspace['name'] = f'LKA_{agent.type}_breach'
        if agent.type == 'RL':
            self.eng.workspace['agent'] = self.eng.load(agent.path)['agent']
        else:
            self.eng.workspace['agent'] = 0
        self.eng.workspace['turn_pos1'] = delta[0]
        self.eng.workspace['turn_pos2'] = delta[1]
        self.eng.InitLKA(nargout=0)
        return self.eng.workspace['model']

    def get_dev_bounds(self):
        return self.dev_bounds
    
    def get_delta_0(self):
        return self.delta_0


class DevWTK(DeviatableEnv):
    def __init__(self, eng, in_rates, out_rates, delta_0):
        self.eng = eng
        self.in_rates = in_rates
        self.out_rates = out_rates
        self.dev_bounds = np.array([in_rates, out_rates])
        self.delta_0 = np.array(delta_0)
    
    def instantiate(self, delta, agent):
        self.eng.workspace['name'] = f'WTK_{agent.type}_breach'
        if agent.type == 'RL':
            self.eng.workspace['agent'] = self.eng.load(agent.path)['agent']
        else:
            self.eng.workspace['agent'] = 0
        self.eng.workspace['in_rate'] = delta[0]
        self.eng.workspace['out_rate'] = delta[1]
        self.eng.InitWTK(nargout=0)
        return self.eng.workspace['model']
    
    def get_dev_bounds(self):
        return self.dev_bounds
    
    def get_delta_0(self):
        return self.delta_0


class DevWTKBaseline(DevWTK):
    def instantiate(self, delta, agent):
        self.eng.workspace['name'] = f'WTK_{agent.type}_breach'
        if agent.type == 'RL':
            self.eng.workspace['agent'] = self.eng.load(agent.path)['agent']
        else:
            self.eng.workspace['agent'] = 0
        self.eng.workspace['in_rate'] = delta[0]
        self.eng.workspace['out_rate'] = delta[1]
        self.eng.workspace['in_rate_range'] = matlab.double(self.in_rates)
        self.eng.workspace['out_rate_range'] = matlab.double(self.out_rates)
        self.eng.InitWTK_baseline(nargout=0)
        return self.eng.workspace['model']


class DevAFC(DeviatableEnv):
    def __init__(self, eng, MAF_sensor_tols, AF_sensor_tols, delta_0):
        self.eng = eng
        self.MAF_sensor_tols = MAF_sensor_tols
        self.AF_sensor_tols = AF_sensor_tols
        self.dev_bounds = np.array([MAF_sensor_tols, AF_sensor_tols])
        self.delta_0 = np.array(delta_0)
    
    def instantiate(self, delta, agent):
        self.eng.workspace['name'] = f'AFC_{agent.type}_breach'
        if agent.type == 'RL':
            self.eng.workspace['agent'] = self.eng.load(agent.path)['agent']
        else:
            self.eng.workspace['agent'] = 0
        self.eng.workspace['MAF_sensor_tol'] = delta[0]
        self.eng.workspace['AF_sensor_tol'] = delta[1]
        self.eng.InitAFC(nargout=0)
        return self.eng.workspace['model']

    def get_dev_bounds(self):
        return self.dev_bounds
    
    def get_delta_0(self):
        return self.delta_0
