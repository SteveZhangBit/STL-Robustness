from rsrl.evaluator import Evaluator
from rsrl.util.logger import EpochLogger, setup_logger_kwargs
from rsrl.util.run_util import setup_eval_configs

from robustness.agents import Agent


class PPOVanilla(Agent):
    def __init__(self, load_dir):
        super().__init__()

        model_path, config = setup_eval_configs(load_dir)
        evaluator = Evaluator(**config, config_dict=config)
        evaluator._init_env()
        logger_kwargs = setup_logger_kwargs(evaluator.exp_name, evaluator.seed, data_dir=evaluator.data_dir,
                                            datestamp=False, use_tensor_board=False)
        evaluator.logger = EpochLogger(**logger_kwargs)
        evaluator._init_policy()
        evaluator.policy.load_model(model_path)
        
        self.policy = evaluator.policy

    def next_action(self, obs):
        _, action = self.policy.get_risk_estimation(obs)
        return action
    
    def reset(self):
        pass
