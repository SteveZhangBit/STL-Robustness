
import torch
import torch.nn as nn

import utils.utils as utils
from hypernet_model import HyperNetwork


class HyperPolicy(nn.Module):
    """
    Approximates the mapping R(\phi) -> \pi^* (a|s)
    """
    def __init__(self, input_param_dim, state_dim, action_dim, embed_dim, hidden_dim):
        super().__init__()

        self.hyper_policy = HyperNetwork(
            meta_v_dim=input_param_dim,
            z_dim=embed_dim,
            base_v_input_dim=action_dim,
            base_v_output_dim=action_dim,
            dynamic_layer_dim=hidden_dim,
            base_output_activation=torch.tanh
        )

    def forward(self, input_param, nom_action):
        z, safe_action = self.hyper_policy(input_param, nom_action)
        return safe_action