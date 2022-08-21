import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
from util import Deque
from logger import Logger
from alg.alg_base import Alg


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128], final_activation=None):
        super(MLP, self).__init__()

        self.final_activation = final_activation
        dimensions = [input_dim] + hidden_dims + [output_dim]
        self.layers = [
            nn.Linear(dimensions[i], dimensions[i + 1])
            for i in range(len(dimensions) - 1)
        ]

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        if self.final_activation == 'softmax':
            return nn.Softmax(self.layers[-1](x))
        return self.layers[-1](x)


class COptiDICE(Alg):
    def __init__(self, n_states, n_actions, config):
        self.config = config
        self.n_actions = n_actions

        self.memory_size = config.get('memory_size', 1e5)

        self.nu_network = MLP(n_states, 1, [128, 128])
        self.chi_network = MLP(n_states, 1, [128, 128])
        self.policy_network = MLP(n_states, n_actions, [128, 128], 'softmax')

        self.memory = Deque(capacity=int(self.memory_size), multi_steps=1)

    def update(self):
        # nu loss
        """
        nu_loss = (1 - gamma) * jnp.mean(f_init['nu'])
        nu_loss += -alpha * jnp.mean(f_fn(w_sa))
        nu_loss += jnp.mean(w_sa * e_nu_lamb)
        nu_loss += gradient_penalty * jnp.mean(jax.nn.relu(nu_grad_norm - 5)**2)
        """
        pass
