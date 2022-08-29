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

        self.gamma = config.get('gamma', 0.95)
        self.alpha = config.get('alpha', 1)
        self.memory_size = config.get('memory_size', 1e5)
        self.f_type = config.get('f_type', 'kl')
        self.num_costs = config.get('num_costs', 3) # TODO: this is hacky
        self.c_hats = torch.ones(self.num_costs) * config.get('cost_thresholds', 1)
        self.cost_ub_eps = config.get('cost_ub_epsilon', 0)

        self.nu_network = MLP(n_states, 1, [128, 128])
        self.chi_network = MLP(n_states, 1, [128, 128])
        self.policy_network = MLP(n_states, n_actions, [128, 128], 'softmax')

        self.lamb = torch.zeros(self.num_costs, requires_grad=True)
        self.tau = torch.zeros(self.num_costs, requires_grad=True)

        self.memory = Deque(capacity=int(self.memory_size), multi_steps=1)

    def f(self, x):
        if self.f_type == 'chisquare':
            return 0.5 * ((x - 1) ** 2)
        if self.f_type == 'softchi':
            return np.where(
                x < 1,
                x * (np.log(x + 1e-10) - 1) + 1,
                0.5 * ((x - 1) ** 2)
            )
        if self.f_type == 'kl':
            x * np.log(x + 1e-10)

        raise NotImplementedError('undefined f_type', self.f_type)

    def f_prime_inv(self, x):
        if self.f_type == 'chisquare':
            return x + 1
        if self.f_type == 'softchi':
            return np.where(
                x < 0,
                np.exp(np.minimum(x, 0)),
                x + 1
            )
        if self.f_type == 'kl':
            return np.exp(x - 1)

        raise NotImplementedError('undefined f_type', self.f_type)

    def update(self):
        s, a, r, c, s_next, dones, s_0 = self.memory.sample(self.batch_size)
        n = s.shape[0]

        nu_loss = (
            (1 - gamma) * torch.mean(self.nu_network(s_0))
            - alpha * torch.mean(self.f())
        )

        nu = self.nu_network(s)
        nu_next = self.nu_network(s_next)
        nu_0 = self.nu_network(s_0)

        lamb = torch.clip(torch.exp(self.lamb), 0, 1e3)
        chi = self.chi_network(s)

        e_nu_lamb = (
            r - torch.sum(c * lamb.detach(), axis=-1)
            + self.gamma * nu_next
            - nu
        )

        w = np.maximum(self.f_prime_inv(e_nu_lamb / self.alpha), 0)

        # nu loss [Eq 23]
        nu_loss = (
            (1 - gamma) * torch.mean(nu_0)
            - self.alpha * torch.mean(self.f(w))
            + torch.mean(w * e_nu_lamb)
        )

        # chi tau loss
        chi_0 = self.chi_network(s_0)
        chi = self.chi_network(s)
        chi_next = self.chi_network(s_next)

        tau = torch.exp(self.tau) + 1e-6

        ell = (
            (1 - self.gamma) * chi_0
            + w.detach()[:, None] * (
                c + self.gamma * chi_next - chi
            )
        )
        logits = ell / tau.detach()
        weights = torch.nn.functional.softmax(logits, axis=0) * n
        log_weights = torch.nn.functional.log_softmax(logits, axis=0) + torch.log(n)
        kl_divergence = torch.mean(
            weights * log_weights - weights + 1, axis=0
        )
        cost_ub = torch.mean(weights * w.detach()[:, None] * c, axis=0)
        chi_tau_loss = (
            torch.sum(torch.mean(weights * ell, axis=0))
            + torch.sum(-tau * (kl_divergence.detach() - self.cost_ub_eps))
        )

        # lambda loss [Eq 26]
        lamb_loss = torch.dot(
            lamb,
            self.c_hats - cost_ub.deatch()
        )

        loss = nu_loss + lamb_loss + chi_tau_loss


if __name__ == '__main__':
    a = COptiDICE(2, 2, {})
