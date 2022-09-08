import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
from util import Deque
from util import load_offline_data
from logger import Logger
from alg.alg_base import Alg


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128], final_activation=None):
        super(MLP, self).__init__()

        if final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None
        dimensions = [input_dim] + hidden_dims + [output_dim]
        self.layers = [
            nn.Linear(dimensions[i], dimensions[i + 1])
            for i in range(len(dimensions) - 1)
        ]

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        if self.final_activation is not None:
            return self.final_activation(self.layers[-1](x))
        return self.layers[-1](x)


class COptiDICE(Alg):
    def __init__(self, n_states, n_actions, config):
        super(COptiDICE, self).__init__()

        self.config = config
        self.n_actions = n_actions

        self.gamma = config.get('gamma', 0.95)
        self.alpha = config.get('alpha', 1)
        self.batch_size = config.get('batch_size', 7)
        self.f_type = config.get('f_type', 'kl')
        self.num_costs = config.get('num_costs', 2) # TODO: this is hacky
        self.c_hats = torch.ones(self.num_costs) * config.get('cost_thresholds', 1)
        self.cost_ub_eps = config.get('cost_ub_epsilon', 0)

        self.nu_network = MLP(n_states, 1, [128, 128])
        self.chi_network = MLP(n_states, self.num_costs, [128, 128])
        self.policy_network = MLP(n_states, n_actions, [128, 128], 'softmax')

        self.lamb = torch.zeros(self.num_costs, requires_grad=True)
        self.tau = torch.zeros(self.num_costs, requires_grad=True)

        self.memory = load_offline_data(config['offline_data'], capacity=int(1e6))

    def f(self, x):
        if self.f_type == 'chisquare':
            return 0.5 * ((x - 1) ** 2)
        if self.f_type == 'softchi':
            return torch.where(
                x < 1,
                x * (torch.log(x + 1e-10) - 1) + 1,
                0.5 * ((x - 1) ** 2)
            )
        if self.f_type == 'kl':
            return x * torch.log(x + 1e-10)

        raise NotImplementedError('undefined f_type', self.f_type)

    def f_prime_inv(self, x):
        if self.f_type == 'chisquare':
            return x + 1
        if self.f_type == 'softchi':
            return torch.where(
                x < 0,
                torch.exp(torch.minimum(x, 0)),
                x + 1
            )
        if self.f_type == 'kl':
            return torch.exp(x - 1)

        raise NotImplementedError('undefined f_type', self.f_type)

    def update(self):
        s, a, r, c, s_next, dones = self.memory.sample(self.batch_size)
        s_0 = s # TODO: hack since we don't log s_0 yet.
        n = len(a)

        s = torch.tensor(np.array(s), device=self.device, dtype=torch.float)
        a = torch.tensor(a, device=self.device, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, device=self.device, dtype=torch.float).unsqueeze(1)

        c = [[x['crash'], x['off_road']] for x in c]
        c = torch.tensor(np.array(c), device=self.device, dtype=torch.float)
        s_next = torch.tensor(np.array(s_next), device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)
        s_0 = torch.tensor(np.array(s_0), device=self.device, dtype=torch.float)

        gamma = self.gamma
        alpha = self.alpha

        nu = self.nu_network(s)
        nu_next = self.nu_network(s_next)
        nu_0 = self.nu_network(s_0)

        lamb = torch.clip(torch.exp(self.lamb), 0, 1e3)
        chi = self.chi_network(s)

        e_nu_lamb = (
            r - torch.sum(c * lamb.detach(), axis=-1, keepdim=True)
            + self.gamma * nu_next
            - nu
        )

        w = torch.relu(self.f_prime_inv(e_nu_lamb / self.alpha))

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

        # [Eq 18]
        ell = (
            (1 - gamma) * chi_0
            + w.detach() * (
                c + gamma * chi_next - chi
            )
        )
        logits = ell / tau.detach()
        weights = torch.nn.functional.softmax(logits, dim=0) * n
        log_weights = torch.nn.functional.log_softmax(logits, dim=0) + np.log(n)
        kl_divergence = torch.mean(
            weights * log_weights - weights + 1, axis=0
        )
        cost_ub = torch.mean(weights * w.detach() * c, axis=0)
        chi_tau_loss = (
            torch.sum(torch.mean(weights * ell, axis=0))
            + torch.sum(-tau * (kl_divergence.detach() - self.cost_ub_eps))
        )

        # lambda loss [Eq 26]
        lamb_loss = torch.dot(
            lamb,
            self.c_hats - cost_ub.detach()
        )

        loss = nu_loss + lamb_loss + chi_tau_loss

        # policy loss

        p = self.policy_network(s).gather(1, a).flatten()
        policy_loss = -torch.mean(w.detach() * torch.log(p))


if __name__ == '__main__':
    import yaml
    with open('config/coptidice.yaml', 'r') as f:
        conf = yaml.safe_load(f)
    a = COptiDICE(35, 5, conf['alg'])
    a.update()
