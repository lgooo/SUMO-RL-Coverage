import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
from util import Deque
from logger import Logger
from alg.ddqn import DDQN

class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class SafeDDQN(DDQN):
    def __init__(self, n_states, n_actions, config):
        super(SafeDDQN, self).__init__(n_states, n_actions, config)

        # TODO: make a safety class with net, optimizer, threshold, lambda
        self.safety_nets = {}
        self.safety_optimizers = {}
        self.safety_thresholds = {}
        self.safety_lambdas = {}
        optimizer_name = config.get('optimizer', 'sgd')
        for k in config['safety'].keys():
            self.safety_nets[k] = MLP(
                n_states, n_actions, config.get('network_width', 128)
            ).to(self.device)
            if optimizer_name == 'sgd':
                self.safety_optimizers[k] = optim.SGD(
                    self.safety_nets[k].parameters(),
                    lr=self.lr,
                    weight_decay=config.get('weight_decay', 0),
                )
            elif optimizer_name == 'adam':
                self.safety_optimizers[k] = optim.Adam(
                    self.safety_nets[k].parameters(),
                    lr=self.lr,
                    weight_decay=config.get('weight_decay', 0),
                )
            else:
                raise Exception(f'Unsupported optimizer: {optimizer_name}')
            self.safety_thresholds[k] = config['safety'][k]['threshold']
            self.safety_lambdas[k] = config['safety'][k]['lambda']

        self.initial_state_memory = Deque(capacity=int(self.memory_size))

        safety_config = self.config.get('safety', {})
        self.lambda_update_start_episode = config.get('lambda_update_start_episode', 100)
        self.lambda_learning_rate = config.get('lambda_learning_rate', 0.01)
        self.lambda_update_freq = safety_config.get('lambda_update_freq', 100)

    def calculate_reward(self, reward, safety):
        R = self.config.get('reward', {})
        ret = reward
        for k, v in self.safety_lambdas.items():
            ret -= v * safety[k]
        return ret

    def cost_network_update(self, state_batch, action_batch, next_state_batch, next_q_actions, safety_batches, done_batch):
        for k, net in self.safety_nets.items():
            values = net(state_batch).gather(dim=1, index=action_batch)
            next_values = net(next_state_batch).gather(dim=1, index=next_q_actions).detach().squeeze(1)
            expected_values = safety_batches[k] + self.gamma * next_values * (1 - done_batch)
            loss = nn.MSELoss()(values, expected_values.unsqueeze(1))
            self.safety_optimizers[k].zero_grad()
            loss.backward()
            for param in net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.safety_optimizers[k].step()

        if self.frame_idx % self.lambda_update_freq == 0:
            self.update_lambdas()

    def update_lambdas(self):
        if len(self.initial_state_memory) < self.lambda_update_start_episode:
            return
        data = self.initial_state_memory._buffer[0:len(self.initial_state_memory)]
        states, actions = zip(*data)
        state_batch = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(1)

        for k in self.safety_lambdas.keys():
            values = self.safety_nets[k](state_batch).gather(dim=1, index=action_batch)
            self.safety_lambdas[k] += self.lambda_learning_rate * (values.mean().item() - self.safety_thresholds[k])
            self.safety_lambdas[k] = max(0, self.safety_lambdas[k])

    def log_tensorboard(self, writer, epi):
        for k, v in self.safety_lambdas.items():
            writer.add_scalar(f'data/lambda_{k}', v, epi)

    def observe(self, experience: Experience):
        self.memory.append(experience)