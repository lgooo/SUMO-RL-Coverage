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

        self.off_road_net = MLP(n_states, n_actions, config.get('network_width', 128)).to(self.device)
        self.crash_net = MLP(n_states, n_actions, config.get('network_width', 128)).to(self.device)

        optimizer_name = config.get('optimizer', 'sgd')
        if optimizer_name == 'sgd':
            self.off_road_optimizer = optim.SGD(
                self.off_road_net.parameters(),
                lr=self.lr,
                weight_decay=config.get('weight_decay', 0),
            )
            self.crash_optimizer = optim.SGD(
                self.crash_net.parameters(),
                lr=self.lr,
                weight_decay=config.get('weight_decay', 0),
            )
        elif optimizer_name == 'adam':
            self.off_road_optimizer = optim.Adam(
                self.off_road_net.parameters(),
                lr=self.lr,
                weight_decay=config.get('weight_decay', 0),
            )
            self.crash_optimizer = optim.Adam(
                self.crash_net.parameters(),
                lr=self.lr,
                weight_decay=config.get('weight_decay', 0),
            )
        else:
            raise Exception(f'Unsupported optimizer: {optimizer_name}')

        self.initial_state_memory = Deque(capacity=int(self.memory_size))

        safety_config = self.config.get('safety', {})
        self.lambda_update_start_time = safety_config.get('min_num_initial_states_for_lambda_update', 100)
        self.alpha = safety_config.get('alpha', 0.01)
        self.crash_threshold = safety_config.get('crash_threshold', 0.01)
        self.off_road_threshold = safety_config.get('off_road_threshold', 0.01)
        self.lambda_off_road = safety_config.get('lambda_off_road', 1)
        self.lambda_crash = safety_config.get('lambda_crash', 1)
        self.lambda_update_freq = safety_config.get('lambda_update_freq', 100)

    def calculate_reward(self, reward, off_road, crash):
        R = self.config.get('reward', {})
        return (
            reward
            - self.lambda_off_road * off_road
            - self.lambda_crash * crash
        )

    def cost_network_update(self, state_batch, action_batch, next_state_batch, next_q_actions, off_road_batch, crash_batch, done_batch):
        off_road_values = self.off_road_net(state_batch).gather(dim=1, index=action_batch)
        next_off_road_values = self.off_road_net(next_state_batch).gather(dim=1, index=next_q_actions).detach().squeeze(1)
        expected_off_road_values = off_road_batch + self.gamma * next_off_road_values * (1 - done_batch)
        loss = nn.MSELoss()(off_road_values, expected_off_road_values.unsqueeze(1))
        self.off_road_optimizer.zero_grad()
        loss.backward()
        for param in self.off_road_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.off_road_optimizer.step()

        crash_values = self.crash_net(state_batch).gather(dim=1, index=action_batch)
        next_crash_values = self.crash_net(next_state_batch).gather(dim=1, index=next_q_actions).detach().squeeze(1)
        expected_crash_values = crash_batch + self.gamma * next_crash_values * (1 - done_batch)
        loss = nn.MSELoss()(crash_values, expected_crash_values.unsqueeze(1))
        self.crash_optimizer.zero_grad()
        loss.backward()
        for param in self.crash_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.crash_optimizer.step()

        if self.frame_idx % self.lambda_update_freq == 0:
            self.update_lambdas()

    def update_lambdas(self):
        if len(self.initial_state_memory) < self.lambda_update_start_time:
            return
        data = self.initial_state_memory._buffer[0:len(self.initial_state_memory)]
        states, actions = zip(*data)
        state_batch = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(1)

        off_road_values = self.off_road_net(state_batch).gather(dim=1, index=action_batch)
        crash_values = self.crash_net(state_batch).gather(dim=1, index=action_batch)

        self.lambda_crash += self.alpha * (crash_values.mean().item() - self.crash_threshold)
        self.lambda_off_road += self.alpha * (off_road_values.mean().item() - self.off_road_threshold)
        self.lambda_crash = max(0, self.lambda_crash)
        self.lambda_off_road = max(0, self.lambda_off_road)

    def log_tensorboard(self, writer, epi):
        writer.add_scalar('data/lambda_crash', self.lambda_crash, epi)
        writer.add_scalar('data/lambda_off_road', self.lambda_off_road, epi)