import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
from util import Deque
from logger import Logger


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


class DDQN:
    def __init__(self, n_states, n_actions, config, seed):
        self.config = config
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.gamma = config.get('gamma',0.99)
        # epsilon-greedy
        self.frame_idx = 0
        self.epsilon_start = config.get('epsilon_start',1.0)
        self.epsilon_end = config.get('epsilon_end',0.01)
        self.epsilon_decay = config.get('epsilon_decay',500)
        # creat DDQN model
        self.update_freq = config.get('update_freq',512)
        self.batch_size = config.get('batch_size',256)
        self.lr=config.get('lr',0.00005)
        self.memory_size=config.get('memory_size',1e5)
        self.policy_net = MLP(n_states, n_actions, config.get('network_width', 128)).to(self.device)
        self.target_net = MLP(n_states, n_actions, config.get('network_width', 128)).to(self.device)
        # initialize target_net and policy_net with same parameters
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        optimizer_name = config.get('optimizer', 'sgd')
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.policy_net.parameters(),
                lr=self.lr,
                weight_decay=config.get('weight_decay', 0),
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=self.lr,
                weight_decay=config.get('weight_decay', 0),
            )
        else:
            raise Exception(f'Unsupported optimizer: {optimizer_name}')
        self.memory = Deque(capacity=int(self.memory_size))
        self.logger = None


    def continuous_action(self, act):
        ax = self.config.get('action_x_acc', 2)
        ay = self.config.get('action_y_acc', 2)
        action = [0, 0]
        # move forward
        if act == 0:
            action = [0, 0]
        # accelerate
        elif act == 1:
            action = [ax, 0]
        # brake
        elif act == 2:
            action = [-ax, 0]
        # turn right
        elif act == 3:
            action = [0, ay]
        # turn left
        elif act == 4:
            action = [0, -ay]
        return action

    def get_epsilon(self):
        return (
            self.epsilon_end +
            (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.frame_idx / self.epsilon_decay)
        )

    def get_norm(self):
        total = 0
        for param in self.policy_net.parameters():
            total += param.norm().item() ** 2
        return np.sqrt(total)

    def choose_action(self, state):
        self.frame_idx += 1
        epsilon = self.get_epsilon()
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        return action

    def set_logger(self, logger):
        self.logger = logger

    def log(self, message):
        if self.logger is not None:
            self.logger.log(message)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        data = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*data)

        state_batch = np.array(states)
        action_batch = np.array(actions)
        reward_batch = np.array(rewards, dtype=np.float32)
        next_state_batch = np.array(next_states)
        done_batch = np.array(dones, dtype=np.uint8)

        self.log('memory_sample')
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        self.log('tensor_preparation')
        # Double DQN
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        self.log('calculate_q_values')
        next_q_actions = torch.max(self.policy_net(next_state_batch), dim=1)[1].unsqueeze(1)
        self.log('calculate_next_q_actions')
        next_q_values = self.target_net(next_state_batch).gather(dim=1, index=next_q_actions).detach().squeeze(1)
        self.log('calculate_next_q_values')
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.log('calculate_loss')
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.log('optimizer_step')

        if self.frame_idx % self.update_freq == 0:  # update target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path)

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
