import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
from util import ExperienceReplay


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
    def __init__(self, n_states, n_actions):
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.95
        # epsilon-greedy
        self.frame_idx = 0
        self.epsilon_start = 0.90
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        # creat DDQN model
        self.update_freq = 10
        self.batch_size = 64
        self.policy_net = MLP(n_states, n_actions).to(self.device)
        self.target_net = MLP(n_states, n_actions).to(self.device)
        # initialize target_net and policy_net with same parameters
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.001)
        self.memory = ExperienceReplay(capacity=1e5)

    def continuous_action(self, act):
        action = [0, 0]
        # move forward
        if act == 0:
            action = [0, 0]
        # accelerate
        elif act == 1:
            action = [2, 0]
        # brake
        elif act == 2:
            action = [-2, 0]
        # turn right
        elif act == 3:
            action = [0, 2]
        # turn left
        elif act == 4:
            action = [0, -2]
        return action

    def choose_action(self, state):
        self.frame_idx += 1
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.frame_idx / self.epsilon_decay)
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        # Double DQN
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        next_q_actions = torch.max(self.policy_net(next_state_batch), dim=1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_state_batch).gather(dim=1, index=next_q_actions).detach().squeeze(1)
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.frame_idx % self.update_freq == 0:  # update target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
