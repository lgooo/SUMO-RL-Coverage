import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
from util import Deque
from util import Experience
from logger import Logger
from alg.alg_base import Alg


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


class DDQN(Alg):
    def __init__(self, n_states, n_actions, config):
        super(DDQN, self).__init__()

        self.config = config
        self.n_actions = n_actions
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
        self.multi_steps=config.get('multi_steps', 1)
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
        if self.multi_steps > 1:
            self.short_memory = Deque(capacity=self.multi_steps)


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

    def calculate_reward(self, reward, safety):
        R = self.config.get('reward', {})
        ret = reward
        for k, v in safety.items():
            ret -= v * R.get(k, 0)
        return ret

    def cost_network_update(self, state_batch, action_batch, next_state_batch, next_q_actions, safety_batches, done_batch):
        pass

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        experiences = self.memory.sample(self.batch_size)
        extract_fields = lambda experience: (
            experience.obs,
            experience.action,
            experience.reward,
            experience.safety,
            experience.next_obs,
            experience.done,
        )
        states, actions, rewards, safety_data, next_states, dones = \
            list(zip(*map(extract_fields, experiences)))

        self.log('memory_sample')
        safety_batches = {}
        for k in safety_data[0].keys():
            safety_batches[k] = torch.tensor(
                list(map(lambda x: x[k], safety_data)),
                device=self.device,
                dtype=torch.float)
        self.log('zip_data')

        state_batch = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = self.calculate_reward(
            torch.tensor(rewards, device=self.device, dtype=torch.float),
            safety_batches,
        )
        next_state_batch = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(dones, device=self.device, dtype=torch.float)
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

        self.cost_network_update(
            state_batch,
            action_batch,
            next_state_batch,
            next_q_actions,
            safety_batches,
            done_batch)

        return loss.item()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path)

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)

    def log_tensorboard(self, writer, epi):
        pass

    def observe(self, experience: Experience):
        if self.multi_steps == 1:
            self.memory.append(experience)
        else:
            self.short_memory.append(experience)
            if len(self.short_memory) >= self.multi_steps:
                past_experience = self.short_memory[-self.multi_steps]
                new_experience = Experience(
                    initial_state=past_experience.initial_state,
                    obs=past_experience.obs,
                    action=past_experience.action,
                    reward=sum([
                        (self.gamma ** n) * self.short_memory[-(self.multi_steps - n)].reward
                        for n in range(self.multi_steps)
                    ]),
                    safety=past_experience.safety, # TODO: discounted sum for safety signals
                    next_obs=experience.next_obs,
                    done=experience.done,
                )
                self.memory.append(new_experience)

    def new_episode(self):
        if self.multi_steps > 1:
            self.short_memory.clear()
