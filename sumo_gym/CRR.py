import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import yaml
from sumo_gym import SumoGym
import numpy as np
import datetime
import argparse
from util import load_offline_data
import os

parser = argparse.ArgumentParser(
    description='test env for CRR')
parser.add_argument(
    '--config',
    default='config/crr.yaml',
    help='config file path')
parser.add_argument(
    '--num_episodes',
    type=int,
    default=50000,
    help='number of episodes to run'
)
parser.add_argument(
    '--render',
    action='store_true',
    default=False,
    help='whether to use gui rendering',
)
parser.add_argument(
    '--delta_t',
    type=float,
    default=0.1,
    help='simulation time step')
parser.add_argument(
    '--experiment',
    help='experiment name')
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='random seed',
)
args = parser.parse_args()

with open(args.config, 'r') as f:
    conf = yaml.safe_load(f)

env = SumoGym(
    config=conf['env'],
    delta_t=args.delta_t,
    render_flag=args.render,
    seed=args.seed
)

alg_config = conf['alg']
learning_rate = alg_config['lr']
gamma = alg_config['gamma']
beta = alg_config['beta']
n_updates = args.num_episodes
target_update_steps = alg_config['update_freq']
n_states = alg_config.get('n_states', 35)
n_actions = alg_config.get('n_actions', 5)
batch_size = alg_config['batch_size']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_width= alg_config.get('network_width', 128)

experiment_name = args.experiment
config_name = args.config.split('/')[-1].rsplit('.', 1)[0]
if not experiment_name:
    datestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    experiment_name = f'{config_name}_{datestamp}'
writer = SummaryWriter(f'runs/{experiment_name}')


class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.fc_a1 = nn.Linear(n_actions, network_width//2)
        self.fc_s1 = nn.Linear(n_states, network_width)
        self.fc_s2 = nn.Linear(network_width, network_width//2)
        self.fc2 = nn.Linear(network_width, network_width//2)
        self.fc3 = nn.Linear(network_width//2, 1)

    def forward(self, state, action):
        a = F.relu(self.fc_a1(action))
        s = torch.flatten(state, start_dim=1)
        s = F.relu(self.fc_s1(s))
        s = F.relu(self.fc_s2(s))
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, network_width)
        self.fc2 = nn.Linear(network_width, network_width)
        self.fc3 = nn.Linear(network_width, network_width)
        self.fc4 = nn.Linear(network_width, n_actions)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)


def continuous_action(act):
    ax = alg_config.get('action_x_acc', 2)
    ay = alg_config.get('action_y_acc', 2)
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


def test(actor):
    score = 0.0
    num_test = 3
    step = 0
    x = 0

    for _ in range(num_test):
        terminate = False
        s = env.reset()
        max_x = 0
        while not terminate:
            s = torch.tensor(s, device=device, dtype=torch.float32).unsqueeze(dim=0)
            action = actor(s)
            action = action.max(1)[1].item()
            next_obs, reward, safety, terminate, done, info = env.step(action=continuous_action(action))
            score += reward
            s = next_obs
            step += 1
            if len(s):
                max_x = env.ego_state['lane_x']
        x += max_x
        env.close()
    return score / num_test, x / num_test


def hard_update(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def action_matrix(a):
    actions = np.zeros((len(a), n_actions))
    for c, index in enumerate(a):
        actions[c][index] = 1
    actions = torch.tensor(actions, device=device, dtype=torch.float)
    return actions


def function_f_binary(q, pi, s, a):
    q_value = q(s, action_matrix(a))
    all_actions = np.diag(np.full(n_actions, 1))
    all_actions = torch.tensor(all_actions, device=device, dtype=torch.float)
    mean_estimate = []
    for state in s:
        state = torch.stack([state] * n_actions)
        r = q(state, all_actions)
        mean_estimate.append(torch.mean(r))
    mean_estimate = torch.stack(mean_estimate, dim=0).unsqueeze(1)
    A_mean = q_value - mean_estimate
    return (A_mean > 0)


def function_f_exp(q, pi, s, a):
    q_value = q(s, action_matrix(a))
    all_actions = np.diag(np.full(n_actions, 1))
    all_actions = torch.tensor(all_actions, device=device, dtype=torch.float)
    mean_estimate = []
    for state in s:
        state = torch.stack([state] * n_actions)
        r = q(state, all_actions)
        mean_estimate.append(torch.mean(r))
    mean_estimate = torch.stack(mean_estimate, dim=0).unsqueeze(1)
    A_mean = q_value - mean_estimate
    return torch.clamp(torch.exp(A_mean / beta), max=20.)


function_f = function_f_exp if alg_config['f'] == 'exp' else function_f_binary

if __name__ == '__main__':
    offline_data = load_offline_data("./data/dataset/beta80.txt")
    actor = Actor(n_states, n_actions).to(device)
    target_actor = Actor(n_states, n_actions).to(device)
    critic = Critic(n_states, n_actions).to(device)
    target_critic = Critic(n_states, n_actions).to(device)


    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    hard_update(actor, target_actor)
    hard_update(critic, target_critic)

    if not os.path.exists(f'data/{experiment_name}/model'):
        os.makedirs(f'data/{experiment_name}/model')

    for n in range(n_updates):
        experiences = offline_data.sample(batch_size)
        extract_fields = lambda experience: (
            experience.obs,
            experience.action,
            experience.reward,
            experience.next_obs,
            experience.done,
        )
        states, actions, rewards, next_states, dones = \
            list(zip(*map(extract_fields, experiences)))
        state_batch = torch.tensor(np.array(states), device=device, dtype=torch.float)
        action_batch = torch.tensor(actions, device=device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(np.array(rewards), device=device, dtype=torch.float).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(next_states), device=device, dtype=torch.float)

        # update actor with gradient
        loss_actor = -1 * torch.log(actor(state_batch).gather(dim=1, index=action_batch))  # log(pi(a_t|s_t))
        f = function_f(critic, actor, state_batch, action_batch)  # binary f function
        loss_actor = torch.mean(loss_actor * f)
        actor_optimizer.zero_grad()
        loss_actor.backward()
        for param in actor.parameters():
            param.grad.data.clamp_(-1, 1)
        actor_optimizer.step()

        # update critic with gradient
        q_values = critic(state_batch, action_matrix(action_batch))
        next_actions = target_actor(next_state_batch)
        next_actions = action_matrix(torch.argmax(next_actions, dim=1))
        expect_q_values = reward_batch + gamma * target_critic(next_state_batch, next_actions)
        critic_optimizer.zero_grad()
        loss_critic = nn.MSELoss()(q_values, expect_q_values)
        loss_critic.backward()
        for param in critic.parameters():
            param.grad.data.clamp_(-1, 1)
        critic_optimizer.step()

        writer.add_scalar('data/actor loss', loss_actor, n)
        writer.add_scalar('data/critic loss', loss_critic, n)

        if n > 0 and n % target_update_steps == 0:
            hard_update(actor, target_actor)
            hard_update(critic, target_critic)

            score, max_x = test(actor)
            writer.add_scalar('data/test score', score, n)
            writer.add_scalar('data/test step', max_x, n)

            actor_path = f'data/{experiment_name}/model/actor_{n}.pth'
            critic_path = f'data/{experiment_name}/model/critic_{n}.pth'
            torch.save(target_actor.state_dict(), actor_path)
            torch.save(target_critic.state_dict(), critic_path)
