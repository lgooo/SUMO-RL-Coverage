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
    description='test env for simple offline CMDP')
parser.add_argument(
    '--config',
    default='config/cmdp.yaml',
    help='config file path')
parser.add_argument(
    '--num_episodes',
    type=int,
    default=10000,
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
n_updates = args.num_episodes
n_states = alg_config.get('n_states', 35)
n_actions = alg_config.get('n_actions', 5)
n_action_space = 2
batch_size = alg_config['batch_size']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_width = alg_config.get('network_width', 128)

experiment_name = args.experiment
config_name = args.config.split('/')[-1].rsplit('.', 1)[0]
if not experiment_name:
    datestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    experiment_name = f'{config_name}_{datestamp}'
writer = SummaryWriter(f'runs/{experiment_name}')


class Q(nn.Module):
    def __init__(self, n_states, n_action_space):
        super(Q, self).__init__()
        self.fc_a1 = nn.Linear(n_action_space, network_width // 2)
        self.fc_s1 = nn.Linear(n_states, network_width)
        self.fc_s2 = nn.Linear(network_width, network_width // 2)
        self.fc2 = nn.Linear(network_width, network_width // 2)
        self.fc3 = nn.Linear(network_width // 2, 1)

    def forward(self, state, action):
        a = F.relu(self.fc_a1(action))
        s = torch.flatten(state, start_dim=1)
        s = F.relu(self.fc_s1(s))
        s = F.relu(self.fc_s2(s))
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor:
    def __init__(self, learning_rate=0.01, alpha=1., threshold=0.5):
        self.alpha = alpha
        self.Y = 0.5
        self.learning_rate = learning_rate
        self.threshold = torch.tensor([threshold], device=device)

    def take_action(self, Q_r, Q_c, s):
        batch_size = s.shape[0]
        Q_r_batch, Q_c_batch = np.zeros((batch_size, n_actions, 1)), np.zeros((batch_size, n_actions, 1))
        Q_r_batch, Q_c_batch = torch.tensor(Q_r_batch, device=device), torch.tensor(Q_c_batch, device=device)
        for act in range(n_actions):
            Q_r_batch[:, act, :] = Q_r(s, action_to_batch(act, batch_size))
            Q_c_batch[:, act, :] = Q_c(s, action_to_batch(act, batch_size))

        rewards_exp = torch.exp(self.alpha * (Q_r_batch + self.Y * Q_c_batch))
        rewards_exp_sum = torch.sum(rewards_exp, axis=1)
        rewards_exp_sum = torch.unsqueeze(rewards_exp_sum, axis=-1)
        rewards_softmax = torch.div(rewards_exp, rewards_exp_sum)
        return rewards_softmax

    def update(self, Q_r, Q_c, s_1):
        with torch.no_grad():
            estimate_constraint = np.zeros((batch_size, n_actions, 1))
            estimate_constraint = torch.tensor(estimate_constraint, device=device)
            for act in range(n_actions):
                estimate_constraint[:, act, :] = Q_c(s_1, action_to_batch(act, batch_size))

            action_softmax = self.take_action(Q_r, Q_c, s_1)
            loss = torch.mean(self.threshold - torch.sum(torch.mul(action_softmax, estimate_constraint), axis=1))
            self.Y -= self.learning_rate * loss
            return loss


def action_to_batch(act, batch_size):
    act = continuous_action(act)
    act_batch = np.vstack([act] * batch_size)
    act_batch = torch.tensor(act_batch, device=device, dtype=torch.float)
    return act_batch


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


def test(actor, Q_r, Q_c):
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
            action = actor.take_action(Q_r, Q_c, s)
            action = action.max(1)[1].item()
            try:
                next_obs, reward, safety, terminate, done, info = env.step(action=continuous_action(action))
            except:
                terminate = True
            else:
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


def action_matrix(action_batch):
    actions = np.zeros((len(action_batch), n_action_space))
    for c, act in enumerate(action_batch):
        actions[c] = continuous_action(act)
    actions = torch.tensor(actions, device=device, dtype=torch.float)
    return actions


def get_constraint(safety_batch):
    constraint_batch = np.zeros((len(safety_batch), 1))
    for i in range(len(safety_batch)):
        if safety_batch[i]['off_road'] == 1 or safety_batch[i]['crash'] == 1:
            constraint_batch[i] += 1
        if safety_batch[i]['near_off_road'] == 1 or safety_batch[i]['near_crash'] == 1:
            constraint_batch[i] += 0.5
    return constraint_batch


if __name__ == '__main__':
    offline_data = load_offline_data("./data/dataset/beta80_no_constraint.txt")
    actor = Actor(alg_config['actor_lr'], alg_config['alpha'], alg_config['actor_threshold'])
    Q_reward = Q(n_states, n_action_space).to(device)
    Q_reward_target = Q(n_states, n_action_space).to(device)
    Q_constraint = Q(n_states, n_action_space).to(device)
    Q_constraint_target = Q(n_states, n_action_space).to(device)

    reward_optimizer = optim.Adam(Q_reward.parameters(), lr=alg_config['lr'])
    constraint_optimizer = optim.Adam(Q_constraint.parameters(), lr=alg_config['lr'])
    loss = nn.MSELoss()

    hard_update(Q_reward, Q_reward_target)
    hard_update(Q_constraint, Q_constraint_target)

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
            experience.initial_state,
            experience.safety
        )
        states, actions, rewards, next_states, dones, initial_states, safety = \
            list(zip(*map(extract_fields, experiences)))
        state_batch = torch.tensor(np.array(states), device=device, dtype=torch.float)
        action_batch = torch.tensor(actions, device=device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(np.array(rewards), device=device, dtype=torch.float).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(next_states), device=device, dtype=torch.float)
        initial_state_batch = torch.tensor(np.array(initial_states), device=device, dtype=torch.float)
        constraint_batch = torch.tensor(get_constraint(safety), device=device, dtype=torch.float)
        done_batch = torch.tensor(dones, device=device, dtype=torch.float).unsqueeze(1)

        Q_r_prime, Q_c_prime = np.zeros((batch_size, n_actions, 1)), np.zeros((batch_size, n_actions, 1))
        Q_r_prime, Q_c_prime = torch.tensor(Q_r_prime, device=device), torch.tensor(Q_c_prime, device=device)
        for act in range(n_actions):
            Q_r_prime[:, act, :] = Q_reward_target(next_state_batch, action_to_batch(act, batch_size))
            Q_c_prime[:, act, :] = Q_constraint_target(next_state_batch, action_to_batch(act, batch_size))

        with torch.no_grad():
            action_softmax = actor.take_action(Q_reward, Q_constraint, next_state_batch)

        loss_reward = loss(Q_reward(state_batch, action_matrix(action_batch)),
                           reward_batch + (1 - done_batch) * alg_config['gamma'] * torch.sum(
                               torch.mul(action_softmax, Q_r_prime), dim=1).float())
        loss_constraint = loss(Q_constraint(state_batch, action_matrix(action_batch)),
                               constraint_batch + (1 - done_batch) * alg_config['gamma'] * torch.sum(
                                   torch.mul(action_softmax, Q_c_prime),
                                   dim=1).float())

        reward_optimizer.zero_grad()
        constraint_optimizer.zero_grad()

        loss_reward.backward()
        for param in Q_reward.parameters():
            param.grad.data.clamp_(-1, 1)
        reward_optimizer.step()

        loss_constraint.backward()
        for param in Q_constraint.parameters():
            param.grad.data.clamp_(-1, 1)
        constraint_optimizer.step()

        loss_actor = actor.update(Q_reward, Q_constraint, initial_state_batch)

        writer.add_scalar('data/reward loss', loss_reward, n)
        writer.add_scalar('data/constraint loss', loss_constraint, n)
        writer.add_scalar('data/constraint weight', actor.Y, n)
        writer.add_scalar('data/actor loss', loss_actor, n)

        del loss_reward
        del loss_constraint

        if n > 0 and n % alg_config['update_freq'] == 0:
            hard_update(Q_reward, Q_reward_target)
            hard_update(Q_constraint, Q_constraint_target)

            score, max_x = test(actor, Q_reward, Q_constraint)
            writer.add_scalar('data/test score', score, n)
            writer.add_scalar('data/test step', max_x, n)

            reward_path = f'data/{experiment_name}/model/r_{n}.pth'
            constraint_path = f'data/{experiment_name}/model/c_{n}.pth'
            torch.save(Q_reward.state_dict(), reward_path)
            torch.save(Q_constraint.state_dict(), constraint_path)
