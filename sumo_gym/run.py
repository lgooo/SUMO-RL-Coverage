# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import argparse
from sumo_gym import SumoGym
import numpy as np
import yaml
from alg.alg_base import Alg
from tensorboardX import SummaryWriter
import os
import sys
import shutil
import datetime
from logger import Logger
from collections import defaultdict
from collections import namedtuple
import util
from util import Experience

Observation = np.ndarray
Action = np.ndarray

parser = argparse.ArgumentParser(
    description='SUMO Gym Tester')
parser.add_argument(
    '--config',
    default='config/simple.yaml',
    help='config file path')
parser.add_argument(
    '--experiment',
    help='experiment name')
parser.add_argument(
    '--num_episodes',
    type=int,
    default=1000,
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
    '--seed',
    type=int,
    default=None,
    help='random seed',
)
parser.add_argument(
    '--model_path',
    help='path of the model to be tested'
)
parser.add_argument(
    '--profile',
    action='store_true',
    default=False,
    help='whether to profile performance',
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

agent = Alg.create(conf['alg'])
if args.seed is not None:
    agent.set_seed(args.seed)

def obs_filter(obs:Observation):
    if len(obs):
        if obs.max()>1e3:
            print(obs.max())
            return False
        elif obs.min()<-1e3:
            print(obs.min())
            return False
        else:
            return True
    return False

def policy(obs: Observation) -> Action:
    return agent.choose_action(obs)


experience_tuple = namedtuple('experience','obs action reward safety next_obs done')
experiment_name = args.experiment
config_name = args.config.split('/')[-1].rsplit('.', 1)[0]
if not experiment_name:
    datestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    experiment_name = f'{config_name}_{datestamp}'

if not os.path.exists(f'data/{experiment_name}/model'):
    os.makedirs(f'data/{experiment_name}/model')

shutil.copy(args.config, f'data/{experiment_name}/{config_name}.yaml')

writer = SummaryWriter(f'runs/{experiment_name}')

logger = Logger(args.profile)
agent.set_logger(logger)
counter=util.dangerous_pair_counter()

for epi in range(args.num_episodes):
    obs = env.reset()
    agent.new_episode()
    initial_state = obs.copy()
    terminate = False
    episode_reward = 0
    episode_steps = 0
    max_x = 0
    loss = None
    log_time_sum = defaultdict(float)
    log_num = defaultdict(int)

    first = True
    while not terminate:
        logger.reset()
        action = policy(obs)
        if first and hasattr(agent, 'initial_state_memory'):
            agent.initial_state_memory.append((obs, action))
            first = False

        logger.log('choose_action')
        next_obs, reward, safety, terminate, done, info = env.step(action=agent.continuous_action(action))
        logger.log('environment_step')
        if not done:
            if obs_filter(next_obs):
                agent.observe(Experience(initial_state, obs, action, reward, safety, next_obs, done))
                counter.count_dangerous(next_obs)
        else:
            if obs_filter(obs):
                agent.observe(Experience(initial_state, obs, action, reward, safety, obs, done))
        logger.log('memory_append')
        episode_steps += 1
        episode_reward += reward
        obs = next_obs
        if len(obs):
            max_x = env.ego_state['lane_x']
        loss = agent.update()
        if episode_steps > 1000:
            break
        one_log = logger.digest()
        for k, v in one_log:
            log_time_sum[k] += v
            log_num[k] += 1
    writer.add_scalar('data/step', episode_steps, epi)
    writer.add_scalar('data/x', max_x, epi)
    writer.add_scalar('data/reward', episode_reward, epi)
    writer.add_scalar('data/network-norm', agent.get_norm(), epi)
    writer.add_scalar('data/epsilon', agent.get_epsilon(), epi)
    writer.add_scalar('data/dangerous-states', len(counter), epi)
    agent.log_tensorboard(writer, epi)

    if args.profile:
        for k, v in log_time_sum.items():
            writer.add_scalar(f'data/profile_{k}', v / log_num[k], epi)

    if loss is not None:
        writer.add_scalar('data/loss', loss, epi)
    env.close()

    # save model every 100 episodes
    if (epi + 1) % 100 == 0:
        agent.save(f'data/{experiment_name}/model/{epi + 1}.pth')

counter.save(experiment_name)
