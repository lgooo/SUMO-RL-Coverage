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
    '--num_iterations',
    type=int,
    help='number of iterations to run'
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
    '--data_path',
    help='path of the data to use for training'
)
args = parser.parse_args()

experiment_name = args.experiment
config_name = args.config.split('/')[-1].rsplit('.', 1)[0]
if not experiment_name:
    datestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    experiment_name = f'{config_name}_{datestamp}'

if not os.path.exists(f'data/{experiment_name}/model'):
    os.makedirs(f'data/{experiment_name}/model')

config_file = f'data/{experiment_name}/{config_name}.yaml'
shutil.copy(args.config, f'data/{experiment_name}/{config_name}.yaml')
with open(config_file, 'r') as f:
    conf = yaml.safe_load(f)

if args.data_path:
    conf['alg']['offline_data'] = args.data_path

agent = Alg.create(conf['alg'])
if args.seed is not None:
    agent.set_seed(args.seed)

def policy(obs: Observation) -> Action:
    return agent.choose_action(obs)


with open(f'data/{experiment_name}/data_path.txt', 'w') as f:
    print(conf['alg']['offline_data'], file=f)

writer = SummaryWriter(f'runs/{experiment_name}')

num_iterations = args.num_iterations
if not num_iterations:
    num_iterations = conf['alg']['num_iterations']

for i in range(num_iterations):
    loss = agent.update()

    agent.log_tensorboard(writer, i)

    writer.add_scalar('data/network-norm', agent.get_norm(), (i + 1))
    if loss is not None:
        writer.add_scalar('data/loss', loss, (i + 1))

    # save model every 1000 episodes
    if (i + 1) % 1000 == 0:
        agent.save(f'data/{experiment_name}/model/{i + 1}.pth')
