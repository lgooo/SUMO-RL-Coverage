# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import argparse
from sumo_gym import SumoGym
import numpy as np
import yaml
from alg.alg import Alg
from tensorboardX import SummaryWriter
import os
import sys
import shutil
import datetime
from logger import Logger
from collections import defaultdict

Observation = np.ndarray
Action = np.ndarray

parser = argparse.ArgumentParser(
    description='SUMO Gym Tester')
parser.add_argument(
    '--config',
    default='config/simple.yaml',
    help='config file path')
parser.add_argument(
    '--num_episodes',
    type=int,
    default=500,
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

agent.load(args.model_path)
# no e-greedy
agent.epsilon_start = 0
agent.epsilon_end = 0
num_crashes = 0
num_out_of_roads = 0
for _ in range(args.num_episodes):
    obs = env.reset()
    terminate = False
    episode_reward = 0
    episode_steps = 0
    while not terminate:
        action = policy(obs)
        obs, reward, terminate, done, info = env.step(action=agent.continuous_action(action))
        if info.get('crash'):
            num_crashes += 1
        if info.get('out_of_road'):
            num_out_of_roads += 1
        episode_steps += 1
    env.close()
print(num_crashes / args.num_episodes)