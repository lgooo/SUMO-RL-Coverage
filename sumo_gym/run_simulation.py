# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import argparse
from sumo_gym import SumoGym
import numpy as np
import yaml
from alg.alg_base import Alg
import os
import json

Observation = np.ndarray
Action = np.ndarray

parser = argparse.ArgumentParser(
    description='offline data generator')
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
    default='',
    required=True,
    help=(
        "path of a single model checkpoint, "
        "e.g. './data/{experiment_name}/model/1000.pth'"
    ),
)
parser.add_argument(
    '--output_dir',
    default='output',
    help="directory for storing simulation results",
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

def policy(obs: Observation) -> Action:
    return agent.choose_action(obs)

to_path = args.output_dir
if not os.path.exists(to_path):
    os.makedirs(to_path)

model_path = args.model_path
model_name = os.path.basename(args.model_path).rsplit('.', 1)[0]
results_path = os.path.join(to_path, model_name + '.log')
f = open(results_path, 'w')
# print header
print('\t'.join([
    'ID', 'timestep', 'initial_state', 'obs', 'next_obs',
    'action', 'reward', 'safety', 'terminate', 'done', 'info',
]), file=f)

agent.load(model_path)
# no e-greedy
agent.epsilon_start = 0
agent.epsilon_end = 0
num_crashes = 0
num_out_of_roads = 0
for episode_id in range(args.num_episodes):
    obs = env.reset()
    initial_state = obs.copy()

    terminate = False
    episode_reward = 0
    episode_steps = 0
    while not terminate:
        action = policy(obs)
        try:
            next_obs, reward, safety, terminate, done, info = env.step(action=agent.continuous_action(action))
        except:
            print("Error: Answered with error to command 0xa4: Vehicle 'ego' is not known.")
            terminate = True
        else:
            if done:
                next_obs = obs
            print('\t'.join([
                str(episode_id + 1),
                str(episode_steps + 1),
                json.dumps(initial_state.tolist()),
                json.dumps(obs.tolist()),
                json.dumps(next_obs.tolist()),
                str(action),
                str(reward),
                json.dumps(safety),
                '1' if terminate else '0',
                '1' if done else '0',
                json.dumps(info)
            ]), file=f)
            if info.get('crash'):
                num_crashes += 1
            if info.get('out_of_road'):
                num_out_of_roads += 1
            episode_steps += 1

            obs = next_obs
    env.close()
print(num_crashes / args.num_episodes)

f.close()
