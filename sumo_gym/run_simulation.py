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

agent.load(model_path)
# no e-greedy
agent.epsilon_start = 0
agent.epsilon_end = 0
num_crashes = 0
num_out_of_roads = 0
all_obs, all_next_obs,all_actions, all_reward, all_safety, all_terminate, all_done, all_info, all_ID, all_timestep = [], [], [], [], [], [], [], [], [],[]
for ID in range(args.num_episodes):
    obs = env.reset()
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
            all_obs.append(obs)
            if not done:
                all_next_obs.append(next_obs)
            else:
                all_next_obs.append(obs)
            all_reward.append(reward)
            all_actions.append(action)
            all_safety.append(safety)
            all_terminate.append(terminate)
            all_done.append(done)
            all_info.append(info)
            all_ID.append(ID)
            all_timestep.append(episode_steps)
            if info.get('crash'):
                num_crashes += 1
            if info.get('out_of_road'):
                num_out_of_roads += 1
            episode_steps += 1
            obs = next_obs
    env.close()
print(num_crashes / args.num_episodes)

results_path = os.path.join(to_path, model_name + '.log')
with open(results_path, 'w') as f:
    print('\t'.join(['ID', 'timestep', 'initial_state', 'obs', 'next_obs','action', 'reward', 'safety', 'terminate',
                        'done', 'info']), file=f)
    num_row = len(all_ID)
    initial_state=all_obs[0]
    for i in range(num_row):
        if all_timestep[i]==0:
            initial_state=all_obs[i]
        print('\t'.join([json.dumps(all_ID[i]), json.dumps(all_timestep[i]),
                            json.dumps(initial_state.tolist()),
                            json.dumps(all_obs[i].tolist()), json.dumps(all_next_obs[i].tolist()),
                            json.dumps(all_actions[i]),
                            json.dumps(all_reward[i]), json.dumps(all_safety[i]),
                            json.dumps(all_terminate[i]), json.dumps(all_done[i]),
                            json.dumps(all_info[i])]), file=f)
f.close()
