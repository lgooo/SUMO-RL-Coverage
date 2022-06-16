# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import argparse
from sumo_gym import SumoGym
import numpy as np
import yaml

Observation = np.ndarray
Action = np.ndarray

parser = argparse.ArgumentParser(
    description='SUMO Gym Tester')
parser.add_argument(
    '--config',
    default='config/simple.yaml',
    help='config file path')
parser.add_argument(
    '--delta_t',
    type=float,
    default=0.1,
    help='simulation time step')
args = parser.parse_args()

with open(args.config, 'r') as f:
    conf = yaml.safe_load(f)

env = SumoGym(
    sumo_config=conf['env']['sumo_config'],
    delta_t=args.delta_t,
    render_flag=True,
)

obs = env.reset()
def policy(obs: Observation) -> Action:
    return [0, 0]

done = False
while not done:
    print(obs)
    action = policy(obs)
    obs, reward, done, info = env.step(action=action)

env.close()