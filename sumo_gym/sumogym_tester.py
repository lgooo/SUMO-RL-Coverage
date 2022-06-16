# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import argparse
from sumo_gym import SumoGym
import numpy as np

Observation = np.ndarray
Action = np.ndarray

parser = argparse.ArgumentParser(
    description='SUMO Gym Tester')
parser.add_argument(
    '--config',
    default='sumo_configs/simple.sumocfg',
    help='SUMO config file path')
parser.add_argument(
    '--delta_t',
    type=float,
    default=0.1,
    help='simulation time step')
args = parser.parse_args()

env = SumoGym(sumo_config=args.config, delta_t=args.delta_t, render_flag=True)

obs = env.reset()
def policy(obs: Observation) -> Action:
    return [0, 0]

done = False
while not done:
    print(obs)
    action = policy(obs)
    obs, reward, done, info = env.step(action=action)

env.close()
