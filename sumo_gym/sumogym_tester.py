# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

# @author      Nikhil Punshi
# @supervisor  Arpan Kusari
# @date        08-16-2021

# SumoGym Testing File

import argparse
import os
import math
from sumo_gym import SumoGym
from params import IDMConstants, LCConstants
import numpy as np
import matplotlib.pyplot as plt

Observation = np.ndarray
Action = np.ndarray

parser = argparse.ArgumentParser(
    description='SUMO Gym Tester')
parser.add_argument(
    '--config',
    default='sumo_configs/quickstart.sumocfg',
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

while True:
    action = policy(obs)
    obs, reward, done, info = env.step(action=action)

    if done:
        break
env.close()
