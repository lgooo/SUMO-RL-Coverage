# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import argparse
from sumo_gym import SumoGym
import numpy as np
import yaml
from DDQN import DDQN
from tensorboardX import SummaryWriter

Observation = np.ndarray
Action = np.ndarray

writer = SummaryWriter()

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
    '--test',
    action='store_true',
    default=False,
    help='whether to test the model'
)
args = parser.parse_args()

with open(args.config, 'r') as f:
    conf = yaml.safe_load(f)

env = SumoGym(
    config=conf['env'],
    delta_t=args.delta_t,
    render_flag=args.render,
)

agent = DDQN(n_states=70, n_actions=5)

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

for epi in range(args.num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    max_x = 0
    loss = None
    while not done:
        action = policy(obs)
        next_obs, reward, done, info = env.step(action=agent.continuous_action(action))
        if not done:
            if obs_filter(next_obs):
                agent.memory.append(obs, action, reward, next_obs, done)
        else:
            if obs_filter(next_obs):
                agent.memory.append(obs, action, reward, obs, done)
        episode_steps += 1
        episode_reward += reward
        obs = next_obs
        if len(obs):
            max_x = obs[0][1]
        loss = agent.update()
    writer.add_scalar('data/step', episode_steps, epi)
    writer.add_scalar('data/x', max_x, epi)
    writer.add_scalar('data/reward', episode_reward, epi)
    if loss is not None:
        writer.add_scalar('data/loss', loss, epi)
    env.close()

agent.save(path="./data")

if args.test:
    env.render_flag = True
    agent.load(path="./data")
    # no e-greedy
    agent.epsilon_start = 0
    agent.epsilon_end = 0
    for _ in range(5):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action=agent.continuous_action(action))
            episode_steps += 1
            episode_reward += reward
        env.close()
        print("Steps: {}, Reward: {}".format(episode_steps, episode_reward))
