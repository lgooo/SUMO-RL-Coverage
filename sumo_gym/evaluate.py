# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import argparse
from itertools import groupby
import json
from collections import defaultdict
import os

parser = argparse.ArgumentParser(
    description='SUMO Gym Tester')
parser.add_argument(
    '--log_dir',
    required=True,
    help='direction to the simulation log file')
parser.add_argument(
    '--output',
    help='output directory')
args = parser.parse_args()

def parse_line(stream):
    for line in stream:
        (
            episode_id, time_step, initial_state, obs, next_obs,
            action, reward, safety, terminate, done, info,
        ) = line.rstrip('\n').split('\t')
        episode_id = int(episode_id)
        time_step = int(time_step)
        initial_state = json.loads(initial_state)
        obs = json.loads(obs)
        next_obs = json.loads(next_obs)
        action = int(action)
        reward = float(reward)
        safety = json.loads(safety)
        terminate = int(terminate)
        done = int(done)
        info = json.loads(info)
        yield {
            'episode_id': episode_id,
            'time_step': time_step,
            'initial_state': initial_state,
            'obs': obs,
            'next_obs': next_obs,
            'action': action,
            'reward': reward,
            'safety': safety,
            'terminate': terminate,
            'done': done,
            'info': info,
        }

def evaluate_one(group):
    ret = defaultdict(float)
    for entry in group:
        ret['reward'] += entry['reward'] * (0.99 ** (entry['time_step'] - 1))
        ret['off_road'] += entry['safety']['off_road']
        ret['near_off_road'] += entry['safety']['near_off_road']
        ret['crash'] += entry['safety']['crash']
        ret['near_crash'] += entry['safety']['near_crash']
        ret['time_steps'] += 1
    return ret

def format_one(info):
    columns = [
        'reward',
        'off_road',
        'near_off_road',
        'crash',
        'near_crash',
        'time_steps',
    ]
    return '\t'.join([
        str(info[x]) for x in columns
    ])

g = open(args.output, 'w')
file_list = os.listdir(args.log_dir)
for filename in file_list:
    if not filename.endswith('.log'):
        continue
    with open(os.path.join(args.log_dir, filename), 'r') as f:
        f.readline() # skip header
        for episode_id, group in groupby(parse_line(f), key=lambda x: x['episode_id']):
            ret = format_one(evaluate_one(group))
            print('\t'.join([
                filename, str(episode_id), ret
            ]), file=g)

g.close()