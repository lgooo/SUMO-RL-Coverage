import os
import sys
import math
import collections
import numpy as np

def add_sumo_path():
    if "SUMO_HOME" in os.environ:
        sumo_path = os.path.join(os.environ["SUMO_HOME"], "tools")
        if sumo_path not in sys.path:
            sys.path.append(sumo_path)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

def long_lat_pos_cal(angle, acc_y, distance, heading):
    """
    Function to compute the global SUMO position based on ego vehicle states
    """
    if angle <= 90:
        alpha = 90 - angle
        # consider steering maneuver
        if acc_y >= 0:
            radians = math.radians(alpha) - heading
        else:
            radians = math.radians(alpha) + heading
        dx = distance * math.cos(radians)
        dy = distance * math.sin(radians)
    elif 90 < angle <= 180:
        alpha = angle - 90
        # consider steering maneuver
        if acc_y >= 0:
            radians = math.radians(alpha) - heading
        else:
            radians = math.radians(alpha) + heading

        dx = distance * math.cos(radians)
        dy = -distance * math.sin(radians)
    elif 180 < angle <= 270:
        alpha = 270 - angle
        # consider steering maneuver
        if acc_y >= 0:
            radians = math.radians(alpha) + heading
        else:
            radians = math.radians(alpha) - heading

        dx = -distance * math.cos(radians)
        dy = -distance * math.sin(radians)
    else:
        alpha = angle - 270
        # consider steering maneuver
        if acc_y >= 0:
            radians = math.radians(alpha) + heading
        else:
            radians = math.radians(alpha) - heading

        dx = -distance * math.cos(radians)
        dy = distance * math.sin(radians)

    return dx, dy


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)

        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(next_states),\
               np.array(dones, dtype=np.uint8),indices