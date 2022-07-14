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


class SumoUtil:
    @staticmethod
    def is_dangerous(obs) -> bool:
        assert obs.shape == (7, 5)

        # note that ego_x is always zero since we use relative x position
        ego_state = obs[0, :]
        for i in range(obs.shape[0]):
            if i == 0:
                # skip ego
                continue
            veh_state = obs[i, :]
            if SumoUtil.is_dangerous_pair(ego_state, veh_state):
                return True
        return False

    @staticmethod
    def is_dangerous_pair(ego_state, veh_state) -> bool:
        assert len(ego_state) == 5
        assert len(veh_state) == 5

        _, ego_x, ego_y, ego_vx, ego_vy = ego_state
        present, x, y, vx, vy = veh_state
        if not present:
            return False
        if np.abs(ego_y - y) < 2 and np.abs(ego_x - x) < 3:
            # Ego too close to other vehicle
            return True
        if np.abs(ego_y - y) < 2 and ego_x < x and ego_vx > vx:
            # calculate TTC to the leading vehicle
            ttc = (x - ego_x) / (ego_vx - vx)
            if ttc <= 2:
                return True
        return False


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