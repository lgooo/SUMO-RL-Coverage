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
        if np.abs(ego_y - y) < 2 and np.abs(ego_x - x) < 5.3:
            # Ego too close to other vehicle longitudinally
            return True
        if np.abs(ego_y - y) < 2 and ego_x < x and ego_vx > vx * 0.8:
            # calculate TTC to the leading vehicle
            ttc = (x - ego_x - 5) / (ego_vx - vx * 0.8)
            if ttc <= 2:
                return True
        return False


class Deque:
    def __init__(self, capacity):
        self.capacity = capacity
        self._buffer = [None] * capacity
        self._end = 0
        self._start = 0
        self._size = 0

    def __len__(self):
        return self._size

    def __setitem__(self, index, data):
        assert index < self._size
        self._buffer[(index + self._start) % self.capacity] = data

    def __getitem__(self, index):
        assert index < self._size
        return self._buffer[(index + self._start) % self.capacity]

    def append(self, data):
        self._buffer[self._end] = data
        if self._size == self.capacity:
            self._start = (self._start + 1) % self.capacity
        else:
            self._size += 1
        self._end = (self._end + 1) % self.capacity

    def sample(self, batch_size):
        assert self._size >= batch_size
        indices = np.random.choice(self._size, batch_size, replace=False)

        return [self.__getitem__(index) for index in indices]
