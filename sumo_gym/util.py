import os
import sys
import math
from collections import namedtuple
import numpy as np
import pickle
import json

Experience = namedtuple(
    'Experience',
    ['obs', 'action', 'reward', 'safety', 'next_obs', 'done'],
)

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

    @staticmethod
    def is_near_off_road(obs) -> bool:
        assert obs.shape == (7, 5)

        ego_state = obs[0, :]
        _, ego_x, ego_y, ego_vx, ego_vy = ego_state
        return ego_y < 1.1 or ego_y > 8.5

    @staticmethod
    def is_off_road(obs) -> bool:
        assert obs.shape == (7, 5)

        ego_state = obs[0, :]
        _, ego_x, ego_y, ego_vx, ego_vy = ego_state
        return ego_y < 1 or ego_y > 8.6

class dangerous_pair_counter:
    def __init__(self):
        self.counter = {}

    def __len__(self):
        return len(self.counter)

    def discretized(self,state_pair):
        for i in range(len(state_pair)):
            state_pair[i]=math.floor(state_pair[i])
        return state_pair

    def append_pair(self,ego_state, veh_state):
        #record y, vx,vy for ego, and x,v,vx,vy for veh
        state_pair=np.concatenate((ego_state[2:],veh_state[1:]))
        state_pair=tuple(self.discretized(state_pair))
        self.counter[state_pair]=self.counter.get(state_pair,0)+1

    def count_dangerous(self,obs):
        assert obs.shape == (7, 5)
        ego_state = obs[0, :]
        for i in range(obs.shape[0]):
            if i == 0:
                continue
            veh_state = obs[i, :]
            if SumoUtil.is_dangerous_pair(ego_state, veh_state):
                self.append_pair(ego_state,veh_state)

    def save(self,experiment_name):
        with open(f'data/{experiment_name}/dangerous_pairs.pkl', "wb") as outfile:
            pickle.dump(self.counter, outfile)


class Deque:
    def __init__(self, capacity, multi_steps=1):
        self.capacity = capacity
        self.gamma = 0.99
        self.multi_steps = multi_steps
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

        samples=[self.__getitem__(index) for index in indices]

        obs=[]
        action=[]
        rewards=[]
        safety=[]
        next_obs=[]
        dones=[]

        for t in samples:
            obs.append(t.obs)
            action.append(t.action)
            rewards.append(t.reward)
            safety.append(t.safety)
            next_obs.append(t.next_obs)
            dones.append(t.done)

        for i in range(batch_size):
            for n in range(1,self.multi_steps):
                if indices[i]+n < self._size:
                    rewards[i] += (self.gamma ** n) * self.__getitem__(indices[i]+n).reward
                    next_obs[i] = self.__getitem__(indices[i] + n).next_obs
                    if self.__getitem__(indices[i] + n).done:
                        dones[i] = True
                        break

        return obs,action,rewards,safety, next_obs, dones


        return samples


def load_offline_data(file_path, capacity=1000000):
    """
    Columns are
    - ID : ID of trajectory (integer)
    - timestep : timestemp within trajectory (integer)
    - obs : state observations (json string of nested lists of numbers)
    - next_obs : state observations (json string of nested lists of numbers)
    - reward : numeric
    - safety : json string of dictionary of safety related signals
    - terminate : boolean (true or false)
    - done : boolean (trune or false)
    - info : json string of dictionary of other info
    """
    replay = Deque(capacity=capacity)
    f = open(file_path, 'r')
    f.readline() # skip header
    for line in f:
        (
            _, _, obs, next_obs, reward, safety,
            terminate, done, _
        ) = line.rstrip('\n').split('\t')
        obs = np.array(json.loads(obs))
        next_obs = np.array(json.loads(next_obs))
        reward = float(reward)
        safety = json.loads(safety)
        terminate = 1 if terminate == 'true' else 0
        done = 1 if done == 'true' else 0
        replay.append(Experience(
            obs=obs,
            action=1, # TODO: fix after Yuhang's fix
            reward=reward,
            safety=safety,
            next_obs=next_obs,
            done=done,
        ))
    f.close()
    return replay
