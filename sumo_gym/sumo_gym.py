# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import os
import sys
from util import add_sumo_path
from sumo import Sumo
from util import long_lat_pos_cal

add_sumo_path()

import gym
import math
import constants as C
import traci
import ast
import numpy as np
import pandas as pd
from typing import Tuple
from shapely.geometry import LineString, Point
import traci.constants as tc
from typing import List
import os
import numpy as np

Observation = np.ndarray
Action = np.ndarray


class SumoGym(gym.Env):
    """
    SUMO Environment for testing AV pipeline

    Uses OpenAI Gym
    @Params: delta_t: time-step of running simulation
    @Params: render_flag: whether to utilize SUMO-GUI or SUMO

    returns None
    """

    def __init__(self, config, delta_t, render_flag=True) -> None:
        self.delta_t = delta_t
        self.vehID = []
        self.ego_state = dict({"x": 0, "y": 0, "lane_x": 0, "lane_y": 0, "vx": 0, "vy": 0, "ax": 0, "ay": 0})
        self.config = config
        self.sumo = None
        self.render_flag = render_flag

    def reset(self) -> Observation:
        """
        Function to reset the simulation and return the observation
        """

        self.sumo = Sumo(self.config, self.delta_t, self.render_flag)

        x, y = self.sumo.sumo_handle.vehicle.getPosition(C.EGO_ID)

        lane_id = self.sumo.sumo_handle.vehicle.getLaneID(C.EGO_ID)
        assert lane_id != ''
        lane_index = self.sumo.sumo_handle.vehicle.getLaneIndex(C.EGO_ID)
        lane_width = self.sumo.sumo_handle.lane.getWidth(lane_id)
        lane_y = lane_width * (lane_index + 0.5) + self.sumo.sumo_handle.vehicle.getLateralLanePosition(C.EGO_ID)

        self.ego_state = {
            'x': x,
            'y': y,
            'lane_x': self.sumo.sumo_handle.vehicle.getLanePosition(C.EGO_ID),
            'lane_y': lane_y,
            'vx': self.sumo.sumo_handle.vehicle.getSpeed(C.EGO_ID),
            'vy': 0,
            'ax': self.sumo.sumo_handle.vehicle.getAcceleration(C.EGO_ID),
            'ay': 0,
        }
        self.ego_line = self.get_ego_shape_info()

        return self._compute_observations()

    def _get_features(self, vehID) -> np.ndarray:
        """
        Function to get the position, velocity and length of each vehicle
        """
        if vehID == C.EGO_ID:
            presence = 1
            x = self.ego_state['lane_x']
            y = self.ego_state['lane_y']
            vx = self.ego_state['vx']
            vy = self.ego_state['vy']
            lane_index = self.sumo.sumo_handle.vehicle.getLaneIndex(vehID)
            ego_vx = self.sumo.sumo_handle.vehicle.getSpeed(vehID)
            ego_vy = self.sumo.sumo_handle.vehicle.getLateralSpeed(vehID)
        else:
            presence = 1
            x = self.sumo.sumo_handle.vehicle.getLanePosition(vehID)
            # right is negative and left is positiive
            y = self.sumo.sumo_handle.vehicle.getLateralLanePosition(vehID)
            lane_id = self.sumo.sumo_handle.vehicle.getLaneID(vehID)
            assert lane_id != ''
            lane_index = self.sumo.sumo_handle.vehicle.getLaneIndex(vehID)
            lane_width = self.sumo.sumo_handle.lane.getWidth(lane_id)
            y = lane_width * (lane_index + 0.5) + y
            vx = self.sumo.sumo_handle.vehicle.getSpeed(vehID)
            vy = self.sumo.sumo_handle.vehicle.getLateralSpeed(vehID)
        length = self.sumo.sumo_handle.vehicle.getLength(vehID)
        distance_to_signal, signal_status, remaining_time = self._get_upcoming_signal_information(vehID)
        features = np.array([presence, x, y, vx, vy, lane_index, length, distance_to_signal, signal_status, remaining_time])

        return features

    def _get_upcoming_signal_information(self, vehID):

        signal_information = self.sumo.sumo_handle.vehicle.getNextTLS(vehID)
        if len(signal_information) > 0:
            signal_id = signal_information[0][0]
            distance_to_signal = signal_information[0][2]
            signal_status = signal_information[0][3]
            remaining_time = self.sumo.sumo_handle.trafficlight.getNextSwitch(signal_id) - self.sumo.sumo_handle.simulation.getTime()

            if signal_status in ["G", "g"]:
                signal_status = 0
            elif signal_status in ["R", "r"]:
                signal_status = 1
            else:
                signal_status = 2
        else:
            distance_to_signal, signal_status, remaining_time = 0, 0, 0

        return distance_to_signal, signal_status, remaining_time

    def _compute_observations(self) -> Observation:
        """
        Function to compute the observation space
        Returns: A 7x10 array of Observations
        Key:
        Row 0 - ego and so on
        Columns:
        0 - presence
        1 - x in longitudinal lane position
        2 - y in lateral lane position
        3 - vx
        4 - vy
        5 - lane index
        6 - vehicle length
        7 - distance to next signal
        8 - current signal status, 0: green, 1: red, 2: yellow
        9 - remaning time of the current signal status in seconds
        """
        ego_features = self._get_features(C.EGO_ID)

        neighbor_ids = self.sumo.get_neighbor_ids(C.EGO_ID)
        obs = np.ndarray((len(neighbor_ids)+1, 10))
        obs[0, :] = ego_features
        for i, neighbor_id in enumerate(neighbor_ids):
            if neighbor_id != "":
                features = self._get_features(neighbor_id)
                obs[i + 1, :] = features
            else:
                obs[i + 1, :] = np.zeros((10, ))
        return obs

    def _update_state(self, action: Action) -> Tuple[bool, float, float, float, LineString, float, float, float, float, float, float]:
        """
        Function to update the state of the ego vehicle based on the action (Accleration)
        Returns difference in position and current speed
        """
        lane_id = self.sumo.sumo_handle.vehicle.getLaneID(C.EGO_ID)
        if lane_id == "":
            # off-road.
            return False, [], [], [], [], [], [], [], [], [], []

        angle = self.sumo.sumo_handle.vehicle.getAngle(C.EGO_ID)
        x, y = self.ego_state['x'], self.ego_state['y']
        ax_cmd = action[0]
        ay_cmd = action[1]

        vx, vy = self.ego_state['vx'], self.ego_state['vy']
        speed = math.sqrt(vx ** 2 + vy ** 2)
        # return heading in degrees
        # heading = (math.atan(vy / (vx + 1e-12))
        heading = math.atan(math.radians(angle) + (vy / (vx + 1e-12)))

        acc_x, acc_y = self.ego_state['ax'], self.ego_state['ay']
        acc_x += (ax_cmd - acc_x) * self.delta_t
        acc_y += (ay_cmd - acc_y) * self.delta_t

        vx += acc_x * self.delta_t
        vy += acc_y * self.delta_t

        # stop the vehicle if speed is negative
        vx = max(0, vx)
        speed = math.sqrt(vx ** 2 + vy ** 2)
        distance = speed * self.delta_t
        long_distance = vx * self.delta_t
        lat_distance = vy * self.delta_t
        if lane_id[0] != ":":
            veh_loc = Point(x, y)
            line = self.ego_line
            distance_on_line = line.project(veh_loc)
            if distance_on_line + long_distance < line.length:
                point_on_line_x, point_on_line_y = line.interpolate(distance_on_line).coords[0][0], \
                                                   line.interpolate(distance_on_line).coords[0][1],
                new_x, new_y = line.interpolate(distance_on_line + long_distance).coords[0][0], \
                               line.interpolate(distance_on_line + long_distance).coords[0][1]
                lon_dx, lon_dy = new_x - point_on_line_x, new_y - point_on_line_y
                lat_dx, lat_dy = lat_distance * math.cos(heading), lat_distance * math.sin(heading)
                dx = lon_dx + lat_dx
                dy = lon_dy + lat_dy
            else:
                self.ego_line = self.get_ego_shape_info()
                line = self.ego_line
                dx, dy = long_lat_pos_cal(angle, acc_y, distance, heading)
        else:
            self.ego_line = get_ego_shape_info()
            line = self.ego_line
            dx, dy = long_lat_pos_cal(angle, acc_y, distance, heading)

        return True, dx, dy, speed, line, vx, vy, acc_x, acc_y, long_distance, lat_distance

    def step(self, action: Action) -> Tuple[dict, float, bool, dict]:
        """
        Function to take a single step in the simulation based on action for the ego-vehicle
        """
        # next state, reward, done (bool), info (dict)
        # dict --> useful info in event of crash or out-of-network
        # bool --> false default, true when finishes episode/sims
        # float --> reward = user defined func -- Zero for now (Compute reward functionality)

        reward = self.reward(action)

        curr_pos = self.sumo.sumo_handle.vehicle.getPosition(C.EGO_ID)
        (in_road, dx, dy, speed, line, vx, vy, acc_x, acc_y, long_dist, lat_dist) = self._update_state(action)
        edge = self.sumo.sumo_handle.vehicle.getRoadID(C.EGO_ID)
        lane = self.sumo.sumo_handle.vehicle.getLaneIndex(C.EGO_ID)
        lane_id = self.sumo.sumo_handle.vehicle.getLaneID(C.EGO_ID)

        obs = []
        info = {}
        if in_road == False or lane_id == "":
            info["debug"] = "Ego-vehicle is out of network"
            return obs, 0, True, info
        if C.EGO_ID in self.sumo.sumo_handle.simulation.getCollidingVehiclesIDList():
            info["debug"] = "A crash happened to the Ego-vehicle"
            return obs, self.config['reward']['crash'], True, info

        # compute new state
        self.ego_state['lane_x'] += long_dist
        self.ego_state['lane_y'] += lat_dist
        # print(self.ego_state['lane_y'])
        new_x = self.ego_state['x'] + dx
        new_y = self.ego_state['y'] + dy
        # ego-vehicle is mapped to the exact position in the network by setting keepRoute to 2
        # moveToXY effective after a simulation step
        self.sumo.sumo_handle.vehicle.moveToXY(
            C.EGO_ID, edge, lane, new_x, new_y,
            tc.INVALID_DOUBLE_VALUE, 2
        )
        # remove control from SUMO, may result in very large speed
        self.sumo.sumo_handle.vehicle.setSpeedMode(C.EGO_ID, 0)
        self.sumo.sumo_handle.vehicle.setSpeed(C.EGO_ID, vx)
        self.ego_line = line
        self.ego_state['x'], self.ego_state['y'] = new_x, new_y
        self.ego_state['vx'], self.ego_state['vy'] = vx, vy
        self.ego_state['ax'], self.ego_state['ay'] = acc_x, acc_y

        if self.check_goal(self.ego_state):
            return [], self.config['reward']['goal'], True, info
        # update sumo with new ego state
        self.sumo.step()

        # compute next state observation
        obs = self._compute_observations()

        return obs, reward, False, info

    def check_goal(self, ego_state):
        return self.config['goal']['x'] <= ego_state['x']

    def reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.
        :param action: the last action performed
        :return: the reward
        """
        obs = self._compute_observations()
        _, ego_x, ego_y, ego_vx, ego_vy = obs[0][:5]

        # TODO: make speed limit configurable
        reward = -(ego_vx - 40) ** 2 # encourage staying close to the speed limit
        reward -= (action[0] ** 2) # discourage too much acceleration
        for i in range(1, len(obs)):
            present, x, y, vx, vy = obs[i][:5]
            if not present:
                continue
            if np.abs(y - ego_y) < 1: # same lane
                if ego_x < x and ego_x > x - 10: # too close to the leading vehicle
                    reward -= 10 # discourage getting too close to the leading vehicle

        return reward

    def get_num_lanes(self):
        edgeID = traci.vehicle.getRoadID(C.EGO_ID)
        num_lanes = traci.edge.getLaneNumber(edgeID)
        return num_lanes

    def get_ego_shape_info(self):
        laneID = traci.vehicle.getLaneID(C.EGO_ID)
        lane_shape = traci.lane.getShape(laneID)
        x_list = [x[0] for x in lane_shape]
        y_list = [x[1] for x in lane_shape]
        coords = [(x, y) for x, y in zip(x_list, y_list)]
        lane_shape = LineString(coords)
        return lane_shape

    def get_lane_width(self):
        laneID = traci.vehicle.getLaneID(C.EGO_ID)
        lane_width = traci.lane.getWidth(laneID)
        return lane_width

    def close(self):
        """
        Function to close the simulation environment
        """
        self.sumo.close()
