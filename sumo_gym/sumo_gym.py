# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import os
import sys
from util import add_sumo_path
from util import Sumo
from util import long_lat_pos_cal

add_sumo_path()

import gym
import math
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
        self.egoID = 'ego'
        self.ego_state = dict({"x": 0, "y": 0, "lane_x": 0, "lane_y": 0, "vx": 0, "vy": 0, "ax": 0, "ay": 0})
        self.config = config
        self.sumo = None

    def reset(self) -> Observation:
        """
        Function to reset the simulation and return the observation
        """

        self.sumo = Sumo(self.config, self.delta_t)

        x,  y = traci.vehicle.getPosition(self.egoID)

        lane_id = traci.vehicle.getLaneID(self.egoID)
        assert lane_id != ''
        lane_index = traci.vehicle.getLaneIndex(self.egoID)
        lane_width = traci.lane.getWidth(lane_id)
        lane_y = lane_width * (lane_index + 0.5) + traci.vehicle.getLateralLanePosition(self.egoID)

        self.ego_state = {
            'x': x,
            'y': y,
            'lane_x': traci.vehicle.getLanePosition(self.egoID),
            'lane_y': lane_y,
            'vx': traci.vehicle.getSpeed(self.egoID),
            'vy': 0,
            'ax': traci.vehicle.getAcceleration(self.egoID),
            'ay': 0,
        }
        self.ego_line = self.get_ego_shape_info()

        return self._compute_observations()

    def _get_features(self, vehID) -> np.ndarray:
        """
        Function to get the position, velocity and length of each vehicle
        """
        if vehID == self.egoID:
            presence = 1
            x = self.ego_state['lane_x']
            y = self.ego_state['lane_y']
            vx = self.ego_state['vx']
            vy = self.ego_state['vy']
            lane_index = traci.vehicle.getLaneIndex(vehID)
        else:
            presence = 1
            x = traci.vehicle.getLanePosition(vehID)
            # right is negative and left is positiive
            y = traci.vehicle.getLateralLanePosition(vehID)
            lane_id = traci.vehicle.getLaneID(vehID)
            assert lane_id != ''
            lane_index = traci.vehicle.getLaneIndex(vehID)
            lane_width = traci.lane.getWidth(lane_id)
            y = lane_width * (lane_index + 0.5) + y
            vx = traci.vehicle.getSpeed(vehID)
            vy = traci.vehicle.getLateralSpeed(vehID)
        length = traci.vehicle.getLength(vehID)
        distance_to_signal, signal_status, remaining_time = self._get_upcoming_signal_information(vehID)
        features = np.array([presence, x, y, vx, vy, lane_index, length, distance_to_signal, signal_status, remaining_time])

        return features

    def _get_upcoming_signal_information(self, vehID):

        signal_information = traci.vehicle.getNextTLS(vehID)
        if len(signal_information) > 0:
            signal_id = signal_information[0][0]
            distance_to_signal = signal_information[0][2]
            signal_status = signal_information[0][3]
            remaining_time = traci.trafficlight.getNextSwitch(signal_id) - traci.simulation.getTime()

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
        vehID = self.egoID
        ego_features = self._get_features(vehID)

        neighbor_ids = self.sumo.get_neighbor_ids(vehID)
        obs = np.ndarray((len(neighbor_ids)+1, 10))
        obs[0, :] = ego_features
        for i, neighbor_id in enumerate(neighbor_ids):
            if neighbor_id != "":
                features = self._get_features(neighbor_id)
                obs[i + 1, :] = features
            else:
                obs[i + 1, :] = np.zeros((10, ))
        return obs

    def update_state(self, action: Action) -> Tuple[bool, float, float, float, LineString, float, float, float, float, float, float]:
        """
        Function to update the state of the ego vehicle based on the action (Accleration)
        Returns difference in position and current speed
        """
        angle = traci.vehicle.getAngle(self.egoID)
        lane_id = traci.vehicle.getLaneID(self.egoID)
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
        in_road = True
        # try:
        if lane_id == "":
            in_road = False
            return in_road, [], [], [], [], [], [], [], [], [], []
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

        return in_road, dx, dy, speed, line, vx, vy, acc_x, acc_y, long_distance, lat_distance

    def step(self, action: Action) -> Tuple[dict, float, bool, dict]:
        """
        Function to take a single step in the simulation based on action for the ego-vehicle
        """
        # next state, reward, done (bool), info (dict)
        # dict --> useful info in event of crash or out-of-network
        # bool --> false default, true when finishes episode/sims
        # float --> reward = user defined func -- Zero for now (Compute reward functionality)
        curr_pos = traci.vehicle.getPosition(self.egoID)
        (in_road, dx, dy, speed, line, vx, vy, acc_x, acc_y, long_dist, lat_dist) = self.update_state(action)
        edge = traci.vehicle.getRoadID(self.egoID)
        lane = traci.vehicle.getLaneIndex(self.egoID)
        lane_id = traci.vehicle.getLaneID(self.egoID)

        obs = []
        info = {}
        if in_road == False or lane_id == "":
            info["debug"] = "Ego-vehicle is out of network"
            return obs, 0, True, info
        if self.egoID in traci.simulation.getCollidingVehiclesIDList():
            info["debug"] = "A crash happened to the Ego-vehicle"
            return obs, self.config['reward']['crash'], True, info

        self.ego_state['lane_x'] += long_dist
        self.ego_state['lane_y'] += lat_dist
        # print(self.ego_state['lane_y'])
        new_x = self.ego_state['x'] + dx
        new_y = self.ego_state['y'] + dy
        # ego-vehicle is mapped to the exact position in the network by setting keepRoute to 2
        traci.vehicle.moveToXY(
            self.egoID, edge, lane, new_x, new_y,
            tc.INVALID_DOUBLE_VALUE, 2
        )
        # remove control from SUMO, may result in very large speed
        traci.vehicle.setSpeedMode(self.egoID, 0)
        traci.vehicle.setSpeed(self.egoID, vx)
        self.ego_line = line
        obs = self._compute_observations()
        self.ego_state['x'], self.ego_state['y'] = new_x, new_y
        self.ego_state['vx'], self.ego_state['vy'] = vx, vy
        self.ego_state['ax'], self.ego_state['ay'] = acc_x, acc_y
        info["debug"] = [lat_dist, self.ego_state['lane_y']]
        traci.simulationStep()
        reward = self.reward(action)

        return obs, reward, False, info

    # Reward will not be implemented, user has the option
    def reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.
        :param action: the last action performed
        :return: the reward
        """
        reward = 0
        return reward

    def get_num_lanes(self):
        edgeID = traci.vehicle.getRoadID(self.egoID)
        num_lanes = traci.edge.getLaneNumber(edgeID)
        return num_lanes

    def get_ego_shape_info(self):
        laneID = traci.vehicle.getLaneID(self.egoID)
        lane_shape = traci.lane.getShape(laneID)
        x_list = [x[0] for x in lane_shape]
        y_list = [x[1] for x in lane_shape]
        coords = [(x, y) for x, y in zip(x_list, y_list)]
        lane_shape = LineString(coords)
        return lane_shape

    def get_lane_width(self):
        laneID = traci.vehicle.getLaneID(self.egoID)
        lane_width = traci.lane.getWidth(laneID)
        return lane_width

    def close(self):
        """
        Function to close the simulation environment
        """
        traci.close()
