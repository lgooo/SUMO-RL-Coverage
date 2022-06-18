import constants as C
import numpy as np
import os
import sys
import math

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


class Sumo:
    def __init__(self, config, delta_t=0.1, render_flag=True):
        if "SUMO_HOME" not in os.environ:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        self.config = config

        sumo_path = os.path.join(os.environ["SUMO_HOME"], "tools")
        if sumo_path not in sys.path:
            sys.path.append(sumo_path)
        bin_path = os.path.join(os.environ["SUMO_HOME"], "bin")
        if bin_path not in sys.path:
            sys.path.append(bin_path)

        import traci
        self.sumo_handle = traci

        from sumolib import checkBinary

        if render_flag:
            sumo_binary = checkBinary("sumo-gui")
        else:
            sumo_binary = checkBinary("sumo")
        sumoCmd = [
            sumo_binary,
            "-c", config['sumo_config'],
            "--step-length", str(delta_t),
            "--collision.action", "warn",
            "--collision.mingap-factor", "0",
            "--random", "false",
            "--lateral-resolution", ".1",
        ]
        traci.start(sumoCmd)

        self._init()

        # simulate until ego appears
        vehicle_ids = []
        while C.EGO_ID not in vehicle_ids:
            traci.simulationStep()
            vehicle_ids = traci.vehicle.getIDList()

        traci.vehicle.highlight(C.EGO_ID)

    def _init(self):
        num_random_vehicles = self.config.get('num_random_vehicles', 0)
        vehicle_time_gap = self.config.get('vehicle_time_gap', 1.0)
        routes = self.sumo_handle.route.getIDList()
        lanes = self.sumo_handle.lane.getIDList()
        vehicles = self.sumo_handle.vehicle.getIDList()
        speed_mean = self.config.get('vehicle_speed_mean', 25)
        speed_stdev = self.config.get('vehicle_speed_stdev', 2)

        start_time = np.random.permutation(num_random_vehicles) * vehicle_time_gap

        vehicle_data = self.config.get('vehicle_list', {})
        for i in range(num_random_vehicles):
            veh_id = f'gen_v_{i}' if C.EGO_ID in vehicle_data else C.EGO_ID
            vehicle_data[veh_id] = {
                'depart_time': start_time[i],
                'position': 0,
                'lane': 'random',
            }
        assert C.EGO_ID in vehicle_data

        # order of position matters for some reason.
        for id, data in sorted(vehicle_data.items(), key=lambda x: x[1]['position'], reverse=True):
            self.sumo_handle.vehicle.add(
                vehID=id,
                routeID='', # randomly chosen
                departSpeed=np.random.normal(speed_mean, speed_stdev),
                depart=data.get('depart_time', 0),
                departPos=data['position'],
                departLane=data.get('lane', 'random'),
            )
            lane_ids = self.sumo_handle.lane.getIDList()
            # hack: force insertion by calling moveTo()
            if data['lane'] != 'random':
                self.sumo_handle.vehicle.moveTo(
                    id, laneID=lane_ids[data['lane']], pos=data['position'])

    def get_neighbor_ids(self, vehID):
        """
        Function to extract the ids of the neighbors of a given vehicle
        """
        neighbor_ids = []

        rightFollower = self.sumo_handle.vehicle.getRightFollowers(vehID)
        rightLeader = self.sumo_handle.vehicle.getRightLeaders(vehID)
        leftFollower = self.sumo_handle.vehicle.getLeftFollowers(vehID)
        leftLeader = self.sumo_handle.vehicle.getLeftLeaders(vehID)
        leader = self.sumo_handle.vehicle.getLeader(vehID, dist=50)
        follower = self.sumo_handle.vehicle.getFollower(vehID, dist=50)

        if len(leftLeader) != 0:
            neighbor_ids.append(leftLeader[0][0])
        else:
            neighbor_ids.append("")
        if leader is not None:
            neighbor_ids.append(leader[0])
        else:
            neighbor_ids.append("")
        if len(rightLeader) != 0:
            neighbor_ids.append(rightLeader[0][0])
        else:
            neighbor_ids.append("")
        if len(leftFollower) != 0:
            neighbor_ids.append(leftFollower[0][0])
        else:
            neighbor_ids.append("")
        if follower is not None and follower[0] != "":
            neighbor_ids.append(follower[0])
        else:
            neighbor_ids.append("")
        if len(rightFollower) != 0:
            neighbor_ids.append(rightFollower[0][0])
        else:
            neighbor_ids.append("")
        return neighbor_ids

    def step(self):
        self.sumo_handle.simulationStep()

    def close(self):
        self.sumo_handle.close()
