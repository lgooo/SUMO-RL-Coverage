import unittest

import os
import sys
import yaml

# hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import Sumo

class TestObservation(unittest.TestCase):
    config1 = {
        'env': {
            'sumo_config': '../sumo/simple.sumocfg',
        }
    }

    def test_neighbors(self):
        conf = self.config1.copy()
        conf['env']['vehicle_list'] = {
            'ego': {'position': 100, 'lane': 1},
            'v1': {'position': 120, 'lane': 1},
            'v2': {'position': 110, 'lane': 0},
            'v3': {'position': 130, 'lane': 2},
            'v4': {'position': 70, 'lane': 0},
            'v5': {'position': 80, 'lane': 1},
            'v6': {'position': 80, 'lane': 2},
        }
        sumo = Sumo(conf, render_flag=False)
        print(sumo.get_neighbor_ids('ego'))

        sumo.close()


if __name__ == '__main__':
    unittest.main()