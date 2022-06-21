import unittest

import os
import sys
import yaml

# hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo import Sumo

class TestObservation(unittest.TestCase):
    config1 = {
        'sumo_config': '../sumo/simple.sumocfg',
    }

    def test_neighbors(self):
        conf = self.config1.copy()
        conf['vehicle_list'] = {
            'ego': {'position': 100, 'lane': 1},
            'v1': {'position': 150, 'lane': 1},
            'v2': {'position': 150, 'lane': 0},
            'v3': {'position': 150, 'lane': 2},
            'v4': {'position': 50, 'lane': 0},
            'v5': {'position': 50, 'lane': 1},
            'v6': {'position': 50, 'lane': 2},
        }
        sumo = Sumo(conf, render_flag=False)
        self.assertEqual(
            sumo.get_neighbor_ids('ego'),
            ['v3', 'v1', 'v2', 'v6', 'v5', 'v4']
        )

        sumo.close()


if __name__ == '__main__':
    unittest.main()