import unittest

import os
import sys
import yaml

# hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_handler import Sumo
from sumo_gym import SumoGym
from util import SumoUtil

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

    def test_dangerous(self):
        # very safe
        conf = self.config1.copy()
        conf['vehicle_list'] = {
            'ego': {'position': 100, 'lane': 1, 'speed': 20},
            'v1': {'position': 150, 'lane': 1, 'speed': 20},
        }
        sumo_gym = SumoGym(conf, delta_t=0.1, render_flag=False, seed=1)
        sumo_gym.reset()
        obs = sumo_gym._compute_observations()
        self.assertFalse(SumoUtil.is_dangerous(obs))
        sumo_gym.close()

        # dangerous since leading vehicle very close.
        conf = self.config1.copy()
        conf['vehicle_list'] = {
            'ego': {'position': 100, 'lane': 1, 'speed': 20},
            'v1': {'position': 101, 'lane': 1, 'speed': 20},
        }
        sumo_gym = SumoGym(conf, delta_t=0.1, render_flag=False, seed=1)
        sumo_gym.reset()
        obs = sumo_gym._compute_observations()
        self.assertTrue(SumoUtil.is_dangerous(obs))
        sumo_gym.close()

        # safe because the leading vehicle is close but in another lane.
        conf = self.config1.copy()
        conf['vehicle_list'] = {
            'ego': {'position': 100, 'lane': 1, 'speed': 20},
            'v1': {'position': 101, 'lane': 2, 'speed': 20},
        }
        sumo_gym = SumoGym(conf, delta_t=0.1, render_flag=False, seed=1)
        sumo_gym.reset()
        obs = sumo_gym._compute_observations()
        self.assertFalse(SumoUtil.is_dangerous(obs))
        sumo_gym.close()

        # safe because ttc is infinity
        conf = self.config1.copy()
        conf['vehicle_list'] = {
            'ego': {'position': 100, 'lane': 1, 'speed': 20},
            'v1': {'position': 110, 'lane': 1, 'speed': 20},
        }
        sumo_gym = SumoGym(conf, delta_t=0.1, render_flag=False, seed=1)
        sumo_gym.reset()
        obs = sumo_gym._compute_observations()
        self.assertFalse(SumoUtil.is_dangerous(obs))
        sumo_gym.close()

        # dangerous because ttc is 1 second
        conf = self.config1.copy()
        conf['vehicle_list'] = {
            'ego': {'position': 100, 'lane': 1, 'speed': 30},
            'v1': {'position': 110, 'lane': 1, 'speed': 20},
        }
        sumo_gym = SumoGym(conf, delta_t=0.1, render_flag=False, seed=1)
        sumo_gym.reset()
        obs = sumo_gym._compute_observations()
        self.assertTrue(SumoUtil.is_dangerous(obs))
        sumo_gym.close()

        # dangerous because following vehicle is very close
        conf = self.config1.copy()
        conf['vehicle_list'] = {
            'ego': {'position': 100, 'lane': 1, 'speed': 20},
            'v1': {'position': 99, 'lane': 1, 'speed': 20},
        }
        sumo_gym = SumoGym(conf, delta_t=0.1, render_flag=False, seed=1)
        sumo_gym.reset()
        obs = sumo_gym._compute_observations()
        self.assertTrue(SumoUtil.is_dangerous(obs))
        sumo_gym.close()

    def test_reward(self):
        conf = self.config1.copy()
        conf['vehicle_list'] = {
            'ego': {'position': 100, 'lane': 1, 'speed': 25},
        }
        sumo_gym = SumoGym(conf, delta_t=0.1, render_flag=False, seed=1)
        sumo_gym.reset()
        print(sumo_gym._compute_observations())
        sumo_gym.close()


if __name__ == '__main__':
    unittest.main()
