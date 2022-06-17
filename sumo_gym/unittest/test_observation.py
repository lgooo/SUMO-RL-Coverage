import unittest

import os
import sys

# hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_gym import SumoGym

import traci

class TestObservation(unittest.TestCase):

    def test_things(self):
        pass


if __name__ == '__main__':
    unittest.main()