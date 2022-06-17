import unittest

import os
import sys
import yaml

# hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import Sumo

class TestObservation(unittest.TestCase):

    def test_things(self):
        with open('test_config/config_01.yaml', 'r') as f:
            conf = yaml.safe_load(f)
        sumo = Sumo(conf)
        sumo.close()


if __name__ == '__main__':
    unittest.main()