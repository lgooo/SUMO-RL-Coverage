import unittest

import os
import sys
import yaml

# hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import Deque

class TestDeque(unittest.TestCase):
    def test_deque(self):
        a = Deque(3)
        self.assertTrue(len(a) == 0)
        a.append('a')
        self.assertTrue(len(a) == 1)
        self.assertTrue(a[0] == 'a')
        a.append('b')
        self.assertTrue(len(a) == 2)
        self.assertTrue(a[0] == 'a')
        self.assertTrue(a[1] == 'b')
        a.append('c')
        self.assertTrue(len(a) == 3)
        self.assertTrue(a[0] == 'a')
        self.assertTrue(a[1] == 'b')
        self.assertTrue(a[2] == 'c')
        a.append('d')
        self.assertTrue(len(a) == 3)
        self.assertTrue(a[0] == 'b')
        self.assertTrue(a[1] == 'c')
        self.assertTrue(a[2] == 'd')
        a.append('e')
        self.assertTrue(len(a) == 3)
        self.assertTrue(a[0] == 'c')
        self.assertTrue(a[1] == 'd')
        self.assertTrue(a[2] == 'e')
        a.append('f')
        self.assertTrue(len(a) == 3)
        self.assertTrue(a[0] == 'd')
        self.assertTrue(a[1] == 'e')
        self.assertTrue(a[2] == 'f')
        a.append('g')
        self.assertTrue(len(a) == 3)
        self.assertTrue(a[0] == 'e')
        self.assertTrue(a[1] == 'f')
        self.assertTrue(a[2] == 'g')

if __name__ == '__main__':
    unittest.main()
