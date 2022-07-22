import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
from util import Deque
from logger import Logger
from alg.DDQN import DDQN

class SafeDDQN(DDQN):
    pass