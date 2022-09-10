import numpy as np
import random
import torch
from util import Experience

class Alg:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def create(config):
        # TODO: proper implementation of factory
        if config['name'] == 'ddqn':
            from alg.ddqn import DDQN
            num_states = config.get('num_states', 35)
            num_actions = config.get('num_actions', 5)
            return DDQN(num_states, num_actions, config)

        if config['name'] == 'safe_ddqn':
            from alg.safe_ddqn import SafeDDQN
            num_states = config.get('num_states', 35)
            num_actions = config.get('num_actions', 5)
            return SafeDDQN(num_states, num_actions, config)

        if config['name'] == 'coptidice':
            from alg.coptidice import COptiDICE
            num_states = config.get('num_states', 35)
            num_actions = config.get('num_actions', 5)
            return COptiDICE(num_states, num_actions, config)

    def observe(self, experience: Experience):
        pass

    def new_epsidoe(self):
        pass

    def set_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
