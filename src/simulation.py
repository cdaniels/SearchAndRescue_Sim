import numpy as np
from enum import Enum
import time

import pygame

class Simulation:
    def __init__(self, num_agents) -> None:
        self.agents = np.arange(0, num_agents)

    def run_simulation(self):
        state = self.world.reset()

        proposed_actions = np.zeros(len(self.agents))
        for i, agent in enumerate(self.agents):
            proposed_actions[i] 
            
        pass