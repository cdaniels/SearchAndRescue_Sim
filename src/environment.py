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


default_options = {
    'screen_size': 1000,
    'grid_size': 100,
    'num_agents': 10,
    'num_rescuers': 1,
    'num_victums': 10,
    'render_mode': 'human'
}

# color
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (200,0,0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
    
class Environment:
    Actions = Enum('Actions', ['LEFT', 'DOWN', 'UP', 'RIGHT', 'COMMUNICATE', 'PICKUP', 'DROPOFF'])

    def __init__(self, options) -> None:
        # unpack the options
        self.unpack_options(options)
        # grid representing world 0 wall, 1 movable
        self.world = np.ones((self.grid_size,self.grid_size)).astype(int).flatten()
        self.movable_locations = np.nonzero(self.world)[0]
        # start and goal locations
        self.starts = self.movable_locations
        self.goals = np.arange(98, 100)
        # array of victum locations at random cells in the world
        self.accident_locations = np.array([np.random.choice(self.movable_locations) for _ in range(4)])
        # self.victum_locations = np.array([np.random.choice(self.movable_locations) for _ in range(self.num_victums)])
        self.victum_locations = np.array([np.random.choice(self.accident_locations) for _ in range(self.num_victums)])
        self.agent_locations = np.array([np.random.choice(self.starts) for _ in range(self.num_agents)])
        self.rescuers = np.array([np.random.choice(np.arange(0, len(self.agent_locations))) for _ in range(self.num_rescuers)])
        # inter-agent trust network
        self.suggested_victum_locatons = np.ones((self.num_agents, self.num_victums))
        self.agents_carrying_victum = np.ones((self.num_victums)).astype(int)*(-1)
        self.trust_matrix = np.ones((self.num_agents, self.num_agents))
        # initialize pygame if appropriate
        if self.render_mode == 'human':
            self.init_pygame()

    def __del__(self):
        # close pygame if it was opened
        if self.render_mode == 'human':
            pygame.quit()
        
    def init_pygame(self):
        # setup screen
        pygame.init()
        self.grid2screen = self.screen_size / self.grid_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption('Grid World')

    def unpack_options(self, options):
        # apply all dictionary key-values as object properties
        for option in options:
            setattr(self, option, options[option])
    
    def step_agent(self, agent_i, action):
        loc = self.get_agent_2d_loc(agent_i)
        x = loc[0]
        y = loc[1]
        # check for victums being carried by agent
        carrying_vic = None
        for vic_i, carrying_agent_i in enumerate(self.agents_carrying_victum):
            if agent_i == carrying_agent_i:
                carrying_vic = vic_i
        # return values to be determined by action
        obs = np.array([])
        reward = 0
        done = False
        # apply affects for selected action
        d_loc = 0, 0
        match action:
            case Environment.Actions.LEFT:
                reward = -1
                d_loc = x-1, y
            case Environment.Actions.RIGHT:
                reward = -1
                d_loc = x+1, y
            case Environment.Actions.UP:
                reward = -1
                d_loc = x, y-1
            case Environment.Actions.DOWN:
                reward = -1
                d_loc = x, y+1
            case Environment.Actions.PICKUP:
                # reward is -10 for failed pickup
                reward = -10
                # check if victum is at same location
                for vic_i, _ in enumerate(self.victum_locations):
                    vic_loc = self.get_victum_2d_loc(vic_i)
                    if np.array_equal(loc, vic_loc):
                        self.agents_carrying_victum[vic_i] = agent_i
                        # reward is 10 for successful pickup
                        reward = 10
            case Environment.Actions.DROPOFF:
                # reward is -10 for failed dropoff
                reward = -10
                # stop carrying victum if victum is being carried
                if carrying_vic is not None:
                    self.agents_carrying_victum[carrying_vic] = -1
                    # reward is 10 for successful dropoff
                    reward = 10
                    # end the episode if all victums are in goal zone
                    done = np.all(np.isin(self.victum_locations, self.goals))
            case Environment.Actions.COMMUNICATE:
                # no reward for communication
                reward = 0
            case _:
                pass

        self.set_agent_2d_loc(agent_i, d_loc[0], d_loc[1])
        if carrying_vic is not None:
            self.set_victum_2d_loc(carrying_vic, d_loc[0], d_loc[1])
        return obs, reward, done

        if self.render_mode == 'human':
            self.render_grid()
                
        obs = np.array([])
        reward = 0.0
        done = self.is_terminated()
        return obs, reward, done

    def is_terminated(self):
        # episode terminateds when all victums are in goal zone
        return False

    def render_grid(self):
        # fill the display buffor on the screen
        self.screen.fill(WHITE)
        # draw the agents, victums, and goals in the display buffer
        scale = self.grid2screen
        for goal in self.goals:
            x, y = self.convert_loc_to_2d(goal)
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(x*scale, y*scale, scale, scale))
        for victum_loc in self.victum_locations:
            x, y = self.convert_loc_to_2d(victum_loc)
            pygame.draw.rect(self.screen, RED, pygame.Rect(x*scale, y*scale, scale, scale))
        for agent_loc in self.agent_locations:
            x, y = self.convert_loc_to_2d(agent_loc)
            pygame.draw.rect(self.screen, BLUE, pygame.Rect(x*scale, y*scale, scale, scale))
        # flip the display buffer to make it visible on the screen
        pygame.display.flip()
        # delay so step is clearly visible
        time.sleep(1)

    def set_agent_1d_loc(self, agent_i, loc):
        self.agent_locations[agent_i] = loc

    def set_victum_1d_loc(self, vic_i, loc):
        self.victum_locations[vic_i] = loc

    def set_agent_2d_loc(self, agent_i, x, y):
        self.agent_locations[agent_i] = self.convert_loc_from_2d(x, y)

    def set_victum_2d_loc(self, vic_i, x, y):
        self.victum_locations[vic_i] = self.convert_loc_from_2d(x, y)

    def get_agent_2d_loc(self, agent_i):
        grid_loc_1d = self.agent_locations[agent_i]
        x, y = self.convert_loc_to_2d(grid_loc_1d)
        grid_loc_2d = np.array([x, y])
        return grid_loc_2d

    def get_victum_2d_loc(self, vic_i):
        grid_loc_1d = self.victum_locations[vic_i]
        x, y = self.convert_loc_to_2d(grid_loc_1d)
        grid_loc_2d = np.array([x, y])
        return grid_loc_2d

    def convert_loc_to_2d(self, loc):
        x = loc % self.grid_size
        y = loc // self.grid_size
        return x, y

    def convert_loc_from_2d(self, x, y):
        loc_1d = y * self.grid_size + x
        return loc_1d

    # environment must assign rewards to certain joint action state combinations from the network
    # Joint_Action_Space = ...
    # def step(network_state, action_state):
    #     pass

    # for each turn every ajent chooses an action then recives an observation of the environment from the worldoo
    
    # the world evaluates the reward to deliver based on the joint actions of all the agents
    # conflicting actions (such as attempts to move into the same space can then be resolved)
