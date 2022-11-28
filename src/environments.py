import numpy as np
from enum import Enum
import time

import pygame

default_options = {
    'screen_size': 100,
    'grid_size': 10,
    'num_agents': 3,
    'num_rescuers': 1,
    'num_victums': 1,
    'visible_range': 2,
    'max_pheromone': 10,
    'render_mode': 'human'
}

# color
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (200,0,0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
    
class SARGridWorld:
    Actions = Enum('Actions', ['LEFT', 'DOWN', 'UP', 'RIGHT', 'COMMUNICATE', 'PICKUP', 'DROPOFF'])

    def __init__(self, options) -> None:
        # unpack the options
        self.unpack_options(options)
        # grid representing world 0 wall, 1 movable
        self.world = np.ones((self.grid_size,self.grid_size)).astype(int).flatten()
        self.movable_locations = np.nonzero(self.world)[0]
        self.agent_location_visits = np.zeros((self.num_agents, len(self.movable_locations)))
        # start and goal locations
        self.starts = self.movable_locations[0:2]
        self.goals = self.movable_locations[0:2]
        # array of victum locations at random cells in the world
        self.accident_locations = np.array([np.random.choice(self.movable_locations) for _ in range(4)])
        # self.victum_locations = np.array([np.random.choice(self.movable_locations) for _ in range(self.num_victums)])
        self.victum_locations = np.array([np.random.choice(self.accident_locations) for _ in range(self.num_victums)])
        self.agent_locations = np.array([np.random.choice(self.starts) for _ in range(self.num_agents)])
        # simple arrays for rescuers and scouts
        self.agents = np.arange(0, len(self.agent_locations))
        self.rescuers = np.array([np.random.choice(self.agents) for _ in range(self.num_rescuers)])
        self.scouts = self.agents[np.isin(self.agents, self.rescuers, invert=True)]
        # inter-agent trust network
        self.likely_victum_locations = np.ones((self.num_agents, self.num_victums)).astype(int)*(-1)
        self.agents_carrying_victum = np.ones((self.num_victums)).astype(int)*(-1)
        self.trust_matrix = np.ones((self.num_agents, self.num_agents))
        # initialize pygame if appropriate
        if self.render_mode == 'human':
            self.init_pygame()

    def __del__(self):
        # close pygame if it was opened
        if self.render_mode == 'human':
            pygame.quit()

    def get_scout_actions(self):
        return [self.Actions.LEFT, self.Actions.RIGHT, self.Actions.UP, self.Actions.DOWN, self.Actions.COMMUNICATE]
        # return [action.value for action in self.Actions]
        
    def get_rescuer_actions(self):
        return [self.Actions.LEFT, self.Actions.RIGHT, self.Actions.UP, self.Actions.DOWN, self.Actions.COMMUNICATE, self.Actions.PICKUP, self.Actions.DROPOFF]
        # return [action.value for action in self.Actions]
        
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

    def manhatten_distance(self, loc1, loc2):
        loc1_2d = self.convert_loc_to_2d(loc1)
        loc2_2d = self.convert_loc_to_2d(loc2)
        return sum(abs(value1 - value2) for value1, value2 in zip(loc1_2d, loc2_2d))

    def objects_in_range(self, agent_i):
        # get the agent location
        agent_loc = self.agent_locations[agent_i]
        # test for victums and agents within range then add them to list
        agents_in_range = list()
        victums_in_range = list()
        for other_i, loc in enumerate(self.agent_locations):
            if other_i != agent_i:
                dist = self.manhatten_distance(loc, agent_loc)
                if dist < self.visible_range:
                    agents_in_range.append(other_i)
        for vic_i, loc in enumerate(self.victum_locations):
            dist = self.manhatten_distance(loc, agent_loc)
            if dist < self.visible_range:
                victums_in_range.append(vic_i)
        return agents_in_range, victums_in_range

    def reset_agent(self, agent_i):
        # format observation data
        suggested_locs = self.likely_victum_locations[agent_i]
        visited_locs = self.agent_location_visits[agent_i]
        carrying = False
        # return the initial observation data
        return agent_i, self.agent_locations, suggested_locs, visited_locs, carrying, self.goals
    
    def step_agent(self, agent_i, action):
        loc = self.get_agent_2d_loc(agent_i)
        # check for other agents or victums in range
        agents_in_range, victums_in_range = self.objects_in_range(agent_i)
        # check for victums being carried by agent
        carrying_vic = None
        for vic_i, carrying_agent_i in enumerate(self.agents_carrying_victum):
            if agent_i == carrying_agent_i:
                carrying_vic = vic_i
        # return values to be determined by action
        obs = np.array([])
        # reward is -1 normally for each timestep
        reward = -1
        done = False
        # apply affects for selected action
        dx, dy = 0, 0
        match action:
            case self.Actions.LEFT:
                dx = -1
            case self.Actions.RIGHT:
                dx = +1
            case self.Actions.UP:
                dy = -1
            case self.Actions.DOWN:
                dy = +1
            case self.Actions.PICKUP:
                if self.attempt_agent_pickup(agent_i):
                    # reward is 10 for successful pickup
                    reward = 10
                else:
                    # reward is -10 for failed pickup
                    reward = -10
            case self.Actions.DROPOFF:
                # reward is -10 for failed dropoff
                reward = -10
                # stop carrying victum if victum is being carried
                if carrying_vic is not None:
                    self.agents_carrying_victum[carrying_vic] = -1
                    # reward is 10 for successful dropoff
                    reward = 10
                    # end the episode if all victums are in goal zone
                    done = np.all(np.isin(self.victum_locations, self.goals))
            # case self.Actions.COMMUNICATE:
            # # case 6:
            #     # no reward for communication
            #     # reward = 0
            #     # exchange data with agents in range
            #     for agent in agents_in_range:
            #         if agent_i != agent:
            #             self.likely_victum_locations[agent_i] = self.likely_victum_locations[agent]
            #             self.agent_location_visits[agent_i] += self.agent_location_visits[agent]
            case _:
                pass
        # exchange info automatically
        for agent in agents_in_range:
            if agent_i != agent:
                self.likely_victum_locations[agent_i] = self.likely_victum_locations[agent]
                # self.agent_location_visits[agent_i] = np.add(self.agent_location_visits[agent_i], self.agent_location_visits[agent])

        # change states for movement if selected
        self.move_agent(agent_i, dx, dy, carrying_vic)
        # update data for in range victums
        for vic in victums_in_range:
            vic_loc = self.victum_locations[vic]
            self.likely_victum_locations[agent_i, vic] = vic_loc

        # draw changes to screen if enabled
        if self.render_mode == 'human':
            self.render_grid()

        # format observation data
        suggested_locs = self.likely_victum_locations[agent_i]
        visited_locs = self.agent_location_visits[agent_i]
        carrying = carrying_vic != None
        obs = agent_i, self.agent_locations, suggested_locs, visited_locs, carrying, self.goals
        # return the observation, reward, and termitation state
        return obs, reward, done
    
    def attempt_agent_pickup(self, agent_i):
        """make one agent attempt to pikcup a victum
           if anny are in range
        Args:
            agent_i (int): the id of the attempting agent
        Return:
            result (bool): whether or not a victum was picked up
        """
        # check if victum is at same location
        loc = self.agent_locations[agent_i]
        result = False
        for vic_i, _ in enumerate(self.victum_locations):
            vic_loc = self.victum_locations[vic_i]
            if np.array_equal(loc, vic_loc):
                self.agents_carrying_victum[vic_i] = agent_i
                result = True
        return result

    def move_agent(self, agent_i, dx, dy, carrying_vic):
        x, y = self.get_agent_2d_loc(agent_i)
        new_x = x + dx
        new_y = y + dy
        if new_x < 0 or new_x > self.grid_size-1: new_x = x
        if new_y < 0 or new_y > self.grid_size-1: new_y = y
        new_loc_1d = self.convert_loc_from_2d(new_x, new_y)
        self.set_agent_1d_loc(agent_i, new_loc_1d)
        if carrying_vic is not None:
            self.set_victum_1d_loc(carrying_vic, new_loc_1d)
        # update visited map data
        if self.agent_location_visits[agent_i][new_loc_1d] <= self.max_pheromone:
            self.agent_location_visits[agent_i][new_loc_1d] += 1
    
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
        for scout_i in self.scouts:
            loc = self.agent_locations[scout_i]
            x, y = self.convert_loc_to_2d(loc)
            pygame.draw.rect(self.screen, YELLOW, pygame.Rect(x*scale, y*scale, scale, scale))
        for rescuer_i in self.rescuers:
            loc = self.agent_locations[rescuer_i]
            x, y = self.convert_loc_to_2d(loc)
            pygame.draw.rect(self.screen, BLUE, pygame.Rect(x*scale, y*scale, scale, scale))
        # flip the display buffer to make it visible on the screen
        pygame.display.flip()
        # delay so step is clearly visible
        # time.sleep(1)

    def set_agent_1d_loc(self, agent_i, loc):
        self.agent_locations[agent_i] = loc

    def set_victum_1d_loc(self, vic_i, loc):
        self.victum_locations[vic_i] = loc

    def set_agent_2d_loc(self, agent_i, x, y):
        loc_1d = self.convert_loc_from_2d(x, y)
        self.agent_locations[agent_i] = loc_1d

    def set_victum_2d_loc(self, vic_i, x, y):
        loc_1d = self.convert_loc_from_2d(x, y)
        self.victum_locations[vic_i] = loc_1d

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
