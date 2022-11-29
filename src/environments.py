import numpy as np
from enum import Enum
import time

import pygame

default_options = {
    'screen_size': 100,
    'grid_size': 100,
    'num_agents': 5,
    'num_rescuers': 2,
    'num_victums': 1,
    'visible_range': 2,
    'max_pheromone': 10,
    'render_mode': None,
    'render_delay': 0 # in seconds
}

# color
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (200,0,0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
    
class SARGridWorld:
    Actions = Enum('Actions', ['LEFT', 'DOWN', 'UP', 'RIGHT', 'COMMUNICATE', 'PICKUP', 'DROPOFF'])

    def __init__(self, options, map=None) -> None:
        # unpack the options
        self.unpack_options(options)
        # grid representing world 0 wall, 1 movable
        map = np.ones((self.grid_size,self.grid_size)).astype(int).flatten()
        # map[60:65] = 0
        # map[70:75] = 0
        self.initialize_world(map)
        # initialize pygame if appropriate
        if self.render_mode == 'human':
            self.init_pygame()

    def initialize_world(self, map):
        # grid representing world 0 wall, 1 movable
        self.world = map
        self.movable_locations = np.nonzero(self.world)[0]
        self.agent_location_visits = np.ones((self.num_agents, len(self.world)))*(np.inf)
        # self.agent_location_visits = np.zeros((self.num_agents, len(self.world)))
        self.agent_last_communicated = np.zeros((self.num_agents, self.num_agents))
        # start and goal locations
        self.starts = self.movable_locations[0:50]
        self.goals = self.movable_locations[-3:-1]
        # array of victum locations at random cells in the world
        self.accident_locations = np.array([np.random.choice(self.movable_locations) for _ in range(4)])
        # self.victum_locations = np.array([np.random.choice(self.movable_locations) for _ in range(self.num_victums)])
        self.victum_locations = np.array([np.random.choice(self.accident_locations) for _ in range(self.num_victums)])
        self.agent_locations = np.array([np.random.choice(self.starts) for _ in range(self.num_agents)])
        # simple arrays for rescuers and scouts
        self.agents = np.arange(0, self.num_agents)
        self.rescuers = np.array([np.random.choice(self.agents) for _ in range(self.num_rescuers)])
        self.scouts = self.agents[np.isin(self.agents, self.rescuers, invert=True)]
        # inter-agent trust network
        self.likely_victum_locations = np.ones((self.num_agents, self.num_victums)).astype(int)*(-1)
        self.agents_carrying_victum = np.ones((self.num_agents)).astype(int)*(-1)
        # self.trust_matrix = np.ones((self.num_agents, self.num_agents))
        for agent in self.agents:
            self.reset_agent(agent)

    def __del__(self):
        # close pygame if it was opened
        self.close()

    def close(self):
        if self.render_mode == 'human':
            pygame.display.quit()
            pygame.quit()
    
    def get_scout_actions(self):
        return [self.Actions.LEFT, self.Actions.DOWN, self.Actions.UP, self.Actions.RIGHT, self.Actions.COMMUNICATE]
        # return [action.value for action in self.Actions]
        
    def get_rescuer_actions(self):
        return [self.Actions.LEFT, self.Actions.DOWN, self.Actions.UP, self.Actions.RIGHT, self.Actions.COMMUNICATE, self.Actions.PICKUP, self.Actions.DROPOFF]
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

    def agents_in_range(self, agent_i):
        # get the agent location
        agent_loc = self.agent_locations[agent_i]
        # test for agents within range then add them to list
        agents_in_range = list()
        for other_i, loc in enumerate(self.agent_locations):
            if other_i != agent_i:
                dist = self.manhatten_distance(loc, agent_loc)
                if dist < self.visible_range:
                    agents_in_range.append(other_i)
        return agents_in_range

    def victums_in_range(self, agent_i):
        # get the agent location
        agent_loc = self.agent_locations[agent_i]
        # test for victums within range then add them to list
        victums_in_range = list()
        for vic_i, loc in enumerate(self.victum_locations):
            dist = self.manhatten_distance(loc, agent_loc)
            if dist < self.visible_range:
                victums_in_range.append(vic_i)
        return victums_in_range

    def reset_agent(self, agent_i):
        for space in self.movable_locations:
            self.agent_location_visits[agent_i][space] = 0
        # format observation data
        suggested_locs = self.likely_victum_locations[agent_i]
        visited_locs = self.agent_location_visits[agent_i]
        comm_log = self.agent_last_communicated[agent_i]
        carrying = False
        # return the initial observation data
        return agent_i, self.agent_locations, suggested_locs, visited_locs, comm_log, carrying, self.goals
    
    def step_agent(self, agent_i, action):
        # reward is -1 normally for each timestep
        reward, done = -1, False
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
                    reward = 10 # reward is 10 for successful pickup
                else:
                    reward = -10 # reward is -10 for failed pickup
            case self.Actions.DROPOFF:
                # stop carrying victum if victum is being carried
                if self.attempt_agent_dropoff(agent_i):
                    reward = 10 # reward is 10 for successful dropoff
                    done = self.check_termination_condition()
                    if done:
                        self.close()
                else:
                    reward = -10 # reward is -10 for failed dropoff
            case self.Actions.COMMUNICATE:
                self.exchange_data_with_agents_in_range(agent_i)
            case _:
                pass
        # update state space for selected action
        self.move_agent(agent_i, dx, dy)
        self.update_data_for_victums_in_range(agent_i)
        # self.exchange_data_with_agents_in_range(agent_i)

        # draw changes to screen if enabled
        if self.render_mode == 'human':
            self.render_grid()

        # format observation data
        suggested_locs = self.likely_victum_locations[agent_i]
        visit_log = self.agent_location_visits[agent_i]
        comm_log = self.agent_last_communicated[agent_i]
        carrying = self.check_agent_carrying_victum(agent_i)
        obs = agent_i, self.agent_locations, suggested_locs, visit_log, comm_log, carrying, self.goals
        # return the observation, reward, and termitation state
        return obs, reward, done

    def update_data_for_victums_in_range(self, agent_i):
        # check for victums in range
        victums_in_range = self.victums_in_range(agent_i)
        for vic in victums_in_range:
            vic_loc = self.victum_locations[vic]
            self.likely_victum_locations[agent_i, vic] = vic_loc

    def exchange_data_with_agents_in_range(self, agent_i):
        # check for other agents in range
        agents_in_range = self.agents_in_range(agent_i)
        # exchange info automatically
        for agent in agents_in_range:
            if agent_i != agent:
                self.likely_victum_locations[agent_i] = self.likely_victum_locations[agent]
                # self.agent_location_visits[agent_i] = np.add(self.agent_location_visits[agent_i], self.agent_location_visits[agent])

    def check_agent_carrying_victum(self, agent_i):
        carrying_vic = self.agents_carrying_victum[agent_i]
        # -1 represents no victum
        return carrying_vic >= 0 

    def check_termination_condition(self):
        """ Check whether the game's termination condition has been fulfilled

        Returns:
            (bool): whether or not all victums are in goal
        """
        return np.all(np.isin(self.victum_locations, self.goals))
    
    def attempt_agent_pickup(self, agent_i):
        """ Make one agent attempt to pikcup a victum if anny are in range

        Args:
            agent_i (int): the id of the attempting agent
        Returns:
            (bool): whether or not a victum was picked up
        """
        # check if victum is at same location
        loc = self.agent_locations[agent_i]
        result = False
        for vic_i, _ in enumerate(self.victum_locations):
            vic_loc = self.victum_locations[vic_i]
            if np.array_equal(loc, vic_loc):
                self.agents_carrying_victum[agent_i] = vic_i
                result = True
        return result

    def attempt_agent_dropoff(self, agent_i):
        """ Make one agent attempt to dropoff a victum
           if one is being carried
        Args:
            agent_i (int): the id of the attempting agent
        Return:
            result (bool): whether or not a victum was dropped up
        """
        result = False
        if self.check_agent_carrying_victum(agent_i):
            # -1 represents no victum
            self.agents_carrying_victum[agent_i] = -1
            result = True
        return result

    def move_agent(self, agent_i, dx, dy):
        x, y = self.get_agent_2d_loc(agent_i)
        new_x = x + dx
        new_y = y + dy
        if new_x < 0 or new_x > self.grid_size-1: new_x = x
        if new_y < 0 or new_y > self.grid_size-1: new_y = y
        new_loc_1d = self.convert_loc_from_2d(new_x, new_y)
        self.set_agent_1d_loc(agent_i, new_loc_1d)
        # move victum if being carried
        carrying_vic = self.agents_carrying_victum[agent_i]
        if carrying_vic >= 0:
            self.set_victum_1d_loc(carrying_vic, new_loc_1d)
        self.update_map_with_visit(agent_i, new_loc_1d)

    def update_map_with_visit(self, agent_i, loc):
        # update visited map data
        if self.agent_location_visits[agent_i][loc] < self.max_pheromone:
            self.agent_location_visits[agent_i][loc] += 1
    
    def render_grid(self):
        # fill the display buffor on the screen
        self.screen.fill(BLACK)
        # draw the empty spaces
        visits = np.sum(self.agent_location_visits, axis=0)
        for space in self.movable_locations:
            visit_count = int(visits[space])
            grey_color = self.grey_scale_for_visit_count(visit_count)
            self.draw_color_at_location(grey_color, space)
        # draw the agents, victums, and goals in the display buffer
        for goal in self.goals:
            self.draw_color_at_location(GREEN, goal)
        for victum_loc in self.victum_locations:
            self.draw_color_at_location(RED, victum_loc)
        for scout_i in self.scouts:
            loc = self.agent_locations[scout_i]
            self.draw_color_at_location(YELLOW, loc)
        for rescuer_i in self.rescuers:
            loc = self.agent_locations[rescuer_i]
            if self.agents_carrying_victum[rescuer_i] >= 0:
                self.draw_color_at_location(PURPLE, loc)
            else:
                self.draw_color_at_location(BLUE, loc)
        # flip the display buffer to make it visible on the screen
        pygame.display.flip()
        # delay so step is clearly visible
        if self.render_delay > 0:
            time.sleep(self.render_delay)

    def grey_scale_for_visit_count(self, visit_count):
        grey_scale = 255
        if visit_count > 0 and visit_count < 255:
            grey_scale = 255 - visit_count*10 
        return (grey_scale, grey_scale, grey_scale)

    def draw_color_at_location(self, color, loc):
        scale = self.grid2screen
        x, y = self.convert_loc_to_2d(loc)
        pygame.draw.rect(self.screen, color, pygame.Rect(x*scale, y*scale, scale, scale))

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
        x = int(loc % self.grid_size)
        y = int(loc // self.grid_size)
        return x, y

    def convert_loc_from_2d(self, x, y):
        loc_1d = int(y * self.grid_size + x)
        return loc_1d
