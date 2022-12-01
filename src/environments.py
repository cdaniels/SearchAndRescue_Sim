import numpy as np
from enum import Enum
import time

import pygame
import math
from src.map_factory import ImageGridFactory, SimpleGridFactory

default_options = {
    'screen_size': 100,
    'grid_size': 100,
    'map_file': None,
    'num_agents': 5,
    'num_rescuers': 2,
    'num_victums': 1,
    'scout_visible_range': 2,
    'rescuer_visible_range': 1,
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

    def __init__(self, options) -> None:
        # unpack the options
        self.unpack_options(options)
        grid = self.build_grid()
        self.populate_grid(grid.flatten())

        # initialize pygame if appropriate
        if self.render_mode == 'human':
            self.init_pygame()

    def build_grid(self):
        # by default create a grid world of the appropriate size
        grid = np.array([])
        # grids are loaded with padding equal to the visible range (so that the logic for observable are is cleaner)
        if self.map_file is not None:
            grid = ImageGridFactory.load_grid(self.map_file, self.scout_visible_range)
        else:
            grid = SimpleGridFactory.load_grid(self.grid_size, self.scout_visible_range)
        return grid

    def populate_grid(self, grid):
        # grid representing world 0 wall, 1 movable
        self.agent_location_visits = np.zeros((self.num_agents, len(grid)))
        self.location_visits = np.zeros((len(grid)))
        for loc, occupied in enumerate(grid):
            if occupied == 0:
                self.agent_location_visits[:, loc] = np.inf
                self.location_visits[loc] = np.inf
        self.world = grid
        self.movable_locations = np.nonzero(self.world)[0]
        # self.agent_location_visits = np.zeros((self.num_agents, len(self.world)))
        # start and goal locations
        self.starts = self.movable_locations
        self.goals = self.movable_locations[-3:-1]
        # array of victum locations at random cells in the world
        self.accident_locations = np.array([np.random.choice(self.movable_locations) for _ in range(4)])
        # self.victum_locations = np.array([np.random.choice(self.movable_locations) for _ in range(self.num_victums)])
        self.victum_locations = np.array([np.random.choice(self.accident_locations) for _ in range(self.num_victums)])
        self.agent_locations = np.array([np.random.choice(self.starts) for _ in range(self.num_agents)])
        # simple arrays for rescuers and scouts
        self.agents = np.arange(0, self.num_agents)
        self.rescuers = self.agents[:self.num_rescuers]
        self.scouts = self.agents[self.num_rescuers:]
        # agent knowledge
        self.last_agent_communications = np.ones((self.num_agents, self.num_agents)).astype(int)*(-1)
        self.known_agent_locations = np.ones((self.num_agents, self.num_agents)).astype(int)*(-1)
        self.known_victum_locations = np.ones((self.num_agents, self.num_victums)).astype(int)*(-1)
        self.agents_carrying_victum = np.ones((self.num_agents)).astype(int)*(-1)
        self.step_count = np.zeros((self.num_agents))
        # self.trust_matrix = np.ones((self.num_agents, self.num_agents))
        for agent in self.agents:
            self.reset_agent(agent)

    def __del__(self):
        # close pygame if it was opened
        self.stop_simulation()

    def stop_simulation(self):
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
        self.screen = pygame.display.set_mode((self.screen_size * 2.5, self.screen_size))
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
        # test for agents within range then add them to list
        cells = self.cells_in_range(agent_i)
        agents_in_range = list()
        for cell in cells:
            for other_i, loc in enumerate(self.agent_locations):
                if loc == cell: agents_in_range.append(other_i)
        return  agents_in_range

    def is_agent_rescuer(self, agent_i):
        return agent_i in self.rescuers

    def is_agent_scout(self, agent_i):
        return agent_i in self.scouts

    def victums_in_range(self, agent_i):
        # test for victums within range then add them to list
        cells = self.cells_in_range(agent_i)
        victums_in_range = list()
        for cell in cells:
            for vic_i, loc in enumerate(self.victum_locations):
                if loc == cell: victums_in_range.append(vic_i)
        return victums_in_range

    def cells_in_range(self, agent_i):
        # get the agent location
        agent_loc = self.agent_locations[agent_i]
        # test for cells within range then add them to list
        cells_in_range = list()
        visible_range = self.rescuer_visible_range if agent_i in self.rescuers else self.scout_visible_range
        for loc, occupied in enumerate(self.world):
            dist = self.manhatten_distance(loc, agent_loc)
            if dist <= visible_range:
                cells_in_range.append(loc)
        return cells_in_range

    def cell_visits_in_range(self, agent_i):
        cells = self.cells_in_range(agent_i)
        visits = np.array([self.location_visits[loc] for loc in cells])
        return visits

    def reset_agent(self, agent_i):
        for space in self.movable_locations:
            self.agent_location_visits[agent_i][space] = 0
        self.last_agent_communications[agent_i][:] = 0
        self.known_agent_locations[agent_i][:] = -1
        self.known_victum_locations[agent_i][:] = -1
        # return the initial observation data
        obs = self.get_observation_for_agent(agent_i)
        return obs
    
    def step_agent(self, agent_i, action):
        self.step_count[agent_i] += 1
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
                else:
                    reward = -10 # reward is -10 for failed dropoff
            case self.Actions.COMMUNICATE:
                if not self.attempt_agent_communicate(agent_i):
                    reward = -10 # reward is -10 for failed communication
            case _:
                pass
        # update state space for selected action
        self.move_agent(agent_i, dx, dy)
        self.update_data_for_agents_in_range(agent_i)
        self.update_data_for_victums_in_range(agent_i)
        # draw changes to screen if enabled
        if self.render_mode == 'human':
            self.render()
        # format observation data
        obs = self.get_observation_for_agent(agent_i)
        # return the observation, reward, and termitation state
        return obs, reward, done

    def get_observation_for_agent(self, agent_i):
        known_victum_locs = self.known_victum_locations[agent_i]
        known_agent_locs = self.known_agent_locations[agent_i]
        last_agent_comms = self.last_agent_communications[agent_i]
        observed_area = self.cell_visits_in_range(agent_i)
        carrying = self.check_agent_carrying_victum(agent_i)
        obs = agent_i, known_agent_locs, known_victum_locs, last_agent_comms, observed_area, carrying, self.goals
        return obs

    def update_data_for_victums_in_range(self, agent_i):
        # check for victums in range
        victums_in_range = self.victums_in_range(agent_i)
        for vic in victums_in_range:
            vic_loc = self.victum_locations[vic]
            self.known_victum_locations[agent_i, vic] = vic_loc

    def update_data_for_agents_in_range(self, agent_i):
        # check for agents in range
        agents_in_range = self.agents_in_range(agent_i)
        for agent in agents_in_range:
            agent_loc = self.agent_locations[agent]
            self.known_agent_locations[agent_i, agent] = agent_loc

    def attempt_agent_communicate(self, agent_i):
        # check for other agents in range
        in_range_agents = self.agents_in_range(agent_i)
        # exchange info automatically
        result = False
        for other in in_range_agents:
            if agent_i != other:
                self.exchange_victum_data(agent_i, other)
                self.last_agent_communications[agent_i][other] = self.step_count[agent_i]
                result = True
        return result
            
    def exchange_victum_data(self, agent: int, other: int):
        for i, vic_loc in enumerate(self.known_victum_locations[agent]):
            if vic_loc < 0:
                self.known_victum_locations[agent][i] = self.known_victum_locations[other][i]

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
        # sum all the visits for a global value
        self.location_visits = np.sum(self.agent_location_visits, axis=0)
    
    def render(self):
        self.render_grid()
        self.render_perception_data()
        self.update_screen()
    
    def render_grid(self):
        # fill the display buffor on the screen
        self.screen.fill(BLACK)
        # draw the empty spaces
        for space in self.movable_locations:
            visit_count = int(self.location_visits[space])
            grey_color = self.grey_scale_for_visit_count(visit_count)
            self.draw_color_at_cell(grey_color, space)
        # draw the agents, victums, and goals in the display buffer
        for goal in self.goals:
            self.draw_color_at_cell(GREEN, goal)
        for victum_loc in self.victum_locations:
            self.draw_color_at_cell(RED, victum_loc)
        for scout_i in self.scouts:
            loc = self.agent_locations[scout_i]
            self.draw_color_at_cell(YELLOW, loc)
        for rescuer_i in self.rescuers:
            loc = self.agent_locations[rescuer_i]
            if self.agents_carrying_victum[rescuer_i] >= 0:
                self.draw_color_at_cell(PURPLE, loc)
            else:
                self.draw_color_at_cell(BLUE, loc)
        
    def render_perception_data(self):
        i = 0
        for scout in self.scouts:
            self.render_agent_perception_at_index(scout, i)
            i += 1
        for rescuer in self.rescuers:
            self.render_agent_perception_at_index(rescuer, i)
            i += 1

    def render_agent_perception_at_index(self, agent_i, i):
        x_block_size = 100
        y_block_size = 100
        known_agent_locations = self.known_agent_locations[agent_i]
        known_victum_locations = self.known_victum_locations[agent_i]
        x_margin = 100
        y_margin = 50
        x_offset = x_margin + self.screen_size + ((i*x_block_size) % (self.screen_size))
        y_offset = y_margin + y_block_size * (i*y_block_size // (self.screen_size))
        self.draw_text_at_position(str(known_agent_locations), x_offset, y_offset)
        self.draw_text_at_position(str(known_victum_locations), x_offset, y_offset + 20)
        self.draw_observed_area(agent_i, x_offset, y_offset + 40)
        
    
    def get_row_sizes_for_visible_range(self, vis_range):
        row_sizes = np.array([1])
        if vis_range > 0:
            row_sizes = np.array([(2*n+1) for n in range(0, vis_range + 1)])
            row_sizes = np.append(row_sizes, np.flip(row_sizes[0:-1]))
        return row_sizes
        
    def draw_observed_area(self, agent, x, y):
        visible_range = self.rescuer_visible_range if agent in self.rescuers else self.scout_visible_range
        visible_area = self.cell_visits_in_range(agent)
        scale = 10
        start = 0
        end = 0
        row_sizes = self.get_row_sizes_for_visible_range(visible_range)
        for i, row_size in enumerate(row_sizes):
            end = end + row_size
            cell_row = visible_area[start:end]
            x_shift = (row_size-1) // 2
            self.draw_cell_row(cell_row, x-(x_shift*scale), y+(i*scale), scale)
            start = end

    def draw_cell_row(self, cells, x, y, scale):
        for i, cell_visits in enumerate(cells):
            color = self.grey_scale_for_visit_count(cell_visits)
            self.draw_color_square_at_position(color, scale, x+(i*scale), y)
        
    def update_screen(self):
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

    def draw_color_at_cell(self, color, loc):
        padding = 0
        scale = self.grid2screen
        x, y = self.convert_loc_to_2d(loc)
        # scale the x,= values to the screen size
        x *= scale
        y *= scale
        # shift the x value to the grid location
        x += padding
        # draw the square
        self.draw_color_square_at_position(color, scale, x, y)

    def draw_color_square_at_position(self, color, length: int, x: int, y: int):
        square = pygame.Rect(x, y, length, length)
        pygame.draw.rect(self.screen, color, square)

    def draw_text_at_position(self, text: str, x: int, y: int):
        # create a text surface object,
        font = pygame.font.Font('freesansbold.ttf', 10)
        text = font.render(text, True, GREEN, BLACK)
        textRect = text.get_rect()
        textRect.center = x, y
        # draw the text to the screen
        self.screen.blit(text, textRect)

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
