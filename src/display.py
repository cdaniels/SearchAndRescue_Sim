import numpy as np
from enum import Enum
import time

import pygame
import math

default_options = {
    'screen_size': 100,
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
   

class DisplayVisitor:
    Actions = Enum('Actions', ['LEFT', 'DOWN', 'UP', 'RIGHT', 'COMMUNICATE', 'REASSESS', 'PICKUP', 'DROPOFF'])

    def __init__(self, options) -> None:
        # unpack the options
        self.unpack_options(options)
        self.init_pygame()

    def __del__(self):
        # close pygame if it was opened
        self.stop_simulation()

    def unpack_options(self, options):
        # apply all dictionary key-values as object properties
        for option in options:
            setattr(self, option, options[option])

    def stop_simulation(self):
        if self.render_mode == 'human':
            pygame.display.quit()
            pygame.quit()
    
    def init_pygame(self):
        # setup screen
        pygame.init()
        self.grid2screen = self.screen_size / self.grid_size
        self.screen = pygame.display.set_mode((self.screen_size * 2.5, self.screen_size))
        pygame.display.set_caption('Grid World')


    def visit(self, env):
        self.render_grid(env)
        self.render_perception_data(env)
        self.update_screen()
    
    def render_grid(self, env):
        # fill the display buffor on the screen
        self.screen.fill(BLACK)
        # draw the empty spaces
        for space in env.movable_locations:
            visit_count = int(env.location_visits[space])
            grey_color = self.grey_scale_for_visit_count(visit_count)
            self.draw_color_at_cell(env, grey_color, space)
        # draw the agents, victums, and goals in the display buffer
        for goal in env.goals:
            self.draw_color_at_cell(env, GREEN, goal)
        for victum_loc in env.victum_locations:
            self.draw_color_at_cell(env, RED, victum_loc)
        for scout_i in env.scouts:
            loc = env.agent_locations[scout_i]
            self.draw_color_at_cell(env, YELLOW, loc)
        for rescuer_i in env.rescuers:
            loc = env.agent_locations[rescuer_i]
            if env.agents_carrying_victum[rescuer_i] >= 0:
                self.draw_color_at_cell(env, PURPLE, loc)
            else:
                self.draw_color_at_cell(env, BLUE, loc)
        
    def render_perception_data(self, env):
        i = 0
        for scout in env.scouts:
            self.render_agent_perception_at_index(env, scout, i)
            i += 1
        for rescuer in env.rescuers:
            self.render_agent_perception_at_index(env, rescuer, i)
            i += 1

    def render_agent_perception_at_index(self, env, agent_i, i):
        x_block_size = 100
        y_block_size = 100
        agent_last_communications = env.last_agent_communications[agent_i]
        known_agent_locations = env.known_agent_locations[agent_i]
        known_victum_locations = env.known_victum_locations[agent_i]
        x_margin = 100
        y_margin = 50
        x_offset = x_margin + self.screen_size + ((i*x_block_size) % (self.screen_size))
        y_offset = y_margin + y_block_size * (i*y_block_size // (self.screen_size))
        self.draw_text_at_position(str(agent_last_communications), x_offset, y_offset)
        self.draw_text_at_position(str(known_agent_locations), x_offset, y_offset + 20)
        self.draw_text_at_position(str(known_victum_locations), x_offset, y_offset + 40)
        self.draw_observed_area(env, agent_i, x_offset, y_offset + 60)
        
    
    def get_row_sizes_for_visible_range(self, vis_range):
        row_sizes = np.array([1])
        if vis_range > 0:
            row_sizes = np.array([(2*n+1) for n in range(0, vis_range + 1)])
            row_sizes = np.append(row_sizes, np.flip(row_sizes[0:-1]))
        return row_sizes
        
    def draw_observed_area(self, env, agent, x, y):
        visible_range = self.rescuer_visible_range if agent in env.rescuers else self.scout_visible_range
        visible_area = env.cell_visits_in_range(agent)
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

    def draw_color_at_cell(self, env, color, loc):
        padding = 0
        scale = self.grid2screen
        x, y = env.convert_loc_to_2d(loc)
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
