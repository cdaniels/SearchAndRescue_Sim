import numpy as np
from enum import Enum
import cv2


EMPTY = 0
WALL = 1

class GridFactory:
    def load_grid(data: any) -> np.array:
        """ takes some form of data and returns a grid array

        Args:
            data (any): data to build the grid with

        Returns:
            np.array: the grid array
        """
        # overwritten by subclasses
        pass
    
class SimpleGridFactory(GridFactory):
    def load_grid(grid_length: int, padding: int) -> np.array:
        """tabes an integer representing side length and returns
        a simple grid consisting of a binary numpy array containing 0 to represent walls
        and 1s to represent free space

        Args:
            grid_length (int): the length of the grid

        Returns:
            np.array: the grid array
        """
        grid = np.ones((grid_length, grid_length)).astype(int)
        padded = SimpleGridFactory.pad_grid(grid, grid_length, padding)
        return padded.flatten()

    def pad_grid(map_array, grid_length, pad_length): 
        # 0 wall, 1 movable
        map_array[0:pad_length, :] = 0 # left edge
        map_array[:, 0:pad_length] = 0 # top edge
        map_array[grid_length-pad_length:grid_length, :] = 0 # right edge
        map_array[:, grid_length-pad_length:grid_length] = 0 # bottom edge
        return map_array

class ImageGridFactory(GridFactory):
    def load_grid(grid_file: str, padding: int) -> np.array:
        """takes a file (either saved pixel data or an image) 
        and converts it to a binary numby array containing 0 to represent walls
        and 1s to represent free space

        Args:
            grid_file (str): the file to load

        Returns:
            np.array: the grid array
        """
        grid = np.array([])
        if grid_file.endswith("npy"):
            grid = np.load(grid_file)
        else:
            grid = cv2.imread(grid_file, 0)
        return ImageGridFactory.validate_grid(grid, padding)


    def is_padded(grid: np.array, pad_val: int, pad_length: int):
        """check if grid is padded with pad_val"""
        return np.all(grid[0:pad_length, :] == pad_val) and np.all(grid[:, 0:pad_length] == pad_val) and np.all(grid[-1:-pad_length, :] == pad_val) and np.all(grid[:, -1:-pad_length] == pad_val)


    def pad_grid(grid: np.array, pad_length: int) -> np.array:
        """pad grid with infinities"""
        return np.pad(grid, pad_width=pad_length, mode='constant', constant_values=WALL)


    def validate_grid(grid: np.array, pad_length: int) -> np.array:
        """make sure float grid is padded with infinities
        make sure the grid has only 0 and infinity else."""
        grid = grid.astype(float)
        if len(grid.shape) > 2:
            grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
        if not ImageGridFactory.is_padded(grid, WALL, pad_length):
            grid = ImageGridFactory.pad_grid(grid, pad_length)
        grid[grid != EMPTY] = WALL
        return (1 - grid)