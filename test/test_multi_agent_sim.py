
from src import multi_agent_sim    # The code to test
from src.multi_agent_sim import Environment, default_options


import unittest   # The test framework
import numpy as np

class Test_Environment(unittest.TestCase):

    def setUp(self) -> None:
        self.env = Environment(default_options)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_step(self):
        expected_obs = np.array([])
        obs, r, done  = self.env.step_agent(0, 0)
        self.assertSequenceEqual(obs.tolist(), expected_obs.tolist())

    def test_unpacks_options(self):
        options = {
            'grid_size': np.random.randint(50, 200),
            'num_victums': np.random.randint(1,20),
            'num_scouts': np.random.randint(1,20), 
            'num_rescuers': np.random.randint(1,5)
        }
        env = Environment(options)

        # test if each option is set as an attrubet of the envioronment wi the same name
        attributes = vars(env)
        for opt in options:
            # self.assertIn(opt, attributes)
            self.assertIn(opt, attributes)

    def test_initializes_grid_with_correct_size(self):
        grid_size = default_options['grid_size']
        world = self.env.world
        self.assertEquals(len(world), grid_size*grid_size)

        
    def test_initializes_starting_locations_in_movable_position_in_world(self):
        starts = self.env.starts
        movable = self.env.movable_locations
        for start in starts:
            self.assertIn(start, movable)

    def test_initializes_victums_in_movable_locations(self):
        victums = self.env.victums
        movable = self.env.movable_locations
        for loc in victums:
            self.assertIn(loc, movable)

    def test_initializes_scouts_in_starting_locations(self):
        scouts = self.env.scouts
        starts = self.env.starts
        world = self.env.world
        for loc in scouts:
            self.assertIn(loc, starts)
            self.assertNotEqual(world[loc], 0)

    def test_initializes_rescuers_in_starting_locations(self):
        rescuers = self.env.rescuers
        starts = self.env.starts
        world = self.env.world
        for loc in rescuers:
            self.assertIn(loc, starts)
            self.assertNotEqual(world[loc], 0)

    

        


if __name__ == '__main__':
    unittest.main()
