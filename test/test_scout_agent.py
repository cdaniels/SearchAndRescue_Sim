
from MARL_Sim.src import agents    # The code to test
from MARL_Sim.src.agents import ScoutAgent, RescueAgent, SARGridWorld, default_options


import unittest   # The test framework
import numpy as np

class Test_RescueAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.agent = RescueAgent(self.env.get_rescuer_actions(), self.env)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()


    def test_calculates_manhatten_distance_to_target(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
