
from src import agents    # The code to test
from src.agents import ScoutAgent, RescueAgent, SARGridWorld, default_options


import unittest   # The test framework
import numpy as np


class Test_ScoutAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.agent = RescueAgent(self.env.get_rescuer_actions(), self.env)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()


    def test_policy_moves_toward_smallest_visit_count(self):
        self.assertTrue(False)

    def test_policy_communicates_with_agent_in_range_if_no_recent_comm(self):
        self.assertTrue(False)

    def test_policy_doesnt_communicate_with_agent_in_range_if_recent_comm(self):
        self.assertTrue(False)

class Test_RescueAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.agent = RescueAgent(self.env.get_rescuer_actions(), self.env)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_reverts_to_scout_policy_if_no_known_victum_locations(self):
        self.assertTrue(False)

    def test_policy_moves_toward_victum_if_known_victum_locations(self):
        self.assertTrue(False)

    def test_policy_moves_toward_victum_if_known_victum_location_and_not_carrying(self):
        self.assertTrue(False)

    def test_policy_picks_up_victum_if_at_victum_location_and_not_carrying(self):
        self.assertTrue(False)

    def test_policy_moves_toward_goal_if_carrying(self):
        self.assertTrue(False)

    def test_policy_drops_off_victum_if_at_goal_location_and_carrying(self):
        self.assertTrue(False)

    # def test_calculates_manhatten_distance_to_target(self):
    #     self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
