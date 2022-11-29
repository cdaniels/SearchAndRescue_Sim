
from src import agents    # The code to test
from src.agents import ScoutAgent, RescueAgent, SARGridWorld


import unittest   # The test framework
import numpy as np

default_options = {
    'screen_size': 16,
    'grid_size': 4,
    'num_agents': 5,
    'num_rescuers': 2,
    'num_victums': 1,
    'visible_range': 1,
    'max_pheromone': 10,
    'render_mode': None,
    'render_delay': 0 # in seconds
}

class Test_ScoutAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.agent = ScoutAgent(self.env.get_scout_actions(), self.env)
        grid_size = default_options['grid_size']
        self.obs_dict = {
            'agent_id': 0,
            'agent_locs': np.zeros(default_options['num_agents']),
            'victum_locs': np.array([]),
            'map_visits': np.zeros((grid_size, grid_size)),
            'comm': np.zeros(default_options['num_agents']),
            'carrying': False,
            'goals': np.array([15])
        }
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_policy_moves_toward_smallest_visit_count(self):
        agent_id = 0
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = self.env.convert_loc_from_2d(0, 0)
        obs_dict['map_visits'] = np.array([
            [1,1,1,2],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ]).astype(int).flatten()
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.DOWN)

    def test_policy_communicates_with_agent_in_range_if_no_recent_comm(self):
        self.assertTrue(False)

    def test_policy_doesnt_communicate_with_agent_in_range_if_recent_comm(self):
        self.assertTrue(False)

class Test_RescueAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.agent = RescueAgent(self.env.get_rescuer_actions(), self.env)

        grid_size = default_options['grid_size']
        self.obs_dict = {
            'agent_id': 0,
            'agent_locs': np.zeros(default_options['num_agents']),
            'victum_locs': np.ones(default_options['num_victums']).astype(int)*(-1),
            'map_visits': np.zeros((grid_size, grid_size)),
            'comm': np.zeros(default_options['num_agents']),
            'carrying': False,
            'goals': np.array([15])
        }
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_reverts_to_scout_policy_if_no_known_victum_locations(self):
        agent_id = 0
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = self.env.convert_loc_from_2d(0, 0)
        obs_dict['map_visits'] = np.array([
            [1,1,1,2],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ]).astype(int).flatten()
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.DOWN)
        obs_dict['agent_locs'][agent_id] = self.env.convert_loc_from_2d(1, 3)
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.LEFT)

    def test_policy_moves_toward_victum_if_known_victum_location_and_not_carrying(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(0, 0)
        vic_loc = self.env.convert_loc_from_2d(3, 0)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs_dict['map_visits'] = np.array([
            [1,1,1,2],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ]).astype(int).flatten()
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.RIGHT)

    def test_policy_doesnt_move_toward_victum_if_at_goal(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(0, 0)
        vic_loc = self.env.convert_loc_from_2d(3, 0)
        goal_loc = self.env.convert_loc_from_2d(3, 0)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs_dict['map_visits'] = np.array([
            [1,1,1,2],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ]).astype(int).flatten()
        obs_dict['goals'] = np.array([goal_loc])
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertNotEqual(suggested_act, self.env.Actions.RIGHT)

    def test_policy_picks_up_victum_if_at_victum_location_and_not_carrying(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(0, 0)
        vic_loc = self.env.convert_loc_from_2d(0, 0)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.PICKUP)

    def test_policy_doesnt_pick_up_victum_if_at_goal(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(3, 3)
        vic_loc = self.env.convert_loc_from_2d(3, 3)
        goal_loc = self.env.convert_loc_from_2d(3, 3)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs_dict['map_visits'] = np.array([
            [1,1,1,2],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ]).astype(int).flatten()
        obs_dict['goals'] = np.array([goal_loc])
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertNotEqual(suggested_act, self.env.Actions.PICKUP)

    def test_policy_moves_toward_goal_if_carrying(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(3, 0)
        vic_loc = self.env.convert_loc_from_2d(3, 0)
        goal_loc = self.env.convert_loc_from_2d(3, 3)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs_dict['carrying'] = True
        obs_dict['goals'] = np.array([goal_loc])
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.DOWN)

    def test_policy_drops_off_victum_if_at_goal_location_and_carrying(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(3, 3)
        vic_loc = self.env.convert_loc_from_2d(3, 3)
        goal_loc = self.env.convert_loc_from_2d(3, 3)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs_dict['map_visits'] = np.array([
            [1,1,1,2],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ]).astype(int).flatten()
        obs_dict['carrying'] = True
        obs_dict['goals'] = np.array([goal_loc])
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.DROPOFF)

if __name__ == '__main__':
    unittest.main()
