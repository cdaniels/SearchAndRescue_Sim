
from src import agents    # The code to test
from src.agents import ScoutAgent, RescueAgent, SARGridWorld


import unittest   # The test framework
import numpy as np

default_options =  {
    'screen_size': 10,
    'grid_size': 10,
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

class Test_ScoutAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.agent = ScoutAgent(self.env.get_scout_actions(), self.env)
        grid_size = default_options['grid_size']
        # for vis_range 2, agent pos is at index 6 in visible range
        #         0
        #     1   2   3
        # 4   5   6   7  8
        #     9  10  11
        #        12
        self.obs_dict = {
            'agent_id': 0,
            'agent_locs': np.zeros(default_options['num_agents']),
            'victum_locs': np.ones(default_options['num_victums'])*(-1),
            'last_comms': np.zeros(default_options['num_agents']),
            'map_visits': np.array([
                    1,
                1,  1,  1,
            1,  1,  1,  1,  1,
                0,  0,  0,
                    0
            ]).astype(int).flatten(),
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
        obs_dict['agent_locs'][agent_id] = self.env.convert_loc_from_2d(3, 3)
        obs_dict['map_visits'] = np.array([
                1,
            1,  1,  1,
        1,  1,  1,  1,  1,
            0,  0,  0,
                0
        ]).astype(int).flatten()
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.DOWN)

    def test_policy_communicates_with_agent_in_range_if_no_recent_comm(self):
        agent_id = 0
        other_id = 1
        current_timestep = np.random.randint(0, 5)
        time_diff = 20
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = self.env.convert_loc_from_2d(3, 3)
        obs_dict['agent_locs'][other_id] = self.env.convert_loc_from_2d(4, 3)
        obs_dict['last_comms'] = np.zeros(default_options['num_agents'])
        obs_dict['last_comms'][agent_id] = current_timestep + time_diff
        obs_dict['last_comms'][other_id] = current_timestep
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.COMMUNICATE)

    def test_policy_doesnt_communicate_with_agent_in_range_if_recent_comm(self):
        agent_id = 0
        other_id = 1
        current_timestep = np.random.randint(0, 5)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = self.env.convert_loc_from_2d(3, 3)
        obs_dict['agent_locs'][other_id] = self.env.convert_loc_from_2d(4, 3)
        obs_dict['last_comms'] = np.zeros(default_options['num_agents'])
        obs_dict['last_comms'][agent_id] = current_timestep
        obs_dict['last_comms'][other_id] = current_timestep
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertNotEqual(suggested_act, self.env.Actions.COMMUNICATE)


    def test_policy_doesnt_communicate_itself(self):
        agent_id = 0
        current_timestep = np.random.randint(0, 5)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = self.env.convert_loc_from_2d(3, 3)
        obs_dict['last_comms'] = np.zeros(default_options['num_agents'])
        obs_dict['last_comms'][agent_id] = current_timestep
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertNotEqual(suggested_act, self.env.Actions.COMMUNICATE)

class Test_RescueAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.agent = RescueAgent(self.env.get_rescuer_actions(), self.env)

        grid_size = default_options['grid_size']
        self.obs_dict = {
            'agent_id': 0,
            'agent_locs': np.zeros(default_options['num_agents']),
            'victum_locs': np.ones(default_options['num_victums']).astype(int)*(-1),
            'last_comms': np.zeros(default_options['num_agents']),
            'map_visits': np.array([
                    1,
                1,  1,  1,
            1,  1,  1,  1,  1,
                0,  0,  0,
                    0
            ]).astype(int).flatten(),
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
        obs_dict['agent_locs'][agent_id] = self.env.convert_loc_from_2d(3, 3)
        obs_dict['map_visits'] = np.array([
                1,
            1,  2,  1,
        1,  1,  1,  2,  1,
            0,  2,  0,
                2
        ]).astype(int).flatten()
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.LEFT)

    def test_policy_moves_toward_victum_if_known_victum_location_and_not_carrying(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(3, 3)
        vic_loc = self.env.convert_loc_from_2d(4, 3)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.RIGHT)

    def test_policy_doesnt_move_toward_victum_if_at_goal(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(3, 3)
        vic_loc = self.env.convert_loc_from_2d(4, 4)
        goal_loc = self.env.convert_loc_from_2d(4, 4)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs_dict['map_visits'] = np.array([
                1,
            1,  1,  1,
        1,  1,  1,  1,  1,
            0,  0,  0,
                0
        ]).astype(int).flatten()
        obs_dict['goals'] = np.array([goal_loc])
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertNotEqual(suggested_act, self.env.Actions.RIGHT)

    def test_policy_picks_up_victum_if_at_victum_location_and_not_carrying(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(3, 3)
        vic_loc = self.env.convert_loc_from_2d(3, 3)
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
        obs_dict['goals'] = np.array([goal_loc])
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertNotEqual(suggested_act, self.env.Actions.PICKUP)

    def test_policy_moves_toward_goal_if_carrying(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(4, 4)
        vic_loc = self.env.convert_loc_from_2d(4, 4)
        goal_loc = self.env.convert_loc_from_2d(4, 3)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs_dict['carrying'] = True
        obs_dict['goals'] = np.array([goal_loc])
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.UP)

    def test_policy_drops_off_victum_if_at_goal_location_and_carrying(self):
        agent_id = 0
        agent_loc = self.env.convert_loc_from_2d(3, 3)
        vic_loc = self.env.convert_loc_from_2d(3, 3)
        goal_loc = self.env.convert_loc_from_2d(3, 3)
        obs_dict = self.obs_dict.copy()
        obs_dict['agent_id'] = agent_id
        obs_dict['agent_locs'][agent_id] = agent_loc
        obs_dict['victum_locs'][0] = vic_loc
        obs_dict['carrying'] = True
        obs_dict['goals'] = np.array([goal_loc])
        obs = tuple(list(obs_dict.values()))
        suggested_act = self.agent.policy(obs)
        self.assertEqual(suggested_act, self.env.Actions.DROPOFF)

if __name__ == '__main__':
    unittest.main()
