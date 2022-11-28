
from MARL_Sim.src import environments    # The code to test
from MARL_Sim.src.environments import SARGridWorld, default_options


import unittest   # The test framework
import numpy as np

class Test_Environment(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.agents = np.arange(0, len(self.env.agent_locations))
        self.victums = np.arange(0, len(self.env.victum_locations))
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_unpacks_options(self):
        options = {
            'screen_size': np.random.randint(200, 1000),
            'grid_size': np.random.randint(50, 200),
            'num_victums': np.random.randint(1,20),
            'num_agents': np.random.randint(1,20), 
            'num_rescuers': np.random.randint(1,5),
            'visible_range': np.random.randint(1,4),
            'render_mode': None
        }
        env = SARGridWorld(options)

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
        victums = self.env.victum_locations
        movable = self.env.movable_locations
        for loc in victums:
            self.assertIn(loc, movable)

    def test_initializes_agents_in_starting_locations(self):
        agents = self.env.agent_locations
        starts = self.env.starts
        world = self.env.world
        for loc in agents:
            self.assertIn(loc, starts)
            self.assertNotEqual(world[loc], 0)

    def test_initializes_certain_agents_as_rescuers(self):
        num_rescuers = default_options['num_rescuers']
        rescuers = self.env.rescuers
        self.assertEquals(len(rescuers), num_rescuers)
        for agent in rescuers:
            self.assertIn(agent, self.agents)

    def test_initializes_other_agents_as_scouts(self):
        num_rescuers = default_options['num_rescuers']
        num_scouts = default_options['num_agents'] - num_rescuers
        scouts = self.env.scouts
        self.assertEquals(len(scouts), num_scouts)
        for agent in scouts:
            self.assertIn(agent, self.agents)

    def test_no_scouts_are_rescuers(self):
        rescuers = self.env.rescuers
        scouts = self.env.scouts
        for agent in scouts:
            self.assertNotIn(agent, rescuers)

    def test_agent_left_action_moves_left_normally(self):
        act = SARGridWorld.Actions.LEFT
        agent = np.random.choice(self.agents)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[0] != 0:
            self.assertEqual(initial_loc[0] - 1, next_loc[0])
        else:
            self.assertEqual(initial_loc[0], next_loc[0])

    def test_agent_left_action_stops_at_boundary(self):
        act = SARGridWorld.Actions.LEFT
        agent = np.random.choice(self.agents)
        # put agent on left boundary and move left
        x = 0
        y = np.random.randint(0, self.env.grid_size-1)
        self.env.set_agent_2d_loc(agent, x, y)
        self.env.step_agent(agent, act)
        # agent should be in same location after step
        next_loc = self.env.get_agent_2d_loc(agent)
        self.assertEquals(x, next_loc[0])

    def test_agent_right_action_moves_right_normally(self):
        act = SARGridWorld.Actions.RIGHT
        agent = np.random.choice(self.agents)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[0] != self.env.grid_size-1:
            self.assertEquals(initial_loc[0] + 1, next_loc[0])
        else:
            self.assertEquals(initial_loc[0], next_loc[0])

    def test_agent_right_action_stops_at_boundary(self):
        act = SARGridWorld.Actions.RIGHT
        agent = np.random.choice(self.agents)
        # put agent on right boundary and move left
        x = self.env.grid_size-1
        y = np.random.randint(0, self.env.grid_size-1)
        self.env.set_agent_2d_loc(agent, x, y)
        self.env.step_agent(agent, act)
        # agent should be in same location after step
        next_loc = self.env.get_agent_2d_loc(agent)
        self.assertEquals(x, next_loc[0])

    def test_agent_up_action_moves_up_normally(self):
        act = SARGridWorld.Actions.UP
        agent = np.random.choice(self.agents)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[1] != 0:
            self.assertEqual(initial_loc[1] - 1, next_loc[1])
        else:
            self.assertEqual(initial_loc[1], next_loc[1])

    def test_agent_up_action_stops_at_boundary(self):
        act = SARGridWorld.Actions.UP
        agent = np.random.choice(self.agents)
        # put agent on top boundary and move up
        x = np.random.randint(0, self.env.grid_size-1)
        y = 0
        self.env.set_agent_2d_loc(agent, x, y)
        self.env.step_agent(agent, act)
        # agent should be in same location after step
        next_loc = self.env.get_agent_2d_loc(agent)
        self.assertEquals(y, next_loc[1])

    def test_agent_down_action_moves_down_normally(self):
        act = SARGridWorld.Actions.DOWN
        agent = np.random.choice(self.agents)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[1] != self.env.grid_size-1:
            self.assertEquals(initial_loc[1] + 1, next_loc[1])
        else:
            self.assertEquals(initial_loc[1], next_loc[1])

    def test_agent_down_action_stops_at_boundary(self):
        act = SARGridWorld.Actions.DOWN
        agent = np.random.choice(self.agents)
        # put agent on bottom boundary and move down
        x = np.random.randint(0, self.env.grid_size-1)
        y = self.env.grid_size - 1
        self.env.set_agent_2d_loc(agent, x, y)
        self.env.step_agent(agent, act)
        # agent should be in same location after step
        next_loc = self.env.get_agent_2d_loc(agent)
        self.assertEquals(y, next_loc[1])

    def test_agent_move_actions_form_cycles(self):
        # move the agent in a circle
        up_act = SARGridWorld.Actions.UP
        right_act = SARGridWorld.Actions.RIGHT
        down_act = SARGridWorld.Actions.DOWN
        left_act = SARGridWorld.Actions.LEFT
        agent = np.random.choice(self.agents)
        # put agent in upper left corner
        self.env.set_agent_2d_loc(agent, 0, 0)
        initial_loc = self.env.get_agent_2d_loc(agent).tolist()
        self.env.step_agent(agent, right_act)
        loc_2 = self.env.get_agent_2d_loc(agent).tolist()
        self.env.step_agent(agent, down_act)
        loc_3 = self.env.get_agent_2d_loc(agent).tolist()
        self.env.step_agent(agent, left_act)
        loc_4 = self.env.get_agent_2d_loc(agent).tolist()
        self.env.step_agent(agent, up_act)
        next_loc = self.env.get_agent_2d_loc(agent).tolist()
        # agent should be in the same location afterward
        self.assertEqual(initial_loc, next_loc)

    def test_rescuer_pickup_action_moves_victum(self):
        rescuer = np.random.choice(self.env.rescuers)
        victum = np.random.choice(self.victums)
        pick_act = SARGridWorld.Actions.PICKUP
        right_act = SARGridWorld.Actions.RIGHT
        down_act = SARGridWorld.Actions.DOWN
        # move the rescuer to the victum location
        self.env.set_agent_2d_loc(rescuer, 0, 0)
        self.env.set_victum_2d_loc(victum, 0, 0)
        # pickup the victum and move the rescuer to a new location
        self.env.step_agent(rescuer, pick_act)
        self.env.step_agent(rescuer, right_act)
        self.env.step_agent(rescuer, down_act)
        # check that the victum has moved to the new location too
        vic_loc = self.env.get_victum_2d_loc(victum).tolist()
        agent_loc = self.env.get_agent_2d_loc(rescuer).tolist()
        self.assertEqual(vic_loc, agent_loc)


    def test_rescuer_dropoff_action_stops_moving_victum(self):
        rescuer = np.random.choice(self.env.rescuers)
        victum = np.random.choice(self.victums)
        pick_act = SARGridWorld.Actions.PICKUP
        drop_act = SARGridWorld.Actions.DROPOFF
        right_act = SARGridWorld.Actions.RIGHT
        down_act = SARGridWorld.Actions.DOWN
        # move the rescuer to the victum location
        self.env.set_agent_2d_loc(rescuer, 0, 0)
        self.env.set_victum_2d_loc(victum, 0, 0)
        # pickup the victum and move the rescuer to a new location
        self.env.step_agent(rescuer, pick_act)
        self.env.step_agent(rescuer, right_act)
        self.env.step_agent(rescuer, down_act)
        self.env.step_agent(rescuer, drop_act)
        drop_loc = self.env.get_agent_2d_loc(rescuer).tolist()
        self.env.step_agent(rescuer, right_act)
        self.env.step_agent(rescuer, right_act)
        # check that the victum has moved to the drop loctaion but not further
        vic_loc = self.env.get_victum_2d_loc(victum).tolist()
        agent_loc = self.env.get_agent_2d_loc(rescuer).tolist()
        self.assertEqual(vic_loc, drop_loc)
        self.assertNotEqual(vic_loc, agent_loc)


    def test_reward_for_each_time_step(self):
        agent = np.random.choice(self.agents)
        # make a movement
        up_act = SARGridWorld.Actions.UP
        down_act = SARGridWorld.Actions.DOWN
        right_act = SARGridWorld.Actions.RIGHT
        left_act = SARGridWorld.Actions.LEFT
        # reward for any any movement should be -1
        _, reward, _ = self.env.step_agent(agent, up_act)
        self.assertEqual(reward, -1)
        _, reward, _ = self.env.step_agent(agent, right_act)
        self.assertEqual(reward, -1)
        _, reward, _ = self.env.step_agent(agent, left_act)
        self.assertEqual(reward, -1)
        _, reward, _ = self.env.step_agent(agent, down_act)
        self.assertEqual(reward, -1)

    def test_reward_for_successfull_pickup(self):
        agent = np.random.choice(self.agents)
        victum = np.random.choice(self.victums)
        pick_act = SARGridWorld.Actions.PICKUP
        # move the agent to the victum location
        rand_loc = np.random.choice(self.env.movable_locations)
        self.env.set_agent_1d_loc(agent, rand_loc)
        self.env.set_victum_1d_loc(victum, rand_loc)
        # pickup the victum and move the agent to a new location
        _, reward, _ = self.env.step_agent(agent, pick_act)
        self.assertEqual(reward, 10)

    def test_reward_for_failed_pickup(self):
        agent = np.random.choice(self.agents)
        victum = np.random.choice(self.victums)
        pick_act = SARGridWorld.Actions.PICKUP
        # move the agent to the RRRuuuulocation
        loc1 = self.env.movable_locations[0]
        loc2 = self.env.movable_locations[1]
        self.env.set_agent_1d_loc(agent, loc1)
        self.env.set_victum_1d_loc(victum, loc2)
        # pickup the victum and move the agent to a new location
        _, reward, _ = self.env.step_agent(agent, pick_act)
        self.assertEqual(reward, -10)

    def test_reward_for_successfull_dropoff(self):
        agent = np.random.choice(self.agents)
        victum = np.random.choice(self.victums)
        pick_act = SARGridWorld.Actions.PICKUP
        drop_act = SARGridWorld.Actions.DROPOFF
        # move the agent to the victum location
        goal_loc = np.random.choice(self.env.goals)
        self.env.set_agent_1d_loc(agent, goal_loc)
        self.env.set_victum_1d_loc(victum, goal_loc)
        # pickup the victum and move the agent to a new location
        self.env.step_agent(agent, pick_act)
        _, reward, _ = self.env.step_agent(agent, drop_act)
        self.assertEqual(reward, 10)

    def test_reward_for_failed_dropoff(self):
        agent = np.random.choice(self.agents)
        victum = np.random.choice(self.victums)
        drop_act = SARGridWorld.Actions.DROPOFF
        # move the agent to the RRRuuuulocation
        loc1 = self.env.movable_locations[0]
        loc2 = self.env.movable_locations[1]
        self.env.set_agent_1d_loc(agent, loc1)
        self.env.set_victum_1d_loc(victum, loc2)
        # attempt to drop the victum without them being picked up
        _, reward, _ = self.env.step_agent(agent, drop_act)
        self.assertEqual(reward, -10)

        
    def test_agent_observation_has_propper_structure(self):
        agent = np.random.choice(self.agents)
        right_act = SARGridWorld.Actions.RIGHT
        # move the agent and get the resultant observation
        obs, _, _ = self.env.step_agent(agent, right_act)
        id, agent_locs, victum_loc_suggestions, visited, carrying, goals = obs 
        # check that thet observation has the right structure
        # expected_suggestion_size = default_options['num_agents'] * default_options['num_victums']
        expected_suggestion_size = default_options['num_victums']
        self.assertEqual(agent, id)
        self.assertEqual(agent_locs.tolist(), self.env.agent_locations.tolist())
        self.assertEqual(len(victum_loc_suggestions), expected_suggestion_size)
        self.assertEqual(len(visited), len(self.env.movable_locations))
        self.assertFalse(carrying)
        self.assertEqual(goals.tolist(), self.env.goals.tolist())

    def test_game_doesnt_terminate_when_victums_not_at_goal(self):
        agent = np.random.choice(self.agents)
        pick_act = SARGridWorld.Actions.RIGHT
        # perform an arbitrary action (with victums not at goal)
        _, _, done = self.env.step_agent(agent, pick_act)
        self.assertEqual(done, False)

    def test_game_terminates_when_victums_at_goal(self):
        agent = np.random.choice(self.agents)
        pick_act = SARGridWorld.Actions.PICKUP
        drop_act = SARGridWorld.Actions.DROPOFF
        # select one of the goal locations
        goal_loc = np.random.choice(self.env.goals)
        # move all the victums to the goal location
        for victum in self.victums:
            self.env.set_victum_1d_loc(victum, goal_loc)
        # move the agent to the goal location
        self.env.set_agent_1d_loc(agent, goal_loc)
        # pickup the victum and move the agent to a new location
        self.env.step_agent(agent, pick_act)
        _, _, done = self.env.step_agent(agent, drop_act)
        self.assertEqual(done, True)

    def test_calculates_manhatten_distance_between_locations(self):
        loc1 = self.env.convert_loc_from_2d(0, 2)
        loc2 = self.env.convert_loc_from_2d(3, 0)
        dist = self.env.manhatten_distance(loc1, loc2)
        self.assertEqual(dist, 5)

    # def test_objects_in_range_returns_list_of_agents():

    def test_victum_in_range_updates_likely_location(self):
        self.assertTrue(False)

    def test_agent_movement_updates_location_visit_count(self):
        self.assertTrue(False)

    def test_agent_communication_exchanges_victum_data(self):
        self.assertTrue(False)

    def test_agent_communication_exchanges_visit_data(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
