
from src import environments    # The code to test
from src.environments import SARGridWorld, default_options


import unittest   # The test framework
import numpy as np

class Test_Environment(unittest.TestCase):

    def setUp(self) -> None:
        self.env = SARGridWorld(default_options)
        self.agents = np.arange(0, len(self.env.agent_locations))
        self.scouts = np.arange(0, len(self.env.scouts))
        self.rescuers = np.arange(0, len(self.env.rescuers))
        self.victums = np.arange(0, len(self.env.victum_locations))
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_unpacks_options(self):
        options = {
            'screen_size': np.random.randint(200, 1000),
            'grid_size': np.random.randint(50, 200),
            'map_file': None,
            'num_victums': np.random.randint(1,20),
            'num_agents': np.random.randint(1,20), 
            'num_rescuers': np.random.randint(1,5),
            'scout_visible_range': np.random.randint(2,4),
            'rescuer_visible_range': 1,
            'render_mode': None,
            'render_delay': 0 # in seconds
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
        self.assertEqual(len(world), grid_size*grid_size)

        
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
        self.assertEqual(len(rescuers), num_rescuers)
        for agent in rescuers:
            self.assertIn(agent, self.agents)

    def test_initializes_other_agents_as_scouts(self):
        num_rescuers = default_options['num_rescuers']
        num_scouts = default_options['num_agents'] - num_rescuers
        scouts = self.env.scouts
        self.assertEqual(len(scouts), num_scouts)
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
        self.assertEqual(x, next_loc[0])

    def test_agent_right_action_moves_right_normally(self):
        act = SARGridWorld.Actions.RIGHT
        agent = np.random.choice(self.agents)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[0] != self.env.grid_size-1:
            self.assertEqual(initial_loc[0] + 1, next_loc[0])
        else:
            self.assertEqual(initial_loc[0], next_loc[0])

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
        self.assertEqual(x, next_loc[0])

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
        self.assertEqual(y, next_loc[1])

    def test_agent_down_action_moves_down_normally(self):
        act = SARGridWorld.Actions.DOWN
        agent = np.random.choice(self.agents)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[1] != self.env.grid_size-1:
            self.assertEqual(initial_loc[1] + 1, next_loc[1])
        else:
            self.assertEqual(initial_loc[1], next_loc[1])

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
        self.assertEqual(y, next_loc[1])

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
        id, agent_locs, victum_loc_suggestions, last_comms, observed_area, carrying, goals = obs 
        # check that thet observation has the right structure
        expected_known_victum_num = default_options['num_victums']
        expected_known_agent_num = default_options['num_agents']
        self.assertEqual(agent, id)
        self.assertEqual(len(agent_locs), expected_known_agent_num)
        self.assertEqual(len(victum_loc_suggestions), expected_known_victum_num)
        self.assertEqual(len(last_comms), expected_known_agent_num)
        self.assertFalse(carrying)
        self.assertEqual(goals.tolist(), self.env.goals.tolist())

    def test_scout_observation_cell_visits_have_propper_range(self):
        # give the environment a set visible range
        vis_range = np.random.randint(0, 3)
        custom_options = default_options.copy()
        custom_options['grid_size'] = 100
        custom_options['scout_visible_range'] = vis_range
        env = SARGridWorld(custom_options)
        agent = np.random.choice(env.scouts)
        self.env.set_agent_2d_loc(agent, 50, 50)

        # manhatten distance range size
        # 4, 12, 28, ...
        # 4, (n-1)+8, (n-1)+16, ...
        expected_manhatten_range = 1 + np.sum(np.array([2**(n+1) for n in range(1, vis_range+1)]))
        right_act = SARGridWorld.Actions.RIGHT
        # move the agent and get the resultant observation
        obs, _, _ = env.step_agent(agent, right_act)
        id, agent_locs, victum_loc_suggestions, last_comms, visit_log, carrying, goals = obs 
        # check that the observed cell visits have the right size
        self.assertEqual(len(visit_log), expected_manhatten_range)

    def test_rescuer_observation_cell_visits_have_propper_range(self):
        # give the environment a set visible range
        vis_range = np.random.randint(0, 4)
        custom_options = default_options.copy()
        custom_options['rescuer_visible_range'] = vis_range
        env = SARGridWorld(custom_options)
        agent = np.random.choice(env.rescuers)

        # manhatten distance range size
        # 4, 12, 28, ...
        # 4, (n-1)+8, (n-1)+16, ...
        expected_manhatten_range = 1 + np.sum(np.array([2**(n+1) for n in range(1, vis_range+1)]))
        right_act = SARGridWorld.Actions.RIGHT
        # move the agent and get the resultant observation
        obs, _, _ = env.step_agent(agent, right_act)
        id, agent_locs, victum_loc_suggestions, last_comms, visit_log, carrying, goals = obs 
        # check that the observed cell visits have the right size
        self.assertEqual(len(visit_log), expected_manhatten_range)


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
        self.assertTrue(done)

    def test_calculates_manhatten_distance_between_locations(self):
        loc1 = self.env.convert_loc_from_2d(0, 2)
        loc2 = self.env.convert_loc_from_2d(3, 0)
        dist = self.env.manhatten_distance(loc1, loc2)
        self.assertEqual(dist, 5)

    def test_agent_outside_range_updates_likely_location(self):
        # selecting a victum and agent
        agent = np.random.choice(self.scouts)
        other = np.random.choice(self.rescuers)
        # set the other agent just outside the visible range
        vis_range = default_options['scout_visible_range']
        self.env.set_agent_2d_loc(agent, 4, 4)
        self.env.set_agent_2d_loc(other, 4 + vis_range+1, 4)
        other_loc = self.env.agent_locations[other]
        # moving still out of range should not update the likely location
        down_act = SARGridWorld.Actions.DOWN
        obs, _, _ = self.env.step_agent(agent, down_act)
        _, known_agent_locs, _, _, _, _, _ = obs 
        self.assertNotIn(other_loc, known_agent_locs)

    def test_agent_inside_range_updates_likely_location(self):
        # selecting a victum and agent
        agent = np.random.choice(self.scouts)
        other = np.random.choice(self.rescuers)
        # set the agent just outside the visible range
        vis_range = self.env.scout_visible_range
        x = 5
        y = 5
        self.env.set_agent_2d_loc(agent, x, y)
        self.env.set_agent_2d_loc(other, x + 2, y)
        other_loc = self.env.agent_locations[other]
        # moving still out of range should not update the likely location
        right_act = SARGridWorld.Actions.RIGHT
        obs, _, _ = self.env.step_agent(agent, right_act)
        _, known_agent_locs, _, _, _, _, _ = obs 
        self.assertIn(other_loc, known_agent_locs)


    def test_victum_outside_range_updates_likely_location(self):
        # selecting a victum and agent
        agent = np.random.choice(self.scouts)
        victum = np.random.choice(self.victums)
        # set the victum just outside the visible range
        vis_range = default_options['scout_visible_range']
        self.env.set_agent_2d_loc(agent, 0, 0)
        self.env.set_victum_2d_loc(victum, vis_range, 0)
        vic_loc = self.env.victum_locations[victum]
        # moving still out of range should not update the likely location
        down_act = SARGridWorld.Actions.DOWN
        obs, _, _ = self.env.step_agent(agent, down_act)
        _, _, loc_suggestions, _, _, _, _ = obs 
        self.assertNotIn(vic_loc, loc_suggestions)

    def test_victum_inside_range_updates_likely_location(self):
        # selecting a victum and agent
        agent = np.random.choice(self.scouts)
        victum = np.random.choice(self.victums)
        # set the victum just outside the visible range
        vis_range = self.env.scout_visible_range
        x = 3
        y = 3
        self.env.set_agent_2d_loc(agent, x, y)
        self.env.set_victum_2d_loc(victum, x + (vis_range-1), y)
        vic_loc = self.env.victum_locations[victum]
        # moving still out of range should not update the likely location
        right_act = SARGridWorld.Actions.RIGHT
        obs, _, _ = self.env.step_agent(agent, right_act)
        _, _, loc_suggestions, _, _, _, _ = obs 
        self.assertIn(vic_loc, loc_suggestions)

    def test_agent_movement_updates_visit_count(self):
        custom_options = default_options.copy()
        custom_options['scout_visible_range'] = 2
        env = SARGridWorld(custom_options)
        # select an agent
        agent = np.random.choice(self.scouts)
        # position the agent
        env.set_agent_2d_loc(agent, 5, 5)
        # step to the right and get the new visit counts
        # for vis_range 2, agent pos is at index 6 in visible range
        #         0
        #     1   2   3
        # 4   5   6   7  8
        #     9  10  11
        #        12
        # get the initial visit count and set the agents position
        _, _, _, _, init_observed_area, _, _ = env.reset_agent(agent)
        # get the resulting location of the next action and its initial visit count
        left_act = SARGridWorld.Actions.LEFT
        right_act = SARGridWorld.Actions.RIGHT
        initial_visit_count = init_observed_area[7]
        # step to the right and get the new visit counts
        env.step_agent(agent, right_act)
        obs, _, _ = env.step_agent(agent, left_act)
        _, _, _, _, observed_area, _, _ = obs 
        # moving should update the visit count
        next_visit_count = observed_area[7]
        self.assertEqual(next_visit_count, initial_visit_count + 1)

    def test_agent_movement_stops_updating_visit_count_at_max(self):
        # select an agent
        custom_options = default_options.copy()
        custom_options['scout_visible_range'] = 2
        env = SARGridWorld(custom_options)
        agent = np.random.choice(env.agents)
        # get the initial visit count and set the agents position
        env.set_agent_2d_loc(agent, 3, 3)
        # get the resulting location of the next action and its initial visit count
        left_act = SARGridWorld.Actions.LEFT
        right_act = SARGridWorld.Actions.RIGHT
        max_count = default_options['max_pheromone']
        # step to the right and get the new visit counts
        # for vis_range 2, agent pos is at index 6 in visible range
        #         0
        #     1   2   3
        # 4   5   6   7  8
        #     9  10  11
        #        12
        for _ in range(max_count + 1):
            env.step_agent(agent, right_act)
            env.step_agent(agent, left_act)
        obs, _, _ = env.step_agent(agent, right_act)
        _, _, _, _, next_visited, _, _ = obs 
        # moving should update the visit count
        next_loc_index = 6
        next_visit_count = next_visited[next_loc_index]
        self.assertEqual(next_visit_count, max_count)

    def test_agent_communication_succeeds_for_agents_in_range(self):
        # select a scout and a rescuer
        scout = np.random.choice(self.scouts)
        rescuer = np.random.choice(self.rescuers)
        comm_acc = SARGridWorld.Actions.COMMUNICATE
        vis_range = default_options['scout_visible_range']
        x, y = 5, 5
        self.env.set_agent_2d_loc(scout, x, y)
        self.env.set_agent_2d_loc(rescuer, x+(vis_range-1), y)

        # perform the communication action
        _, reward, _ = self.env.step_agent(scout, comm_acc)
        # reward for successful communication should be the same as nothing
        self.assertEqual(reward, -1)

    def test_agent_communication_fails_for_agents_out_of_range(self):
        # select a scout and a rescuer
        scout = np.random.choice(self.scouts)
        rescuer = np.random.choice(self.rescuers)
        comm_acc = SARGridWorld.Actions.COMMUNICATE
        vis_range = default_options['scout_visible_range']
        x, y = 4, 4
        self.env.set_agent_2d_loc(scout, x, y)
        self.env.set_agent_2d_loc(rescuer, (x+vis_range+1), y)

        # perform the communication action
        _, reward, _ = self.env.step_agent(scout, comm_acc)
        # reward should be -10 for a failed communication
        self.assertEqual(reward, -10)

    def test_agent_communication_exchanges_victum_data(self):
        # select a scout and a rescuer
        scout = np.random.choice(self.scouts)
        rescuer = np.random.choice(self.rescuers)
        victum = np.random.choice(self.victums)
        comm_acc = SARGridWorld.Actions.COMMUNICATE
        vis_range = default_options['scout_visible_range']
        x, y = 4, 4
        self.env.set_agent_2d_loc(scout, x, y)
        self.env.set_agent_2d_loc(rescuer, x+1, y)
        # make the location known by the scout but not the rescuer
        new_loc = self.env.convert_loc_from_2d(5, 5)
        self.env.known_victum_locations[scout][victum] = new_loc
        self.env.known_victum_locations[rescuer][victum] = -1

        # perform the communication action
        obs, _, _ = self.env.step_agent(scout, comm_acc)
        _, _, known_victum_locs, _, _, _, _ = obs 

        # the rescuer should now know the new location
        self.assertEqual(known_victum_locs[victum], new_loc)

    def test_agent_communication_updates_last_communication_data(self):
        # select a scout and a rescuer
        scout = np.random.choice(self.scouts)
        rescuer = np.random.choice(self.rescuers)
        comm_acc = SARGridWorld.Actions.COMMUNICATE
        x, y = 5, 5
        self.env.set_agent_2d_loc(scout, x, y)
        self.env.set_agent_2d_loc(rescuer, x+1, y)
        old_step_count = self.env.last_agent_communications[scout][rescuer]

        # perform the communication action
        obs, _, _ = self.env.step_agent(scout, comm_acc)
        _, _, _, last_agent_comm, _, _, _ = obs 

        # the rescuer should now have registered a communication one step later
        self.assertEqual(last_agent_comm[rescuer], old_step_count + 1)

    def test_convert_loc_returns_int(self):
        # convert function should even convert float coordinates to ints
        loc_1d = self.env.convert_loc_from_2d(0.0, 1.0)
        self.assertIsInstance(loc_1d, int)

    def test_calculates_propper_number_of_rows_for_visible_range(self):
        vis_ranges = np.arange(0,5)
        # rows should be of length, 1, 3, 5, 7, ...
        for i, vis_range in enumerate(vis_ranges):
            row_sizes = self.env.get_row_sizes_for_visible_range(vis_range)
            self.assertEqual(len(row_sizes), (2*i+1))

    def test_calculates_propper_row_sizes_for_visible_range(self):
        vis_range = 2
        row_sizes = self.env.get_row_sizes_for_visible_range(vis_range)
        self.assertEqual(row_sizes[0], 1)
        self.assertEqual(row_sizes[1], 3)
        self.assertEqual(row_sizes[2], 5)
        self.assertEqual(row_sizes[3], 3)
        self.assertEqual(row_sizes[4], 1)
        

if __name__ == '__main__':
    unittest.main()
