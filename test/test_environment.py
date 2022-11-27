
from MARL_Sim.src import environment    # The code to test
from MARL_Sim.src.environment import Environment, default_options


import unittest   # The test framework
import numpy as np

class Test_Environment(unittest.TestCase):

    def setUp(self) -> None:
        self.env = Environment(default_options)
        self.agents = np.arange(0, len(self.env.agent_locations))
        self.victums = np.arange(0, len(self.env.victum_locations))
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_step(self):
        expected_obs = np.array([])
        obs, r, done  = self.env.step_agent(0, 0)
        self.assertSequenceEqual(obs.tolist(), expected_obs.tolist())

    def test_unpacks_options(self):
        options = {
            'screen_size': np.random.randint(200, 1000),
            'grid_size': np.random.randint(50, 200),
            'num_victums': np.random.randint(1,20),
            'num_agents': np.random.randint(1,20), 
            'num_rescuers': np.random.randint(1,5),
            'render_mode': 'human'
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
        for agent in rescuers:
            self.assertIn(agent, self.agents)
            self.assertEquals(len(rescuers), num_rescuers)

    def test_agent_selects_left_action(self):
        act = Environment.Actions.LEFT
        agent = np.random.choice(self.agents)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[0] != 0:
            self.assertEquals(initial_loc[0] - 1, next_loc[0])
        else:
            self.assertEquals(initial_loc[0], next_loc[0])

    def test_agent_selects_right_action(self):
        act = Environment.Actions.RIGHT
        agent = np.random.choice(self.agents)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[0] != self.env.grid_size:
            self.assertEquals(initial_loc[0] + 1, next_loc[0])
        else:
            self.assertEquals(initial_loc[0], next_loc[0])

    def test_agent_selects_up_action(self):
        act = Environment.Actions.UP
        agent = np.random.choice(self.agents)
        self.env.set_agent_2d_loc(agent, 2, 1)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[1] != 0:
            self.assertEquals(initial_loc[1] - 1, next_loc[1])
        else:
            self.assertEquals(initial_loc[1], next_loc[1])

    def test_agent_selects_down_action(self):
        act = Environment.Actions.DOWN
        agent = np.random.choice(self.agents)
        initial_loc = self.env.get_agent_2d_loc(agent)
        self.env.step_agent(agent, act)
        next_loc = self.env.get_agent_2d_loc(agent)
        if initial_loc[1] != self.env.grid_size:
            self.assertEquals(initial_loc[1] + 1, next_loc[1])
        else:
            self.assertEquals(initial_loc[1], next_loc[1])


    def test_agent_move_actions_form_cycles(self):
        # move the agent in a circle
        up_act = Environment.Actions.UP
        right_act = Environment.Actions.RIGHT
        down_act = Environment.Actions.DOWN
        left_act = Environment.Actions.LEFT
        agent = np.random.choice(self.agents)
        # put agent in upper left corner
        self.env.set_agent_2d_loc(agent, 0, 0)
        initial_loc = self.env.get_agent_2d_loc(agent).tolist()
        self.env.step_agent(agent, down_act)
        self.env.step_agent(agent, right_act)
        self.env.step_agent(agent, left_act)
        self.env.step_agent(agent, up_act)
        next_loc = self.env.get_agent_2d_loc(agent).tolist()
        # agent should be in the same location afterward
        self.assertEquals(initial_loc, next_loc)

    def test_rescuer_pickup_action_moves_victum(self):
        agent = np.random.choice(self.agents)
        victum = np.random.choice(self.victums)
        pick_act = Environment.Actions.PICKUP
        right_act = Environment.Actions.RIGHT
        down_act = Environment.Actions.DOWN
        # move the agent to the victum location
        self.env.set_agent_2d_loc(agent, 0, 0)
        self.env.set_victum_2d_loc(victum, 0, 0)
        # pickup the victum and move the agent to a new location
        self.env.step_agent(agent, pick_act)
        self.env.step_agent(agent, right_act)
        self.env.step_agent(agent, down_act)
        # check that the victum has moved to the new location too
        vic_loc = self.env.get_victum_2d_loc(victum).tolist()
        agent_loc = self.env.get_agent_2d_loc(agent).tolist()
        self.assertEqual(vic_loc, agent_loc)


    def test_rescuer_dropoff_action_stops_moving_victum(self):
        agent = np.random.choice(self.agents)
        victum = np.random.choice(self.victums)
        pick_act = Environment.Actions.PICKUP
        drop_act = Environment.Actions.DROPOFF
        right_act = Environment.Actions.RIGHT
        down_act = Environment.Actions.DOWN
        # move the agent to the victum location
        self.env.set_agent_2d_loc(agent, 0, 0)
        self.env.set_victum_2d_loc(victum, 0, 0)
        # pickup the victum and move the agent to a new location
        self.env.step_agent(agent, pick_act)
        self.env.step_agent(agent, right_act)
        self.env.step_agent(agent, down_act)
        self.env.step_agent(agent, drop_act)
        drop_loc = self.env.get_agent_2d_loc(agent).tolist()
        self.env.step_agent(agent, right_act)
        self.env.step_agent(agent, right_act)
        # check that the victum has moved to the drop loctaion but not further
        vic_loc = self.env.get_victum_2d_loc(victum).tolist()
        agent_loc = self.env.get_agent_2d_loc(agent).tolist()
        self.assertEqual(vic_loc, drop_loc)
        self.assertNotEqual(vic_loc, agent_loc)


    def test_reward_for_each_time_step(self):
        agent = np.random.choice(self.agents)
        # make a movement
        up_act = Environment.Actions.UP
        down_act = Environment.Actions.DOWN
        right_act = Environment.Actions.RIGHT
        left_act = Environment.Actions.LEFT
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
        pick_act = Environment.Actions.PICKUP
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
        pick_act = Environment.Actions.PICKUP
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
        pick_act = Environment.Actions.PICKUP
        drop_act = Environment.Actions.DROPOFF
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
        drop_act = Environment.Actions.DROPOFF
        # move the agent to the RRRuuuulocation
        loc1 = self.env.movable_locations[0]
        loc2 = self.env.movable_locations[1]
        self.env.set_agent_1d_loc(agent, loc1)
        self.env.set_victum_1d_loc(victum, loc2)
        # attempt to drop the victum without them being picked up
        _, reward, _ = self.env.step_agent(agent, drop_act)
        self.assertEqual(reward, -10)

    def test_game_doesnt_terminate_when_victums_not_at_goal(self):
        agent = np.random.choice(self.agents)
        pick_act = Environment.Actions.RIGHT
        # perform an arbitrary action (with victums not at goal)
        _, _, done = self.env.step_agent(agent, pick_act)
        self.assertEqual(done, False)

    def test_game_terminates_when_victums_at_goal(self):
        agent = np.random.choice(self.agents)
        pick_act = Environment.Actions.PICKUP
        drop_act = Environment.Actions.DROPOFF
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


if __name__ == '__main__':
    unittest.main()
