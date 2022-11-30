
import math
import numpy as np
from src.environments import SARGridWorld, default_options

class Agent:
    def __init__(self) -> None:
        pass

    def random_argmax(self, array):
        return np.random.choice([i for i, v in enumerate(array) if v == np.max(array)])

    def random_argmin(self, array):
        return np.random.choice([i for i, v in enumerate(array) if v == np.min(array)])


class RLAgent(Agent):
    def __init__(self) -> None:
        super().__init__()


class ScoutAgent(Agent):
    def __init__(self, actions=None, env=None) -> None:
        super().__init__()
        #'LEFT', 'DOWN', 'UP', 'RIGHT', 'COMMUNICATE'
        self.A = env.get_scout_actions()
        self.env = env

    #obs = agent_i, self.agent_locations, suggested_locs, visited_locs, carrying, self.goals
    def policy(self, obs):
        id, agent_locs, victum_loc_suggestions, visited, carrying, goals = obs

        action_visit_counts = self.get_action_visit_counts(visited)
        best_action = self.A[self.random_argmin(action_visit_counts)]
        return best_action

    def get_action_visit_counts(self, visited):
        # for vis_range 2, agent pos is at index 6 in visible range
        #         0
        #     1   2   3
        # 4   5   6   7  8
        #     9  10  11
        #        12
        # left, down, up, right
        size_of_range = len(visited)
        center = size_of_range // 2
        left = center - 1
        right = center + 1
        down = center + math.ceil(size_of_range / 4)
        up = center - math.ceil(size_of_range / 4)
        nearby_visits = np.array([visited[left], visited[down], visited[up], visited[right]])
        return nearby_visits

class RescueAgent(ScoutAgent):
    def __init__(self, actions=np.arange(0,6), env=None) -> None:
        super().__init__(actions, env)
        self.A = env.get_rescuer_actions()
    
    #obs = agent_i, self.agent_locations, suggested_locs, visited_locs, carrying, self.goals
    def policy(self, obs):
        id, agent_locs, suggested_locs, visited, carrying, goals = obs
        loc = agent_locs[id]
        
        if carrying:
            if loc in goals:
                best_action = self.env.Actions.DROPOFF
            else:
                action_distances = self.get_action_distances_to_target(loc, goals[0])
                best_action = self.A[self.random_argmin(action_distances)]
        else:
            possible_locs = np.array([i for i in suggested_locs if (i >= 0 and i not in goals)])
            if len(possible_locs) > 0:
                vic_distances = self.get_victum_distances(loc, possible_locs)
                closest_vic = self.random_argmin(vic_distances)
                vic_loc = possible_locs[closest_vic]
                # if the rescuer is at the victum then pick them up
                if vic_distances[closest_vic] == 0:
                    best_action = self.env.Actions.PICKUP
                else:
                    action_distances = self.get_action_distances_to_target(loc, vic_loc)
                    best_action = self.A[self.random_argmin(action_distances)]
            else:
                best_action = super().policy(obs)
        return best_action


    def get_victum_distances(self, loc, victum_locations):
        vic_dists = np.zeros(len(victum_locations))
        for i, vic_loc in enumerate(victum_locations):
            dist = self.env.manhatten_distance(vic_loc, loc)
            vic_dists[i] = dist
        return vic_dists


    def get_action_distances_to_target(self, loc, target):
        loc = self.env.convert_loc_to_2d(loc)
        grid_size = self.env.grid_size

        left_loc = np.add(loc, [-1,0])
        down_loc = np.add(loc, [0,1])
        up_loc = np.add(loc, [0,-1])
        right_loc = np.add(loc, [1,0])

        act_distances = np.empty(4)
        for i, loc_2d in enumerate([left_loc, down_loc, up_loc, right_loc]):
            loc_1d = self.env.convert_loc_from_2d(loc_2d[0], loc_2d[1])
            if loc_2d[0] >= 0 and loc_2d[0] < grid_size and loc_2d[1] >= 0 and loc_2d[1] < grid_size:
                dist = self.env.manhatten_distance(loc_1d, target)
                act_distances[i] = dist
            else:
                act_distances[i] = np.inf
        return np.array(act_distances)


            

