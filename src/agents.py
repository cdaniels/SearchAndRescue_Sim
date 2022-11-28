import numpy as np
from environments import SARGridWorld, default_options

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

        action_visit_counts = self.get_action_visit_counts(agent_locs[id], visited)
        best_action = self.A[self.random_argmax(action_visit_counts)]
        return best_action

    def get_action_visit_counts(self, loc, visited):
        movable = self.env.movable_locations

        loc = self.env.convert_loc_to_2d(loc)

        left_loc = np.add(loc, [-1,0])
        down_loc = np.add(loc, [0,-1])
        up_loc = np.add(loc, [0,1])
        right_loc = np.add(loc, [1,0])

        nearby_visits = list()
        for loc_2d in [left_loc, down_loc, up_loc, right_loc]:
            loc_1d = self.env.convert_loc_from_2d(loc_2d[0], loc_2d[1])
            can_move = False
            for i, movable_loc in enumerate(movable):
                if loc_1d == movable_loc:
                    can_move = True
                    break
            if can_move:
                nearby_visits.append(visited[i])
            else:
                nearby_visits.append(np.inf)
        return np.array(nearby_visits)

            

        

class RescueAgent(ScoutAgent):
    def __init__(self, actions=np.arange(0,6), env=None) -> None:
        super().__init__(actions, env)
        self.A = env.get_rescuer_actions()
    
    #obs = agent_i, self.agent_locations, suggested_locs, visited_locs, carrying, self.goals
    def policy(self, obs):
        id, agent_locs, victum_loc_suggestions, visited, carrying, goals = obs
        loc = agent_locs[id]
        
        possible_victum_locs = np.array([loc for loc in victum_loc_suggestions if loc > 0])
        if len(possible_victum_locs) > 0:
            if not carrying:
                vic_distances = self.get_victum_distances(loc, possible_victum_locs)
                closest_vic = self.random_argmin(vic_distances)
                vic_loc = possible_victum_locs[closest_vic]
                # if the rescuer is at the victum then pick them up
                if vic_distances[closest_vic] == 0:
                    best_action = self.env.Actions.PICKUP
                else:
                    action_distances = self.get_action_distances_to_target(loc, vic_loc)
                    best_action = self.random_argmin(action_distances)
            else: 
                if loc in goals:
                    best_action = self.env.Actions.DROPOFF
                else:
                    action_distances = self.get_action_distances_to_target(loc, goals[0])
                    best_action = self.random_argmin(action_distances)
        else:
            action_visit_counts = self.get_action_visit_counts(loc, visited)
            best_action = self.A[self.random_argmax(action_visit_counts)]
        return best_action


    def get_victum_distances(self, loc, victum_locations):
        vic_dists = np.zeros(len(victum_locations))
        for i, vic_loc in enumerate(victum_locations):
            dist = self.env.manhatten_distance(vic_loc, loc)
            vic_dists[i] = dist
        return vic_dists


    def get_action_distances_to_target(self, loc, target):
        movable = self.env.movable_locations

        loc = self.env.convert_loc_to_2d(loc)

        left_loc = np.add(loc, [-1,0])
        down_loc = np.add(loc, [0,-1])
        up_loc = np.add(loc, [0,1])
        right_loc = np.add(loc, [1,0])

        act_distances = list()
        for loc_2d in [left_loc, down_loc, up_loc, right_loc]:
            loc_1d = self.env.convert_loc_from_2d(loc_2d[0], loc_2d[1])
            can_move = False
            for i, movable_loc in enumerate(movable):
                if loc_1d == movable_loc:
                    can_move = True
                    break
            if can_move:
                dist = self.env.manhatten_distance(loc_1d, target)
                act_distances.append(dist)
            else:
                act_distances.append(np.inf)
        return np.array(act_distances)


            

