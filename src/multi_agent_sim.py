import numpy as np

class Simulation:
    def __init__(self, num_agents) -> None:
        self.agents = np.arange(0, num_agents)

    def run_simulation(self):
        state = self.world.reset()

        proposed_actions = np.zeros(len(self.agents))
        for i, agent in enumerate(self.agents):
            proposed_actions[i] 
            
        pass


default_options = {
    'grid_size': 100,
    'num_scouts': 9,
    'num_rescuers': 1,
    'num_victums': 10,
}
    
class Environment:
    def __init__(self, options) -> None:
        # unpack the options
        self.unpack_options(options)
        # grid representing world 0 wall, 1 movable
        self.world = np.ones((self.grid_size,self.grid_size)).astype(int).flatten()
        self.movable_locations = np.nonzero(self.world)[0]
        # start and goal locations
        self.starts = np.arange(0,10)
        self.goals = np.arange(98, 100)
        # array of victum locations at random cells in the world
        self.victums = np.array([np.random.choice(self.movable_locations) for _ in range(self.num_victums)])
        self.scouts = np.array([np.random.choice(self.starts) for _ in range(self.num_scouts)])
        self.rescuers = np.array([np.random.choice(self.starts) for _ in range(self.num_rescuers)])

    def unpack_options(self, options):
        # apply all dictionary key-values as object properties
        for option in options:
            setattr(self, option, options[option])
    
    def step_agent(self, agent_n, action):
        obs = np.array([])
        reward = 0.0
        done = self.is_terminated()
        return obs, reward, done

    def is_terminated(self):
        # episode terminateds when all victums are in goal zone
        return False

    # environment must assign rewards to certain joint action state combinations from the network
    # Joint_Action_Space = ...
    # def step(network_state, action_state):
    #     pass

    # for each turn every ajent chooses an action then recives an observation of the environment from the worldoo
    
    # the world evaluates the reward to deliver based on the joint actions of all the agents
    # conflicting actions (such as attempts to move into the same space can then be resolved)

# for learning agents
# states are,
# 1. positions of each agent
# 2. positions of each victum
# 
# actions are:
# 1. movement choices for each agent



# agents
# A = np.range(0, 50)

# multi agent system