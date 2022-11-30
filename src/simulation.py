import numpy as np

from src.environments import SARGridWorld, default_options
from src.agents import ScoutAgent, RescueAgent

class Simulation:
    def __init__(self, env=SARGridWorld(default_options)) -> None:
        self.grid_world = env
        # dictionary for holding agent classes
        self.agent_dict = dict()
        self.initialize_agents(self.grid_world)

    def initialize_agents(self, env):
        # get agents for environment
        scouts = env.scouts
        rescuers = env.rescuers
        # initialize scouts
        scout_actions = np.arange(0,4)
        for i in scouts:
            agent = ScoutAgent(scout_actions, env)
            self.agent_dict[i] = agent
        # initialize rescuers
        rescuer_actions = np.arange(0,6)
        for j in rescuers:
            agent = RescueAgent(rescuer_actions, env)
            self.agent_dict[j] = agent
        
    def run_simulation(self):
        env_agents = self.grid_world.agents
        terminated = False

        agent_actions = list()
        for i in self.agent_dict:
            obs = self.grid_world.reset_agent(i)
            act = self.agent_dict[i].policy(obs)
            agent_actions.append(act)

        while not terminated:
            for i in env_agents:
                agent = self.agent_dict[i]
                act = agent_actions[i]
                obs, reward, terminated = self.grid_world.step_agent(i, act)

                # setup the next action for the agent
                agent_actions[i] = agent.policy(obs)
                # termination must break the loop or other agents will reset it
                if(terminated): break
        self.grid_world.stop_simulation()
            

    # environment must assign rewards to certain joint action state combinations from the network
    # Joint_Action_Space = ...
    # def step(network_state, action_state):
    #     pass

    # for each turn every ajent chooses an action then recives an observation of the environment from the worldoo
    
    # the world evaluates the reward to deliver based on the joint actions of all the agents
    # conflicting actions (such as attempts to move into the same space can then be resolved)