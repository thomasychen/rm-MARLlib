import random, math, os
import numpy as np
from enum import Enum

import sys
sys.path.append('../')
sys.path.append('../../')
from pathlib import Path

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict, Tuple, Discrete, Box

ROOT = Path(__file__).parent.parent.parent

policy_mapping_dict = {
    "all_scenario": {
        "description": "buttons all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

def buttons_config():
    env_settings = dict()
    env_settings['Nr'] = 10
    env_settings['Nc'] = 10
    env_settings['initial_states'] = [0, 5, 9]
    env_settings['walls'] = [(0, 2), (1, 2), (3, 2),
                                (1,4), (2,4), (3,4), (4,4), (5,4), (6,4), (7, 4),
                                (4, 2), (4, 3),
                                (1, 6), (2, 6), (3,6), (4, 6), (4, 7), (5, 7), (6, 7)]
    env_settings['yellow_button'] = (6,2)
    env_settings['green_button'] = (5,6)
    env_settings['red_button'] = (5,9)

    env_settings['p'] = 0.98

    return env_settings

class RLlibButtons(MultiAgentEnv):
    def __init__(self, env_config):
        self.env_config = env_config
        self.num_agents = 3
        self.current_step = 0
        self.episode_limit = self.env_config["max_episode_length"]
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        self.auxiliary_rm = self.env_config["joint_rm_file"]
        self.initial_rm_states = self.env_config["initial_rm_states"]
        self.envs = {self.agents[i]: HardButtonsEnv(self.auxiliary_rm, i+1, buttons_config(), self.initial_rm_states) for i in range(self.num_agents)}
        self.action_space = Discrete(5)
        self.observation_space = Dict({"obs": Box(
            low=np.array([0,0]),
            high=np.array([99,6]),
            shape=(2,),
            dtype=int,
        )})
        self.agent_mdp_states = {agent: self.envs[agent].get_initial_state() for agent in self.agents}
        self.agent_rm_states = {self.agents[i]: self.initial_rm_states[i] for i in range(self.num_agents)}
        self.reset()

    def reset(self):
        original_obs = [self.envs[agent].get_initial_state() for agent in self.agents]
        obs = {}
        for x in range(self.num_agents):
            obs["agent_%d" % x] = {
                "obs": (original_obs[x], self.initial_rm_states[x])
            }

            self.agent_mdp_states["agent_%d" % x] = original_obs[x]
            self.agent_rm_states["agent_%d" % x] = self.initial_rm_states[x]
        self.current_step = 0
        for i in range(len(self.agents)):
            self.envs[self.agents[i]] = HardButtonsEnv(self.env_config["joint_rm_file"], i+1, buttons_config(),self.initial_rm_states)
        return obs
    
    def step(self, action_dict):
        rewards = {}
        obs = {}
        terminated = {}
        info = {}
        
        for agent, value in sorted(action_dict.items()):
            r, l, s_next = self.envs[agent].environment_step(self.agent_mdp_states[agent], value)
            self.agent_mdp_states[agent] = s_next
            for e in l:
                # Get the new reward machine state and the reward of this step
                u2 = self.envs[agent].reward_machine.get_next_state(self.agent_rm_states[agent], e)
                # Update the reward machine state
                self.agent_rm_states[agent] = u2
            obs[agent] = {"obs": (s_next, self.agent_rm_states[agent])}
            rewards[agent] = r
            terminated[agent] = self.envs[agent].reward_machine.is_terminal_state(self.agent_rm_states[agent])
        # if any(rewards.values()):
        #     print(rewards)
            
        if all(terminated.values()) or self.current_step > self.episode_limit:
            terminated["__all__"] = True
        else:
            terminated["__all__"] = False

        self.current_step += 1

        return obs, rewards, terminated, info
    
    def render(self):
        for agent, env in self.envs.items():
            print(agent)
            env.show(self.agent_mdp_states[agent])
    
    def close(self):
        ...
    
    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.episode_limit,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
    
    def assign_rm_states(self, new_assignment):
        self.initial_rm_states = new_assignment

"""
Enum with the actions that the agent can execute
"""
class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left
    none  = 4 # none 

class ManagedSparseRewardMachine:
    def __init__(self,file=None):
        # <U,u0,delta_u,delta_r>
        self.U = []       # list of machine states
        self.events = set() # set of events
        self.u0 = None    # initial state
        self.delta_u = {} # state-transition function
        self.delta_r = {} # reward-transition function
        self.T = set()    # set of terminal states (they are automatically detected)
        if file is not None:
            self._load_reward_machine(file)
        
    def __repr__(self):
        s = "MACHINE:\n"
        s += "init: {}\n".format(self.u0)
        for trans_init_state in self.delta_u:
            for event in self.delta_u[trans_init_state]:
                trans_end_state = self.delta_u[trans_init_state][event]
                s += '({} ---({},{})--->{})\n'.format(trans_init_state,
                                                        event,
                                                        self.delta_r[trans_init_state][trans_end_state],
                                                        trans_end_state)
        return s

    # Public methods -----------------------------------

    def load_rm_from_file(self, file):
        self._load_reward_machine(file)

    # def get_initial_state(self):
    #     return self.u0

    def get_next_state(self, u1, event):
        if u1 in self.delta_u:
            if event in self.delta_u[u1]:
                return self.delta_u[u1][event]
        return u1

    def get_reward(self,u1,u2,s1=None,a=None,s2=None):
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            return self.delta_r[u1][u2]
        return 0 # This case occurs when the agent falls from the reward machine

    def get_rewards_and_next_states(self, s1, a, s2, event):
        rewards = []
        next_states = []
        for u1 in self.U:
            u2 = self.get_next_state(u1, event)
            rewards.append(self.get_reward(u1,u2,s1,a,s2))
            next_states.append(u2)
        return rewards, next_states

    def get_states(self):
        return self.U

    def is_terminal_state(self, u1):
        return u1 in self.T

    def get_events(self):
        return self.events

    def is_event_available(self, u, event):
        is_event_available = False
        if u in self.delta_u:
            if event in self.delta_u[u]:
                is_event_available = True
        return is_event_available

    # Private methods -----------------------------------

    def _load_reward_machine(self, file):
        """
        Example:
            0                  # initial state
            (0,0,'r1',0)
            (0,1,'r2',0)
            (0,2,'r',0)
            (1,1,'g1',0)
            (1,2,'g2',1)
            (2,2,'True',0)

            Format: (current state, next state, event, reward)
        """
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()
        # setting the DFA
        self.u0 = eval(lines[0])
        # adding transitions
        for e in lines[1:]:
            self._add_transition(*eval(e))
            self.events.add(eval(e)[2]) # By convention, the event is in the spot indexed by 2
        # adding terminal states
        for u1 in self.U:
            if self._is_terminal(u1):
                self.T.add(u1)
        self.U = sorted(self.U)

        # print("pls", self.T)

    def calculate_reward(self, trace):
        total_reward = 0
        current_state = self.get_initial_state()

        for event in trace:
            next_state = self.get_next_state(current_state, event)
            reward = self.get_reward(current_state, next_state)
            total_reward += reward
            current_state = next_state
        return total_reward

    def _is_terminal(self, u1):
        # Check if reward is given for reaching the state in question
        for u0 in self.delta_r:
            if u1 in self.delta_r[u0]:
                if self.delta_r[u0][u1] == 1:
                    return True
        return False
            
    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U:
                self.U.append(u)

    def _add_transition(self, u1, u2, event, reward):
        # Adding machine state
        self._add_state([u1,u2])
        # Adding state-transition to delta_u
        if u1 not in self.delta_u:
            self.delta_u[u1] = {}
        if event not in self.delta_u[u1]:
            self.delta_u[u1][event] = u2
        else:
            raise Exception('Trying to make rm transition function non-deterministic.')
            # self.delta_u[u1][u2].append(event)
        # Adding reward-transition to delta_r
        if u1 not in self.delta_r:
            self.delta_r[u1] = {}
        self.delta_r[u1][u2] = reward

class HardButtonsEnv:

    def __init__(self, rm_file, agent_id, env_settings, initial_rm_states):
        """
        Initialize environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        agent_id : int
            Index {0,1} indicating which agent
        env_settings : dict
            Dictionary of environment settings
        """
        self.env_settings = env_settings
        self.agent_id = agent_id
        self._load_map()
        self.reward_machine = ManagedSparseRewardMachine(rm_file)
        self.u = initial_rm_states[self.agent_id-1]

        # self.u = self.reward_machine.get_initial_state()
        self.last_action = -1 # Initialize last action to garbage value

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']

        initial_states = self.env_settings['initial_states']

        self.s_i = initial_states[self.agent_id-1]
        self.objects = {}
        # self.objects[self.env_settings['goal_location']] = "g" # goal location
        self.objects[self.env_settings['yellow_button']] = 'yb'
        self.objects[self.env_settings['green_button']] = 'gb'
        self.objects[self.env_settings['red_button']] = 'rb'

        self.p = self.env_settings['p']

        self.num_states = self.Nr * self.Nc

        self.actions = [Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value, Actions.none.value]
        
        # Define forbidden transitions corresponding to map edges
        self.forbidden_transitions = set()
        
        wall_locations = self.env_settings['walls']

        for row in range(self.Nr):
            self.forbidden_transitions.add((row, 0, Actions.left)) # If in left-most column, can't move left.
            self.forbidden_transitions.add((row, self.Nc - 1, Actions.right)) # If in right-most column, can't move right.
        for col in range(self.Nc):
            self.forbidden_transitions.add((0, col, Actions.up)) # If in top row, can't move up
            self.forbidden_transitions.add((self.Nr - 1, col, Actions.down)) # If in bottom row, can't move down

        # Restrict agent from having the option of moving "into" a wall
        for i in range(len(wall_locations)):
            (row, col) = wall_locations[i]
            self.forbidden_transitions.add((row, col + 1, Actions.left))
            self.forbidden_transitions.add((row, col-1, Actions.right))
            self.forbidden_transitions.add((row+1, col, Actions.up))
            self.forbidden_transitions.add((row-1, col, Actions.down))

    def environment_step(self, s, a):
        """
        Execute action a from state s.

        Parameters
        ----------
        s : int
            Index representing the current environment state.
        a : int
            Index representing the action being taken.

        Outputs
        -------
        r : float
            Reward achieved by taking action a from state s.
        l : list
            List of events occuring at this step.
        s_next : int
            Index of next state.
        """
        s_next, last_action = self.get_next_state(s,a)
        self.last_action = last_action

        l = self.get_mdp_label(s, s_next, self.u)
        r = 0

        for e in l:
            # Get the new reward machine state and the reward of this step
            u2 = self.reward_machine.get_next_state(self.u, e)
            r = r + self.reward_machine.get_reward(self.u, u2)
            # Update the reward machine state
            self.u = u2

        return r, l, s_next

    def get_mdp_label(self, s, s_next, u):
        """
        Return the label of the next environment state and current RM state.
        """
        row, col = self.get_state_description(s_next)

        l = []

        thresh = 0.3 #0.3

        if u == 1:
            if (row, col) == self.env_settings['yellow_button']:
                l.append('by')
        if u == 3:
            if (row, col) == self.env_settings['green_button']:
                l.append('bg')
        if u == 5:
            if (row, col) == self.env_settings['red_button']:
                l.append('br')

        return l

    def get_next_state(self, s, a):
        """
        Get the next state in the environment given action a is taken from state s.
        Update the last action that was truly taken due to MDP slip.

        Parameters
        ----------
        s : int
            Index of the current state.
        a : int
            Action to be taken from state s.

        Outputs
        -------
        s_next : int
            Index of the next state.
        last_action :int
            Last action taken by agent due to slip proability.
        """
        slip_p = [self.p, (1-self.p)/2, (1-self.p)/2]
        check = random.random()

        row, col = self.get_state_description(s)

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 

        if (check<=slip_p[0]) or (a == Actions.none.value):
            a_ = a

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
            if a == 0: 
                a_ = 3
            elif a == 2: 
                a_ = 1
            elif a == 3: 
                a_ = 2
            elif a == 1: 
                a_ = 0

        else:
            if a == 0: 
                a_ = 1
            elif a == 2: 
                a_ = 3
            elif a == 3: 
                a_ = 0
            elif a == 1: 
                a_ = 2

        action_ = Actions(a_)
        if (row, col, action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                row -= 1
            if action_ == Actions.down:
                row += 1
            if action_ == Actions.left:
                col -= 1
            if action_ == Actions.right:
                col += 1

        s_next = self.get_state_from_description(row, col)

        last_action = a_
        return s_next, last_action

    def get_state_from_description(self, row, col):
        """
        Given a (row, column) index description of gridworld location, return
        index of corresponding state.

        Parameters
        ----------
        row : int
            Index corresponding to the row location of the state in the gridworld.
        col : int
            Index corresponding to the column location of the state in the gridworld.
        
        Outputs
        -------
        s : int
            The index of the gridworld state corresponding to location (row, col).
        """
        return self.Nc * row + col

    def get_state_description(self, s):
        """
        Return the row and column indeces of state s in the gridworld.

        Parameters
        ----------
        s : int
            Index of the gridworld state.

        Outputs
        -------
        row : int
            The row index of state s in the gridworld.
        col : int
            The column index of state s in the gridworld.
        """
        row = np.floor_divide(s, self.Nr)
        col = np.mod(s, self.Nc)

        return (row, col)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.last_action

    def get_initial_state(self):
        """
        Outputs
        -------
        s_i : int
            Index of agent's initial state.
        """
        return self.s_i

    def show(self, s):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : int
            Index of the current state
        """
        display = np.zeros((self.Nr, self.Nc))
        
        # Display the locations of the walls
        for loc in self.env_settings['walls']:
            display[loc] = -1

        display[self.env_settings['red_button']] = 9
        display[self.env_settings['green_button']] = 9
        display[self.env_settings['yellow_button']] = 9
        # display[self.env_settings['goal_location']] = 9

        # Display the location of the agent in the world
        row, col = self.get_state_description(s)
        display[row,col] = self.agent_id

        print(display)