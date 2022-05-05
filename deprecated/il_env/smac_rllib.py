from ray.rllib.env.multi_agent_env import MultiAgentEnv
from smac.env.starcraft2.starcraft2 import StarCraft2Env
import numpy as np
from gym.spaces import Dict, Discrete, Box


class RLlibSMAC(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(self, map_name):
        map_name = map_name if isinstance(map_name, str) else map_name["map_name"]
        self.env = StarCraft2Env(map_name)

        env_info = self.env.get_env_info()
        self.num_agents = self.env.n_agents
        obs_shape = env_info['obs_shape']
        n_actions = env_info['n_actions']
        state_shape = env_info['state_shape']
        self.observation_space = Dict({
            "obs": Box(-2.0, 2.0, shape=(obs_shape,)),
            "state": Box(-2.0, 2.0, shape=(state_shape,)),
            "action_mask": Box(-2.0, 2.0, shape=(n_actions,))
        })
        self.action_space = Discrete(n_actions)

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self.env.reset()
        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()
        obs_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            action_mask_one_agent = np.array(self.env.get_avail_agent_actions(agent_index)).astype(np.float32)
            agent_index = "agent_{}".format(agent_index)
            obs_dict[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent
            }

        return obs_dict

    def step(self, actions):

        actions_ls = [int(actions[agent_id]) for agent_id in actions.keys()]

        reward, terminated, info = self.env.step(actions_ls)

        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()

        obs_dict = {}
        reward_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            action_mask_one_agent = np.array(self.env.get_avail_agent_actions(agent_index)).astype(np.float32)
            agent_index = "agent_{}".format(agent_index)
            obs_dict[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent
            }
            reward_dict[agent_index] = reward

        dones = {"__all__": terminated}

        return obs_dict, reward_dict, dones, {}

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.episode_limit
        }
        return env_info

    def close(self):
        self.env.close()