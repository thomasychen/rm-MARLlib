from marllib.marl.algos.manager_utils.model_saver import ModelSaver
import random
import itertools
import numpy as np
import torch
import wandb
from collections import defaultdict
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from tqdm import tqdm

class Manager:

    perm_qs = defaultdict(list)
    starting_qs = defaultdict(float)

    @staticmethod
    def dynamic_callback(*args, **kwargs):
        return Manager.call_back(*args, **kwargs)
    
    @staticmethod
    def _nothing(policy,sample_batch,other_agent_batches=None,episode=None):
        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            last_r = sample_batch[SampleBatch.VF_PREDS][-1]

        if "lambda" in policy.config:
            train_batch = compute_advantages(
                sample_batch,
                last_r,
                policy.config["gamma"],
                policy.config["lambda"],
                use_gae=policy.config["use_gae"])
        else:
            train_batch = compute_advantages(
                rollout=sample_batch,
                last_r=0.0,
                gamma=policy.config["gamma"],
                use_gae=False,
                use_critic=False)
        
        return train_batch

    call_back = _nothing

    def her_factory(self):
        def wrapper(policy,sample_batch,other_agent_batches=None,episode=None):
            return Manager.her(policy, sample_batch, other_agent_batches, episode, self)
        
        return wrapper
    

        
    # def add_to_batch(self, sample_batch, obs, next_obs, action, prev_action, reward, prev_reward, done, info, eps_id, unroll_id, t):
        np.append(sample_batch["obs"], obs)
        np.append(sample_batch["new_obs"], next_obs)
        np.append(sample_batch["actions"], action)
        np.append(sample_batch["prev_actions", prev_action])
        np.append(sample_batch["rewards"], reward)
        np.append(sample_batch["prev_rewards"], prev_reward)
        np.append(sample_batch["dones"], done)
        np.append(sample_batch["infos"], info)
        np.append(sample_batch["eps_id"], eps_id)
        np.append(sample_batch["unroll_id"], unroll_id)
        np.append(sample_batch["agent_index"], 0)
        np.append(sample_batch["t"], t)
        np.append(sample_batch["state_out_0"], np.zeros((self.num_agents, 256)))
        np.append(sample_batch["state_in_0"], np.zeros((self.num_agents), 1), axis=1)
    


# def add_to_sample(sample_batch, obs, action, reward, done, info, eps_id, unroll_id, agent_id, vf_preds, action_dist_inputs, action_logp):
#     # Assuming all inputs are numpy arrays and can be reshaped to (-1, 1)
#     new_data = np.hstack([
#         np.reshape(obs, (-1, 1)),
#         np.reshape(action, (-1, 1)),
#         np.reshape(reward, (-1, 1)),
#         np.reshape(done, (-1, 1)),
#         np.reshape(info, (-1, 1)),  # This might need special handling depending on what info contains
#         np.reshape(eps_id, (-1, 1)),
#         np.reshape(unroll_id, (-1, 1)),
#         np.reshape(agent_id, (-1, 1)),
#         np.reshape(vf_preds, (-1, 1)),
#         np.reshape(action_dist_inputs, (-1, 1)),
#         np.reshape(action_logp, (-1, 1))
#     ])

#     # Append new data column-wise
#     if sample_batch["obs"].size == 0:
#         sample_batch["obs"] = new_data
#     else:
#         sample_batch["obs"] = np.vstack([sample_batch["obs"], new_data])       




    ''' obs: num_s x 6
        new_obs: 6
        actions: 3
        prev_actions: 3
        rewards: 1
        prev_rewards: 1
        dones: 1
        infos: {'_group_rewards': [0, 0, 0]} => list
        eps_id: 1
        unroll_id: 
        agent_index: 1 but all 0s (maybe)
        t: 1 in order starting at 0
        state_in_0: 3x num_s
        state_out_0: num_s x 3 x 256
    ''' 
    def add_to_batch(self, sample_batch, obs, next_obs, action, prev_action, reward, prev_reward, done, info, eps_id, unroll_id, t):
        np.append(sample_batch["obs"], obs)
        np.append(sample_batch["new_obs"], next_obs)
        np.append(sample_batch["actions"], action)
        np.append(sample_batch["prev_actions"], prev_action)
        np.append(sample_batch["rewards"], reward)
        np.append(sample_batch["prev_rewards"], prev_reward)
        np.append(sample_batch["dones"], done)
        np.append(sample_batch["infos"], info)
        np.append(sample_batch["eps_id"], eps_id)
        np.append(sample_batch["unroll_id"], unroll_id)
        np.append(sample_batch["agent_index"], 0)
        np.append(sample_batch["t"], t)
        np.append(sample_batch["state_out_0"], np.zeros((self.num_agents, 256)))

    @staticmethod
    def her(policy,sample_batch,other_agent_batches=None,episode=None, self=None):
        # print()
        # print("YAYYYYY \n\n\n\n")

        # import pdb; pdb.set_trace()

        ## mess with sample batch before computing advantages

        assert(all([sample_batch["agent_index"][i] == sample_batch["agent_index"][0] for i in range(len(sample_batch["agent_index"]))]))
        assert(all([sample_batch["eps_id"][i] == sample_batch["eps_id"][0] for i in range(len(sample_batch["eps_id"]))]))


        observations = sample_batch["obs"]
        new_observations = sample_batch["obs"]
        actions = sample_batch['actions']
        prev_actions = sample_batch['prev_actions']
        unroll_id = sample_batch['unroll_id'][0]
        

        for t in tqdm(range(len(sample_batch["agent_index"])-1)):
            fake_obs = [[] for _ in range(self.num_agents)]
            fake_new_obs = [[] for _ in range(self.num_agents)]
            fake_actions = [[] for _ in range(self.num_agents)]
            fake_prev_actions = [[] for _ in range(self.num_agents)]
            fake_rewards = [[] for _ in range(self.num_agents)]
            fake_prev_rewards = [[] for _ in range(self.num_agents)]
            fake_dones = [[] for _ in range(self.num_agents)]
            fake_infos = [[] for _ in range(self.num_agents)]
            # fake_t = [[] for _ in range(self.num_agents)]

            for i in range(self.num_agents):
                    env = self.envs[f"agent_{i}"]
                    rm = env.reward_machine
                    current_u = int(observations[t][i*2 + 1])
                    s = int(observations[t][i*2])
                    s_new = int(new_observations[t][i*2])
                    a = int(actions[t][i])
                    prev_a = int(prev_actions[t][i])
                    

                    for u in rm.U:
                        if not (u == current_u) and not (u in rm.T) and not (u == rm.u0):
                        # if not (u == current_u) and not (u in agent_list[i].rm.T):
                            new_l = env.get_mdp_label(s, s_new, u)
                            new_r = 0
                            u_temp = u
                            u2 = u
                            for e in new_l:
                                # Get the new reward machine state and the reward of this step
                                u2 = rm.get_next_state(u_temp, e)
                                new_r = new_r + rm.get_reward(u_temp, u2)
                                # Update the reward machine state
                                u_temp = u2
                            done = rm.is_terminal_state(u2)
                        
                            fake_obs[i].append([s, u])
                            fake_new_obs[i].append([s_new,u2])
                            fake_actions[i].append(a)
                            fake_prev_actions[i].append(prev_a)
                            if not fake_rewards[i]:
                                fake_prev_rewards[i].append(0)
                            else:
                                fake_prev_rewards[i].append(fake_rewards[i][-1])
                            fake_rewards[i].append(new_r)
                            fake_dones[i].append(done)
                            fake_infos[i].append(new_r)

            # for i in range(len(fake_obs[0])):
            #     for j in range(len(fake_obs[1])):
            #         for k in range(len(fake_obs[2])):
            #             obs = fake_obs[0][i] + fake_obs[1][j] + fake_obs[2][k]
            #             new_obs = fake_new_obs[0][i] + fake_new_obs[1][j] + fake_new_obs[2][k]
            #             action = [fake_actions[0][i], fake_actions[1][j], fake_actions[2][k]]
            #             prev_action = [fake_prev_actions[0][i], fake_prev_actions[1][j], fake_prev_actions[2][k]]
            #             reward = int(fake_rewards[0][i] or fake_rewards[1][j] or fake_rewards[2][k])
            #             prev_reward = int(fake_prev_rewards[0][i] or fake_prev_rewards[1][j] or fake_prev_rewards[2][k])
            #             done = (fake_rewards[0][i] + fake_rewards[1][j] + fake_rewards[2][k]) //3
            #             info = {"_group_rewards": [fake_rewards[0][i] , fake_rewards[1][j] , fake_rewards[2][k]]}
            #             eps_id = np.random.randint(1000000)
            #             self.add_to_batch(sample_batch, obs, new_obs, action, prev_action, reward, prev_reward, done, info, eps_id, unroll_id, t)
        
        return sample_batch


    def __init__(self, num_agents, env_config, envs):
        self.num_agents = num_agents
        
        self.ms = ModelSaver()
        self.epsilon = env_config['manager_epsilon']
        self.epsilon_decay = env_config['manager_epsilon_decay']

        self.assignment_method = env_config["manager_assignment_method"]
        self.env_config = env_config
        self.envs = envs
        self.start_qs = {}

        # print("BUTONSSS\n\n\n\n", self.ms)

        # ModelSaver.call_back = self.ms.vf_preds_factory()
        ModelSaver.call_back = self.ms.save_model


    def assign(self, initial_rm_states, mdp_states):

        if self.assignment_method == "ground_truth":
            self.curr_permutation_qs = self.calculate_permutation_qs(initial_rm_states, mdp_states)

            self.curr_assignment = [0,1,2]
        elif self.assignment_method == "random": 

            self.curr_permutation_qs = self.calculate_permutation_qs(initial_rm_states, mdp_states)

            self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
        elif self.assignment_method == "epsilon_greedy":
            self.curr_permutation_qs = self.calculate_permutation_qs(initial_rm_states, mdp_states)

            if random.random() < self.epsilon:
                self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
            else:
                self.curr_assignment = list(max(self.curr_permutation_qs, key=self.curr_permutation_qs.get))
            self.epsilon *= self.epsilon_decay
        elif self.assignment_method == "multiply":
            self.curr_permutation_qs = self.calculate_permutation_qs(initial_rm_states, mdp_states, True)

            if random.random() < self.epsilon:
                self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
            else:
                self.curr_assignment = list(max(self.curr_permutation_qs, key=self.curr_permutation_qs.get))
            self.epsilon *= self.epsilon_decay
        else:
            raise Exception("fucker")

        new_assignment = [0]*len(initial_rm_states)

        for i in range(self.num_agents):
            i_assigned = self.curr_assignment[i]
            new_assignment[i] = initial_rm_states[i_assigned]

        # print(self.curr_permutation_qs)

        # For logging
        for perm, v in self.curr_permutation_qs.items():
            Manager.perm_qs[perm].append(v)

        self.env_config["initial_rm_states"] = new_assignment
        # print("manager", self.curr_assignment)


    def calculate_permutation_qs(self, starting_rm_states, mdp_states, multiply=False):
        res = {}
        for permutation in itertools.permutations(list(range(self.num_agents))):
            accumulator = 1 if multiply else 0

            obs_batch = [[[mdp_states[i], starting_rm_states[permutation[i]]] for i in range(len(permutation))]]

            with torch.no_grad():
                q_values, hiddens = self.ms.mac_func(
                    self.ms.model,
                    torch.as_tensor(
                        obs_batch, dtype=torch.float, device=self.ms.device), [
                        torch.as_tensor(
                            np.array(s), dtype=torch.float, device=self.ms.device)
                        for s in self.ms.state_batches
                    ])
            max_values = q_values.max(dim=2).values
            res[tuple(permutation)] = max_values.prod() if multiply else max_values.sum()
        return res
            
        

            

            # for i in range(len(permutation)):
                
            #     starting_rm_state = starting_rm_states[permutation[i]]
            #     curr_state = np.row_stack(([mdp_states[i]], [starting_rm_state])).T

            #     # self.ms.model_input_dicts[i]["obs"] = torch.tensor(curr_state)
            #     # temp = self.ms.model_input_dicts[i]["obs"]
            #     # import pdb; pdb.set_trace()
            #     self.ms.model_input_dicts[i]["obs"] = {"obs": torch.tensor(curr_state)}

                # if "obs" in self.ms.model_input_dicts[i]["obs"]:
                #     self.ms.model_input_dicts[i]["obs"]["obs"] = torch.tensor(curr_state)
                # print(self.ms.model_input_dicts[0]["obs"])



                # self.model_input_dicts[i]["obs"]["obs"] = torch.tensor(curr_state)
        #         self.ms.models[i].forward(self.ms.model_input_dicts[i], [], None)
        #         # self.models[i].forward(self.model_input_dicts[i], [])

        #         value = self.ms.models[i].value_function().item()

        #         if multiply:
        #             accumulator *= value
        #         else:
        #             accumulator += value
            
        #     res[tuple(permutation)] = accumulator
        # return res

