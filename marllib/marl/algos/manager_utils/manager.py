from marllib.marl.algos.manager_utils.model_saver import ModelSaver
import random
import itertools
import numpy as np
import torch
import wandb
from collections import defaultdict
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch

class Manager:

    perm_qs = defaultdict(list)

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
    
        
    def add_to_batch(self, sample_batch, obs, action, reward, done, info, eps_id, unroll_id, agent_id, vf_preds, action_dist_inputs, action_logp):
        np.append(sample_batch["obs"], obs)
        np.append(sample_batch["actions"], action)
        np.append(sample_batch["rewards"], reward)
        np.append(sample_batch["dones"], done)
        np.append(sample_batch["infos"], info)
        np.append(sample_batch["eps_id"], eps_id)
        np.append(sample_batch["unroll_id"], unroll_id)
        np.append(sample_batch["agent_index"], agent_id)
        np.append(sample_batch[SampleBatch.VF_PREDS], vf_preds)
        np.append(sample_batch["action_dist_inputs"], action_dist_inputs)
        np.append(sample_batch["action_logp"], action_logp)

    @staticmethod
    def her(policy,sample_batch,other_agent_batches=None,episode=None, self=None):

        # import pdb; pdb.set_trace()

        ## mess with sample batch before computing advantages

        assert(all([sample_batch["agent_index"][i] == sample_batch["agent_index"][0] for i in range(len(sample_batch["agent_index"]))]))
        assert(all([sample_batch["eps_id"][i] == sample_batch["eps_id"][0] for i in range(len(sample_batch["eps_id"]))]))

        agent_id = sample_batch["agent_index"][0]

        env = self.envs[f"agent_{int(agent_id)}"]
        rm = env.reward_machine
        obs = sample_batch["obs"]
        actual_start_rm = obs[0][1]
        unroll_id = sample_batch["unroll_id"][0]
        ## optional safety: group batch by agent_id and rollout_id
        for u in rm.U:
            if not (u == actual_start_rm) and not (u in rm.T) and not (u == rm.u0):
                eps_id = np.random.randint(1000000) # nikhil thinks we might collide
                curr_rm_state = u
                for i in range(len(sample_batch["agent_index"])-1):
                    curr_mdp_state = obs[i][0]
                    next_mdp_state = obs[i+1][0]
                    new_l = env.get_mdp_label(curr_mdp_state, next_mdp_state, curr_rm_state)
                    new_r = 0
                    u_temp = curr_rm_state
                    u2 = curr_rm_state
                    for e in new_l:
                        # Get the new reward machine state and the reward of this step
                        u2 = rm.get_next_state(u_temp, e)
                        new_r = new_r + rm.get_reward(u_temp, u2)
                        # Update the reward machine state
                        u_temp = u2

                    done = rm.is_terminal_state(u2)# idk how to tell

                    ## add new row to sample_batch
                    self.add_to_batch(sample_batch, [next_mdp_state, u2], sample_batch["actions"][i+1], new_r, done, {}, eps_id, unroll_id, agent_id, 0, [0,0,0,0,0], 0)

                    if done:
                        break

                    curr_rm_state = u2
        
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


    def __init__(self, num_agents, env_config, envs):
        self.num_agents = num_agents
        
        self.ms = ModelSaver()
        self.epsilon = env_config['manager_epsilon']
        self.epsilon_decay = env_config['manager_epsilon_decay']

        self.assignment_method = env_config["manager_assignment_method"]
        self.env_config = env_config
        self.envs = envs

        # print("BUTONSSS\n\n\n\n", self.ms)

        ModelSaver.call_back = self.ms.vf_preds_factory()


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

            for i in range(len(permutation)):
                
                starting_rm_state = starting_rm_states[permutation[i]]
                curr_state = np.row_stack(([mdp_states[i]], [starting_rm_state])).T
                

                # self.ms.model_input_dicts[i]["obs"] = torch.tensor(curr_state)
                # temp = self.ms.model_input_dicts[i]["obs"]
                self.ms.model_input_dicts[i]["obs"] = {"obs": torch.tensor(curr_state)}

                # if "obs" in self.ms.model_input_dicts[i]["obs"]:
                #     self.ms.model_input_dicts[i]["obs"]["obs"] = torch.tensor(curr_state)
                # print(self.ms.model_input_dicts[0]["obs"])



                # self.model_input_dicts[i]["obs"]["obs"] = torch.tensor(curr_state)
                self.ms.models[i].forward(self.ms.model_input_dicts[i], [], None)
                # self.models[i].forward(self.model_input_dicts[i], [])

                value = self.ms.models[i].value_function().item()

                if multiply:
                    accumulator *= value
                else:
                    accumulator += value
            
            res[tuple(permutation)] = accumulator
        return res

