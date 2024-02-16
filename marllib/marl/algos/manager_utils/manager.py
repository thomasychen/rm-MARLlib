from marllib.marl.algos.manager_utils.model_saver import ModelSaver
import random
import itertools
import numpy as np
import torch
import wandb
from collections import defaultdict

class Manager:

    call_back = ModelSaver._nothing
    perm_qs = defaultdict(list)

    def __init__(self, num_agents, env_config):
        self.num_agents = num_agents
        # self.env = env
        
        self.ms = ModelSaver()
        self.epsilon = env_config['manager_epsilon']
        self.epsilon_decay = env_config['manager_epsilon_decay']

        self.assignment_method = env_config["manager_assignment_method"]
        self.env_config = env_config

        # print("BUTONSSS\n\n\n\n", self.ms)

        ModelSaver.call_back = self.ms.vf_preds_factory()


    def assign(self, initial_rm_states, mdp_states):

        # print(self.ms.id)
        # print("gangn", self.ms)

        self.curr_permutation_qs = self.calculate_permutation_qs(initial_rm_states, mdp_states)

        for perm, v in self.curr_permutation_qs.items():
            Manager.perm_qs[perm].append(v)

        print(self.curr_permutation_qs)

        if self.assignment_method == "ground_truth":
            self.curr_assignment = [0,1,2]
        elif self.assignment_method == "random": 
            self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
        elif self.assignment_method == "epsilon_greedy":
            # self.curr_permutation_qs = self.calculate_permutation_qs()

            if random.random() < self.epsilon:
                self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
            else:
                self.curr_assignment = list(max(self.curr_permutation_qs, key=self.curr_permutation_qs.get))
            self.epsilon *= self.epsilon_decay
        elif self.assignment_method == "multiply":
            self.curr_permutation_qs = self.calculate_permutation_qs(True)

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

        self.env_config["initial_rm_states"] = new_assignment
        print("manager", self.curr_assignment)


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

