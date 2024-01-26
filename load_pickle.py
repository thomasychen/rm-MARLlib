import pickle
import ray
from gym.spaces import Dict, Discrete, Box
import numpy as np
from marllib.marl.algos.core.IL.ppo import IPPOTrainer

with open("/Users/thomaschen/rm-MARLlib/test_pkl.pkl", "rb") as file:
    loaded_object = pickle.load(file)
print(loaded_object)
print(loaded_object.value_function())