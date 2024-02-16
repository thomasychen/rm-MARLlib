# from reward_machines.managed_sparse_reward_machine import ManagedSparseRewardMachine
import numpy as np
# import infrastructure.pytorch_utils as ptu
import torch
import random
import itertools

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer as PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.policy.policy import Policy
from typing import Dict, List
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
import torch
import pickle

from typing import Optional


class ModelSaver:
    @staticmethod
    def dynamic_callback(*args, **kwargs):
        return ModelSaver.call_back(*args, **kwargs)
    
    @staticmethod
    def _nothing(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
        return {
            SampleBatch.VF_PREDS: model.value_function(),
        }

    call_back = _nothing

    def __init__(self, id=0):
        self.models = {}
        self.model_input_dicts = {}
        self.id = id

    def vf_preds_factory(self):
        def wrapper(policy: Policy, input_dict: Dict[str, TensorType],
                    state_batches: List[TensorType], model: ModelV2,
                    action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:

            return ModelSaver._generate_values(policy, input_dict, state_batches, model, action_dist, self)
        
        return wrapper

    
    @staticmethod
    def _generate_values(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper,
        self: Optional['ModelSaver'] = None) -> Dict[str, TensorType]:
        """Defines extra fetches per action computation.

        Args:
            policy (Policy): The Policy to perform the extra action fetch on.
            input_dict (Dict[str, TensorType]): The input dict used for the action
                computing forward pass.
            state_batches (List[TensorType]): List of state tensors (empty for
                non-RNNs).
            model (ModelV2): The Model object of the Policy.
            action_dist (TorchDistributionWrapper): The instantiated distribution
                object, resulting from the model's outputs and the given
                distribution class.

        Returns:
            Dict[str, TensorType]: Dict with extra tf fetches to perform per
                action computation.
        """
        # Return value function outputs. VF estimates will hence be added to the
        # SampleBatches produced by the sampler(s) to generate the train batches
        # going into the loss function.

        # print("\n\n BTUHHHH")

        self.models[int(input_dict["agent_index"][0].item())] = model
        # print("\n\n", model)
        self.model_input_dicts[int(input_dict["agent_index"][0].item())] = input_dict

        # self.id += 1

        # print("\n\nyuhhh", self.id)

        # print(self.models, self.model_input_dicts)

        return {
            SampleBatch.VF_PREDS: model.value_function(),
        }