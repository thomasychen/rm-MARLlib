# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer as PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.policy.policy import Policy
from typing import Dict, List
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
import torch
from marllib.marl.algos.manager_utils.model_saver import ModelSaver

###########
### PPO ###
###########

def vf_preds_fetches(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
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
    # import pdb; pdb.set_trace()
    # with open("/Users/nikhil/Desktop/RL_Research/test_pkl.pkl", "wb") as file:
    #     pickle.dump(model, file)
    print("\n\n", input_dict["agent_index"][0])
    return {
        SampleBatch.VF_PREDS: model.value_function(),
    }

# IPPOTorchPolicy = PPOTorchPolicy.with_updates(
#     name="IPPOTorchPolicy",
#     get_default_config=lambda: PPO_CONFIG,
#     extra_action_out_fn=vf_preds_fetches,
# )

# test_manager = Manager()

IPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="IPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    extra_action_out_fn = ModelSaver.dynamic_callback,# extra_action_out_fn=RLlibButtons.dynamic_callback
)


def get_policy_class_ppo(config_):
    if config_["framework"] == "torch":
        return IPPOTorchPolicy


IPPOTrainer = PPOTrainer.with_updates(
    name="IPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ppo,
)
