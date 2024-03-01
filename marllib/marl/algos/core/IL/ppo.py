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
from typing import Dict, List, Optional
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
import torch
from marllib.marl.algos.manager_utils.model_saver import ModelSaver
from marllib.marl.algos.manager_utils.manager import Manager
from ray.rllib.evaluation import MultiAgentEpisode  # noqa

# TEST
import numpy as np
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.policy.sample_batch import SampleBatch
from marllib.marl.algos.utils.centralized_Q import get_dim
from marllib.marl.algos.utils.mixing_Q import align_batch
###
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

def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    
    print("\n\n\nPOST PROCESSING")
    print(sample_batch)
    import pdb; pdb.set_trace()
    if episode:
        print(episode.length)
        print(episode.total_reward)
    # custom_config = policy.config["model"]["custom_model_config"]
    # pytorch = custom_config["framework"] == "torch"
    # obs_dim = get_dim(custom_config["space_obs"]["obs"].shape)
    # algorithm = custom_config["algorithm"]
    # opp_action_in_cc = custom_config["opp_action_in_cc"]
    # global_state_flag = custom_config["global_state_flag"]
    # mask_flag = custom_config["mask_flag"]

    # if mask_flag:
    #     action_mask_dim = custom_config["space_act"].n
    # else:
    #     action_mask_dim = 0

    # n_agents = custom_config["num_agents"]
    # opponent_agents_num = n_agents - 1

    # if (pytorch and hasattr(policy, "compute_central_vf")) or \
    #         (not pytorch and policy.loss_initialized()):

    #     if not opp_action_in_cc and global_state_flag:
    #         sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:]
    #         sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
    #             convert_to_torch_tensor(
    #                 sample_batch["state"], policy.device),
    #         ).cpu().detach().numpy()
    #     else:  # need opponent info
    #         assert other_agent_batches is not None
    #         opponent_batch_list = list(other_agent_batches.values())
    #         raw_opponent_batch = [opponent_batch_list[i][1] for i in range(opponent_agents_num)]
    #         opponent_batch = []
    #         for one_opponent_batch in raw_opponent_batch:
    #             one_opponent_batch = align_batch(one_opponent_batch, sample_batch)
    #             opponent_batch.append(one_opponent_batch)

    #         # all other agent obs as state
    #         # sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]
    #         if global_state_flag:  # include self obs and global state
    #             sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:]
    #         else:
    #             # must stack in order for the consistency
    #             state_batch_list = []
    #             for agent_name in custom_config['agent_name_ls']:
    #                 if agent_name in other_agent_batches:
    #                     index = list(other_agent_batches).index(agent_name)
    #                     state_batch_list.append(
    #                         opponent_batch[index]["obs"][:, action_mask_dim:action_mask_dim + obs_dim])
    #                 else:
    #                     state_batch_list.append(sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim])
    #             sample_batch["state"] = np.stack(state_batch_list, 1)

    #         sample_batch["opponent_actions"] = np.stack(
    #             [opponent_batch[i]["actions"] for i in range(opponent_agents_num)],
    #             1)

    #         if algorithm in ["coma"]:
    #             sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
    #                 convert_to_torch_tensor(
    #                     sample_batch["state"], policy.device),
    #                 convert_to_torch_tensor(
    #                     sample_batch["opponent_actions"], policy.device) if opp_action_in_cc else None,
    #             ) \
    #                 .cpu().detach().numpy()
    #             sample_batch[SampleBatch.VF_PREDS] = np.take(sample_batch[SampleBatch.VF_PREDS],
    #                                                          np.expand_dims(sample_batch["actions"], axis=1)).squeeze(
    #                 axis=1)
    #         else:
    #             sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
    #                 convert_to_torch_tensor(
    #                     sample_batch["state"], policy.device),
    #                 convert_to_torch_tensor(
    #                     sample_batch["opponent_actions"], policy.device) if opp_action_in_cc else None,
    #             ) \
    #                 .cpu().detach().numpy()

    # else:
    #     # Policy hasn't been initialized yet, use zeros.
    #     o = sample_batch[SampleBatch.CUR_OBS]
    #     if global_state_flag:
    #         sample_batch["state"] = np.zeros((o.shape[0], get_dim(custom_config["space_obs"]["state"].shape) + get_dim(
    #             custom_config["space_obs"]["obs"].shape)),
    #                                          dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
    #     else:
    #         sample_batch["state"] = np.zeros((o.shape[0], n_agents, obs_dim),
    #                                          dtype=sample_batch[SampleBatch.CUR_OBS].dtype)

    #     sample_batch["vf_preds"] = np.zeros_like(
    #         sample_batch[SampleBatch.REWARDS], dtype=np.float32)
    #     sample_batch["opponent_actions"] = np.stack(
    #         [np.zeros_like(sample_batch["actions"], dtype=sample_batch["actions"].dtype) for _ in
    #          range(opponent_agents_num)], axis=1)

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
    postprocess_fn=Manager.dynamic_callback,
)


def get_policy_class_ppo(config_):
    if config_["framework"] == "torch":
        return IPPOTorchPolicy


IPPOTrainer = PPOTrainer.with_updates(
    name="IPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ppo,
)
