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

"""
support commandline parameter insert to running
# instances
python api_example.py --ray_args.local_mode --env_args.difficulty=6  --algo_args.num_sgd_iter=6
----------------- rules -----------------
1 ray/rllib config: --ray_args.local_mode
2 environments: --env_args.difficulty=6
3 algorithms: --algo_args.num_sgd_iter=6
order insensitive
-----------------------------------------

-------------------------------available env-map pairs-------------------------------------
- smac: (https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/maps/smac_maps.py)
- mpe: (https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/mpe.py)
- mamujoco: (https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/mamujoco.py)
- football: (https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/football.py)
- magent: (https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/magent.py)
- lbf: use (https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/config/lbf.yaml) to generate the map.
Details can be found https://github.com/semitable/lb-foraging#usage
- rware: use (https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/config/rware.yaml) to generate the map.
Details can be found https://github.com/semitable/robotic-warehouse#naming-scheme
- pommerman: OneVsOne-v0, PommeFFACompetition-v0, PommeTeamCompetition-v0
- metadrive: Bottleneck, ParkingLot, Intersection, Roundabout, Tollgate
- hanabi: Hanabi-Very-Small, Hanabi-Full, Hanabi-Full-Minimal, Hanabi-Small
- mate: MATE-4v2-9-v0 MATE-4v2-0-v0 MATE-4v4-9-v0 MATE-4v4-0-v0 MATE-4v8-9-v0 MATE-4v8-0-v0 MATE-8v8-9-v0 MATE-8v8-0-v0
-------------------------------------------------------------------------------------------


-------------------------------------available algorithms-------------------------------------
- iql ia2c iddpg itrpo ippo
- maa2c coma maddpg matrpo mappo hatrpo happo
- vdn qmix facmac vda2c vdppo
----------------------------------------------------------------------------------------------
"""



from marllib import marl
from marllib.marl.algos.core.IL.ppo import IPPOTorchPolicy


test_env = marl.make_env(environment_name="buttons", map_name='all_scenarios', force_coop=True)
env = marl.make_env(environment_name="buttons", map_name='all_scenarios', force_coop=False)

# ippo = marl.algos.ippo(hyperparam_source="common")
ippo = marl.algos.ippo(hyperparam_source="common")
model = marl.build_model(env, ippo, model_preference={"core_arch": "mlp"})

# num_epochs = 100

# for i in range(num_epochs):

#     ippo.fit(env, model, checkpoint_end=True, stop={"timesteps_total": 1000})


ippo.fit(env, model, checkpoint_end=True, stop={"timesteps_total": 1000})
# print("TEST 2", ippo.config_dict)
# print("BRUHHH", IPPOTorchPolicy)


# print("\n\n\nHELLO", model)
policy = model[0].policy

# Call value_function
values = policy.value_function()
# print("BRUH", values)


# test_env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
# ippo.render(test_env, model, local_mode = True, restore_path={'params_path': "/Users/nikhil/Desktop/RL_Research/examples/exp_results/ippo_mlp_all_scenarios/IPPOTrainer_buttons_all_scenarios_0f4a6_00000_0_2024-01-10_15-07-43/params.json",  # experiment configuration

# print("BRUH", model)
# print("critics", model[1].critics)
# print(model[1].value_function())

# restore_path = {"model_path": "/Users/nikhil/Desktop/RL_Research/examples/exp_results/ippo_mlp_all_scenarios/IPPOTrainer_buttons_all_scenarios_0f4a6_00000_0_2024-01-10_15-07-43/checkpoint_000001", 
#                                                                 "params_path": "/Users/nikhil/Desktop/RL_Research/examples/exp_results/ippo_mlp_all_scenarios/IPPOTrainer_buttons_all_scenarios_0f4a6_00000_0_2024-01-10_15-07-43/params.json"}

# # # prepare the environment academy_pass_and_shoot_with_keeper
# # #env = marl.make_env(environment_name="hanabi", map_name="Hanabi-Very-Small")
# env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# # can add extra env params. remember to check env configuration before use
# # env = marl.make_env(environment_name='smac', map_name='3m', difficulty="6", reward_scale_rate=15)

# # initialize algorithm and load hyperparameters
# mappo = marl.algos.mappo(hyperparam_source="mpe")

# # can add extra algorithm params. remember to check algo_config hyperparams before use
# # mappo = marl.algos.MAPPO(hyperparam_source='common', use_gae=True,  batch_episode=10, kl_coeff=0.2, num_sgd_iter=3)

# # build agent model based on env + algorithms + user preference if checked available
# model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# # start learning + extra experiment settings if needed. remember to check ray.yaml before use
# mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, share_policy='all')
