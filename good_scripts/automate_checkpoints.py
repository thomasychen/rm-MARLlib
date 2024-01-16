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


import yaml
from marllib import marl
import os
import time
import wandb

ray_path = "/Users/nikhil/Desktop/RL_Research/marllib/marl/ray/ray.yaml"
checkpoint_folder = "/Users/nikhil/Desktop/RL_Research/temp_checkpoints"
# os.mkdir(checkpoint_folder)

with open(ray_path, 'r') as ymlfile:
    ray_config = yaml.safe_load(ymlfile)

folder_name = f"{checkpoint_folder}/{time.time()}"

ray_config['local_dir'] = folder_name
with open(ray_path, 'w') as ymlfile:
    yaml.dump(ray_config, ymlfile)

num_epochs = 10



# ippo = marl.algos.ippo(hyperparam_source="common")

latest_subdir = ""
for i in range(num_epochs):

    test_env = marl.make_env(environment_name="buttons", map_name='all_scenarios', force_coop=True)
    env = marl.make_env(environment_name="buttons", map_name='all_scenarios', force_coop=False)

    ippo = marl.algos.ippo(hyperparam_source="common")
    model = marl.build_model(env, ippo, model_preference={"core_arch": "mlp"})

    print("FITTING MODEL\n\n")
    if i == 0:
        ippo.fit(env, model, checkpoint_end=True, stop={"timesteps_total": 10000})
    else:
        ippo.render(env, model, local_mode = True, restore_path={'params_path': f"{latest_subdir}/params.json",  # experiment configuration
                           'model_path': f"{latest_subdir}/checkpoint_00000{i}/checkpoint-{i}"}, stop={"timesteps_total": 10000})


    # time.sleep(1)
    # print("hello", f'{folder_name}/ippo_mlp_all_scenarios')

    main_path = f'{folder_name}/ippo_mlp_all_scenarios'
    all_subdirs = [os.path.join(main_path, d) for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]

    latest_subdir = max(all_subdirs, key=os.path.getmtime)

    test_env = marl.make_env(environment_name="buttons", map_name='all_scenarios', force_coop=True)
    env = marl.make_env(environment_name="buttons", map_name='all_scenarios', force_coop=False)

    ippo = marl.algos.ippo(hyperparam_source="common")
    model = marl.build_model(env, ippo, model_preference={"core_arch": "mlp"})

    print("TESTING MODEL\n\n")
    ippo.render(test_env, model, local_mode = True, restore_path={'params_path': f"{latest_subdir}/params.json",
                           'model_path': f"{latest_subdir}/checkpoint_00000{i+1}/checkpoint-{i+1}"}, stop={"timesteps_total": 1000})


