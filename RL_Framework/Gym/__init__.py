'''
Copyright (C) 2022  Jan-Philipp Kaiser

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
'''

from re import M
from RL_Framework.Gym.Cov_Env import CoverageVP
from RL_Framework.Gym.NBV_Env import NBV_VP
from RL_Framework.Gym.Ins_Env import InspectionVP
import gym
import sys, yaml

list = sys.argv[1:]
config_arg = None
max_step_override = False

for item in list:
    if item.endswith(".yaml"):
        config_arg = item
    if item.startswith("--max_steps"):
        idx = list.index(item)
        max_step_override = True
    if item.startswith("--env"):
        idx_e = list.index(item)
        env = list[idx_e + 1]
    else:
        env = "Cov" 

if config_arg == None:
    raise Exception("No config found")    

with open(config_arg, "r") as config_file:
    config = yaml.safe_load(config_file)

if max_step_override:
    max_steps = int(list[idx + 1])
else:
    max_steps = config[env]["max_iterations"]

print(f"Environment {env}-v0 max steps per epoch: {max_steps}")

gym.envs.register(
     id='Cov-v0',
     entry_point='RL_Framework.Gym.Cov_Env:CoverageVP',
     max_episode_steps=max_steps,
)

gym.envs.register(
    id="NBV-v0",
    entry_point="RL_Framework.Gym.NBV_Env:NBV_VP",
    max_episode_steps=max_steps,
)

gym.envs.register(
    id="Ins-v0",
    entry_point="RL_Framework.Gym.Ins_Env:InspectionVP",
    max_episode_steps=max_steps,
)
