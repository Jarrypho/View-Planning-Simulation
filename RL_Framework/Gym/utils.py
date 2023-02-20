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

from typing import Callable, Tuple, Union, Dict, Any
import argparse, yaml
from stable_baselines3.common.utils import constant_fn


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear rate schedule

    Args:
        initial_value (float): Initial Value

    Returns:
        Callable[[float], float]: schedule that computes current rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0.

        Args:
            progress_remaining (float): 

        Returns:
            float: current rate
        """
        return progress_remaining * initial_value
    return func

class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.
    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            try:
                arg_dict[key] = eval(value)
            except:
                arg_dict[key] = value #Probleme mit eval(value) wenn Erstellung eines linearen Schedules, daher so Abhilfe
        setattr(namespace, self.dest, arg_dict)

def preprocess_hyperparams(config: dict, args: dict) -> Tuple[Dict[str, Any]]:
    """updates hyperparameters from ArgParser to update config. 
    For more information visit stable-baselines3 zoo
    """
    agent_type = config["RL_params"]["agent_type"]
    yaml_file = args.yaml_file or f"RL_Framework/Gym/Agent_hyperparameters/{agent_type}.yaml"
    print(f"Loading hyperparameters from: {yaml_file}")
    with open(yaml_file) as f:
        hyperparameters_dict = yaml.safe_load(f)
        if f"{args.env}-v0" in list(hyperparameters_dict.keys()):
            hyperparams = hyperparameters_dict[f"{args.env}-v0"]
        else:
            raise ValueError(f"Hyperparameters not found for {agent_type}-{args.env}-v0")
    if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])
    if args.hyperparams is not None:
        hyperparams.update(args.hyperparams)

    return hyperparams

def preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """updates hyperparameters from ArgParser to update config. 
        For more information visit stable-baselines3 zoo
        """
        # Create schedules
        for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                if schedule == "lin":
                    hyperparams[key] = linear_schedule(initial_value)
                else:
                    raise ValueError(f"Invalid value for schedule {schedule}: {hyperparams[key]}")
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams
