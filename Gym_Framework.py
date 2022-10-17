import os
import argparse
import yaml
import wandb
import torch
import gym
import copy
from PointCloud_Generator.pc_generator_coverage import PcdGen_Cov
from PointCloud_Generator.pc_generator_inspection import PcdGen_Ins
from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo.policies import MlpPolicy as PPOMlpPolicy
from stable_baselines3.sac.policies import MlpPolicy as SACMlpPolicy
from RL_Framework.Models.pytorch.CustomFeatureExtractor_PCN import CustomPCN
from RL_Framework.Gym.utils import StoreDict, preprocess_hyperparams, preprocess_schedules


class Gym_VPP():
    def __init__(self, config:dict, args):
        """initializes Framework

        Args:
            config (dict): config file
            args (_type_): ArgParser arguments
        """
        self.agent = None
        self.env = None
        self.env_type = args.env

        self.RL_params = config["RL_params"]
        self.env_params = config["env"]
        wandb_params = config["wandb"]
        self.env_specific_params = config[self.env_type]

        self.entity = wandb_params["entity"]
        self.train_project = wandb_params["project"]
        self.eval_project = wandb_params["eval"]
        self.run_name = wandb_params["run_name"]

        self.data_set = self.env_params["dataset"]
        self.file_type = self.env_params["file_type"]
        self.encoding = self.env_params["encoding"]
        self.action_type = self.env_params["action_type"]
        self.downsampling_factor = self.env_params["downsampling_factor"]
        self.dim = self.env_params["dim"]


        self.agent_type = self.RL_params["agent_type"]
        self.num_timesteps = self.RL_params["num_steps"]
        
        self.verbose =args.verbose
        self.track = args.track

        if self.encoding == "2048x4":
            self.in_channel = 4
        elif self.encoding == "in_policy":
            self.in_channel = 3
        
        self.data_path = None
        self.outdir_name = None
        self.outdir = None
        self.checkpoint_dir = None

        self.hyperparams = preprocess_hyperparams(config, args)
        self.prepare_dir(config)
        
        if self.track >= 0:
            wandb.init(
                project=self.train_project, 
                entity = self.entity, 
                sync_tensorboard=True, 
                reinit=True,
                dir=self.outdir,
                name=self.outdir_name,
                config={
                    "RL_params": self.RL_params,
                    "Env_Params": self.env_params,
                    "Env_specific": self.env_specific_params,
                    "hyperparameters": self.hyperparams
                    }
                )
        self.hyperparams = preprocess_schedules(self.hyperparams)


    def prepare_dir(self, config: dict):
        """creates directorys based on run_name
        """
        BASE_PATH = os.path.abspath(os.getcwd())
        self.data_path = os.path.join(BASE_PATH,'Data', self.data_set)
        # Create Output Dir:
        idx = 1
        self.outdir_name = f"{self.run_name}_{idx}"
        self.outdir =os.path.join(BASE_PATH, 'Runs', self.outdir_name)

        while os.path.isdir(self.outdir):
            idx += 1
            self.outdir_name = f"{self.run_name}_{idx}"
            self.outdir = os.path.join(BASE_PATH, 'Runs', self.outdir_name)
        os.mkdir(self.outdir)
        self.checkpoint_dir = os.path.join(self.outdir,'model-checkpoints')
        os.mkdir(self.checkpoint_dir)
        print(f"create output dir: '{self.outdir_name}'")

        #save config and agent hyperparams in run folder
        with open(os.path.join(self.outdir, 'config.yaml'), 'w') as outfile:
            yaml.safe_dump(config, outfile, default_flow_style=False)
        with open(os.path.join(self.outdir, 'agent_hyperparams.yaml'), 'w') as outfile:
            yaml.safe_dump(self.hyperparams, outfile, default_flow_style=False)


    def create_scanner(self):
        """creates scanner module based on specified environment

        Raises:
            ValueError: if Scanner module is not implemented
        """
        if "scanner" in self.env_specific_params:
            kwargs = copy.deepcopy(self.env_specific_params["scanner"])
            del(self.env_specific_params["scanner"])
        else:
            kwargs = self.env_specific_params

        if self.env_type == "Cov" or self.env_type == "NBV":
            self.scanner = PcdGen_Cov(
                downsampling_factor=self.downsampling_factor,
                file_type=self.file_type,
                dim=self.dim,
                **kwargs
            )
        elif self.env_type == "Ins":
            self.scanner = PcdGen_Ins(
                downsampling_factor=self.downsampling_factor,
                file_type=self.file_type,
                dim=self.dim,
                **kwargs
            )
        else:
            raise ValueError("Wrong scanner type, not implemented error")

    def create_env(self):
        """creates environment specified via argparser in terminal
        Raises:
            ValueError: if Environment is not implemented
        """
        kwargs = self.env_specific_params
        try:
            self.env = gym.make(
                f"{self.env_type}-v0",
                scanner=self.scanner,
                data_dir=self.data_path,
                file_type=self.file_type,
                encoding=self.encoding,                      # [pcn_512 / pcn_1024 / in_policy, 2048x4], path to encoder model needs to be set in 'Environment.py'
                action_type=self.action_type,                     # working: 2T0R, 3T0R, 2T2R, 3T2R, XYZ_0R, XYZ_2R
                outdir=self.outdir,                          # pass the logging dir 
                track = self.track,                        
                **kwargs)
        except ValueError:
            print(f"Environment {self.env_type}-v0 not implemented")
    
    def create_agent(self):
        """creates agent
        policy gets updated with custom feature extractor
        hyperparameters are stored in specified agent_config.yaml
        """
        env = self.env
        policy_kwargs = dict(
                features_extractor_class = CustomPCN,
                features_extractor_kwargs=dict(in_channels=self.in_channel, features_dim=1024),
                #net_arch=[dict(pi=[64,64], vf=[64,32])] #TODO: austesten ob funktional?
            )
        kwargs = self.hyperparams.copy()
        if self.agent_type == "ppo":
            self.agent = PPO(
                PPOMlpPolicy, 
                env,
                policy_kwargs=policy_kwargs,
                use_sde=False,
                verbose=self.verbose, 
                tensorboard_log=self.outdir,
                **kwargs
            )
        elif self.agent_type == "sac":
            
            self.agent=SAC(
                SACMlpPolicy,
                env,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose, 
                tensorboard_log=self.outdir,
                **kwargs
            )
        else:
            raise NotImplementedError
    
    def train(self):
        """trains agent"""
        model = self.agent
        if self.verbose > 0:
            print(model.policy)
        print(f"Training started with {self.num_timesteps} steps:")
        model.learn(total_timesteps=self.num_timesteps)

def main(config: dict, args):
    """main function to run Framework

    Args:
        config (dict): config file to read, if specified via Argparser updated config
        args (ArgParser): Arguments specified via Argparser 
    """
    print("CUDA: ", torch.cuda.is_available())
    framework = Gym_VPP(config=config, args=args)
    framework.create_scanner()
    framework.create_env()
    framework.create_agent()
    framework.train()

if __name__ == "__main__":
    #add flags for terminal,for more information type in terminal: python rl_Framework.py -h 
    parser = argparse.ArgumentParser(description="Reinforcement Learning Framework for ViewPlanningEnvironment")
    parser.add_argument("config", type=str, help="path to the config file", default="config.yaml")
    parser.add_argument("--steps", type=int, help="number of steps to train the agent on")
    parser.add_argument("--max_steps", type=int, help="max number of steps until env terminates")
    parser.add_argument("--env", type=str, help="Environment to use", choices=["Cov", "Ins", "NBV"], default="Cov")
    parser.add_argument(
        "--verbose", 
        type=int, 
        help="gives information about training, 0: none, 1: training information, 2: tf debugg", 
        choices=[0, 1, 2], 
        default=0)
    parser.add_argument("--id", type=str,  help="overrides run_name in config", default="")
    parser.add_argument("--agent", type=str, help="override agent_type in config", choices=["sac", "ppo"])
    parser.add_argument("--dataset", type=str, help="override dataset in config")
    parser.add_argument("--encoding", type=str, help="overrides encoding in config", choices=["2048x4", "in_policy"])
    parser.add_argument("--action", type=str, help="overrides action_type in config file", choices=["2T0R", "2T2R", "XYZ_0R", "XYZ_2R", "3T0R", "3T2R"])
    parser.add_argument("--reward", type=str, help="overrides reward_type in config_file")
    parser.add_argument(
        "-yaml", "--yaml-file", type=str, default=None, help="Custom yaml file from which the hyperparameters will be loaded")
    parser.add_argument(
        "-params", 
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10) or for schedule: learning_rate:linear_0.001 <- but later as string)"
    )
    parser.add_argument(
        "-scanner", 
        "--scanner_params",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite scanner_hyperparameters (e.g. cam_workspace:[30.0,50.0])",
    )
    parser.add_argument(
        "--track", type=int, help="types of plots to track! -1: no tracking, 0: only metrics, 1: only cam position (with density), 2: interactive_plots (consume a lot of storage space on wandb)",
        choices=[-1, 0, 1, 2], default=1
    )
    
    #read specified config and specified args
    args = parser.parse_args()
    with open(args.config, "r") as config:
        config = yaml.safe_load(config)

    #update config of run with specific values if specified in terminal, otherwise config values are used
    if args.agent != None: config["RL_params"].update({"agent_type":args.agent})
    if args.dataset != None: config["env"].update({"dataset":args.dataset})
    if args.encoding != None: config["env"].update({"encoding":args.encoding})
    if args.action != None: config["env"].update({"action_type":args.action})
    if args.reward != None: config[args.env].update({"rewardtype":args.reward})
    if args.max_steps != None: config[args.env].update({"max_iterations":args.max_steps})
    if args.steps != None: config["RL_params"].update({"num_steps": args.steps})
    if args.scanner_params != None: config[args.env]["scanner"].update(args.scanner_params)

    action_type = config["env"]["action_type"]
    agent_type = config["RL_params"]["agent_type"]
    project = config["wandb"]["project"]
    config["wandb"].update({"run_name": f"{action_type}_{agent_type}_{args.id}"})
    config["wandb"].update({"project": f"{project}_{args.env}"})
    #runs programm
    main(config, args)