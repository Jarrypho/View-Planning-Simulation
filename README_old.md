# In short
This Repository aims to solve the View-Planning problem for unknown products. A scanner module can be placed around an object and performs raytracing to generate a pointcloud. The goal is to maximize the covered area in the *Coverage Problem* or inspect certain parts of the object in the *Inspection Problem*. 

We use Reinforcement Learning to tackle this problem. Two frameworks are viable in this repository, Tensorforce and OpenAi Gym.

This Repository contains all the files and methods developed during the theses from Jonas Schmid, Jonas Gaebele and Dominik Koch.

Gym version is in the General_Framework Branch, old versions can be found in the backup branches. 

# Requirements
Using tensorforce: python v3.6.13. 
Using gym: python v3.7.13
Required packages are written down in requirements.txt. They can also be installed manually

- Open3d v0.12
- trimesh v3.9.35
- pyembree v0.1.6
- tensorforce v0.6.5 (only with python v3.6.13)
- gym v0.21.0
- tensorflow v2.6
- numba v0.53
- keras v2.6
- stable-baselines3 v1.6
- wandb 0.13
- pytorch 1.12 (only required for gym environment)


# Quickstart
Bellow, a installation guide is given as well as a routine for training agents.        

Tested with Linux (Ubuntu 20.04.4 LTS) with root, Windows does not support the raytracing optimization package'embree'


## Install and Setup

Download Repository
``` 
git clone https://gitlab.com/jarryphi/view-planning-simulation.git
```
Login with gitlab

Switch to main folder in which the repository was saved
```
cd ma-jonas
```

Create new Environment
X: custom name of the environment
```
conda create -n X python=3.7
```
activate conda environment
```
conda activate XXX
```
Install requirements
```
pip install -r requirements.txt
```
To use optimization of raytracing embree/pyembree must be installed:
```
conda install -c conda-forge pyembree
```
## Using CUDA if available
To check if CUDA is available for graphic card and which version:
```
nvidia-smi
```
If CUDA is available, go to https://pytorch.org/get-started/locally/ and choose the right options (pytorch 1.12). Use the official command from the website to install pytorch

If cuda is not available, install pytorch 1.12 for CPU



## RL-Framework
To start training, switch to main folder and activate conda environment
```
python Gym_Framework.py config.yaml
```
Gym_Framework.py - file that will be executed

config.yaml - config file which is needed for training, containing parameters for agent and environment (if config is not saved in main folder, path must be given in form path/config_name.yaml)

Tracking of the training is performed in Weight And Biases: https://wandb.ai/agi-probot_ma

# Repository Structure
Below the most important parts of the repository

## Path structure

**View-Planning-Simulation**
- Data
- PointCloud_Generator
- RL-Framework
    - Gym
        - Agent_configs
        - *gym_env.py*
        - *__init__.py*
    - Tensorforce
        - *Agents.py*
        - Environments
    - Models
        - keras
        - pytorch
    - utils
- *Gym_Framework.py*
- *sup_framework.py*

# RL-Framework
## Gym_Framework.py
Framework for creating the gym-environment and training the agent.

Available agents are the official stable-baselines3 agents.
https://stable-baselines3.readthedocs.io/en/master/modules/base.html

Hyperparameters for agents are default ones from stable-baselines. 

If Gym/Agent_hyperparameters contains a file "agent.yaml", the hyperparameters as specified in the config-file are used. For allowed parameters look at stable-baseline documentation. 

Agent-Config file contains paragraphs for each environment, so different agents can be specified for each environment.

Furthermore, hyperparameters of the config file can be overwritten by flags:
```
python Gym_Framework.py config.yaml -params learning_rate:"linear_0.001" use_sde:False --action 3T2R
```
- param: all the values for the agent
    - learning_rate:"Linear_0.001" -> chooses linear schedule with initial value of 0.001. For constant learning_rate use e.g. learning_rate:0.001
- other flags: overwrite config.yaml or are used to control programm, e.g. choose the environment 

for more information about flags type
``` 
python Gym_Framework.py -h
```


## gym_env.py
contains the gym environment with the following structure 
### step()
performs one action in the environment and returns observation, reward and terminal

in this repo, one action is a scan, which is performed with the scanner module

### reset()
resets the environment to start a new epoch

resets also scanner, which means clearing all point-clouds and loading a new mesh

also the initial_scan will be performed if specified in the config

returns an observation

### _get_state()
returns the encoded state

2048x4: point cloud is encoded as [x, y, z, 0/1] -> 0: unseen point, 1: seen
in_policy: in this scan seen points [x, y, z]

input is normalized with MinMaxScaler

### _perform_scan()
performs a scan according to given action
updates all pointsclouds

### _get_reward()
calculates reward and if a terminal state was reached

## __init_.py
used to register the environment as gym environment.

## PC_Generator (pc_generator3.py)
Contains all necessary methods to perform a scan (given a position and angles), update all pointclouds

parent class



## pc_generator_coverage.py/ pc_generator_inspection.py
Inherit from pc_generatpr3.py and contain the environment specific methods like calculating the coverage or encoding the state arrays

### single_scan(translation) (1)
### single_scan(alpha, beta, gamma, degrees=False) (2)
To perform a scan, single_scan(...) must be performed. 

Two options to specifiy the viewpose:
1. Homogenous transformation matrix
2. Position (x, y, z) and euler angles (option degrees=True if specified as \[°])

To use the scan, call scanner.array, which contains either the coordinates of the scanned points (encoding=in_policy) or the coordinates of all points with the according encoding of 0/1 (encoding=2048x4)

A sensor model can be specified in the scanner_config.
Options are:
- cam_workspace_bool (1)
- proj_bool (2)

1. - if True, will calculate for each ray the distance and delete points which are not inside the specified cam_workspace
    - if False, will delete all points scanned if origin of scanner was outside specified cam_workspace
2. - if False: no projector will be used
    - if True: a second scan will be performed with a scanner translated by the specified trans_proj relative to the scanner. Only points which were seen by both scanner and projector will be further used, others get deleted

## Models
contains the keras and pytorch models which are used as policy for the agent or for the supervised problem

## sup_framework.py
contains everything necessary to create_datasets, load them, create, save and load the network and also methods to train the net and evaluate it.

Only the __main__() must be performed

## Data
contains the relevant .stl files and the full pointclouds for each file

samples from ABC-dataset and StaterMotors are possible

also contains the supervised datasets

# How to cite
Please consider citing when using our simulation
@misc{ViewPlanningSimulation,
  author={Kaiser, Jan-Philipp and Koch, Dominik and Schmid, Jonas and Gaebele, Jonas},
  title={View-Planning-Simulation},
  year={2022},
  url={https://github.com/Jarrypho/View-Planning-Simulation},
}


