import os
import random
import wandb
import time
import numpy as np
from RL_Framework.Gym.Parent_Env import gym_parent
from PointCloud_Generator.pc_generator_coverage import PcdGen_Cov
from RL_Framework.utils.env_utils import action_mapping, distance_to_origin, get_cam_pose_and_rotation
from RL_Framework.utils.pointcloud_utils import scatter_pcd_interactive, scatter_interactive_nbv


class NBV_VP(gym_parent):
    def __init__(
        self,
        data_dir: os.path, 
        outdir: str,
        scanner: object = PcdGen_Cov,
        file_type: str = 'stl',
        encoding: str = '2048x4',
        action_type: str = '2T0R',
        max_iterations = 10,
        CamRadius = 50,
        track = 1,
        inverted_state = False,
        initial_scan = [1.42, 0.71],
        rewardtype = "dcov_each",
        scaling = True

    ):
        super().__init__(action_type, encoding, scaling)
        self.action_type = action_type
        self.file_type = file_type
        self.encoding = encoding
        self.outdir = outdir
        self.max_iterations = max_iterations
        self.CamRadius = CamRadius
        self.data_dir = data_dir
        self.scanner = scanner
        self.track = track
        self.initial_scan = initial_scan
        self.inverted_state = inverted_state
        self.rewardtype = rewardtype
        
        self.start = 0
        self.poses = []
        self.nbv_poses = []

    def step(self, action):
        s2 = time.perf_counter()
        info = {}
        self._perform_scan(action, init=False)
        if len(self.scanner.current_pcd.points) == 0:
            self.empty_scans +=1
        observation = self._get_state()
        reward, terminal = self._get_reward()
        self.cum_reward += reward
        done = terminal
        e2 = time.perf_counter()
        step_time = e2-s2
        if self.track >= 0:
            wandb.log({"reward_step":reward, "runtime/step_time":step_time})
        
        
        if self.track >= 0 and (terminal or self.iteration == 2):
            self._log()
            if self.track == 2:
                fig = scatter_interactive_nbv(np.asarray(self.nbv_poses))
                wandb.log({"nbv_pose - [red: initial; cyan: nbv, green: object origin]": fig})

        return observation, reward, done, info

    def reset(self):
        self.start = time.time()
        self.iteration = 0
        self.coverage = 0
        self.cum_reward = 0
        self.empty_scans = 0
        self.nbv_poses = []
        if self.epochs % 200 == 0:
            self.poses= []

        mesh_dir = os.path.join(self.data_dir, self.file_type)
        names = [n for n in os.listdir(mesh_dir) if n.endswith(self.file_type)]
        name = random.choice(names)
        # if self.dataset == "abc" and self.track >= 0:
        #     wandb.log({"utils/object_nr":int(name[:-4])}, commit=False)
        # setup scanner
        mesh_path = os.path.join(mesh_dir, name)
        
        #reset scanner parameters
        self.scanner.reset()
        #load object
        self.scanner.setup(mesh_path)
        #create state encoding array from full_pcd
        self.scanner.create_encoding_array()
        #perform initial scan
        if self.initial_scan != None:
            action = self.initial_scan
            self._perform_scan(action=action, init=True, First=True)
        else:
            action = np.random.default_rng().uniform(low=-1.0, high=1.0, size=(2, ))
            self._perform_scan(action=action, init=False, First=True)
        
        self.coverage = self.scanner.calc_coverage_binary()
        #observation is encoded state
        observation = self._get_state()
        return observation

    def render(self):
        pass

    def close(self):
        super().close()

    def _perform_scan(self, action, init=False, First: bool=False):
        """performs a scan with the defined scanner module

        Args:
            action (np.array): action based on the action_encoding
            init (bool, optional): if the scan is a initial scan (2T0R) -> centered scan. Defaults to False.
            First (bool, optional): if scan is first scan . Defaults to False.
        """
        self.iteration += 1

        if First:
            if init==False:
                action = action_mapping("2T0R", action)
            pos, alpha, beta, gamma = get_cam_pose_and_rotation(
                action_type="2T0R",
                actions=action,
                radius=self.CamRadius,
                initial=init)
        else:
            action = action_mapping(self.action_type, action)
            pos, alpha, beta, gamma = get_cam_pose_and_rotation(
                action_type=self.action_type,
                actions=action,
                radius=self.CamRadius,
                initial=init)

        #append cam position for plotting
        self.nbv_poses.append(pos)
        if self.iteration == 2:
            self.poses.append(pos)
        if self.track >= 0:
            dist = distance_to_origin(pos)
            wandb.log({"utils/distance_to_origin":dist}, commit=False)
        
        #Performing the scan
        scan = self.scanner.single_scan(
            translation=pos, 
            alpha=alpha, 
            beta=beta, 
            gamma=gamma)
        if self.encoding == "2048x4" and len(self.scanner.current_pcd.points) > 0:
            self.scanner.update_binary_state_array()
        self.scanner.update_pointclouds()
        
        #nbv plot
        if self.track >= 0 and (self.track == 2 and self.epochs % 10 == 0):
            data = np.array(self.scanner.array)
            fig = scatter_pcd_interactive(data, inspection=False)
            wandb.log({"pos and pcd: [lime: seen poins, grey: unseen]":fig})


    def _get_reward(self):
        """calculates the reward as specified

        Raises:
            Exception: Reward type not implemented 

        Returns:
            float: calculated reward for a step
        """
        new_coverage = self.scanner.calc_coverage_binary()
        dif_coverage = new_coverage - self.coverage
        if self.track >= 0:
            wandb.log({f"Coverage/Dif_Coverage_in_Scan_{str(self.iteration).zfill(2)}": dif_coverage}, commit=False)
        if new_coverage > self.coverage:
            self.coverage = new_coverage
        
        terminal = False
        
        #linear reward based on the difference of the coverage
        #if terminal (goal reached), extra reward
        if self.rewardtype == 'dcov_each':
            if dif_coverage > 0:
                reward = dif_coverage
            elif dif_coverage <= 0:
                reward = -1
            if terminal:
                reward += 1
            return reward, terminal
        #same as dcov_each, but scaled to exponential function
        elif self.rewardtype == 'exp_dcov':
            if dif_coverage > 0.0:
                reward = np.exp(dif_coverage / 100. - 0.5) * (dif_coverage/100)**2
            else:
                reward = -0.02
            if terminal:
                reward += 1
            return reward, terminal
        #returns coverage/number of scans only if terminal, else negative reward
        elif self.rewardtype == "cov_all":
            if not terminal:
                reward = -1
            else:
                if self.iteration < 1:
                    reward = 0
                else:
                    reward = self.coverage / self.iteration
            return reward, terminal
        else:
            raise Exception(f"reward type {self.rewardtype} not valid")


        
