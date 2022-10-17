import os
import random
import wandb
import time
import numpy as np
import matplotlib.pyplot as plt
from RL_Framework.Gym.Parent_Env import gym_parent
from PointCloud_Generator.pc_generator_inspection import PcdGen_Ins
from RL_Framework.utils.env_utils import action_mapping, distance_to_origin, get_cam_pose_and_rotation
from RL_Framework.utils.pointcloud_utils import plot_spherical, scatter_pcd_interactive


class InspectionVP(gym_parent):
    def __init__(
        self,
        data_dir: os.path, 
        outdir: str,
        scanner: object = PcdGen_Ins,
        file_type: str = 'stl',
        encoding: str = '2048x4',
        action_type: str = '2T0R',
        max_iterations: int = 10,
        CamRadius: float = 50.0,
        track : int = 1,
        num_points = 50,
        num_areas = 1,
        initial_scan = [1.42, 0.71],
        rewardtype = "dcov",
        desired_coverage = 95,
        scaling = True
        ):
        super().__init__(
            encoding=encoding,
            action_type=action_type,
            scaling=scaling
        )
        assert file_type == 'stl' or file_type == 'obj'
        # Constants
        self.inverted_state = False
        self.CamRadius = CamRadius  # Sphere for Camera Positions
        self.outdir = outdir
        self.rewardtype = rewardtype
        self.max_iterations = max_iterations
        self.data_dir = data_dir
        self.file_type = file_type
        self.track = track
        self.initial_scan = initial_scan
        self.desired_coverage = desired_coverage

        self.encoding = encoding
        self.action_type = action_type

        # Variables
        self.scan_positions = []
        self.coverage = 0
        self.iteration = 0
        self.current_pose = []
        self.last_pose = []  # stores the last pose for travel cost calculation
        self.cumulated_distance = 0
        # Setup PointNet for State encoding
        self.scanner = scanner
        self.num_points = num_points
        self.num_areas = num_areas

        if encoding == 'in_policy':
            return

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

        if self.track >= 0 and (terminal or self.iteration == self.max_iterations):
            self.epochs += 1
            end = time.time()
            epoch_time = end - self.start
            wandb.log({
                "coverage": round(self.coverage,2),
                "scans": self.iteration,
                "epochs": self.epochs,
                "reward": self.cum_reward,
                "runtime/epoch_time": epoch_time,
                "utils/empty_scans": self.empty_scans
            })
            if self.track > 0 and self.epochs % 10 == 0:
                try:
                    fig = plot_spherical(np.asarray(self.poses))
                    plot = wandb.Image(fig)
                    wandb.log({"all_poses (colorscale density), green: object, red: initial": plot})
                    plt.close()
                except: 
                    print("plotting failed")
        return observation, reward, done, info

    def reset(self):
        self.start = time.time()
        self.iteration = 0
        self.coverage = 0
        self.cum_reward = 0
        self.empty_scans = 0
        if self.epochs % 200 == 0:
            self.poses= []

        mesh_dir = os.path.join(self.data_dir, self.file_type)
        names = [n for n in os.listdir(mesh_dir) if n.endswith(self.file_type)]
        name = random.choice(names)
        # setup scanner
        mesh_path = os.path.join(mesh_dir, name)
        
        #reset scanner parameters
        self.scanner.reset()
        #load object
        self.scanner.setup(mesh_path)
        #create state encoding array from full_pcd
        self.scanner.create_encoding_array(self.num_points, self.num_areas)
        #perform initial scan
        if self.initial_scan != None:
            action = self.initial_scan
            self._perform_scan(action=action, init=True)
            self.coverage = self.scanner.calc_coverage()
        #observation is encoded state
        observation = self._get_state()
        
        #data = self.scanner.array
        #fig = scatter_pcd_interactive(data, inspection=True)
        #fig.show()

        return observation

    def render(self):
        pass

    def close(self):
        super().close()

    

    def _perform_scan(self, action, init=False):
        """performs scan

        Args:
            action (np.array): action of agent
            init (bool, optional): if scan is a initial scan (2T0R) -> centered scan. Defaults to False.
        """
        if init==False:
            self.iteration += 1
            action = action_mapping(self.action_type, action)
        pos, alpha, beta, gamma = get_cam_pose_and_rotation(
            action_type=self.action_type,
            actions=action,
            radius=self.CamRadius,
            initial=init)
        if init==False:
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
            self.scanner.update_encoding_array()
        #self.scanner.update_pointclouds()
        
        if self.track >= 0 and (self.track == 2 and self.epochs % 10 == 0):
            data = self.scanner.array
            fig = scatter_pcd_interactive(data, inspection=True)
            wandb.log({"pos and pcd: [red: unseen ROI; lime: seen ROI, grey: uninterested]": fig})
            plt.close()


    def _get_reward(self):
        """calculates reward as specified

        Raises:
            NotImplementedError: if reward nor implemented

        Returns:
            float: calculated reward
        """
        #new coverage percentage
        new_coverage = self.scanner.calc_ROI_coverage()
        
        #if terminal stage is reached
        if self.initial_scan == None:
            terminal = (new_coverage > self.desired_coverage)
        else:
            terminal = (new_coverage > self.desired_coverage)

        #difference of coverage between two scans
        dif_coverage = new_coverage - self.coverage
        if self.track >= 0:
            wandb.log({f"Coverage/dif_cov_per_scan_{str(self.iteration).zfill(2)}":dif_coverage}, commit=False)

        #only increases coverage is new is bigger
        if new_coverage > self.coverage:
            self.coverage = new_coverage
        
        #based on dif_coverage scaled reward, negative if dif_coverage not bigger. If goal is reached, big reward
        if self.rewardtype == "dcov":
            if dif_coverage > 0:
                reward = dif_coverage/100
            else:
                reward = -0.1
            if len(self.scanner.current_pcd.points) > 0:
                reward += 0.01
            else:
                reward += 0
            if terminal:
                reward += 1
            if self.epochs == 0:
                print(f"Rewardfunc: dif_cov [dif_cov/100; -0.25], len_pcd.points [0.01; -0.25], terminal: 1")
            return reward, terminal
        if self.rewardtype == "sparse":
            if terminal:
                reward = 1
            else:
                reward = -0.25
        else:
            raise NotImplementedError(f"reward type {self.rewardtype} not valid")

    
            
