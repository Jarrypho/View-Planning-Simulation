from gym import Env, spaces
from RL_Framework.utils.env_utils import get_output_dim
from PointCloud_Generator.utils import vox_downsample_numpoint
from RL_Framework.utils.pointcloud_utils import encode_voxel, plot_spherical
from sklearn import preprocessing
import numpy as np
import open3d as o3d
import time, wandb
import matplotlib.pyplot as plt

class gym_parent(Env):
    """Parent Gym Environment in View-Planning-Simulation

    Args:
        Env (Gym.Env): Gym Environment
    """
    def __init__(self, action_type: str, encoding: str, scaling: bool):
        """defines action and observation space according to gym documentation

        Args:
            action_type (str): action_type
            encoding (str): type of encoding
        """
        self.action_type = action_type
        self.encoding = encoding
        self.scaling = scaling
        self.epochs = 0
        self.action_type_variable = get_output_dim(self.action_type)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_type_variable, ))    

        if encoding == "2048x4":
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2048,4))
        elif encoding =="in_policy":
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2048,3))
        else:
            raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        super().close

    def render(self):
        pass

    def _get_state(self):
        """creates encoding of state and normalizes values"""
        pointcloud = self.scanner.combined_pcd
        
        if self.inverted_state:
            """invertes state point cloud - further reference Thesis Jonas Schmid"""
            pointcloud = self.scanner.inverted_state()
        if self.encoding == "2048x4":
            """encoding with 2048 x [x, y, z, 0/1]"""
            x = np.asarray(self.scanner.array)
            if self.scaling:
                min_max_scaler = preprocessing.MinMaxScaler()
                x = min_max_scaler.fit_transform(x)
            return x
        if self.encoding == "in_policy":
            """ true pointcloud but downsampled to match fixed input dimension of policy"""
            points = pointcloud.points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd = vox_downsample_numpoint(pcd, self.dim)
            points = np.asarray(pcd.points)
            min_max_scaler = preprocessing.MinMaxScaler()
            points = min_max_scaler.fit_transform(points)
            return points
        if self.encoding == "voxel":
            return encode_voxel(points=self.combined_pcd.points)
    
    def _log(self):
        """logs metrics to wandb
        """
        self.epochs += 1
        end = time.time()
        epoch_time = end - self.start
        wandb.log({
            "coverage": round(self.coverage,2),
            "scans": self.iteration + 1,
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
                print("logging plot failed")