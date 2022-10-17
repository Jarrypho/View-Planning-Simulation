from PointCloud_Generator.pc_generator3 import PointcloudGenerator
import open3d as o3d
from PointCloud_Generator.utils import get_nearest_points, vox_downsample_numpoint, get_seen_indices
import numpy as np
import sys

class PcdGen_Ins(PointcloudGenerator):
    def __init__(self, 
        downsampling_factor: float = 0.01,
        file_type: str = "stl",
        dim: int = 2048,
        **kwargs
    ):
        self.downsampling_factor = downsampling_factor
        self.file_type = file_type
        super().__init__(
            downsampling_factor = downsampling_factor,
            file_type = file_type,
            **kwargs
        )
        self.dim = dim
        self.num_roi_points = 0

    def create_encoding_array(self, num_points, num_areas):
        """creates the array for binary encoding
            last row all zeros
            [x, y, z, 0]
        """
        self.pcd_full_ds_norm = vox_downsample_numpoint(self.pcd_full_ds, self.dim)
        help_array = np.asarray(self.pcd_full_ds_norm.points)
        zeros = np.zeros((len(help_array), 4))
        zeros[:, :-1] = help_array
        indices = get_nearest_points(self.pcd_full_ds_norm, num_points, num_areas)
        for idx in indices:
            zeros[idx, 3] = -1
        self.array = zeros.tolist()
        self.num_roi_points = np.sum(np.asarray(self.array)[:, 3] < 0, axis=0)
        return indices
    
    def update_encoding_array(self):
        """updates encoding area, unseen ROI points get updated if seen
        """
        if len(self.current_pcd.points) > 0:
            indices = get_seen_indices(self.pcd_full_ds_norm, self.current_pcd, thresh_adjust=1)
            if len(indices) > 0:
                for index in indices:
                    if self.array[index][3] == 0:
                        continue
                    else:
                        self.array[index][3] = 1
                    
    def calc_ROI_coverage(self):
        """calculates the covered areas. 
        In inspection env only the points in ROI is considered to calculates the covered area

        Returns:
            float: percentage of covered ROI area
        """
        unseen = np.sum(np.asarray(self.array)[:, 3] < 0, axis=0)

        new_coverage = (self.num_roi_points - unseen) / (self.num_roi_points) * 100
        return new_coverage

    def vis_encoding(self):
        """visualizes encoding matrix as pcds with different colors
        """
        ind_1 = np.where(np.asarray(self.array)[:, 3] == 1)
        ind_0 = np.where(np.asarray(self.array)[:, 3] == 0)
        ind_neg1 = np.where(np.asarray(self.array)[:, 3] == -1)
        
        self.pcd_full_ds_norm.paint_uniform_color([0.0, 0.0, 0.0])
        np.asarray(self.pcd_full_ds_norm.colors)[ind_1[0], :] = [1, 0, 0]
        np.asarray(self.pcd_full_ds_norm.colors)[ind_0[0], :] = [0, 1, 0]
        np.asarray(self.pcd_full_ds_norm.colors)[ind_neg1[0], :] = [0, 0, 1]
        o3d.visualization.draw_geometries([self.pcd_full_ds_norm], window_name="Full with choosen areas")


        


if __name__=='__main__':
    sys.path.append("/home/dominik/Documents/Git/ba_jonas")

    mesh_path = '/home/dominik/Documents/Git/ba_jonas/Data/StarterMotors/stl/Starter_Engine_10.stl'
    pcd_path = '/home/dominik/Documents/Git/ba_jonas/Data/StarterMotors/pcd/Starter_Engine_10.pcd'
    CamRadius = 50
    # Read Full Pointcloud
    pcd_full = o3d.io.read_point_cloud(pcd_path)
    pcd_full = vox_downsample_numpoint(pcd_full, 2048)
    # Create Partial Pointcloud
    scanner = PcdGen_Ins(resolution=[120, 120])
    scanner.reset()
    scanner.setup(mesh_path)
    ind = scanner.create_encoding_array(50, 2)
    print(ind)
    scanner.visualizue_pcd(scanner.pcd_full_ds_norm)
    print(np.asarray(scanner.array)[:50, 3])