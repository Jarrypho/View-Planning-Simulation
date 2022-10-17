from PointCloud_Generator.pc_generator3 import PointcloudGenerator
from PointCloud_Generator.utils import subtract, vox_downsample_numpoint, get_seen_indices
import numpy as np

class PcdGen_Cov(PointcloudGenerator):
    def __init__(self, 
        downsampling_factor: float = 0.01,
        file_type: str = "stl",
        dim: int = 2048,
        **kwargs
    ):
        super().__init__(
            downsampling_factor = downsampling_factor,
            file_type = file_type,
            **kwargs
        )
        self.dim = dim

    def calc_coverage(self):
        """calculates coverage percentage of the object based on pcds

        Returns:
            float: coverage in percent
        """
        pcd_covered = self.combined_pcd.voxel_down_sample(self.downsampling_factor)
        pcd_not_covered = subtract(complete=self.pcd_full_ds, partial=pcd_covered, thresh_adjust=2)
        new_coverage = (len(self.pcd_full_ds.points) - len(pcd_not_covered.points)) / len(self.pcd_full_ds.points) * 100
        return new_coverage

    def calc_coverage_binary(self):
        """calculates the coverage percentage of the object based on the binary encoding matrix
        Returns:
            float: coverage in percent
        """
        seen_points = np.sum(np.asarray(self.array)[:, 3] > 0, axis=0)
        new_coverage = seen_points / 2048.0 * 100.0

        return new_coverage

    def create_encoding_array(self):
        """creates the array for binary encoding
            last row all zeros
            [x, y, z, 0]
        """
        self.pcd_full_ds_norm = vox_downsample_numpoint(self.pcd_full_ds, self.dim)
        help_array = np.asarray(self.pcd_full_ds_norm.points)
        zeros = np.zeros((len(help_array), 4))
        zeros[:, :-1] = help_array
        self.array = zeros.tolist()

    def update_binary_state_array(self):
        """updates binary array
            [x, y, z, 0] - all unseen points
            [x, y, z, 1] - all seen points
        """
        indices = get_seen_indices(self.pcd_full_ds_norm, self.current_pcd, thresh_adjust=1)
        for idx in indices:
            self.array[idx][3] = 1.0


    def inverted_state(self):
        """if state representation should be inverted
           more information in theses Jonas Schmid
        Returns:
            o3d.point_cloud: inverted_pointcloud
        """
        inverted_pcd = subtract(self.pcd_full_ds, self.combined_pcd, thres_adjust=2)
        return inverted_pcd