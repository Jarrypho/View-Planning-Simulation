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

# Contribution: This code is based on https://github.com/mikedh/trimesh/issues/620
import trimesh
import numpy as np
import open3d as o3d
from PointCloud_Generator.utils import scanner_laser_overlapp, calc_dist, vis_visible_and_nonvisible_pcd, combine_scans, subtract, as_mesh

from RL_Framework.utils.env_utils import get_cam_rotation, get_camera_transform_from_euler, distance_to_origin

"""class for creating pointcloud from meshes and manage the pointclouds
    -resolution: resolution of the scanner
    -fov_degree: fov of the scanner
    -binary_encoding: if state representation is used as binary (2048x4) with [x, y, z, 0/1]
    -cam_workspace: distance in which the scanner can detect points
    -downsampling_factor: factor with which the pointcloud is downsampled
    -file_type: file_type of the object which should be loaded (.stl for mesh)
    -dim: number of points which with which the area of the loaded object is represented
    -sensor_debug: option for debugging projector (visualization of scanned pointsclouds)
    -proj_bool: whether a projector is used to use only the scanned points which are also viewed by a projector
    -trans_proj: translation on [x, y, z]-Axis to the cam of the scanner 
    -cam_workspace_bool: wheter the points which are not in the cam_workspace should be deleted from a scan
"""

class PointcloudGenerator:
    def __init__(
        self, 
        downsampling_factor: float,
        file_type: str,
        resolution = [384, 240],
        fov_degree = [27, 25],
        cam_workspace = [30.0, 50.0],
        cam_workspace_bool = True,
        sensor_debug = False,
        proj_bool =False,
        trans_proj = [0.0, 5.0, 0.0]
    ):
        self.resolution = resolution
        self.fov_degrees = fov_degree
        self.cam_workspace = cam_workspace
        self.sensor_debug = sensor_debug
        self.proj_bool = proj_bool
        self.trans_proj = trans_proj
        self.cam_workspace_bool = cam_workspace_bool
        self.downsampling_factor = downsampling_factor
        self.file_type = file_type

    def reset(self):
        """resets the pointsclouds of the scanner and initializes them
           call first 
        """
        self.pcd_full_ds_norm = o3d.geometry.PointCloud()
        self.array = []
        self.pcd_full_ds = o3d.geometry.PointCloud()
        self.current_pcd = o3d.geometry.PointCloud()
        self.combined_pcd = o3d.geometry.PointCloud()
        self.combined_pcd_old = o3d.geometry.PointCloud()
        self.dif_new_old_combined = o3d.geometry.PointCloud()
        self.possible_pcd = o3d.geometry.PointCloud()

        
    def setup(self, mesh_path: str):
        """loads mesh and full pcd of a object. 

        Args:
            mesh_path (str): path to mesh
        """
        # Setup Scene
        self.mesh = as_mesh(trimesh.load(mesh_path))

        # center mesh:
        box = self.mesh.bounding_box
        center = box.center_mass
        # print('Center: ', center)
        self.mesh.apply_translation(center * -1)
        self.box = self.mesh.bounding_box
        # Create Scene from centerd Mesh
        self.scene = self.mesh.scene()

        # Creatre Camera
        self.scene.camera.resolution = self.resolution
        self.scene.camera.fov = self.fov_degrees
        
        pcd_dir = mesh_path.replace(self.file_type, "pcd")
        pcd_full = o3d.io.read_point_cloud(pcd_dir)
        self.possible_pcd = pcd_full
        self.pcd_full_ds = pcd_full.voxel_down_sample(self.downsampling_factor)
    
    def update_pointclouds(self):
        """updates the pointsclouds based on the performed scan
        """
        if len(self.combined_pcd.points) == 0:
            self.possible_pcd = self.pcd_full_ds
            self.combined_pcd = self.current_pcd
            self.dif_new_old_combined = self.current_pcd
        else:
            if len(self.current_pcd.points) > 0:
                new_combined_pcd = combine_scans(self.combined_pcd, self.current_pcd)
                self.combined_pcd = new_combined_pcd

    def update_dif_new_old_pcd(self):
        """updates the difference between the new and old pointcloud
           - only necessary for calculation of specific rewards
        """
        if len(self.current_pcd.points) > 0:
            if len(self.combined_pcd.points) == 0:
                self.dif_new_old_combined = self.current_pcd
            else:
                self.dif_new_old_combined = subtract(self.combined_pcd, self.combined_pcd_old, thres_adjust=2)  
                self.combined_pcd_old = self.combined_pcd
        else:
            self.dif_new_old_combined = self.current_pcd

    def visualizue_pcd(self, pointcloud: o3d.geometry.PointCloud()):
        """visualizes a pointcloud
        Args:
            pointcloud (o3d.geometry.PointCloud()): pointcloud to visualize
        """
        o3d.visualization.draw_geometries([pointcloud])

    
    def _general_scan(self, transformation_matrix=None, translation=[0, 0, 0], alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0, degrees: bool = False):
        """internal function to perform a single scan of either cam or projector

        Args:
            transformation_matrix (float, optional): shape(4x4). Defaults to None.
            translation (list, optional): translation of cam from origin. Defaults to [0, 0, 0].
            alpha (float, optional): first rotation. Defaults to 0.0.
            beta (float, optional): second rotation. Defaults to 0.0.
            gamma (float, optional): third rotation. Defaults to 0.0.
            degrees (bool, optional): _use degree or radian interpretation for angles. Defaults to False.

        Returns:
            points (list): coordinates of scanned points
            index_ray (list): indices of rays - which ray index was responsible for corresponding point
            index_tri (list): which triangle of the mesh the ray intersected
            origins (list): coordinates of the cameras origin
            transformation_matrix (np.array): transformation matrix of the camera
        """

        if transformation_matrix == None:
            transformation_matrix = get_camera_transform_from_euler(translation=translation, alpha=alpha, beta=beta, gamma=gamma, degrees=degrees)
        # Set Camera Pose as 4x4 Array (Transformation Matrix)
        self.scene.camera_transform = transformation_matrix


        # perform Scan (Raytracing)
        # vectors [x,y,z] 1 vector = 1 pixel of the Depth Camera
        # origins [x,y,z] :origin of the camera, one for each vector but all the same values
        origins, vectors, _ = self.scene.camera_rays()
        # do the actual ray- mesh queries
        points, index_ray, index_tri = self.mesh.ray.intersects_location(
            origins, vectors, multiple_hits=False)
        origins = np.float32(origins)
        points = np.float32(points)
        index_ray = np.float32(index_ray)
        index_tri = np.float32(index_tri)

        return points, index_ray, index_tri, origins, transformation_matrix
    
    def single_scan(self, transformation_matrix=None, translation=[0, 0, 0], alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0, degrees: bool = False):
        """function to call to perform a scan of a mesh object given a position and rotation of the scanner

        Args:
            transformation_matrix (float, optional): shape(4x4). Defaults to None.
            translation (list, optional): translation of the scanner in reference to origin. Defaults to [0, 0, 0].
            alpha (float, optional): first rotation. Defaults to 0.0.
            beta (float, optional): second rotation. Defaults to 0.0.
            gamma (float, optional): third rotation. Defaults to 0.0.
            degrees (bool, optional): use degree or radian interpretation for angles. Defaults to False.

        Returns:
            dict: returns scan dict, including the captured pointcloud and the transformation matrix of the scanner
        """
        #performs one scan for camera
        if transformation_matrix == None:
            points_scanner, _ , index_tri_scanner, origins, transformation_matrix = self._general_scan(translation=translation, alpha=alpha, beta=beta, gamma=gamma, degrees=degrees)
        else:
            points_scanner, _, index_tri_scanner, origins, transformation_matrix = self._general_scan(transformation_matrix=transformation_matrix, degrees=degrees)
        #performs a scan for projector if set to True
        if self.proj_bool:
            pos_proj = translation.copy()
            trans_proj = self.trans_proj    
            for idx in range(3):
                pos_proj[idx] = pos_proj[idx] + trans_proj[idx]
            alpha_proj, beta_proj, gamma_proj = get_cam_rotation(pos_proj)
            
            _, _, index_tri_laser, _, _ = self._general_scan(translation=pos_proj, alpha=alpha_proj, beta=beta_proj, gamma=gamma_proj, degrees=degrees)

            #calculate the indices of points which are visible (1) and non-visible (0)
            ind_del_idx = scanner_laser_overlapp(index_tri_scanner, index_tri_laser)

            #calculate points which are not visible, for debug purpose or visibility
            if self.sensor_debug: 
                inv_idx_np = np.where(ind_del_idx==0)[0]
                inverse_idx = inv_idx_np.tolist()
                points_scanner_inverse = np.delete(points_scanner, inverse_idx, axis=0)
                pcd_inv = o3d.geometry.PointCloud()
                pcd_inv.points=o3d.utility.Vector3dVector(points_scanner_inverse)

            #calculate points which are visible by camera and projector    
            ind_del_np = np.where(ind_del_idx==1)[0]
            ind_del = ind_del_np.tolist()
            if len(ind_del) != 0:
                points_scanner = np.delete(points_scanner, ind_del, axis=0)

        pcd = o3d.geometry.PointCloud()
        #calculates for each ray the distance of the intersection point to the cam and deletes points which are not in workspace
        if self.cam_workspace_bool:
            if len(points_scanner) > 0:
                #delete points which are not in viewfield of zivid one cam
                points_scanner = calc_dist(points_scanner, origins, self.cam_workspace)
                pcd.points = o3d.utility.Vector3dVector(points_scanner) 
        #deletes complete pcd if cam is not in workspace, if in workspace every scanned point is used
        else:
            dist = distance_to_origin(translation)
            if (self.cam_workspace[0] > dist) or (self.cam_workspace[1] < dist):
                points_scanner = []
                pcd.points = o3d.utility.Vector3dVector()
            else:
                pcd.points = o3d.utility.Vector3dVector(points_scanner)

        if self.sensor_debug and self.proj_bool:
            vis_visible_and_nonvisible_pcd(pcd, pcd_inv, self.mesh)
        
        #update current pointcloud
        self.current_pcd = pcd
        scan = {
            'transformation': transformation_matrix,
            'pcd': pcd
        }
        
        return scan

    



