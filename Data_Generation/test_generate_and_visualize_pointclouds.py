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

import sys
sys.path.append("/home/dominik/Documents/Git/ba_jonas")
import RL_Framework.utils.pointcloud_utils as pc_util
import RL_Framework.utils.env_utils as env_util
from PointCloud_Generator.pc_generator3 import PointcloudGenerator
import random
from math import pi, sin, cos
import open3d as o3d
import numpy as np
import time
import os

"""test file for some features"""
BASE_PATH = os.path.abspath(os.getcwd())
# Dir: /workspace/kaiser/GaebeleJonas/ba-gaebele/Data/StarterEngines/pcd
#motor_path = os.path.join(BASE_PATH,'Data',"abc", "stl", "00000053.stl")
motor_path = "/media/dominik/Backup Plus/Masterarbeit/abc_dataset/stl/00000072.stl"
#motor_path = "/home/dominik/Documents/Git/ba_jonas/Data/StarterMotors/stl/Starter_Engine_120.stl"
out_path = "/media/dominik/Backup Plus/Masterarbeit/abc_dataset/scaled/00000000.stl"

scanner = PointcloudGenerator(mesh_path=motor_path, resolution=[900,600])

def log(txt:str):
    pass
    #print(txt)
    
cam_radius = 50

scans = []
vis = []
for i in range(10):
    phi = random.uniform(0, 2*pi) # radians
    theta = random.uniform(0, pi) # radians
    log("phi: " + str(phi) + "theta:" + str(theta))
    pos = [cam_radius * sin(theta) * cos(phi), cam_radius * sin(theta) * sin(phi),
           cam_radius * cos(theta)]
    #pos = [random.uniform(-100,100), random.uniform(-100,100), random.uniform(-100,100)]
    #pos = [0, 50, 0]
    alpha, beta, gamma = env_util.get_cam_rotation(pos)
    beta = 0
    print(alpha, beta)
    pos_proj = [cam_radius * sin(theta) * cos(phi), cam_radius * sin(theta) * sin(phi),
           cam_radius * cos(theta)]
    pos_proj[1] = pos_proj[1] + 5.0
    alpha_proj, beta_proj, gamma_proj = env_util.get_cam_rotation(pos_proj)

    log("Pos [x,y,z] = " + str(pos))
    log("Rotation: " + str(alpha) + str(beta) + str(gamma))
    # trans_matrix = pc_util.get_camera_transform_from_euler(pos, alpha, beta, gamma, degrees=False)

    log("    - scan")
    s = time.time()
    scan = scanner.single_scan(translation=pos, alpha=alpha, beta=beta, gamma=gamma, trans_2=pos_proj, alpha_2=alpha_proj, beta_2=beta_proj, gamma_2=gamma_proj)
    e = time.time()
    #print("Scantime: ", e-s)
    scans.append(scan['pcd'])
    #cam_pos = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=3) 
    cam_pos = o3d.geometry.TriangleMesh.create_arrow()
    cam_pos = cam_pos.rotate([[1,0,0], [0,-1,0],[0,0,-1]])
    transform = env_util.get_camera_transform_from_euler(pos, alpha, beta, gamma, degrees=False)
    
    cam_pos_rot = cam_pos.transform(transform)
    
    
    # cam_pos_rot = cam_pos.rotate(transform[:3, :3])
    #cam_pos_rot = cam_pos.translate(transform[:3, 3])
    # print(transform)
    # print(transform[:3, :3])
    vis.append(cam_pos_rot)

#pc_util.vis_pcd(scans)
cam = o3d.geometry.TriangleMesh.create_sphere(radius=cam_radius, resolution=20)
cam_line = o3d.geometry.LineSet.create_from_triangle_mesh(cam)

mesh = o3d.io.read_triangle_mesh(motor_path)
vis.append(mesh)
vis.append(cam_line)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
vis.append(mesh_frame)
o3d.visualization.draw_geometries(vis,mesh_show_wireframe=True, window_name="mesh")

# pcd = mesh.sample_points_poisson_disk(number_of_points=2048)
# pcd_ds = pcd.voxel_down_sample(0.005)
# #o3d.visualization.draw_geometries([pcd])

bb = mesh.get_oriented_bounding_box()
o3d.visualization.draw_geometries([mesh, bb], mesh_show_wireframe=True)
print(bb.get_center())
bb_points = np.asarray(bb.get_box_points())
print(bb_points)
# print(np.amax(bb_points))
# mesh_2 = mesh.scale(scale=10*1/abs(np.amax(bb_points)), center=bb.get_center())
# #mesh_2 = mesh.scale(scale=100, center=bb.get_center())

# bb_2 = mesh_2.get_oriented_bounding_box()

# o3d.visualization.draw_geometries([mesh_2, bb_2, cam_line])
# print(bb_2.get_center())
# bb_points_2 = np.asarray(bb_2.get_box_points())
# print(bb_points_2)
# print(np.amax(bb_points_2))
# mesh_2.compute_triangle_normals()
# #o3d.io.write_triangle_mesh(out_path, mesh_2)






