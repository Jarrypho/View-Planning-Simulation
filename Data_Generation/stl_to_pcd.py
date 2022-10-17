import numpy as np
from scipy.spatial.transform import Rotation as R
from math import sin, cos, pi
import os
import open3d as o3d
import time
import sys
sys.path.append("/home/dominik/Documents/Git/ba_jonas")
from RL_Framework.utils.env_utils import get_cam_rotation
from PointCloud_Generator.pc_generator3 import PointcloudGenerator

from configparser import SafeConfigParser

################################################################
### file to create pcds for the dataset out of the stl file ####
################################################################

config = SafeConfigParser()
config.read("config.ini")

in_path = os.path.join(config.get("paths", "abc_dataset"), "stl")
out_bin = os.path.join(config.get("paths", "abc_dataset"), "pcd")
out_ascii = os.path.join(config.get("paths", "abc_dataset"), "ascii_pcd")

#log_file = "/home/jonas/MA_Software/ma-jonas/Data/Motors_2/Pointcloud.info"
log_file = os.path.join(config.get("paths", "abc_dataset"), "Pointcloud.info")

files = os.listdir(in_path)
files.sort()

with open(log_file, 'w') as log:
    log.write("filename, points downsampled, points captured\n")

for filename in files:
    print(filename)
    start = time.time()
    mesh = os.path.join(in_path, filename)
    scanner = PointcloudGenerator(mesh_path=mesh,
                                    #resolution=[384, 240],
                                    resolution=[768, 480])
    radius = 30
    points = []
    for i in range(1, 12):
        for j in range(12):
            theta = pi / 6 * i
            phi = pi / 6 * j
            pos = [radius * sin(theta) * cos(phi), radius * sin(theta) * sin(phi),
                    radius * cos(theta)]
            alpha, beta, gamma = get_cam_rotation(pos)
            scan = scanner.single_scan(translation=pos, alpha=alpha, beta=beta, gamma=gamma)['pcd']
            if len(points) == 0:
                points = np.asarray(scan.points)
            else:
                points = np.append(points, np.asarray(scan.points), axis=0)
    full_pc = o3d.geometry.PointCloud()
    full_pc.points = o3d.utility.Vector3dVector(points)
    voxel_size = 0.05
    check = False
    while check == False:
        ds_pc = full_pc.voxel_down_sample(voxel_size)
        if len(ds_pc.points) < 3200:
            voxel_size = voxel_size/2
        elif len(ds_pc.points) >= 3200:
            check = True
    print("Voxel Size", voxel_size)
    delta1 = time.time() - start

    name = filename.replace('.stl', '')

    with open(log_file, 'a') as log:
        log.write(name + ',' + str(len(ds_pc.points)) + ',' + str(len(full_pc.points)) + '\n')
    pcd_name = name + '.pcd'
    print("Start PC writing ascii false")
    o3d.io.write_point_cloud(os.path.join(out_bin, pcd_name), ds_pc, write_ascii=False, print_progress=True)
    delta2 = time.time() - start
    print("Start PC writing ascii true")
    o3d.io.write_point_cloud(os.path.join(out_ascii, pcd_name), ds_pc, write_ascii=True, print_progress=True)
    delta3 = time.time() - start
    print(delta1, delta2, delta3)
    #o3d.visualization.draw_geometries([ds_pc])