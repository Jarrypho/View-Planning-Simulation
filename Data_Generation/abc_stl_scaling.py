import sys
sys.path.append("/home/dominik/Documents/Git/ba_jonas")
import RL_Framework.utils.pointcloud_utils as pc_util
import RL_Framework.utils.env_utils as env_util
from PointCloud_Generator.pc_generator3 import PointcloudGenerator
import random
from math import pi, sin, cos
import open3d as o3d
import numpy as np

"""scales meshes of abc_set up and moves them to new folder"""
"""only use scaled models which have 2 < max_bb_point < 20 and have had least 25.000 triangles to ensure high resolution of mesh"""
models = 0
info = []
used = []
for j in range(0,1000):
    name = str(j).zfill(8)
    motor_path = f"/media/dominik/Backup Plus/Masterarbeit/abc_dataset/stl_unscaled/{name}.stl"

    mesh = o3d.io.read_triangle_mesh(motor_path)
    #o3d.visualization.draw_geometries([mesh],mesh_show_wireframe=True, window_name="mesh")
    bb = mesh.get_oriented_bounding_box()
    bb_points = np.asarray(bb.get_box_points())
    mesh_2 = mesh.scale(scale=10*1/abs(np.amax(bb_points)), center=bb.get_center())
    mesh_2.compute_triangle_normals()
    bb_scaled = mesh_2.get_oriented_bounding_box()
    bb_points_scaled = np.asarray(bb_scaled.get_box_points())
    ids = np.asarray(mesh_2.triangles)
    
    usable = True
    if abs(np.amax(bb_points_scaled)) > 20:
        usable = False
    elif abs(np.amax(bb_points_scaled)) < 2:
        usable = False
    if len(ids) < 25000:
        usable = False
    info.append([j, len(ids), abs(np.amax(bb_points_scaled)), usable])
    if usable == False:
        print("out of range")
        next
    else:
        model_nr = str(models).zfill(8)
        out_path = "/media/dominik/Backup Plus/Masterarbeit/abc_dataset/stl/{}.stl".format(model_nr)
        used.append([model_nr, str(name), abs(np.amax(bb_points_scaled)), len(ids)])
        o3d.io.write_triangle_mesh(out_path, mesh_2)
        models += 1
    print(j)
    print('###################################################################')
    
    visual = False
    if visual:
        vis = []
        cam = o3d.geometry.TriangleMesh.create_sphere(radius=30, resolution=20)
        cam_line = o3d.geometry.LineSet.create_from_triangle_mesh(cam)

        vis.append(mesh_2)
        vis.append(cam_line)
        o3d.visualization.draw_geometries(vis,mesh_show_wireframe=True, window_name="mesh")
    if models > 99:
        break

print(models)
with open("info.txt", "w") as log:
    for row in info:
        log.write(str(row) + '\n')

with open("used.txt", "w") as file:
    for row in used:
        file.write(str(row) + "\n")

        # scanner = PointcloudGenerator(mesh_path=out_path)

        # def log(txt:str):
        #     print(txt)
            
        # cam_radius = 30

        # scans = []
        # for i in range(20):
        #     phi = random.uniform(0, 2*pi) # radians
        #     theta = random.uniform(0, 2*pi) # radians
        #     log("phi: " + str(phi) + "theta:" + str(theta))
        #     pos = [cam_radius * sin(theta) * cos(phi), cam_radius * sin(theta) * sin(phi),
        #         cam_radius * cos(theta)]
        #     alpha, beta, gamma = env_util.get_cam_rotation(pos)
        #     log("Pos [x,y,z] = " + str(pos))
        #     log("Rotation: " + str(alpha) + str(beta) + str(gamma))
        #     # trans_matrix = env_util.get_camera_transform_from_euler(pos, alpha, beta, gamma, degrees=False)

        #     log("    - scan")
        #     scan = scanner.single_scan(translation=pos, alpha=alpha, beta=beta, gamma=gamma)
        #     scans.append(scan['pcd'])

        # pc_util.vis_pcd(scans)








