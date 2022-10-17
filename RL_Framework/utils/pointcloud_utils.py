import numpy as np
import open3d as o3d
import matplotlib.colors as colors
import sys
from PointCloud_Generator.utils import vox_downsample_numpoint, subtract
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

colorlist = [colors.to_rgb(c) for c in
             ['red', 'blue', 'green', 'orange', 'yellow', 'black', 'brown', 'cyan', 'magenta', 'gold', 'olive', 'khaki',
              'lime', 'grey', 'sienna']]
black = colors.to_rgb('black')

def vis_mesh_and_cam_pos(mesh_path, pos):
    """visualizes mesh and cam pose for ground truth and predicted value"""
    vis = []
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vis.append(mesh)
    i = 0
    j = 0
    while i < len(pos):
        cam_pos = o3d.geometry.TriangleMesh.create_arrow()
        cam_pos = cam_pos.rotate([[1,0,0], [0,-1,0],[0,0,-1]])
        cam_sphere = cam_pos.transform(pos[i])
        cam_sphere = cam_sphere.paint_uniform_color(colorlist[j])
        vis.append(cam_sphere)
        cam_pos = o3d.geometry.TriangleMesh.create_arrow()
        cam_pos = cam_pos.rotate([[1,0,0], [0,-1,0],[0,0,-1]])
        cam_sphere = cam_pos.transform(pos[i+1])
        cam_sphere = cam_sphere.paint_uniform_color(colorlist[j])
        vis.append(cam_sphere)
        i += 2
        j += 1

    o3d.visualization.draw_geometries(vis,mesh_show_wireframe=True, window_name="mesh")


def encode_voxel(points, voxel_size=0.05):
    """
    Simple Voxel Encoding
    
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()
    np_voxel = np.zeros(shape=(50, 50, 50))
    for voxel in voxels:
        x, y, z = voxel.grid_index
        np_voxel[x, y, z] = 1

    return np.reshape(np_voxel, 50 * 50 * 50)

def plot_spherical(data: np.array):
    "https://stackoverflow.com/questions/25286811/how-to-plot-a-3d-density-map-in-python-with-matplotlib"
    x = data[1:, 0]
    y = data[1:, 1]
    z = data[1:, 2]
    xyz = np.vstack([x, y, z])
    density = stats. gaussian_kde(xyz)(xyz)
    idx = density.argsort()
    x, y, z, density = x[idx], y[idx], z[idx], density[idx]
    fig = plt.figure(figsize=[18,5], constrained_layout=True)

    for j in range(1,7):
        position = 160 + j
        ax = fig.add_subplot(position, projection='3d')
        ax.scatter(x, y, z, c=density)
        ax.scatter(data[0, 0], data[0, 1], data[0, 2], c="red")
        ax.scatter(0, 0, 0, c="lime")
        ax.view_init(30, 60*j)
        ax.set_xlim(-100.0, 100.0)
        ax.set_ylim(-100.0, 100.0)
        ax.set_zlim(-100.0, 100.0)   
    return fig

def scatter_interactive_nbv(data: np.array):
    """
    next-best-view plot
    interactive
    one point for object centre, one for initial scan and one for nbv scan
    """
    all = np.vstack([
        np.hstack([data[0, :3], 1]), 
        np.hstack([0,0,0,0]), 
        np.hstack([data[1, :3], 2])
        ])
    df = pd.DataFrame(data=all, columns=["x", "y", "z", "encoding"])
    fig = go.Figure(data=go.Scatter3d(
        x=df["x"], 
        y=df["y"], 
        z=df["z"], 
        mode="markers", 
        marker=dict(
            size=2,
            color=df["encoding"],
            colorscale=["lime", "red", "cyan"],
            opacity=1
        )))
    fig.update_layout(
        dict(
            scene = dict(
                xaxis = dict(nticks=5, range=[-100, 100]),
                yaxis = dict(nticks=5, range=[-100, 100]),
                zaxis = dict(nticks=5, range=[-100, 100])
            ),
            scene_aspectmode="cube"
        )
    )
    return fig


def scatter_pcd_interactive(data: np.array, inspection: bool=False):
    """
    visualizes pointcloud based on information in encoding matrix
    plot is interactive
    """
    if inspection:
        df = pd.DataFrame(data=data, columns=["x", "y", "z", "encoding"])
        df.loc[df.encoding == 1.0, "encoding"] = "lime"
        df.loc[df.encoding == -1.0, "encoding"] = "red"
        df.loc[df.encoding == 0.0, "encoding"] = "grey"
    else:
        df = pd.DataFrame(data=data, columns=["x", "y", "z", "encoding"])
        df.loc[df.encoding == 1.0, "encoding"] = "lime"
        df.loc[df.encoding == 0.0, "encoding"] = "grey"

    fig = go.Figure(data=go.Scatter3d(
        x=df["x"], 
        y=df["y"], 
        z=df["z"], 
        mode="markers", 
        marker=dict(
            size=2,
            color=df["encoding"],
            opacity=1
        )))
    return fig
    


    
    