from numba import njit
import numpy as np
import open3d as o3d
import matplotlib.colors as colors
import random
import trimesh


@njit
def scanner_laser_overlapp(index_tri_scanner, index_tri_laser):
    """calculates the indices of the points which are visible both by camera and projector
    further information in thesis Koch          

    Args:
        index_tri_scanner (numpy array): indices of triangles hit by scanner rays
        index_tri_laser (numpy array): indices of triangles hit by projector rays

    Returns:
        numpy array: indices of visible points returned as 1, nonvisible returned as 0
    """
    index_tri_laser_reduced = set(np.unique(index_tri_laser))

    ind_del_np = [0 if val in index_tri_laser_reduced else 1 for val in index_tri_scanner]

    return ind_del_np

def calc_dist(points, origins, cam_workspace=[30.0,50.0]):
    """deletes all points which are not in the defined distance of the cameras viewfield

    Args:
        points (numpy array): point coordinates visible by camera
        origins (numpy array): origin coordinates of camera

    Returns:
        numpy array: coordinates of visible points
    """
    if len(points) > 0:
        temp = np.abs(points - origins[:len(points)])
        dist = np.sqrt(np.einsum('ij,ij->i', temp, temp))
        points = np.delete(
            points, 
            np.where(
                (dist < cam_workspace[0]) | (dist > cam_workspace[1]))[0], 
                axis=0)
    return points

def vis_visible_and_nonvisible_pcd(pcd, pcd_inv, mesh):
    """visualization of points visible (and nonvisible)

    Args:
        pcd (o3d pointcloud): visible points (care: some points also get deleted because of distance, can be turned of in config)
        pcd_inv (o3d pointcloud): nonvisible points, i.e. only visible by cam, not projector
        mesh (trimesh mesh): mesh
    """
    # o3d.visualization.draw_geometries([pcd_inv], window_name='inverse')
    # o3d.visualization.draw_geometries([pcd], window_name='full')
    pcd.paint_uniform_color([0.0,0.8,0.0])
    pcd_inv.paint_uniform_color([0.2,0.2,0.8])
    o3d.visualization.draw_geometries([pcd,pcd_inv], window_name='pcd (green) & pcd_inverted (blue)')
    mesh.show()

def get_nearest_points(pcd: o3d.geometry.PointCloud, n_points: int = 100, n_areas: int = 1):
    """takes n_areas random points and calculates for each the n_points nearest neighbours

    Args:
        pcd (o3d.geometry.PointCloud): PointCloud of the object
        n_points (int): number of points for each area
        n_areas (int): number of areas (region of interests)

    Returns:
        points: returns points of a pointcloud which belong to ROI
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = []
    target = n_areas * n_points
    while len(points) < target:
        rand = random.randint(0,len(pcd.points)-1)
        if rand in points:
            continue
        else:
            dif = target - len(points)
            if dif >= n_points:
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[rand], n_points)
                points.extend(idx)
            else:
                if dif > 20:
                    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[rand], dif)
                    points.extend(idx)
                else:
                    break
            points = list(set(points))
    return points

@njit
def get_idx(array, threshold, operator="bigger"):
    if operator == "bigger":
        keep_idx = [idx for idx, e in enumerate(array) if e > threshold]
    elif operator == "smaller":
        keep_idx = [idx for idx, e in enumerate(array) if e < threshold]
    return keep_idx

def get_seen_indices(pcd_2048x3: o3d.geometry.PointCloud, current_scan:o3d.geometry.PointCloud,thresh_adjust=1.0):
    '''
    Methode, die indizes (des Array) von self.pcd_full_ds_2048 ausgibt, die in self.current_pcd drin sind
    danach in self.array im 4. Eintrag bei jedem der erhaltenen Indizes eine 1 reinschreiben
    wir arbeiten mit extra variablen um Trennung zu haben, dann kann reward gleich bleiben. State Ausgabe muss angepast werden
    :param pcd_2048x3: Full, downsampled pointcloud of the current motor
    :param current_scan: scan of current iteration
    :return: Indices of the points contained in full pcd
    '''
    current_scan = current_scan.voxel_down_sample(0.2)
    PARTIAL_dist_nn = current_scan.compute_nearest_neighbor_distance()
    dist_COMPLETE_PARTIAL = pcd_2048x3.compute_point_cloud_distance(current_scan)
    threshold = np.mean(PARTIAL_dist_nn) * thresh_adjust
    seen_idx = get_idx(np.array(dist_COMPLETE_PARTIAL), threshold=threshold, operator="smaller")
    return seen_idx

def subtract(complete: o3d.geometry.PointCloud, partial:o3d.geometry.PointCloud,thresh_adjust=1.0):
    """
    returns all points in FULL, which are not covered in 'PARTIAL'
    :param complete: Full Pointcloud
    :param partial:  partial Pointcloud
    :return: FULL - PARTIAL
    """
    if len(partial.points) == 0:
        return complete
    else:
        PARTIAL_dist_nn = partial.compute_nearest_neighbor_distance()
        dist_COMPLETE_PARTIAL = complete.compute_point_cloud_distance(partial)
        threshold = np.mean(PARTIAL_dist_nn)*thresh_adjust
        keep_idx = get_idx(np.array(dist_COMPLETE_PARTIAL), threshold, operator="bigger")
        result = complete.select_by_index(keep_idx)
    return result

def combine_scans(a: o3d.geometry.PointCloud, b: o3d.geometry.PointCloud, vis=False, threshadjust=1.1):
    """
    combines two pointclouds a,b after removing dupicates / overlapping Points in secondary(b)
    :param a: o3d.geometry.Pointcloud of original Pointcloud (remain unchanged)
    :param b: o3d.geometry.Pointcloud of new Pointcloud (cut overlap)
    :return: o3d.geometry.Pointcloud combined scan without overlap
    """
    if len(a.points) == 0:
        return b
    if len(b.points) == 0:
        return a

    # vector with len=NumPoints_primary with distance to next neighbour in Pointcloud
    dist_nn = a.compute_nearest_neighbor_distance()
    # vector with len=NumPoints_secondary with distance to next neighbour in primary(a) Pointcloud
    dist_b_a = b.compute_point_cloud_distance(a)

    # threshold for keep / remove a point in Pointcloud b
    threshold = np.mean(dist_nn) / 2 * threshadjust

    keep_idx = get_idx(np.array(dist_b_a), threshold, operator="bigger")
    b_cropped = b.select_by_index(keep_idx)

    # combine origninal a and cropped b pointcloud
    combined = o3d.geometry.PointCloud()
    points = np.concatenate((a.points, b_cropped.points), axis=0)
    combined.points = o3d.utility.Vector3dVector(points)
    if vis:
        vis_pcd([a, b])
        vis_pcd([a, b_cropped])
    return combined

def vox_downsample_numpoint(pcd: o3d.geometry.PointCloud, n_target):
    """
    Downsampling einer Punktwolke zu einer gewünschten Punktzahl per Voxeldownsampling

    Zunächst eine zu große und zu kleine Voxel-Größe Wählen, dann Downsampling mit Mittelwert testen.
    Iterativ wird der Abstand weiter Halbiert bis die gewünschte Punktzahl erreicht ist. 
    Falls dies nach 20 Iterationen nicht gelingt, werden überschüssige Punkte zufällig gelöscht

    """
    voxel_size_lower = 0.01
    voxel_size_upper = 1.0
    for i in range(20):
        voxel_size = (voxel_size_lower + voxel_size_upper) / 2
        ds = pcd.voxel_down_sample(voxel_size)
        if len(ds.points) == n_target:
            return ds
        elif len(ds.points) > n_target:
            voxel_size_lower = voxel_size
        elif len(ds.points) < n_target:
            voxel_size_upper = voxel_size
    # No voxel size for n_target found, random downsample
    ds = pcd.voxel_down_sample(voxel_size_lower)
    n_points = len(ds.points)
    indices = random.sample(range(n_points), k=n_points - n_target)
    ds = ds.select_by_index(indices, invert=True)
    return ds

colorlist = [colors.to_rgb(c) for c in
             ['red', 'blue', 'green', 'orange', 'yellow', 'black', 'brown', 'cyan', 'magenta', 'gold', 'olive', 'khaki',
              'lime', 'grey', 'sienna']]
black = colors.to_rgb('black')

def vis_pcd(pointclouds=[], colors=False):

    """
    :param pointclouds: list of o3d.geometry.Pointcloud
    :return: -
    """

    if colors:
        colors = iter(colorlist)
        for pcd in pointclouds:
            pcd.paint_uniform_color(next(colors))

    zero = o3d.geometry.PointCloud()
    zero.points = o3d.utility.Vector3dVector([[0, 0, 0]])
    zero.paint_uniform_color(black)
    pointclouds.append(zero)

    o3d.visualization.draw_geometries(pointclouds)

def as_mesh(scene_or_mesh):

    """
    Convert a possible scene to a mesh.
    If conversion occurs, the returned mesh has only vertex and face data.
    contributuions: https://github.com/mikedh/trimesh/issues/507 - jackd
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh