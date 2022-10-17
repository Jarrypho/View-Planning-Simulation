import numpy as np
from math import sin, cos, atan, pi, sqrt
import math, warnings
from scipy.spatial.transform import Rotation as R

def action_mapping(action_type, actions):
    """action is defined from [-1, 1], mapping to real coordinates needed"""
    mapped = []
    #spherical coordinates for translation, euler angles for cam rotation
    if action_type == "3T2R":
        mapped.append((actions[0]+1)*pi)
        mapped.append((actions[1]+1)*pi/2)
        mapped.append((actions[2]+1)*50)
        mapped.append((actions[3]+1)*pi)
        mapped.append((actions[4])*pi/2)
    #spherical coordinates for translation, rotation fixed, focused on object centre
    elif action_type == "3T0R":
        mapped.append((actions[0]+1)*pi)
        mapped.append((actions[1]+1)*pi/2)
        mapped.append((actions[2]+1)*50)
    #cartesian coordinate system for translation, rotation fixed, focused on object centre
    elif action_type == "XYZ_0R":
        mapped.append((actions[0])*50)
        mapped.append((actions[1])*50)
        mapped.append((actions[2])*50)
    #cartesian coordinate system for translation, euler angles for cam rotation
    elif action_type == "XYZ_2R":
        mapped.append((actions[0])*100)
        mapped.append((actions[1])*100)
        mapped.append((actions[2])*100)
        mapped.append((actions[3]+1)*pi)
        mapped.append((actions[4])*pi/2)
    #polar coordinates for translation, rotation fixed, focused on object centre
    elif action_type == "2T0R":
        mapped.append((actions[0]+1)*pi)
        mapped.append((actions[1]+1)*pi/2)
    #polar coordinates for translation, euler angles for cam rotation
    elif action_type == "2T2R":
        mapped.append((actions[0]+1)*pi)
        mapped.append((actions[1]+1)*pi/2)
        mapped.append((actions[2]+1)*pi)
        mapped.append((actions[3])*pi/2)
    else:
        raise Exception(f"Action type {action_type} not implemented")
    return mapped

def get_output_dim(action_type:str):
    """number of variables for each action type"""
    output_dim = {
        "XYZ_0R": 3,
        "XYZ_2R": 5,
        "2T0R": 2,
        "2T2R": 4,
        "3T0R": 3,
        "3T2R": 5
    }
    if action_type not in output_dim:
        raise ValueError(f"action type {action_type} not implemented")
    
    return output_dim[action_type]

def get_cam_rotation(pos=[0.0, 0.0, 0.0]):
    """
    :param pos: Camera Position as [x,y,z]
    :return: Camera Rotation towards [0,0,0]
    """
    x, y, z = pos
    alpha, beta, gamma = 0.0, 0.0, 0.0
    # Alpha: Cam rotation around x-Achsis so that it's focussed on a point on [a,0,0]
    if y == 0:
        if z >= 0:
            alpha = 0.0
        else:
            alpha = pi  # 180°  = 2(pi/2)
    elif y >= 0 and z >= 0:  # Quadrant 2
        alpha = pi / 2 - atan(z / y)
    elif y >= 0 and z < 0:  # Quadrant 3
        alpha = pi / 2 - atan(z / y)
    elif y < 0 and z < 0:  # Quadrant 4
        alpha = 3 * pi / 2 - atan(z / y)
    elif y < 0 and z >= 0:  # Quadrant 1
        alpha = 3 * pi / 2 - atan(z / y)

    # Beta: Cam rotation around y'-achsis (Cam coordinat system after the first rotation around x), so that the cam is focused on [0,0,0]
    d = sqrt(y * y + z * z)  # Distance from cam to x-Achsis
    if d != 0:
        beta = atan(x / d)

    return -alpha, beta, gamma

def get_cam_pose_and_rotation(action_type, actions, radius=50.0, initial=False, random = False):
    """
    :param initial: initial scan of the episode?
    :param radius: if 2T action, radius is needed
    :param action_type: String containing action-type, e.g. 2TOR, 3TOR, 2T2R, 3T2R, XYZ_0R, XYZ_2R
    :param actions: action from the agent
    :return:
    """
    # If initial scan perform 2TOR bc of fixed inital scan coming as 2TOR
    if initial:
        phi = actions[0]  # radians
        theta = actions[1]  # radians
        pos = [radius * sin(theta) * cos(phi), radius * sin(theta) * sin(phi),
               radius * cos(theta)]
        alpha, beta, gamma = get_cam_rotation(pos)
        return pos, alpha, beta, gamma

    elif not initial:
        if action_type == '2T0R':
            phi = actions[0]  # radians
            theta = actions[1]  # radians
            pos = [radius * sin(theta) * cos(phi), radius * sin(theta) * sin(phi),
                   radius * cos(theta)]
            alpha, beta, gamma = get_cam_rotation(pos)
            return pos, alpha, beta, gamma

        elif action_type == '3T0R' or action_type == '3T0R*':
            phi = actions[0]  # radians
            theta = actions[1]  # radians
            radius = actions[2]  # radius from actions input, non-radians
            pos = [radius * sin(theta) * cos(phi), radius * sin(theta) * sin(phi),
                   radius * cos(theta)]
            alpha, beta, gamma = get_cam_rotation(pos)
            return pos, alpha, beta, gamma

        elif action_type == '2T2R':
            phi = actions[0]  # radians
            theta = actions[1]  # radians
            pos = [radius * sin(theta) * cos(phi), radius * sin(theta) * sin(phi),
                   radius * cos(theta)]
            alpha = actions[2]
            beta = actions[3]
            gamma = 0.0
            return pos, alpha, beta, gamma

        elif action_type == '3T2R':
            phi = actions[0]  # radians
            theta = actions[1]  # radians
            radius = actions[2]  # radius from actions input, non-radians
            pos = [radius * sin(theta) * cos(phi), radius * sin(theta) * sin(phi),
                   radius * cos(theta)]
            alpha = actions[3]
            beta = actions[4]
            gamma = 0.0
            return pos, alpha, beta, gamma

        elif action_type == 'XYZ_0R':
            pos = [actions[0], actions[1], actions[2]]
            alpha, beta, gamma = get_cam_rotation(pos)
            return pos, alpha, beta, gamma

        elif action_type == 'XYZ_2R':
            pos = [actions[0], actions[1], actions[2]]
            alpha = actions[3]
            beta = actions[4]
            gamma = 0.0
            return pos, alpha, beta, gamma

        else:  
            raise ValueError(f"action type {action_type} not implemented")

def get_camera_transform_from_euler(translation=[0, 0, 0], alpha=0, beta=0, gamma=0, degrees=True):
    """calculates transformation matrix from euler angles"""
    translation = [e * 1.0 for e in translation]  # convert int to float
    rotation = R.from_euler('XYZ', [alpha, beta, gamma], degrees=degrees)
    rot_matrix = rotation.as_matrix()
    transform = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 1]
                 ]
    for i in range(3):
        for j in range(4):
            if j == 3:
                transform[i][j] = translation[i]
            else:
                transform[i][j] = rot_matrix[i][j]
    return np.array(transform)

def distance_to_origin(pose):
    """calculates distance from origin to cam pose"""
    p2 = np.array([pose[0], pose[1], pose[2]])
    squared_dist = np.sum(p2 ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist


def orthodrome_distance(point_a, point_b, radius):
    """
    :param point_a: First point on sphere surface with r = radius
    :param point_b: Second point on sphere surface with r = radius
    :return: distance, see: https://math.stackexchange.com/questions/1304169/distance-between-two-points-on-a-sphere#:~:text=If%20a%3D(a1%2C,3b3r2)
    """
    p1 = np.array(point_a)
    p2 = np.array(point_b)
    p1p2 = np.sum(p1*p2) / (radius**2)
    if p1p2 > 1:
        p1p2 = 1
        warnings.warn("p1p2 > 1")
    elif p1p2 < -1:
        p1p2 = -1
        warnings.warn("p1p2 < -1 ")
    distance = radius * math.acos(p1p2)
    return distance

def get_travel_cost(point_a, point_b, action_type, radius):
    """
    :param point_a: Point A :param point_b: Point B :param action_type: Action Type, for example 2T0R :param radius:
    :return: (Approx.) travel cost. If both points on same sphere use orthodrome distance. If radius is different we
    use an approximation.
    # orthodrome distance: 2T0R, 2T2R
    # special distance: XYZ_0R, XYZ_2R, 3T0R, 3T2R
    """
    if point_a == point_b:
        return 0.0
    else:
        if action_type == '2T0R' or action_type == '2T2R':
            # both points on same sphere. Use radius.
            return orthodrome_distance(point_a=point_a, point_b=point_b, radius=radius)
        elif action_type == 'XYZ_0R' or action_type == 'XYZ_2R' or action_type == '3T2R' or action_type == '3T0R':
            # points probably on different spheres.
            d1 = distance_to_origin(point_a)
            #print(d1)
            d2 = distance_to_origin(point_b)
            if d1 == d2:
                return orthodrome_distance(point_a=point_a, point_b=point_b, radius=d1)
            if d1 < d2:
                b_norm = point_b / d2
                b_on_sphere = b_norm * d1
                sphere_distance = orthodrome_distance(point_a=point_a, point_b=b_on_sphere, radius=d1)
                b_on_sphere_to_point_b = np.sqrt(np.sum(np.square(b_on_sphere - point_b)))
                distance = sphere_distance + b_on_sphere_to_point_b
                return distance
            if d1 > d2:
                a_norm = point_a / d1
                a_on_sphere = a_norm * d2
                sphere_distance = orthodrome_distance(point_a=a_on_sphere, point_b=point_b, radius=d2)
                a_on_sphere_to_point_a = np.sqrt(np.sum(np.square(a_on_sphere - point_a)))
                distance = sphere_distance + a_on_sphere_to_point_a
                return distance