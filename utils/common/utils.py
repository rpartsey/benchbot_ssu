import numpy as np
from scipy.spatial.transform import Rotation as Rot


def compute_fov(res, f):
    return 2 * np.arctan2(res, 2 * f)


def fov_bounds(fov, d=3):
    fov_bounds_polar = np.array([
        [d, -fov / 2],
        [d,  fov / 2]
    ])
    fov_bounds_cart = list(zip(
        [x[0] * np.cos(x[1]) for x in fov_bounds_polar],
        [x[0] * np.sin(x[1]) for x in fov_bounds_polar], 
        [0] * len(fov_bounds_polar)
    ))
    
    return fov_bounds_cart

    
def polar_to_cart(polar_coords):
    return np.concatenate(
        (
            polar_coords[:, :1] * np.cos(polar_coords[:, 1:]),
            polar_coords[:, :1] * np.sin(polar_coords[:, 1:]),
        ),
        axis=1
    )


def compute_normals(scan_points, scale_by_x=False):
    scan_points_1 = scan_points
    
    scan_points_2 = np.copy(scan_points_1)
    scan_points_2[:,2] += 1
    
    x = np.array([*scan_points_1[1:], scan_points_1[0]]) - scan_points_1
    y = scan_points_2 - scan_points_1
    z = np.cross(x, y)

    z_norm = z / np.linalg.norm(z, axis=1).reshape((z.shape[0], 1))
    
    if scale_by_x:
        z_norm = z_norm * np.linalg.norm(x, axis=1).reshape((x.shape[0], 1))
    
    return z_norm


def compute_intersection(p1, n1, p2, n2):
    x = (n1[0] * n2[0] * (p2[1] - p1[1]) + p1[0] * n1[1] * n2[0] - p2[0] * n2[1] * n1[0]) / (n1[1] * n2[0] - n2[1] * n1[0]) 
    y = (x - p1[0]) * n1[1] / n1[0] + p1[1]
    
    return (x, y)


def local_to_global(translation, rotation, points):
    rotation_obj = Rot.from_euler('xyz', rotation)
    
    points_rot = np.array([
        rotation_obj.apply(p) 
        for p in points
    ])
    
    points_rot_trans = np.array([
        (np.array(p) + translation).tolist() 
        for p in points_rot
    ])
    
    return points_rot_trans


def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]
    ])


def compute_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(np.clip(dot_product, -1, 1))

    return angle
