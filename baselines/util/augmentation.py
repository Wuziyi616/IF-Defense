"""Functions for point cloud data augmentation"""
import numpy as np


###########################################
# numpy based functions
###########################################

def rotate_point_cloud(pc):
    """
    Rotate the point cloud along up direction with certain angle.
    Input:
        pc: Nx3 array of original point clouds
    Return:
        rotated_pc: Nx3 array of point clouds after rotation
    """
    angle = np.random.uniform(0, np.pi * 2)
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_pc = np.dot(pc, rotation_matrix)

    return rotated_pc


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter point cloud per point.
    Input:
        pc: Nx3 array of original point clouds
    Return:
        jittered_pc: Nx3 array of point clouds after jitter
    """
    N, C = pc.shape
    assert clip > 0
    jittered_pc = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_pc += pc

    return jittered_pc


def translate_point_cloud(pc):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pc = np.add(np.multiply(pc, xyz1), xyz2).astype('float32')
    return translated_pc
