import numpy as np
from typing import List, Tuple
import cv2
import scipy.linalg as scipy_linalg

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners


#--------------------------------------------------------------

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    Rot_1 = np.array([[np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0],[np.sin(np.radians(alpha)),             np.cos(np.radians(alpha)), 0],[0, 0, 1]])
    Rot_2 = np.array([[1, 0, 0],[0, np.cos(np.radians(beta)), -np.sin(np.radians(beta))],[0, np.sin(np.radians(beta)), np.cos(np.radians(beta))]])
    Rot_3 = np.array([[np.cos(np.radians(gamma)), -np.sin(np.radians(gamma)), 0],[np.sin(np.radians(gamma)), np.cos(np.radians(gamma)), 0],[0, 0, 1]])
    
    rot_xyz2XYZ = np.dot(Rot_3, np.dot(Rot_2,Rot_1))

    return rot_xyz2XYZ

#--------------------------------------------------------------#

def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation
    Rot_1 = np.array([[np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0],[np.sin(np.radians(alpha)), np.cos(np.radians(alpha)), 0],[0, 0, 1]])
    Rot_2 = np.array([[1, 0, 0],[0, np.cos(np.radians(beta)), -np.sin(np.radians(beta))],[0, np.sin(np.radians(beta)), np.cos(np.radians(beta))]])
    Rot_3 = np.array([[np.cos(np.radians(gamma)), -np.sin(np.radians(gamma)), 0],[np.sin(np.radians(gamma)), np.cos(np.radians(gamma)), 0],[0, 0, 1]])

    
    rot_xyz2XYZ = np.dot(Rot_3, np.dot(Rot_2,Rot_1))
    rot_XYZ2xyz = rot_xyz2XYZ.T

    
    return rot_XYZ2xyz
#--------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------
