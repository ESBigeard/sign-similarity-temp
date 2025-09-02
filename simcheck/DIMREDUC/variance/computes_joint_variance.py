import numpy as np
from collections import defaultdict

"""
This script computes the joint variance of keypoints in a video dataset.
It calculates the variance for each joint across all frames in a video.
"""

def get_joint_coordinates(data):

    """
    Extracts joint coordinates from the video data.
    """

    dict_for_variances = defaultdict(list)
    for frame in data:
        for joint, coordinates in enumerate(frame):
            dict_for_variances[joint].append(coordinates)
    
    return dict_for_variances

def get_embedding_variances(data):

    """
    Computes variance for the keypoints of the given video data.
    """

    dict_for_variances = get_joint_coordinates(data)
    variance_pose = []
    for joint, coordinates in dict_for_variances.items():
        variance = np.var(coordinates, axis=0)
        variance_pose.append(variance)
    
    return np.array(variance_pose)

