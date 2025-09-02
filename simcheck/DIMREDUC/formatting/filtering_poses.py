import numpy as np

"""
This script formats the data based on the body parts specified by the user.
It filters the keypoints to keep only the desired elements, so if the user wants to keep only the dominant hand,
it will keep only the keypoints corresponding to the dominant hand.
"""

def keep_desired_bodyparts(elements):

    """
    This function filters the embeddings to keep only the elements indicated by the user.
    """

    if elements == 'all':
        return 0, 49
    if elements == 'hands':
       return 8, 49
    if elements == 'domhand':
        
        return 29, 49

def filter_keypoints(data, elements):

    """
    Formats the keypoints data from JSON into a sequence of poses.
    """

    Llimit, Rlimit = keep_desired_bodyparts(elements)

    pose_sequence = []
    for frame, keypoints in data.items():
        pose = []
        for idx, joint in enumerate(keypoints):
            if idx < Llimit or idx > Rlimit:
                continue
            pose.extend([joint['x'], joint['y']])
        pose_sequence.append(pose)
     
    return np.array(pose_sequence, dtype=np.float32)

