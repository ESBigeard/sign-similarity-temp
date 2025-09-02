import numpy as np
from keyframe_extraction.utils import PoseProcessor

"""
Retrieve Mediapipe landmarks from a video and flatten each frame's coordinates.
These coordinates are concatenated with the ViT embeddings in the concatenation script. 
These concatenated embeddings are used for clustering and key frame extraction.
"""

def retrieve_poses(video_path, mp):

    """
    Retrieve the Mediapipe poses from a video using the PoseProcessor class.
    """

    pp = PoseProcessor(video_path, mp)
    pose_file = pp.pose_file
    poses = pp.load_poses_from_json(pose_file)

    return poses

def single_frame_pose(frame_coordinates):

    """
    Get the coordinates of a single frame and return them as a flattened array.
    """

    frame_array = []
    for couple in frame_coordinates:
        coordinates_array = np.array(list(couple.values()))
        frame_array.append(coordinates_array)
    
    frame_array = np.array(frame_array).flatten()
    frame_array = frame_array[29:]  # Keep dominant hand coordinates
    return frame_array

def all_frames_poses(video_path, mp):

    """
    Get the coordinates of all frames and return them as a dictionary.
    The keys are the frame numbers and the values are the flattened arrays of coordinates.
    """

    poses = retrieve_poses(video_path, mp)
    all_frames_poses = {}
    for frame_num, frame_coordinates in poses.items():
        frame_array = single_frame_pose(frame_coordinates)
        all_frames_poses[frame_num] = frame_array.tolist()

    return all_frames_poses