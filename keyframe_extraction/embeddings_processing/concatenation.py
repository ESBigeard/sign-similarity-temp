import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keyframe_extraction.utils import ImageFeaturesProcessor
from keyframe_extraction.embeddings_processing.pose_embeddings import all_frames_poses

def retrieve_features(video_path, vit):

    """
    Retrieve the ViT features from a video using the ImageFeaturesProcessor class.
    """

    print("Retrieving ViT features...")
    print("Video path:", video_path)
    print("ViT model:", vit)
    ifp = ImageFeaturesProcessor(video_path, vit)
    features_file = ifp.features_file
    features = ifp.load_features_from_json(features_file)

    return features

def min_max_scaling(pose_coordinates):

    """
    Apply min-max scaling to the pose coordinates.
    Min-max scaling is a normalization technique that scales the data between -1 and 1.
    This scale is the one used in the ViT model.
    """
    
    scaler = MinMaxScaler()
    pose_coordinates = pose_coordinates.astype(float)
    # Mask the 0 values
    mask = pose_coordinates == 0.0
    pose_coordinates[mask] = np.nan

    if np.all(np.isnan(pose_coordinates)):
        pose_coordinates[mask] = 0.0
        return pose_coordinates
    
    # Apply scaling to non-zero values
    pose_coordinates = scaler.fit_transform(pose_coordinates)
    # Restore the 0 values
    pose_coordinates[mask] = 0.0

    return pose_coordinates

def concatenate_mediapipe_vit(video_path, concat, mp, vit):

    """
    Reshape the ViT features and the Mediapipe landmarks to be 1D arrays.
    Rescale the pose coordinates using min-max scaling.
    Concatenate the ViT features and the pose coordinates if specified.
    """

    concatenated_embeddings = {}
    id_counter = 0

    vit_features = retrieve_features(video_path, vit)

    if concat:
        # If concatenating, retrieve the poses and concatenate them with the ViT features
        poses = all_frames_poses(video_path, mp)
        for pose_frame, vit_frame in zip(poses.values(), vit_features.values()):
            vit_frame = np.array(vit_frame).reshape(-1, 1)
            pose_frame = np.array(pose_frame).reshape(-1, 1)
            # Rescale the pose frame
            pose_frame = min_max_scaling(pose_frame)
            # Concatenate the pose frame and the ViT frame if specified
            concatenated_frame = np.concatenate((pose_frame, vit_frame), axis=0)
            concatenated_embeddings[id_counter] = concatenated_frame.flatten().tolist()
            id_counter += 1
    else:
        # If not concatenating, just flatten the ViT features
        for vit_frame in vit_features.values():
            vit_frame = np.array(vit_frame).reshape(-1, 1)
            concatenated_embeddings[id_counter] = vit_frame.flatten().tolist()
            id_counter += 1

    return concatenated_embeddings