import umap
import numpy as np
from sklearn.preprocessing import StandardScaler


"""
This script applies the UMAP dimensionality reduction technique to the given matrix of data.
It uses Manolis Fragkiadakis' method, which flattens the data into a 1D array of keypoints.
Since videos can have different lengths, it pads the data to ensure all videos have the same number of frames.
"""

def length_to_86_frames(pose_embedding):

    print(f"Length of pose_embedding: {len(pose_embedding)}")
    if len(pose_embedding) < 86:
        padding = np.zeros((86 - len(pose_embedding), 100))
        pose_embedding = np.concatenate((pose_embedding, padding), axis=0)
    elif len(pose_embedding) > 86:
        
        start = (len(pose_embedding) - 86) // 2
        pose_embedding = pose_embedding[start:start + 86, :]
    
    print(f"New length of pose_embedding: {len(pose_embedding)}")
    return pose_embedding
    
# For one video
def get_flattened_embeddings(data):

    """
    Flattens the multidimensional data into a 1D array.
    Each video is represented as a 1D array of keypoints.
    """

    data = length_to_86_frames(data)
    return data.flatten()


# Dimensionality reduction !
def padding(matrix):

    """
    Pads each video in the matrix to have the same number of frames (max_frames).
    Ensures the output is a 2D array where rows are videos and columns are features.
    """

    max_frames = max([len(data) for data in matrix])
    padded_data = []
    for video in matrix:
        num_frames = len(video)
        if num_frames < max_frames:
            padding = np.zeros((max_frames - num_frames))
            video = np.concatenate((video, padding), axis=0)
        padded_data.append(video)
    
    return np.array(padded_data, dtype=np.float32)

def truncate_embeddings(matrix):

    """
    Truncates each video in the matrix to have the same number of frames (max_frames).
    Ensures the output is a 2D array where rows are videos and columns are features.
    """

    max_frames = min([len(data) for data in matrix])
    truncated_data = []
    for video in matrix:
        num_frames = len(video)
        if num_frames > max_frames:
            video = video[:max_frames]
        truncated_data.append(video)
    
    return np.array(truncated_data, dtype=np.float32)

def get_results_fragkiadakis(matrix):

    """
    This function computes the UMAP results for the given matrix.
    """
    
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_result = umap_model.fit_transform(matrix)

    return umap_result