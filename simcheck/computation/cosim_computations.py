"""
This module handles COSIM computation.
These computations are used in SIMCHECK subpackages to compare different features.
It expects a list of arrays.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def cosim_gpu(matrix):

    """
    Computes a similarity matrix using cosine similarity on GPU.
    """
    
    data_tensor = torch.tensor(matrix, device='cuda', dtype=torch.float32)
    normalized_tensor = torch.nn.functional.normalize(data_tensor, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.T)
    
    return similarity_matrix.cpu().numpy()

def cosim(matrix):

    """
    Computes a similarity matrix using cosine similarity.
    """

    return cosine_similarity(matrix)

def COSIM_formatting_high_dim(poses):

    """
    Flattens and applies padding to the high_dimensional pose data
    embeddings for COSIM computation.
    """

    max_length = max(len(pose[1].flatten()) for pose in poses)
    padded_poses = []

    print("Padding poses to the maximum length for COSIM computation...")
    for pose in tqdm(poses):
        keypoints = pose[1].flatten()
        if len(keypoints) < max_length:
            padding = np.zeros(max_length - len(keypoints))
            keypoints = np.concatenate((keypoints, padding))

        padded_poses.append(keypoints)

    return padded_poses

def COSIM_computation(arrays, use_gpu=False):

    """
    Computes the cosine similarity between all pairs of videos in the provided list of arrays.
    If use_gpu is True, it uses GPU for comput  ation.
    """

    matrix = np.stack(arrays)

    if use_gpu:
        print("GPU requested for COSIM computation.")
        print("Checking if GPU is available...")
        if not torch.cuda.is_available():
            print("GPU not available, falling back to CPU.")
            use_gpu = False

    if use_gpu:
        print("GPU available, using it for COSIM computation.")
        return cosim_gpu(matrix)
    else:
        print("Using CPU for COSIM computation.")
        return cosim(matrix)