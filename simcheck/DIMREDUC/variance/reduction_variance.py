import umap
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
This script applies the UMAP dimensionality reduction technique to the given matrix of data.
It uses the variance as input, each row of the matrix represents a video, and each column represents a joint's variance.
All rows have the same length.
"""

def get_results_variance(matrix):

    """
    This function computes the UMAP results for the given matrix.
    """

    matrix = np.array(matrix, dtype=np.float32)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_result = umap_model.fit_transform(matrix)

    return umap_result

