from simcheck.DIMREDUC.variance.computes_joint_variance import get_embedding_variances
from simcheck.DIMREDUC.fragkiadakis.reduction_fragkiadakis import get_flattened_embeddings

""" 
This scripts formats the data based on the method provided as an argument by the user.
"""

def format_fragkiadakis(data):

    """
    If Fragkiadakis' method is used, the data is flattened.
    Each video is represented as a 1D array of keypoints of size (num_frames * num_keypoints * 2).
    """
  
    flattened_arrays = get_flattened_embeddings(data)
    return flattened_arrays


def format_variances(data):

    """
    If variance method is used, the data is formatted to compute the variances of the keypoints.
    Each video is therefore represented as a 1D array 100 values, representing the variances of the keypoints
    throughout the video.
    """

    variances = get_embedding_variances(data)
    return variances

def format_data(data, method):

    """
    Formats the data based on the specified method.
    """

    if method == 'fragkiadakis':
        return format_fragkiadakis(data)
    elif method == 'variance':
        return format_variances(data)
    else:
        raise ValueError(f"Unknown method: {method}")