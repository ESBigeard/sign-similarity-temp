import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def apply_kmeans_clustering(features, clusters):

    """
    Apply KMeans clustering on a set of embeddings to find keyframes.
    """

    optimal_k = clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(features)
    clustered_frames = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return clustered_frames, centroids

# Not used for now
# def find_optimal_clusters_number(features_list):

#     """
#     This function finds the optimal number of clusters for k-means clustering.
#     It uses the elbow method. We test the k-means algorithm with multiple values
#     of k and get the intertia. The inertia is the sum of squared distances of
#     each data point to its closest cluster center. The lower it is the better
#     the clustering is.
#     """

#     inertias = []
#     for k in range(2, 5):
#         print(f"Trying k = {k}")
#         kmeans = KMeans(n_clusters=k, random_state=0)
#         kmeans = kmeans.fit(features_list)
#         inertias.append(kmeans.inertia_)

#     kl = KneeLocator(range(2, 5), inertias, curve="convex", direction="decreasing")
#     optimal_k = kl.elbow

#     return optimal_k

def choose_keyframes(labels, centroids, features):
    
    """
    This function finds each point (frame) that is the closest to the centroid of its cluster.
    It returns a dictionary with the cluster id as the key and the frame id as the value.
    """

    # Retrieve the number of clusters
    n_clusters = len(np.unique(labels))
    # we then get the label with the most data points
    # Retrieve the distances between each point and all clusters
    # If we have 5 clusters, we'll get a list of arrays with 5 values.
    # There will be as many arrays as points. Each value is the distance
    # between the point and a specific cluster.
    # distances[0][0] => distance between the first frame point and the first cluster centroid
    # distances[0][1] => distance between the first frame point and the second cluster centroid
    distances = pairwise_distances(features, centroids)

    # We store key_frames in a dictionnary with the key being the cluster id
    # and the value being the key frame id
    key_frames = {}

    # For each cluster
    for cluster_number in range(n_clusters):
        # print(f"Pour le cluster numéro {cluster_number} :")

        # We find the index of the frames belonging to the cluster
        cluster_indices = np.where(labels == cluster_number)[0]
        # We convert the distances array to a list
        # Since this array is a list of list (with 5 distances each)
        # We only keep the distance that concern the given cluster in the loop
        distances_list = distances[cluster_indices].tolist()
        distances_list = [distance[cluster_number] for distance in distances_list]
        # We find the smallest distance
        min_distance = np.min(np.array(distances_list))
        # We retrieve the index of the frame with the smallest distance
        min_distance_index = distances_list.index(min_distance)
        # Based on this point, we retrieve the index of the frame and its name
        key_frame = cluster_indices[min_distance_index]
        # We store the key frame in the dictionnary
        key_frames[f"cluster_{cluster_number}"] = int(key_frame)
       
    return key_frames