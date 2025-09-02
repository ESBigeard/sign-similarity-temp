"""
This module handles DTW parallel computations.
These computations are used in SIMCHECK subpackages to compare different features.
It expects a list of tuples, each containing a video ID and its corresponding feature array.
"""

import os
from tqdm import tqdm
from dtaidistance import dtw_ndim
from multiprocessing import Pool, cpu_count


def DTW_computation(items):

    """
    Computes the DTW distance between all pairs of videos in the provided list of tuples.
    """


    tasks = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            video1, pose1 = items[i]
            video2, pose2 = items[j]
            tasks.append((video1, pose1, video2, pose2))

    print(f"Comparing {len(tasks)} pairs... using {cpu_count()} cores")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(compare_pair, tasks), total=len(tasks)))

    return results

def compare_pair(args):

    """
    Computes the distance between a pair of videos using DTW.
    """

    video1, pose1, video2, pose2 = args
    distance = dtw_ndim.distance(pose1, pose2)
    video1 = os.path.basename(video1).replace(".mp4", "")
    video2 = os.path.basename(video2).replace(".mp4", "")

    return (video1, video2, distance)