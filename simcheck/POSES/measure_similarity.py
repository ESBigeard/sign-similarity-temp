import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from simcheck.utils import EmbeddingsProcessor as ep
from simcheck.computation.dtw_computations import DTW_computation
from simcheck.computation.cosim_computations import COSIM_computation, COSIM_formatting_high_dim
from simcheck.computation.scores import save_scores_DTW, save_scores_COSIM

"""
This script is uses extracted keypoints from videos (poses) to compute similarity between videos. 
Scores are computed based on cosinus similarity and DTW, this similarity is computed between all pairs of videos in the dataset.

Seeing how exponentially long the computation can take on a large dataset (WLASL), 
this script allows to use a GPU if available, otherwise it will use the CPU.
The results are saved in a CSV file with the top 100 most similar videos for each video
in the dataset.

--- 

Requirements:
    - A directory containing JSON files with keypoints of the videos, where each file corresponds to
    a video and contains the keypoints data. These keypoints should have been extracted using our 
    feature_extraction/pose_estimation subpackage.

Command to run the script:

    python -m simcheck.POSES.measure_similarity -exp <experiment_name> -kpd <keypoints_dir> -m <method> [-fast]
"""


def retrieve_keypoints(json_file):

    """
    Retrieves keypoints from a JSON file, the returned array is of shape (T, D),
    where T is the number of frames and D is the number of keypoints (each joint
    * 2 since we have x and y coordinates).
    """

    with open(json_file, 'r') as json_file:
        keypoints = json.load(json_file)
     
    return np.array(ep.load_pose(keypoints), dtype=np.float32)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DTW similarity check")
    parser.add_argument("-exp", "--experiment_name", type=str, required=True, help="Experiment name (for saving results).")
    parser.add_argument("-kpd", "--keypoints_dir", type=str, required=True, help="Directory containing the mediapipe pose files.")
    parser.add_argument("-m", "--method", type=str, choices=["cosim", "dtw"], required=True, help="Method to use for measuring similarity: 'cosim' for cosine similarity, 'dtw' for dynamic time warping.")
    parser.add_argument("-fast", "--use_gpu", action='store_true', help="Use GPU for COSIM computation if available.")
    args = parser.parse_args()

    keypoints_dir, method, experiment_name, use_gpu = args.keypoints_dir, args.method, args.experiment_name, args.use_gpu
    pose_paths = [os.path.join(keypoints_dir, f) for f in os.listdir(keypoints_dir)]

    poses_dict = {}
    print(f"Found {len(pose_paths)} pose files in {keypoints_dir}.")
    print("Retrieving keypoints from pose files...")
    for pose_path in tqdm(pose_paths):
        poses_dict[pose_path] = retrieve_keypoints(pose_path)
    
    video_names = list(poses_dict.keys())
    items = list(poses_dict.items())
    
    if method == 'dtw':

        print("Computing DTW distances...")
        results = DTW_computation(items)
        print("DTW distances computed and saved.")
        save_scores_DTW(results, f'simcheck/POSES/results/similarity_measures/{experiment_name}_dtw_scores.csv')

    if method == 'cosim':

        print("Computing COSIM similarity matrix...")
        arrays = COSIM_formatting_high_dim(items)
        results = COSIM_computation(arrays, use_gpu=use_gpu)
        print("COSIM similarity matrix computed and saved.")
        save_scores_COSIM(results, video_names, f'simcheck/POSES/results/similarity_measures/{experiment_name}_cosim_scores.csv', direction=True)    

       
