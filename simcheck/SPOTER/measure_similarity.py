"""
This script uses extracted embeddings from SPOTER to compute similarity between videos.
Scores are computed based on the cosine similarity of the embeddings, this similarity
is computed between all pairs of videos in the dataset.

Seeing how exponentially long the computation can take on a large dataset (WLASL),
this script allows to use a GPU if available, otherwise it will use the CPU.
The results are saved in a CSV file with the top 100 most similar videos for each video
in the dataset.

---

Requirements:
    - A JSON file containing the embeddings of the videos, where each key is a video name
    and the value is an embedding. These embeddings act as a compressed representation of each
    sign.

Command to run the script:

    python -m simcheck.SPOTER.measure_similarity -en <experiment_name> -ef <embeddings_file> [-gpu]
"""

import json
import argparse
import numpy as np

from simcheck.computation.dtw_computations import DTW_computation
from simcheck.computation.cosim_computations import COSIM_computation
from simcheck.computation.scores import save_scores_DTW, save_scores_COSIM


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Measure similarity between two sequences of poses using Spoter embeddings.")
    parser.add_argument('-exp', '--experiment_name', type=str, required=True, help='Name of the experiment to load embeddings from.')
    parser.add_argument('-ef', '--embeddings_file', type=str, required=True, help='Path to the folder containing keypoints files.')
    parser.add_argument('-m', '--method', type=str, required=True, choices=['cosim', 'dtw'], help='Method to use for measuring similarity.')
    parser.add_argument('-fast', '--use_gpu', action='store_true', help='Use GPU for COSIM computation.')

    args = parser.parse_args()
    exp_name, embeddings_file, method, fast = args.experiment_name, args.embeddings_file, args.method, args.use_gpu
    embeddings_data = json.load(open(embeddings_file, 'r'))
    video_names = list(embeddings_data.keys())[:100]
    embeddings = [np.array(embeddings_data[video][0]) for video in video_names]

    print(f"Loaded {len(video_names)} videos and their embeddings.")
   
    if method == 'dtw':

        print("Computing DTW distances...")
        results = DTW_computation(list(zip(video_names, embeddings)))
       
        save_scores_DTW(results, f'simcheck/SPOTER/results/similarity_measures/{exp_name}_dtw_scores.csv')
        print("DTW distances computed and saved.")

    elif method == 'cosim':
        
        print("Computing COSIM similarity matrix...")
        results = COSIM_computation(embeddings, use_gpu=fast)

        save_scores_COSIM(results, video_names, f'simcheck/SPOTER/results/similarity_measures/{exp_name}_cosim_scores.csv', direction=True)
        print("COSIM similarity matrix computed and saved.")
  