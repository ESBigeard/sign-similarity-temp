"""
This script measures similarity between videos using dimensionality reduction methods.

With two different formatting of data:
- Inspired by Manolis Fragkiadakis
- Using joints variance

It furthermore allows you to only use specific keypoints as input, so only the dominant
hand for instance.
"""

import os 
import json
import argparse    
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from simcheck.DIMREDUC.variance.reduction_variance import get_results_variance
from simcheck.DIMREDUC.fragkiadakis.reduction_fragkiadakis import get_results_fragkiadakis
from simcheck.DIMREDUC.formatting.filtering_poses import filter_keypoints
from simcheck.DIMREDUC.formatting.formatting_poses import format_data
from simcheck.DIMREDUC.visualisation.create_html_file import visualisation
from simcheck.computation.scores import save_scores_COSIM


def compute_similarity(matrix):

    """
    This function computes the euclidean distance between each 2D point in the matrix.
    """

    distances = np.zeros((matrix.shape[0], matrix.shape[0]))

    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            distances[i, j] = np.linalg.norm(matrix[i] - matrix[j])
            distances[j, i] = distances[i, j]  

    return distances


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Measure similarity between videos using dimensionality reduction techniques.")
    parser.add_argument('-exp', '--experiment_name', type=str, required=True, help='Name of the experiment.')
    parser.add_argument('-kpd', '--keypoints_directory', type=str, required=True, help='Path to the directory containing embeddings file (JSON format).')
    parser.add_argument('-e', '--elements', type=str, default='all', choices=['all', 'hands', 'domhand', 'wrists', 'domwrist'], help='Elements to filter embeddings by.')
    parser.add_argument('-m', '--method', type=str, required=True, choices=['fragkiadakis', 'variance'], help='Method to use for measuring similarity.')
    parser.add_argument('-s', '--save', action='store_true', help='Save the visualisation to a HTML file.')
    parser.add_argument('-gold', '--gold_standard', type=str, help="""Path to the gold standard JSON file for video pairs.
                        If provided, videos from the same groups will be coloured the same in the visualisation.""")

    args = parser.parse_args()
    experiment_name, keypoints_files, elements, method, save, gold = args.experiment_name, args.keypoints_directory, args.elements, args.method, args.save, args.gold_standard
    videos = [file.replace('.json', '') for file in os.listdir(keypoints_files) if file.endswith('.json')]
    matrix = []

    for file in os.listdir(keypoints_files):
        embeddings_path = os.path.join(keypoints_files, file)
        data = json.load(open(embeddings_path, 'r'))
        data = filter_keypoints(data, elements)
        
        formatted_data = format_data(data, method)


        matrix.append(formatted_data)
    
    if method == 'fragkiadakis':
        coordinates = get_results_fragkiadakis(matrix)
    else:
        coordinates = get_results_variance(matrix)

    distances = compute_similarity(coordinates)
    save_scores_COSIM(distances, videos, f'simcheck/DIMREDUC/results/similarity_measures/{experiment_name}_cosim_scores.csv', direction=False)
    
    if save:
        visualisation(coordinates, videos, method, experiment_name, gold)
        print(f"Visualisation saved to simcheck/DIMREDUC/results/visualisations/{experiment_name}_visualisation.html")