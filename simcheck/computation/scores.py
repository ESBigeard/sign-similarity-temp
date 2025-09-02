"""
This module handles the formatting and storing of similarity scores.
It retrieves the scores obtained with DTW or COSIM, formats them, and saves them in a CSV file.
This is used in SIMCHECK subpackages to compare different features.
"""

import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def save_scores_DTW(results, output_path):
    

    similarity_results = defaultdict(list)
    for video1, video2, distance in results:
        similarity_results[video1].append((video2, distance))
        similarity_results[video2].append((video1, distance))  

    df_dict = defaultdict(list)
    for video, res in similarity_results.items():
        res.sort(key=lambda x: x[1])
        for other, dist in res:
            df_dict[f"{video}"].append((other, dist))

  
    df = pd.DataFrame.from_dict(df_dict, orient="index").transpose()
    df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")

def save_scores_COSIM(similarity_matrix, video_names, output_path):
    
    dict_similarity = defaultdict(list)
    for i, video1 in tqdm(enumerate(video_names)):
        for j, video2 in enumerate(video_names):
            if i != j:
                dict_similarity[video1].append((video2, similarity_matrix[i][j]))
    
    for video, similarities in tqdm(dict_similarity.items()):
        similarities.sort(key=lambda x: x[1], reverse=True)
        dict_similarity[video] = similarities[:100]

    df = pd.DataFrame(dict_similarity)
    df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")
