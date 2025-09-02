import pandas as pd
import os
import matplotlib.pyplot as plt

"""
This allows to evaluate methods applied to the wordnet dataset.
"""


kS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

def load_csv(csv_file):

    df = pd.read_csv(csv_file)
    return df

def get_dtw_results(df, K):

    dtw_results = {}
    videos = []

    for column, idx in enumerate(df.columns):
        # Get the name of the column as the key
        key = df.columns[column]
        key = key.split("/")[-1][:-7]
        # get the rest of the column as the value
        values = df.iloc[0:, column].tolist()[:K]
        values = [value.strip('"').strip("(").strip(")").strip("'") for value in values]
        values = [tuple(value.split(",")) for value in values]
        values = [(value[0][:-8].split("/")[-1], value[1]) for value in values]
        # get the first cell of the column as the key
        # add the key and values to the dictionary
        dtw_results[key] = values
        videos.append(key)
        videos.extend([value[0] for value in values])
    
    return dtw_results, set(videos)

def get_matches(matches_df):

    matches = {}
    Target_videos = matches_df["Target Video"].tolist()
    Identical_videos = matches_df["Identical Videos"].tolist()
    for src, match in zip(Target_videos, Identical_videos):
        match = match.replace("[", "").replace("]", "").replace("'", "")
        match = match.split(",")
        match = [video.strip() for video in match]
        matches[src] = match
    return matches


if __name__ == "__main__":
   
    
  
    from math import comb

    N = 511
    results_dtw = {}
    results_random = {}

    for K in kS:
        TP = 0
        baseline_sum = 0
            
        df = load_csv("results/WORDNET_similarity_results_raw.csv")
        dtw_results, videos = get_dtw_results(df,K)
        matches = load_csv("../../../../evaluation_data/WordNet/identical.csv")
        matches = get_matches(matches)

        for video, results in dtw_results.items():
            results = [result[0] for result in results]
            real_matches = matches.get(video, [])
            
            for result in results:
                if result in real_matches:
                    TP += 1
                    break
            k = len(real_matches)
            p_k = 1 - comb(N - 1 - k, K) / comb(N - 1, K)
            baseline_sum += p_k
        
        results_dtw[K] = TP / N
        results_random[K] = baseline_sum / N

        accuracy = TP / len(dtw_results)
        baseline_avg = baseline_sum / len(dtw_results)

        print(f"K = {K}")
        print(f"True Positives: {TP}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Expected accuracy by chance (avg baseline): {baseline_avg:.2f}")


    # We plot the results so on peut compare the difference between the DTW results and the random baseline
    plt.figure(figsize=(10, 5))
    plt.plot(kS, list(results_dtw.values()), label="DTW Results", marker='o')
    plt.plot(kS, list(results_random.values()), label="Random Baseline", marker='x')
    plt.xlabel("Top-k returned videos")
    plt.ylabel("Accuracy")
    plt.title("DTW Results vs Random Baseline on the WordNet Dataset")
    plt.xticks(kS)
    plt.legend()
    plt.grid()
    plt.savefig("results/wordnet_dtw_vs_random.png")
