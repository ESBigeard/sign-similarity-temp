import pandas as pd
import json
from collections import defaultdict, Counter
import os
import random

def load_csv(csv_file):

    df = pd.read_csv(csv_file)
    return df

def load_json(json_file, videos):

    with open(json_file, "r") as f:
        data = json.load(f)

    data = {video : data[video] for video in data.keys() if video in videos}
    
    return data

def get_videos(dir):

    return [video.split(".")[0] for video in os.listdir(dir)]

def get_dtw_results(df):

    dtw_results = {}
    videos = []

    for column, idx in enumerate(df.columns):
        # Get the name of the column as the key
        key = df.columns[column]
        key = key.split("/")[-1][:-7]
        # get the rest of the column as the value
        values = df.iloc[0:, column].tolist()[:20]
        values = [value.strip('"').strip("(").strip(")").strip("'") for value in values]
        values = [tuple(value.split(",")) for value in values]
        values = [(value[0][:-8].split("/")[-1], value[1]) for value in values]
        # get the first cell of the column as the key
        # add the key and values to the dictionary
        dtw_results[key] = values
        videos.append(key)
        videos.extend([value[0] for value in values])
    
    return dtw_results, set(videos)

def count_configs(data):

    configs_dict = Counter()

    for video, configs in data.items():
        for config in configs:
            configs_dict[config] += 1
    
    return configs_dict

if __name__ == "__main__":

    df = load_csv("results/ASLLVD_similarity_results_raw.csv")
    dtw_results, videos = get_dtw_results(df)
    data = load_json("../../../../data/ASLLVD/data.json", videos)
    one_arm_videos = get_videos("../../../../test_data/ASLLVD_one_arm")
    both_arm_videos = get_videos("../../../../test_data/ASLLVD_both_arms")

    TP_one_arm = 0
    TP_both_arm = 0

    # Only keep the 500 first videos
    videos = list(dtw_results.keys())[:511]
    dtw_results = {video: dtw_results[video] for video in videos}

    for video, values in dtw_results.items():
        video_config_ND = data[video][1]
        video_config_D = data[video][0]
        
        if video in one_arm_videos:
            video_config_D = data[video][0]

            for value in values:
                video2 = value[0]
                video2_config_ND = data[video2][1]
                video2_config_D = data[video2][0]

                if video_config_D == video2_config_D or video_config_D == video2_config_ND:
                    TP_one_arm += 1
                    break
        
        elif video in both_arm_videos:
            video_config_D = data[video][0]
            video_config_ND = data[video][1]

            for value in values:
                video2 = value[0]
                video2_config_ND = data[video2][1]
                video2_config_D = data[video2][0]

                if video_config_D == video2_config_D and video_config_ND == video2_config_ND:
                    TP_both_arm += 1
                    break
                elif video_config_ND == video2_config_D and video_config_D == video2_config_ND:
                    TP_both_arm += 1
                    break

    TP = TP_one_arm + TP_both_arm
    print("TP one arm: ", TP_one_arm)
    print("TP both arm: ", TP_both_arm)
    print("Total TP: ", TP)
    print("TP: ", TP)
    accurac_one_arm = TP_one_arm / len(one_arm_videos)
    print(f"Accuracy one arm: {accurac_one_arm:.2f}")
    accurac_both_arm = TP_both_arm / len(both_arm_videos)
    print(f"Accuracy both arm: {accurac_both_arm:.2f}")
    accuracy = TP / len(dtw_results)
    print(f"Accuracy: {accuracy:.2f}")
            