"""
This script can be used to format pose data extracted from videos into a structured format suitable for SPOTER training.
Training data is expected to be in a specific JSON format, that can be obtained using the extract_poses module.
This script loads the keypoints from the JSON files and formats them into a structured dictionary.
The data is then saved into a CSV file for further processing or training.

The WLASL dataset is used as an example, but the script can be adapted for other datasets as well.
The WLASL dataset already has three folders: train, val, and test; containing the keypoints extracted from the videos.
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

def get_target_format():

    """
    This function dictates the number of columns in the input data and their
    names.
    """

    format = {
        "Label": [], "Video": [],
        "noseX": [], "noseY": [], "neckX": [], "neckY": [], 
        "leftShoulderX": [], "leftShoulderY": [], "leftElbowX": [], "leftElbowY": [], 
        "leftWristX": [], "leftWristY": [], "rightShoulderX": [], "rightShoulderY": [], 
        "rightElbowX": [], "rightElbowY": [], "rightWristX": [], "rightWristY": [],
        "leftHandWristX": [], "leftHandWristY": [], "leftHandThumb1X": [], "leftHandThumb1Y": [],
        "leftHandThumb2X": [], "leftHandThumb2Y": [], "leftHandThumb3X": [], "leftHandThumb3Y": [],
        "leftHandThumb4X": [], "leftHandThumb4Y": [],
        "leftHandIndex1X": [], "leftHandIndex1Y": [], "leftHandIndex2X": [], "leftHandIndex2Y": [],
        "leftHandIndex3X": [], "leftHandIndex3Y": [], "leftHandIndex4X": [], "leftHandIndex4Y": [],
        "leftHandMiddle1X": [], "leftHandMiddle1Y": [], "leftHandMiddle2X": [], "leftHandMiddle2Y": [],
        "leftHandMiddle3X": [], "leftHandMiddle3Y": [], "leftHandMiddle4X": [], "leftHandMiddle4Y": [],
        "leftHandRing1X": [], "leftHandRing1Y": [], "leftHandRing2X": [], "leftHandRing2Y": [],
        "leftHandRing3X": [], "leftHandRing3Y": [], "leftHandRing4X": [], "leftHandRing4Y": [],
        "leftHandPinky1X": [], "leftHandPinky1Y": [], "leftHandPinky2X": [], "leftHandPinky2Y": [],
        "leftHandPinky3X": [], "leftHandPinky3Y": [], "leftHandPinky4X": [], "leftHandPinky4Y": [],
        "rightHandWristX": [], "rightHandWristY": [], "rightHandThumb1X": [], "rightHandThumb1Y": [],
        "rightHandThumb2X": [], "rightHandThumb2Y": [], "rightHandThumb3X": [], "rightHandThumb3Y": [],
        "rightHandThumb4X": [], "rightHandThumb4Y": [],
        "rightHandIndex1X": [], "rightHandIndex1Y": [], "rightHandIndex2X": [], "rightHandIndex2Y": [],
        "rightHandIndex3X": [], "rightHandIndex3Y": [], "rightHandIndex4X": [], "rightHandIndex4Y": [],
        "rightHandMiddle1X": [], "rightHandMiddle1Y": [], "rightHandMiddle2X": [], "rightHandMiddle2Y": [],
        "rightHandMiddle3X": [], "rightHandMiddle3Y": [], "rightHandMiddle4X": [], "rightHandMiddle4Y": [],
        "rightHandRing1X": [], "rightHandRing1Y": [], "rightHandRing2X": [], "rightHandRing2Y": [],
        "rightHandRing3X": [], "rightHandRing3Y": [], "rightHandRing4X": [], "rightHandRing4Y": [],
        "rightHandPinky1X": [], "rightHandPinky1Y": [], "rightHandPinky2X": [], "rightHandPinky2Y": [],
        "rightHandPinky3X": [], "rightHandPinky3Y": [], "rightHandPinky4X": [], "rightHandPinky4Y": []}

    indexes = {joint : i for i, joint in enumerate(list(format.keys())[2:])}

    return format, indexes

def organise_data(json_labels, subset):

    """
    This function organises the data into a dataframe format.
    Each is expected to have an ID (its name) and a label.
    """
   
    labels = {"Label": [], "Video": []}

    for idx, class_data in enumerate(json_labels):
        videos_info = class_data["instances"]
        for video_info in videos_info:
            if video_info["split"] == subset:
                labels["Label"].append(idx)
                labels["Video"].append(video_info["video_id"])  
            
    videos = labels["Video"]
    classes = labels["Label"]
    labels = dict(zip(videos, classes))

    return labels

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Format pose data for training")
    parser.add_argument("--pose_folder", type=str, help="Path to the folder containing pose data in JSON format")
    parser.add_argument("--json_labels", type=str, help="Path to the JSON file containing labels for the videos", required=True)
    parser.add_argument("--set", type=str, choices=["train", "val", "test"], default="train",
                        help="Specify the dataset split to process (train, val, test)")
    args = parser.parse_args()
    
    pose_folder, subset, json_labels = args.pose_folder, args.set, args.json_labels
    format, indexes = get_target_format()

    keypoints_files = os.listdir(pose_folder)
    keypoints_files = [os.path.join(pose_folder, file) for file in keypoints_files]

    json_labels =  json.load(open(json_labels, 'r'))
    labels = organise_data(json_labels, subset)

    keypoints_files = [file for file in keypoints_files if os.path.basename(file).split(".")[0].replace("_2", "") in labels.keys()]

    for keypoint_file in tqdm(keypoints_files):
        video_name = os.path.basename(keypoint_file).split(".")[0].replace("_2", "")
        data = json.load(open(keypoint_file, 'r'))
        formatted_data, indexes = get_target_format()

        for frame, keypoints in data.items():

            for video, label in labels.items():
                    if video_name == video:
                        formatted_data["Label"] = label
                        formatted_data["Video"] = video_name
                        break
                        
            xi = 0
            for keypoint in keypoints:
                x, y = keypoint.get("x", 0), keypoint.get("y", 0)
                yi = xi + 1
                for joint, index in indexes.items():
                    if index == xi:
                        formatted_data[joint].append(x)
                    if index == yi:
                        formatted_data[joint].append(y)
                
                xi = yi + 1
            
        for key, value in formatted_data.items():
          
            format[key].append(value)

    df = pd.DataFrame.from_dict(format)
    print(df.head())
    df.to_csv(f"../datasets/{subset}.csv", index=False)
                
            

