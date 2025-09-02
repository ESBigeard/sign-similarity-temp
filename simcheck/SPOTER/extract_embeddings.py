"""
This script is used to extract embeddings from videos by given their extracted keyopoints as
input to a pre-trained SPOTER model. The model should be trained on the same keypoints format.
For each video, embeddings are extracted from the latent space of the model, and more precisely
from the penultimate layer of the model, right before the linear layer. 

These embeddings aim to represent each video in a compressed form, which can then be used
for similarity matching or other downstream tasks.

---

Requirements:
    - A pre-trained SPOTER model checkpoint. Can be trained using the train.sh script
    in the retrain_SPOTER directory.
    - A folder containing keypoints files in JSON format, where each file corresponds to a video
    and contains keypoints. These keypoints should be in the format expected by the SPOTER model.
    So, in the format (same normalisation, preprocessing, and number of keypoints) as the one used
    during training.

Command to run the script:

    All given paths should be relative to the project root directory.

    python3 -m simcheck.SPOTER.extract_embeddings.py -exp <experiment_name> -mp <model_path> -kf <keypoints_folder>
"""

import os
import sys
import glob
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Setting the subdirectory for the SPOTER retraining scripts
retrain_dir = Path(__file__).resolve().parents[0] / "retrain_SPOTER"
if str(retrain_dir) not in sys.path:
    sys.path.insert(0, str(retrain_dir))


def format_keypoints(data):

    """
    Formats the keypoints data from JSON into a sequence of poses.
    """

    pose_sequence = []
    for frame, keypoints in data.items():
        pose = []
        for keypoint in keypoints:
            pose.extend([keypoint['x'], keypoint['y']])
        pose_sequence.append(pose)
    return np.array(pose_sequence, dtype=np.float32)


def load_model(model_path):

    """
    Loads the pre-trained SPOTER model from the given path.
    """
    
    model = torch.load(model_path, map_location='cpu')
    model.eval()

    return model


def extract_embedding_from_file(keypoints_file, model):

    """
    Extracts the embedding from a single keypoints file using the pre-trained SPOTER model.
    """

    try:

        file_id = os.path.basename(keypoints_file).split('.')[0].replace("_", "")
        data = json.load(open(keypoints_file, 'r'))
        pose_sequence = format_keypoints(data)

        with torch.no_grad():
            pose_tensor = torch.tensor(pose_sequence, dtype=torch.float32)
            # Returning the linear layer embeddings and the h representations
            # h being the penultimate layer output
            h, _ = model(pose_tensor, return_embeddings=True)
            embedding = h.squeeze(0).cpu().numpy().tolist()

        return file_id, embedding
    
    except Exception as e:

        print(f"[ERROR] Failed on {keypoints_file} with the error: {e}")

        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract embeddings from a pre-trained SPOTER model.")
    parser.add_argument("-exp", "--exp_name", type=str, required=True, help="Name of the experiment (used for output file).")
    parser.add_argument("-mp", "--model_path", type=str, required=True, help="Path to the best model checkpoint.")
    parser.add_argument("-kf", "--keypoints_folder", type=str, required=True, help="Path to the keypoints folder.")
    args = parser.parse_args()

    exp_name, model_path, keypoints_folder = args.exp_name, args.model_path, args.keypoints_folder
    keypoints_files = glob.glob(keypoints_folder + "/*.json")
    embeddings_dict = {}

    # Loading the pre-trained SPOTER model
    model = load_model(model_path)

    for keypoints_file in tqdm(keypoints_files):
        result = extract_embedding_from_file(keypoints_file, model)
        if result:
            video_id, embedding = result
            embeddings_dict[video_id] = embedding

    # Saving the embeddings and naming them after the experiment name
    output_path = f"simcheck/SPOTER/results/embeddings/{exp_name}_embeddings.json"
    with open(output_path, "w") as f:
        json.dump(embeddings_dict, f, indent=4)

    print(f"Embeddings extracted and saved to {output_path}")
