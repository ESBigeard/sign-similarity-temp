import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from keyframe_extraction.utils import VideoProcessor
from keyframe_extraction.embeddings_processing.concatenation import concatenate_mediapipe_vit
from keyframe_extraction.clustering.kmeans import apply_kmeans_clustering, choose_keyframes


"""
This script is used to extract keyframes from a video using the ViT features and the Mediapipe landmarks.
The landmarks can be used separately or in combination. By default, only the ViT features are used.
The keyframes are extracted using KMeans clustering and the centroids of the clusters.
A set number of keyframes can be specified.
The keyframes are saved in the specified directory.

"""


def process_video(video_path, concat, clusters, mp, vit):

    """
    Process a video and extract keyframes using KMeans clustering.
    """

    logging.info(f"Processing video: {video_path}")

    # Retrieving the Mediapipe landmarks and the ViT features
    # and concatenating them if specified
    concatenated_embeddings = concatenate_mediapipe_vit(video_path, concat, mp, vit)
    features = np.array([concatenated_embeddings[frame] for frame in concatenated_embeddings.keys()])
    features = features.reshape(features.shape[0], -1)

    # Apply KMeans clustering to the features
    # and choose the keyframes based on the centroids of the clusters
    clustered_frames, centroids = apply_kmeans_clustering(features, clusters)
    key_frames = choose_keyframes(clustered_frames, centroids, features)
    
    return dict(sorted(key_frames.items(), key=lambda item: item[1]))


def retrieve_and_save_keyframes(video_path, keyframes, saving_dir):
    
    """
    Retrieve the keyframes from the video and save them to the specified directory.
    """

    # Extract keyframes from the video
    with VideoProcessor(video_path) as vp:
        for frame, frame_number in vp.frames():
            # Check if the frame number is in the keyframes dictionary
            # and save the frame if it is
            if frame_number in keyframes.values():
                vp.save_frame(frame, frame_number, saving_dir + "/" + os.path.basename(video_path).split(".")[0])
                logging.info(f"Keyframe {frame_number} saved to {saving_dir}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""Extract keyframes from a video using the ViT features and the Mediapipe landmarks.
                                                    The landmarrks can be used separately or in combination. By default, only the ViT features are used.
                                                    The keyframes are extracted using KMeans clustering and the centroids of the clusters.
                                                    A set number of keyframes can be specified.
                                                    The keyframes are saved in the specified directory.
                                                    """)

    parser.add_argument("-vp", "--video_path", type=str, help="Path to video file.")
    parser.add_argument("-vd", "--videos_dir", type=str, help="Path to directory containing videos.")
    parser.add_argument("-sd", "--saving_dir", type=str, help="Path to directory where the extracted features will be saved.",
                        required=True)
    
    parser.add_argument("-mp", "--mediapipe", type=str, help="Path to the Mediapipe landmarks file.")
    parser.add_argument("-vit", "--vit_features", type=str, help="Path to the ViT features file.", required=True)
    parser.add_argument("-cl", "--clusters", type=int, help="Number of clusters to use for KMeans clustering.", choices=range(1, 15), default=5)

    parser.add_argument("-log", "--logging", action="store_true", help="Enable logging.")

    args = parser.parse_args()

    video_path, videos_dir, saving_dir = args.video_path, args.videos_dir, args.saving_dir
    mp, vit, clusters = args.mediapipe, args.vit_features, args.clusters
   
    log = args.logging


    if log:
        logging.basicConfig(
            filename="extract_poses.log", 
            level=logging.INFO, 
            format='%(asctime)s %(levelname)s %(message)s', 
            filemode='w')
        
    if mp and vit:
        logging.info("Using both the ViT features and the Mediapipe landmarks.")
        logging.info(f"Number of clusters: {clusters}")
        concat = True
    else:
        logging.info("Using only the ViT features.")
        logging.info(f"Number of clusters: {clusters}")
        concat = False
        
    
    if video_path:
        # Process a single video
        keyframes = process_video(video_path, concat, clusters, mp, vit)
        logging.info(f"Keyframes: {keyframes}")
        retrieve_and_save_keyframes(video_path, keyframes, saving_dir)


    if videos_dir:
        # Process all videos in the directory
        video_paths = [os.path.join(args.videos_dir, video) for video in os.listdir(args.videos_dir)]
        for video_path in tqdm(video_paths):
            keyframes = process_video(video_path, concat, clusters, mp, vit)
            logging.info(f"Keyframes: {keyframes}")
            retrieve_and_save_keyframes(video_path, keyframes, saving_dir)
