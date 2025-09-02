import cv2
import os
import numpy as np
import torch
from collections import defaultdict
import json



class BaseProcessor:
    """
    This class contains methods that are common to all processors.
    Notably including methods related to file and directory handling.
    """

    @staticmethod
    def ensure_directory(path):

        """
        Ensures that a directory exists. If it does not, it creates it.
        """

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def ensure_file(path):

        """
        Ensures that a file exists. If it does not, it raises an error.

        """

        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")

class VideoProcessor(BaseProcessor):
    
    """
    This class is responsible for processing video files, info about them and frames
    """

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.video_name = os.path.basename(video_path).split(".")[0]
    
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
    # Context manager methods
    # Closing the video file when exiting the context
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    # Generator method to iterate over frames
    # Frames are yielded as tuples containing the frame and the frame number
    def frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            yield frame, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1


    # Method to save a frame to a directory
    def save_frame(self, frame, frame_number, target_directory):
        self.ensure_directory(target_directory)
        frame_path = f"{target_directory}/{self.video_name}_{frame_number}.jpg"
        cv2.imwrite(frame_path, frame)


    # Method to load frames from a directory
    def load_frames(self, frames_path):
        frames = []
        frames_directory = os.listdir(frames_path)
        frames_directory = sorted(frames_directory, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for frame in frames_directory:
            print(frame)
            frame_path = os.path.join(frames_path, frame)
            frames.append(cv2.imread(frame_path))

        return frames

class AnnotationsProcessor(BaseProcessor):

    """
    This class is reponsible for processing annotations, loading, saving and transforming them.
    """

    def __init__(self, video_path, saving_dir):
        super().__init__()
        self.video_name = os.path.basename(video_path).split(".")[0]
        self.saving_dir = saving_dir
        self.ensure_directory(self.saving_dir)

    def save_poses(self, poses, dimensions):

        """
        Saves poses to a JSON file. 'poses' is expected to be a list of tuples
        where each tuple represents a frame and contains lists of coordinates for that frame.
        """

        pose_data = defaultdict(list)
        for frame_index, frame_poses in enumerate(poses):
            for pose in zip(*frame_poses): 
                if dimensions == 2:
                    pose_data[frame_index].append({"x": float(pose[0]), "y": float(pose[1])})
                elif dimensions == 3:
                    pose_data[frame_index].append({"x": float(pose[0]), "y": float(pose[1]), "z": float(pose[2])})

        
        if dimensions == 3:
            json_path = os.path.join(self.saving_dir, f"{self.video_name}_{dimensions}.json")
        elif dimensions == 2:
            json_path = os.path.join(self.saving_dir, f"{self.video_name}.json")

        print(f"Saving poses to {json_path}")
        with open(json_path, "w") as json_file:
            json.dump(pose_data, json_file)

        return pose_data
    
    def load_poses(self, dimensions):

        """
        Loads poses from a JSON file. 'dimensions' is expected to be an integer
        representing the number of dimensions of the poses.
        """

        json_path = os.path.join(self.target_directory, f"{self.video_name}{dimensions}.json")
        print(f"Loading poses from {json_path}")
        with open(json_path, "r") as json_file:
            pose_data = json.load(json_file)

        return pose_data
    

class ImageFeaturesProcessor(BaseProcessor):
    
    """
    This class is responsible for processing image features, loading, saving and transforming them.
    """

    def __init__(self, video_path, target_directory):
        super().__init__()
        self.video_name = os.path.basename(video_path).split(".")[0]
        self.target_directory = target_directory
        self.features_path = os.path.join(target_directory, f"{self.video_name}.json")
        self.ensure_directory(self.target_directory)

    def save_features(self, features):

        # if the features are np or torch, convert them to a dict
      
        features = {k : v.tolist() for k, v in features.items()}
        print(f"Saving features to {self.target_directory}/{self.video_name}.json")

        with open(os.path.join(self.target_directory, f"{self.video_name}.json"), "w") as json_file:
            json.dump(features, json_file)