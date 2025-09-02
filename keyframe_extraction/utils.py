import os 
import json
import cv2


class BaseProcessor:
    """
    This class contains methods that are common to all processors.
    Notably including methods related to file and directory handling.
    """

    def ensure_directory(self, path):

        """
        Ensures that a directory exists. If it does not, it creates it.
        """

        if not os.path.exists(path):
            os.makedirs(path)

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
    
class PoseProcessor(BaseProcessor):

    """
    This class is responsible for processing pose files.
    """

    def __init__(self, video_path, mp_dir):
        super().__init__()
        self.video_path = video_path
        self.video_name = os.path.basename(video_path).split(".")[0]
        self.pose_file = os.path.join(mp_dir, f"{self.video_name}.json")

    def load_poses_from_json(self, json_path):

        """
        Load poses from a json file.
        """
        with open(json_path, "r") as f:
            poses = json.load(f)
        return poses
    
class ImageFeaturesProcessor(BaseProcessor):

    """
    This class is responsible for processing image features files.
    """

    def __init__(self, video_path, vit_dir):
        super().__init__()
        self.video_path = video_path
        self.video_name = os.path.basename(video_path).split(".")[0]
        self.features_file = os.path.join(vit_dir, f"{self.video_name}.json")

    def load_features_from_json(self, json_path):

        """
        Load features from a json file.
        """

        with open(json_path, "r") as f:
            features = json.load(f)
        return features