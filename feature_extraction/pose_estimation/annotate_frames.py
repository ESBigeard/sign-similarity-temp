import os
import cv2
import json
import argparse
import subprocess
from feature_extraction.utils import VideoProcessor
from feature_extraction.utils import BaseProcessor as bp
from feature_extraction.pose_estimation.tree import annotation_skeletal_structure

"""
This script is used to annotate video frames with pose estimation coordinates.
It reads the pose coordinates from a JSON file and draws them on the frames.
The annotated frames are saved as a video file.

Thie script can be used for visualisation and debugging purposes.
"""


def annotate_video(video_name, poses, frames, output_video_path):
  
    connections = annotation_skeletal_structure()
    original_fps = 20.0
    temp_dir = "../corpus/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    for frame_index, frame in enumerate(frames):
        frame_copy = frame.copy()
        keypoints = {}
        for idx, pose in enumerate(poses[str(frame_index)]):
            x, y = pose["x"], pose["y"]
            if idx in range(8, 29): #non dominating hand
                current_point = (int(x * frame_copy.shape[1]), int(y * frame_copy.shape[0]))
                keypoints[idx] = current_point
                cv2.circle(frame_copy, current_point, 2, (0, 255, 255), -1)
            elif idx in range(29, 50):
                current_point = (int(x * frame_copy.shape[1]), int(y * frame_copy.shape[0]))
                keypoints[idx] = current_point
                cv2.circle(frame_copy, current_point, 2, (0, 0, 255), -1)
            else: #<8 : body pose
                current_point = (int(x * frame_copy.shape[1]), int(y * frame_copy.shape[0]))
                keypoints[idx] = current_point
                cv2.circle(frame_copy, current_point, 2, (255, 0, 255), -1)

        
        for joint_a, joint_b in connections.items():
            if joint_a in keypoints:
                for b_point in joint_b:
                    if b_point in keypoints:
                        ## if in coordinates 0, 0, skip
                        if keypoints[joint_a] == (0, 0) or keypoints[b_point] == (0, 0):
                            continue
                        cv2.line(frame_copy, keypoints[joint_a], keypoints[b_point], (0, 255, 0), 1)
             
        frame_path = os.path.join(temp_dir, f"frame_{frame_index:04d}.png")
        cv2.imwrite(frame_path, frame_copy)

    ffmpeg_command = [
        "ffmpeg",
        "-y",  
        "-framerate", str(original_fps),
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video_path,
        "-loglevel", "quiet",
    ]
    subprocess.run(ffmpeg_command, check=True)

    for frame_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, frame_file))
    os.rmdir(temp_dir)

    print(f"Video saved successfully to {output_video_path}")

def annotate_video_native_format(video_name, poses, frames, output_video_path):

    """ same as annotate_video() but keeps mediapipe's native json format """
  
    connections = annotation_skeletal_structure()
    original_fps = 20.0
    temp_dir = "../corpus/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    for frame_index, frame in enumerate(frames):
        frame_copy = frame.copy()
        keypoints = {}
        #for pose_frame in poses:
        #print(frame_index,len(frame))
        pose_frame=poses[frame_index]
        if True:
            id_pose=pose_frame["frame"]
            
            #here the idx of each point is changed to match the order of the other format, so it can match "connections" below to draw the bones between the correct joints
            
            #body pose
            points=pose_frame["pose_landmarks"]
            for idx,pose in enumerate(points): 
                #idx = id of the point    
                if idx >10 and idx <17: #only keep the points for the arms
                    d={11:2, 12:5, 13:3, 14:6, 15:4, 16:7} #they completely rearranged the order of those points. native_mediapipe -> their order
                    idx=d[idx]
                    x, y = pose["x"], pose["y"]
                    current_point = (int(x * frame_copy.shape[1]), int(y * frame_copy.shape[0]))
                    keypoints[idx] = current_point
                    cv2.circle(frame_copy, current_point, 2, (0, 0, 0), -1)

                
            #hands
            for hand in pose_frame["hand_landmarks"]:
                points=hand["landmarks"]
                if hand["label"]=="Left":
                    for idx,pose in enumerate(points): 
                        idx=idx+8 #the order of the points is the same in both formats, but each hand starts at 0 in native mediapipe. converts the starting id.
                        x, y = pose["x"], pose["y"]
                        current_point = (int(x * frame_copy.shape[1]), int(y * frame_copy.shape[0]))
                        keypoints[idx] = current_point
                        cv2.circle(frame_copy, current_point, 2, (0, 0, 255), -1)
                else:
                    for idx,pose in enumerate(points): 
                        idx=idx+29
                        x, y = pose["x"], pose["y"] 
                        current_point = (int(x * frame_copy.shape[1]), int(y * frame_copy.shape[0]))
                        keypoints[idx] = current_point
                        cv2.circle(frame_copy, current_point, 2, (255, 255, 255), -1)

                

        #connections is a dict that lists which points connects to which. for example 1:[2,5]
        for joint_a, joint_b in connections.items():
            if joint_a in keypoints:
                for b_point in joint_b:
                    if b_point in keypoints:
                        cv2.line(frame_copy, keypoints[joint_a], keypoints[b_point], (0, 255, 0), 1)
             
        frame_path = os.path.join(temp_dir, f"frame_{frame_index:04d}.png")
        cv2.imwrite(frame_path, frame_copy)

    ffmpeg_command = [
        "ffmpeg",
        "-y",  
        "-framerate", str(original_fps),
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video_path,
        "-loglevel", "quiet",
    ]
    subprocess.run(ffmpeg_command, check=True)

    for frame_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, frame_file))
    os.rmdir(temp_dir)

    print(f"Video saved successfully to {output_video_path}")    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Annotate frames with mediapipe pose estimation coordinates')
    parser.add_argument('--videos_dir', type=str, help='Path to directory containing videos', required=True)
    parser.add_argument('--poses_dir', type=str, help='Path to directory containing poses (json) files', required=True)
    parser.add_argument('--saving_dir', type=str, help='Path to directory where the annotated videos will be saved', required=True)

    args = parser.parse_args()

    poses_dir, saving_dir = args.poses_dir, args.saving_dir
    bp.ensure_directory(saving_dir)

    video_paths = [os.path.join(args.videos_dir, video) for video in os.listdir(args.videos_dir)]
    for video_path in video_paths:
    
        print(f"Processing video: {video_path}...")
        video_name = os.path.basename(video_path).split(".")[0]
        poses_file = poses_dir + "/" + video_name + ".json"
        output_video_path = saving_dir + f"/{video_name}.mp4"
        frames = []

        with VideoProcessor(video_path) as vp:
            for frame, frame_index in vp.frames():
                frames.append(frame)

        try:
            bp.ensure_file(poses_file)
            poses = json.load(open(poses_file))
            if type(poses)==dict:
                annotate_video(video_name, poses, frames, output_video_path)
            else:
                annotate_video_native_format(video_name, poses, frames, output_video_path)

        except FileNotFoundError as err:
            print(err)

    