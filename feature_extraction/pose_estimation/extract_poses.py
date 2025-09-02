import os, logging, argparse
from tqdm import tqdm
from multiprocessing import Pool

from feature_extraction.utils import AnnotationsProcessor, VideoProcessor
from feature_extraction.pose_estimation.mediapipe.landmarks import extract_2D_pose_from_frames
from feature_extraction.pose_estimation.zelinka.correct_2D_pose import get_corrected_2D_pose
# from feature_extraction.pose_estimation.zelinka.optimization import get_corrected_3D_pose


"""
This script is used to extract poses from videos using the mediapipe framework.
The extracted poses can be saved to a json file.

Poses can be saved using 2D coordinates. 
Original mediapipe coordinates can be normalized and interpolated.

Example usage:

    python3 -m feature_extraction.pose_estimation.extract_poses -vd ../test -sd ../testmediapipe --normalize --interpolate
    python3 -m feature_extraction.pose_estimation.extract_poses -vp ../test/video.mp4 -sd ../testmediapipe --normalize -log

    with -vd indicating the directory containing the videos to be processed.
    with -sd indicating the directory where the extracted poses will be saved.
    with --normalize and --interpolate indicating the flags for the processing steps.

    The combination of flags will determine the processing steps applied to the extracted poses.
"""

def process_video(video_path, saving_dir, norm, interp):

    """
    Get mediapipe coordinates from video frames and save them to a json file.
    """
    try:

        # Loading video and annotations processors
        logging.info(f"##################### Processing video {video_path} with normalization={norm} and interpolation={interp} #####################")
        vp = VideoProcessor(video_path)
        ap = AnnotationsProcessor(video_path, saving_dir)
        fps = vp.fps
        print(f"Processing video: {video_path} with fps: {fps}")
        logging.info(" --------------------- 1. Extracting 2D pose landmarks from video frames...")
        all_skeletons = extract_2D_pose_from_frames(video_path)
        logging.info(f"{len(all_skeletons)} skeletons have been extracted with 2D pose landmarks.")
        logging.info(" --------------------- 2. Correcting and processing 2D pose landmarks...")
        Xx, Xy, Xw = get_corrected_2D_pose(all_skeletons, norm, interp, fps, video_path.split("/")[-1].split(".")[0])

        pose_2D = list(zip(Xx, Xy))
        ap.save_poses(pose_2D, 2)

        # Extract 3D coordinates if indicated with the threeD flag
        # Those and the 2D coordinates are saved in different files
        # if threeD:
        #     Xx, Xy, Xz = get_corrected_3D_pose(Xx, Xy, Xw)
        #     pose_3D = list(zip(Xx, Xy, Xz))
        #     ap.save_poses(pose_3D, 3)
        # UNCOMMENT TO EXTRACT 3D COORDINATES
        # Not used in the rest of the internship

    except Exception as e:

        logging.error(f"Error processing video {video_path}: {e}")
        
                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""Extract Mediapipe poses from videos, these poses can be retrieved
    in 2D or 3D coordinates. You can also normalize and interpolate them if needed using the Zelinka algorithm.""")

    parser.add_argument("-vp", "--video_path", type=str, help="Path to video file.")
    parser.add_argument("-vd", "--videos_dir", type=str, help="Path to directory containing videos.")
    parser.add_argument("-sd", "--saving_dir", type=str, help="Path to directory where the extracted features will be saved.",
                        required=True)

    parser.add_argument("--normalize", action="store_true", help="Normalize the 2D coordinates.")
    parser.add_argument("--interpolate", action="store_true", help="Interpolate the 2D coordinates.")
    # parser.add_argument("--threeD", action="store_true", help="Extract 3D coordinates.")

    parser.add_argument("-log", "--logging", action="store_true", help="Enable logging.")
    parser.add_argument("-p", "--parallel", action="store_true", help="Enable parallel processing.")

    args = parser.parse_args()

    video_path, videos_dir, saving_dir = args.video_path, args.videos_dir, args.saving_dir
    norm, interp = args.normalize, args.interpolate
    log = args.logging
    p = args.parallel


    if log:
        logging.basicConfig(
            filename="feature_extraction/pose_estimation/extract_poses.log", 
            level=logging.INFO, 
            format='%(asctime)s %(levelname)s %(message)s', 
            filemode='w')


    # Process only one video, good for testing
    if args.video_path:
        print(f"Processing video: {video_path}")
        logging.info(f"Processing a single video: {video_path}")
        process_video(args.video_path, saving_dir, norm, interp)

    # Process all videos in a directory 
    if args.videos_dir:
        print(f"Processing videos in directory: {videos_dir}")
        videos = [os.path.join(args.videos_dir, video) for video in os.listdir(args.videos_dir)]
        if p:
            logging.info("Parallel processing enabled.")
           
            n = len(videos)
            cpu = min(n, os.cpu_count())
            k = n // cpu
           
            j = 0
            for i in range(k + 1):
                glob_distances = {}
                videos_dirs_sublist = videos[i * cpu:(i + 1) * cpu]

                names = videos_dirs_sublist
                args_list = [(video, saving_dir, norm, interp) for video in names]
                logging.info(f"Processing videos {args_list}...")
                logging.info("###############################################################")
                with Pool(processes=cpu) as pool:
                    
                    pool.starmap(process_video, args_list)
                    
        else:

            logging.info("Parallel processing disabled.")
            logging.info(f"Processing videos in directory: {args.videos_dir}")
            video_paths = [os.path.join(args.videos_dir, video) for video in os.listdir(args.videos_dir)]
            for video_path in tqdm(video_paths):
                process_video(video_path, saving_dir, norm, interp)
   