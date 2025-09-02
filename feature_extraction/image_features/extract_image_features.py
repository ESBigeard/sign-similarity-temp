import os, logging, argparse
from tqdm import tqdm
from multiprocessing import Pool

from feature_extraction.utils import ImageFeaturesProcessor, VideoProcessor
from feature_extraction.image_features.ViT.embeddings import load_model, extract_features_batched


"""
This script is used to extract features for each frame in a video using the Vision Transformer (ViT) model.
The extracted features are saved to a JSON file.

Example usage:

    python3 -m feature_extraction.image_features.extract_image_features.py -vp corpus/videos/gsl_nothing.mp4 -sd features/ViT/test
    python3 -m feature_extraction.image_features.extract_image_features.py -vd corpus/videos -sd features/ViT/test -p

    with -vp indicating the path to a video file. => Used for testing
    with -vd indicating the directory containing the videos to be processed.
    with -sd indicating the directory where the extracted features will be saved.
"""


def process_video(video_path, saving_dir, image_processor, model):

    """
    Get VIT features from video frames and save them to a json file.
    """
    
    ifp = ImageFeaturesProcessor(video_path, saving_dir)

    frames = {}
    with VideoProcessor(video_path) as vp:
        for frame, frame_number in vp.frames():
            frames[frame_number] = frame
               
    features = extract_features_batched(frames, image_processor, model)
    ifp.save_features(features)
    print(f"Finished processing video {video_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""Extract features from videos using the Vision Transformer (ViT) model.
    The extracted features are saved to a JSON file.""")

    parser.add_argument("-vp", "--video_path", type=str, help="Path to a single video file.")
    parser.add_argument("-vd", "--videos_dir", type=str, help="Path to directory containing videos.")
    parser.add_argument("-sd", "--saving_dir", type=str, help="Path to directory where the extracted features will be saved.",
                        required=True)
    parser.add_argument("-p", "--parallel", action="store_true", help="Enable parallel processing.")
    parser.add_argument("-log", "--logging", action="store_true", help="Enable logging.")

    args = parser.parse_args()

    video_path, videos_dir, saving_dir = args.video_path, args.videos_dir, args.saving_dir
    log = args.logging
    p = args.parallel

    # Load the model and image processor
    image_processor, model = load_model()

    if log:
        logging.basicConfig(
            filename="feature_extraction/image_features/extract_ViT.log", 
            level=logging.INFO, 
            format='%(asctime)s %(levelname)s %(message)s', 
            filemode='w')

    # Process only one video, good for testing
    if args.video_path:
        logging.info(f"Processing a single video: {video_path}")
        process_video(args.video_path, saving_dir, image_processor, model)

    # Process all videos in a directory 
    if args.videos_dir:
        logging.info(f"Processing all videos in directory: {args.videos_dir}")
        videos = [os.path.join(args.videos_dir, video) for video in os.listdir(args.videos_dir)]
        if p:
            logging.info("Parallel processing enabled.")
           
            n = len(videos)
            cpu = min(n, os.cpu_count())
            k = n // cpu
           
            j = 0
            for i in range(k + 1):
                glob_distances = {}
                # Create a sublist of videos to process
                videos_dirs_sublist = videos[i * cpu:(i + 1) * cpu]
            
                logging.info(f"Processing {len(videos_dirs_sublist)} videos...")
                logging.info(f"Processing videos {i * cpu} to {(i + 1) * cpu}...")
                logging.info(f"Processing videos {videos_dirs_sublist}...")

                names = videos_dirs_sublist
                args_list = [(video, saving_dir, image_processor, model) for video in names]
                logging.info(f"Processing videos {args_list}...")
                logging.info("###############################################################")
                with Pool(processes=cpu) as pool:
                    
                    pool.starmap(process_video, args_list)
                    
        video_paths = [os.path.join(args.videos_dir, video) for video in os.listdir(args.videos_dir)]
        for video_path in tqdm(video_paths):
            process_video(video_path, saving_dir, image_processor, model)


