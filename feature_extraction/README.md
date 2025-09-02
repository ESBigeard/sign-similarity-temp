# Feature Extraction

**This package handles the extraction of two distinct types of features:**

- **Pose estimation** (joint annotations)
- **Image-based features**

The `feature_extraction` package therefore contains two subpackages : `pose_estimation` and `image_features`.

## `pose_estimation`

- This subpackage extracts upper-body joint annotations using the [MediaPipe](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html) framework. 
This extraction is made possible by the `extract_poses` module located at the root of the subpackage. All other scripts are used for processing data and improving keypoint consistency. This module expects video(s) in mp4 format as input to extract keypoints from joints and store them in resulting JSON files. It also provides the `annotate_frames` module to draw keypoints on video, useful for visualisation and debugging purposes. 

- The `extract_poses` should be run from the `root_directory` (`manseri-sign-similarity`) directory. All paths should be provided relative to this directory. All keypoints will be saved in JSON files named accordingly to their corresponding videos, and stored in the directory specified as argument.

- Here is how the module should be run using an illustrative command : 

    ```bash
    python3 -m feature_extraction.pose_estimation.extract_poses -vd data/WLASL/all -sd features/mediapipe/WLASL_complete -p --interpolate --normalize
👉 For more details on the preprocessing steps and how to run the module, see [`pose_estimation/README.md`](./pose_estimation/README.md).


## `image_features`

- This packages computes embeddings for each frame of a video using a [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929). These embeddings are used for keyframe extraction. Extracting these embeddings is made possible by the `extract_image_features` module located in the root of the subpackage. All other scripts are used to process data. This module expects video(s) in mp4 format as input to extract ViT embeddings and store them in resulting JSON files. 

- The `extract_image_features` should be run from the `root_directory` (`manseri-sign-similarity`) directory. All paths should be provided relative to this directory. All embeddings will be saved in JSON files named accordingly to their corresponding videos, and stored in the directory specified as argument.

- Here is how the module should be run using an illustrative command : 

    ```bash
    python3 -m feature_extraction.image_features.extract_image_features -vd data/WordNet/all -sd features/ViT/WordNet
👉 For more information on the processing pipeline and on how the module shoule be run, see [`image_features/README.md`](./image_features/README.md).
