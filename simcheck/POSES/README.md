# POSES for phonological similarity

One of the similarity methods explored during this internship involves directly comparing poses extracted from videos using the `feature_extraction/pose_estimation` subpackage.  
More details about the feature extraction and the entire process can be found below.

- [1. Introduction](#introduction)
- [2. Poses](#poses)
- [3. Manual](#manual)
- [4. Evaluation](#evaluation)

# Introduction

This method consists of directly comparing pose representations from videos.  
It is inspired by previous work on sign language and action recognition/classification.

This README briefly summarises the pose extraction process. For full details, and to actually run the extraction, which is required before using `POSES`, please refer to the `feature_extraction/pose_estimation` [README](../../feature_extraction/pose_estimation/README.md).

**The approach is as follows:**

- Provide the `measure_similarity` module with a folder of keypoint files.
- The module compares every possible pair of videos, each represented as a high-dimensional array, using DTW or cosine similarity.
- This subpackage works similarly to `KEYFRAMES`, except it uses the full sequence of frames instead of a subset.  
  As a result, video representations vary in length depending on the number of frames.

## Contents

Here's a brief overview of what’s included in this subpackage (see the [Manual](#manual) for more details):

- A `measure_similarity` module for computing similarities between videos using their keypoints.
- A `results` directory where similarity scores and evaluation results are stored.

👉 To learn more about evaluation, check the [`simcheck/README.md`](../README.md).

All DTW and cosine similarity computations are implemented in the scripts located in `simcheck/computation`.

# Poses

Poses are extracted using the `feature_extraction/pose_estimation` subpackage.  
This package uses MediaPipe to extract 2D body keypoints, coordinates corresponding to the positions of major joints, for each frame in a video.

The subpackage also includes a preprocessing pipeline to reduce bias in similarity computation. For example:

- Missing keypoints (undetected by MediaPipe) are interpolated.
- Videos of left-handed signers are flipped to ensure consistency.

Each video’s poses are saved in a single JSON file and are represented as an array of shape `(num_frames, 50, 2)`, where:
- `num_frames` is the number of frames in the video,
- `50` is the number of joints,
- `2` corresponds to the x and y coordinates.

These pose sequences can be directly compared using DTW, which handles high-dimensional sequences.  
For cosine similarity, however, the sequences must be flattened and padded to a uniform length, making it less suitable for sequences with temporal variation.

# Manual

This section explains the requirements and how to run the similarity script.

### Measure similarity

Similarity is computed using the `measure_similarity` module located in the root of the `POSES` subpackage.

- **Requirements:**

    - **A folder of JSON keypoint files:**

      Keypoints must be extracted from your video dataset using the `extract_poses` module from the `feature_extraction/pose_estimation` subpackage.

      👉 If you haven’t done so, refer to the [`feature_extraction/pose_estimation/README.md`](../../feature_extraction/pose_estimation/README.md).

- **Running the script:**

    Run the `measure_similarity` module from the project root (`manseri-sign-similarity`).  
    All file paths should be provided relative to this root.  
    Similarity results will be saved in the `results/similarity_measures` directory as a CSV file.

    **Required arguments:**
    
    - `-exp`: name of the experiment (used as the output filename)
    - `-kpd`: path to the folder containing keypoint files
    - `-m`: method for similarity computation (`cosim` or `dtw`)

    **Example command:**

    ```bash
    python3 -m simcheck.POSES.measure_similarity -kpd WordNet/test -exp TEST -m cosim
    ```

# Evaluation

This method is evaluated using the same top-*k* retrieval strategy described in the main `simcheck` README.  
For each query video, the top-*k* most similar videos are retrieved based on pose similarity. A correct prediction is counted if at least one true match appears among the top-*k* results.

👉 For full details on the evaluation strategy, refer to the [Evaluation section in `simcheck/README.md`](../README.md#evaluations).

## todo
