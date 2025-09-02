# Images features 

This subpackage enables **frame-based features extraction** using the [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929). ViT extracts features related to shapes, colours, and absolute positions. 

These features can be used for various tasks, however, in this repository, they'll be used to extract keyframes from videos using the `keyframe_extraction` package. For more info, see the corresponding [README file](../../keyframe_extraction/README.md).

## MANUAL

The module allowing for ViT embeddings extraction is `extract_image_features`. 

- **Requirements:**

    - A folder of videos saved in `mp4` format, or, a single video in the same format.

- **Running:**

    The `extract_image_features` should be run from the `root_directory` (`manseri-sign-similarity`) directory. All paths should be provided relative to this directory. All embeddings will be saved in JSON files named accordingly to their corresponding videos, and stored in the directory specified as argument.

    It expects the following arguments:

    - `-vp` : path to a single video, the one from which the embeddings will be extracted.
    - `-vd` : path to a folder containing multiple videos, from which the embeddings will be extracted. 
    - `-sd` : path to the directory where the embeddings will be saved in JSON files.

    - `-p` : to allow parallel processing.
    - `-log` : to enable logging.

    Command:

    ```bash
    python3 -m feature_extraction.image_features.extract_image_features -vd data/WordNet/all -sd features/ViT/WordNet