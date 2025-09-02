import gc
import cv2
import torch
import logging
from transformers import AutoImageProcessor, AutoModel

"""
Creates embeddings for each frame in a video using the Vision Transformer (ViT) model.
This script extracts features from a single frame.
"""

def load_model():
    
    """
    Load the image processor and model from Hugging Face Hub.
    """

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
    model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
    logging.info("Model and image processor loaded successfully.")
    return image_processor, model


def extract_features_from_frame(frame, image_processor, model):
    
    """
    Loads the image processor and model, processes the frame, and extracts features.
    """

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB

    # Feature extraction
    inputs = image_processor(images=image, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(device)

    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    # We retrieve the features using the pooler_output
    # Using the LastHiddenState (after averaging) is also possible
    # features = outputs.last_hidden_state.mean(dim=1)
    # As explained by the ViT cretors, the pooler output seems to be the best option
    features = outputs.pooler_output

    return features


def extract_features_batched(frames, image_processor, model, batch_size=16):
    
    """
    Process a batch of frames and extract features.
    """

    all_features = {}
    for i in range(0, len(frames), batch_size):
        batch_frames = list(frames.items())[i : i + batch_size]  # Take a batch of frames
        batch_features = {}
        for frame_name, frame in batch_frames:
            features = extract_features_from_frame(frame, image_processor, model)
            batch_features[frame_name] = features
            del features
            gc.collect()

        all_features.update(batch_features)
        del batch_frames, batch_features
        gc.collect()

    return all_features

