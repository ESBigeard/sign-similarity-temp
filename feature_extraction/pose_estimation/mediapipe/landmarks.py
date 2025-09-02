import cv2
import logging
import numpy as np
from mediapipe.python.solutions.hands import Hands
from mediapipe.python.solutions.pose import Pose
from feature_extraction.pose_estimation.tree import mediapipe_tree
from feature_extraction.utils import VideoProcessor

"""
This module contains functions to extract 2D pose landmarks from video frames using MediaPipe.
It is used in the extract_poses module inside of the feature_extraction package and the 
pose_estimation subpackage.
For each frame in the video, the function extract_2D_pose_from_frame extracts the pose landmarks
and the hand landmarks. The function convert_mediapipe_arrays_to_tree converts the mediapipe arrays
to a tree structure. The function extract_2D_pose_from_frames extracts the landmarks from all the frames
in the video.

A list of tuples is returned, where each tuple contains the landmarks in the order of the skeletal model.
Each tuple is a frame in the video.

In the case of any issue during the extraction, please refer to the logs and do not hesitate to add 
more logging information to the functions.
"""


def extract_2D_pose_from_frame(frame, 
							   body_predictor, 
							   hands_predictor, 
							   MEDIAPIPE_ARRAY_TO_TREE, 
							   TREE_COMPARTMENTS):

	"""
	Extracts 2D pose landmarks from a single frame.
	Returns the landmarks in the order of the skeletal model.
	"""

	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False

	pose_results, hands_results = body_predictor.process(image), hands_predictor.process(image)

	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	missing_points = 0

	pose_pts_scores = []
	pose_landmarks = pose_results.pose_landmarks

	logging.info("Extracting pose landmarks")
	if pose_landmarks:
		logging.info("Pose landmarks detected")
		for lk in pose_landmarks.landmark:
			try:
				pose_pts_scores.append([lk.x, lk.y, lk.visibility])
			except:
				pose_pts_scores.append([0, 0, 0])
				missing_points += 1
	
	else:
		logging.info("No pose landmarks detected")
		pose_pts_scores = [[0, 0, 0] for _ in range(33)]
		missing_points = 33

	logging.info(f"State of pose landmarks: {pose_pts_scores}")
	logging.info(f"Number of missing points: {missing_points}")

	hands_pts_scores = {"left": [], "right": []}
	hands_landmarks = hands_results.multi_hand_landmarks
	detected_hands = False

	logging.info("Extracting hand landmarks")
	if hands_landmarks:
		sides = [h.classification[0].label.lower() for h in hands_results.multi_handedness]
		if len(sides) == 2 and "left" in sides and "right" in sides:
			detected_hands = True
			logging.info("Both hands detected !")

			for side, hand_lamarks in zip(sides, hands_landmarks):
				real_side = "left" if side == "right" else "right"
				for lk in hand_lamarks.landmark:
					try:
						hands_pts_scores[real_side].append([lk.x, lk.y, 0.5])
					except:
						hands_pts_scores[real_side].append([0, 0, 0])
						missing_points += 1

		if len(sides) == 1:
			detected_hands = True
			logging.info("Single hand detected !")
			side = sides[0]
			real_side = "left" if side == "right" else "right"
			logging.info(f"Side of the detected hand: {side}")
			for lk in hands_landmarks[0].landmark:
				try:
					hands_pts_scores[real_side].append([lk.x, lk.y, 0.5])
				except:
					hands_pts_scores[real_side].append([0, 0, 0])
					missing_points += 1
			
			other_side = "right" if real_side == "left" else "left"
			hands_pts_scores[other_side] = [[0, 0, 0] for _ in range(21)]
		
	if not detected_hands:
		logging.info("No hands detected !")
		hands_pts_scores = {"left": [[0, 0, 0] for _ in range(21)], "right": [[0, 0, 0] for _ in range(21)]}
		missing_points += 42

	pose_array = np.array(pose_pts_scores)
	left_hand_array = np.array(hands_pts_scores["left"])
	right_hand_array = np.array(hands_pts_scores["right"])

	tree_tuples = convert_mediapipe_arrays_to_tree(pose_array, 
												   left_hand_array, 
												   right_hand_array, 
												   MEDIAPIPE_ARRAY_TO_TREE, 
												   TREE_COMPARTMENTS)

	return tree_tuples


def detect_parasite_hand(hand_array):

	"""
	Detects if the hand is a parasite hand.
	A parasite hand is a hand that is detected only in 5 % of the frames.
	"""

	num_frames = hand_array.shape[1]
	five_percent = int(num_frames * 0.05)

	logging.info(f"Number of frames: {num_frames}, five percent: {five_percent}")
	logging.info(f"Number of frames with hand: {np.sum(hand_array[:, 0] != 0)}")

	if np.sum(hand_array[:, 0] == 0) > five_percent:
	
		logging.info("Parasite hand detected")
		hand_array = np.zeros(hand_array.shape)
		logging.info("Hand array set to zero")
	else:
		logging.info("No parasite hand detected")

	return hand_array


def convert_mediapipe_arrays_to_tree(pose_array, 
									 left_hand_array, 
									 right_hand_array, 
									 MEDIAPIPE_ARRAY_TO_TREE, 
									 TREE_COMPARTMENTS):

	"""
	Converts the mediapipe arrays to a tree structure.
	Returns the arrays in the order of the skeletal model.
	"""

	skeleton = []

	parts_arr = {
		"body": pose_array,
		"left_hand": left_hand_array,
		"right_hand": right_hand_array
	}

	for part, ids in TREE_COMPARTMENTS.items():
		part_arr = parts_arr[part]
		for i in ids:
			mp_id = MEDIAPIPE_ARRAY_TO_TREE[i]
			if isinstance(mp_id, tuple):
				a, b = mp_id
				point = (part_arr[a] + part_arr[b]) / 2
			else:
				assert isinstance(mp_id, int)
				point = part_arr[mp_id]

			skeleton += point.tolist()

	return tuple(skeleton)

def extract_2D_pose_from_frames(video_path):
	
	"""
	Apply the extract_2D_pose_from_frame function to all the frames in the video.
	Returns a list of tuples, where each tuple contains the landmarks in the order of the
	skeletal model. Each tuple is a frame in the video.
	"""

	config_predictor = {"min_detection_confidence": 0.5, "min_tracking_confidence": 0.5}
	body_predictor, hands_predictor = Pose(**config_predictor), Hands(**config_predictor)

	MEDIAPIPE_ARRAY_TO_TREE, TREE_COMPARTMENTS = mediapipe_tree()

	all_skeleton = []

	with VideoProcessor(video_path) as video:

		fps = video.fps
		logging.info("The following info will concern the Mediapipe pose extraction:")
		for frame, frame_number in video.frames():
			tree_tuples = extract_2D_pose_from_frame(frame, body_predictor, hands_predictor, MEDIAPIPE_ARRAY_TO_TREE, TREE_COMPARTMENTS)
			all_skeleton.append(tree_tuples)

	return all_skeleton




















