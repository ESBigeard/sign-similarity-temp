import numpy as np


def mediapipe_tree():

    MEDIAPIPE_ARRAY_TO_TREE = {
        # format is {ID TREE}: {ID MEDIAPIPE} or ({ID1 MEDIAPIPE, ID2 MEDIAPIPE}) s.t. pt = (pt[ID2] + pt[ID1])/2
        # ============= BODY =============
        0: 0, # (Nose)

        1: (12, 11), # (Between Shoulders)

        5: 12, # (Left Shoulder Joint)
        6: 14, # (Left Elbow Joint)
        7: 16, # (Left Wrist Body Joint)

        2: 11, # (Right Shoulder Joint)
        3: 13, # (Right Elbow Joint)
        4: 15, # (Right Wrist Body Joint)

        # ========== HAND LEFT ==========
        8: 0, # (Left Wrist Hand Joint)

        9: 1, # (Left 1st Joint Thumb)
        10: 2, # (Left 2nd Joint Thumb)
        11: 3, # (Left 3rd Joint Thumb)
        12: 4, # (Left 4th Joint Thumb)

        13: 5, # (Left 1st Joint Index)
        14: 6, # (Left 2nd Joint Index)
        15: 7, # (Left 3rd Joint Index)
        16: 8, # (Left 4th Joint Index)

        17: 9, # (Left 1st Joint Middle Finger)
        18: 10, # (Left 2nd Joint Middle Finger)
        19: 11, # (Left 3rd Joint Middle Finger)
        20: 12, # (Left 4th Joint Middle Finger)

        21: 13, # (Left 1st Joint Ring)
        22: 14, # (Left 2nd Joint Ring)
        23: 15, # (Left 3rd Joint Ring)
        24: 16, # (Left 4th Joint Ring)

        25: 17, # (Left 1st Joint Pinky)
        26: 18, # (Left 2nd Joint Pinky)
        27: 19, # (Left 3rd Joint Pinky)
        28: 20, # (Left 4th Joint Pinky)

        # ========== HAND RIGHT ==========
        29: 0, # (Right Wrist Hand Joint)

        30: 1, # (Right 1st Joint Thumb)
        31: 2, # (Right 2nd Joint Thumb)
        32: 3, # (Right 3rd Joint Thumb)
        33: 4, # (Right 4th Joint Thumb)

        34: 5, # (Right 1st Joint Index)
        35: 6, # (Right 2nd Joint Index)
        36: 7, # (Right 3rd Joint Index)
        37: 8, # (Right 4th Joint Index)

        38: 9, # (Right 1st Joint Middle Finger)
        39: 10, # (Right 2nd Joint Middle Finger)
        40: 11, # (Right 3rd Joint Middle Finger)
        41: 12, # (Right 4th Joint Middle Finger)

        42: 13, # (Right 1st Joint Ring)
        43: 14, # (Right 2nd Joint Ring)
        44: 15, # (Right 3rd Joint Ring)
        45: 16, # (Right 4th Joint Ring)

        46: 17, # (Right 1st Joint Pinky)
        47: 18, # (Right 2nd Joint Pinky)
        48: 19, # (Right 3rd Joint Pinky)
        49: 20, # (Right 4th Joint Pinky)
    }


    TREE_COMPARTMENTS = {
        "body": list(np.arange(0, 8)),
        "left_hand": list(np.arange(8, 29)),
        "right_hand": list(np.arange(29, 50)),
    }

    return MEDIAPIPE_ARRAY_TO_TREE, TREE_COMPARTMENTS

def zelinka_skeletal_structure():

    # Definition of skeleton model structure:
    #   The structure is an n-tuple of:
    #
    #   (index of a start point, index of an end point, index of a bone) 
    #
    #   E.g., this simple skeletal model
    #
    #             (0)
    #              |
    #              |
    #              0
    #              |
    #              |
    #     (2)--1--(1)--1--(3)
    #      |               |
    #      |               |
    #      2               2
    #      |               |
    #      |               |
    #     (4)             (5)
    #
    #   has this structure:
    #
    #   (
    #     (0, 1, 0),
    #     (1, 2, 1),
    #     (1, 3, 1),
    #     (2, 4, 2),
    #     (3, 5, 2),
    #   )
    #
    #  Warning 1: The structure has to be a tree.  
    #
    #  Warning 2: The order isn't random. The order is from a root to lists.
    #

    return ( 
        # head
        (0, 1, 0),

        # left shoulder
        (1, 2, 1),

        # left arm
        (2, 3, 2),
        (3, 4, 3),

        # right shoulder
        (1, 5, 1), 

        # right arm
        (5, 6, 2),
        (6, 7, 3),
    
        # left hand - wrist
        (7, 8, 4),
    
        # left hand - palm
        (8, 9, 5),
        (8, 13, 9),
        (8, 17, 13),
        (8, 21, 17),
        (8, 25, 21),

        # left hand - 1st finger
        (9, 10, 6),
        (10, 11, 7),
        (11, 12, 8),

        # left hand - 2nd finger
        (13, 14, 10),
        (14, 15, 11),
        (15, 16, 12),
    
        # left hand - 3rd finger
        (17, 18, 14),
        (18, 19, 15),
        (19, 20, 16),
    
        # left hand - 4th finger
        (21, 22, 18),
        (22, 23, 19),
        (23, 24, 20),
    
        # left hand - 5th finger
        (25, 26, 22),
        (26, 27, 23),
        (27, 28, 24),
    
        # right hand - wrist
        (4, 29, 4),
    
        # right hand - palm
        (29, 30, 5), 
        (29, 34, 9),
        (29, 38, 13),
        (29, 42, 17),
        (29, 46, 21),

        # right hand - 1st finger
        (30, 31, 6),
        (31, 32, 7),
        (32, 33, 8),
    
        # right hand - 2nd finger
        (34, 35, 10),
        (35, 36, 11),
        (36, 37, 12),
    
        # right hand - 3rd finger
        (38, 39, 14),
        (39, 40, 15),
        (40, 41, 16),
    
        # right hand - 4th finger
        (42, 43, 18),
        (43, 44, 19),
        (44, 45, 20),
    
        # right hand - 5th finger
        (46, 47, 22),
        (47, 48, 23),
        (48, 49, 24), 
    )


def annotation_skeletal_structure():

    """
    Made to link points to their respective connections.
    Used in the annotate_video function in feature_extraction/annotate_frames.py
    """
    return {
    0: [],  # nez, non connecté
    1: [2, 5],
    2: [3],
    3: [4],
    4: [4],
    5: [6],
    6: [7],
    7: [7],

    8: [9, 13, 17, 21, 25],
    9: [10],
    10: [11],
    11: [12],
    12: [12],
    13: [14],
    14: [15],
    15: [16],
    16: [16],
    17: [18],
    18: [19],
    19: [20],
    20: [20],
    21: [22],
    22: [23],
    23: [24],
    24: [24],
    25: [26],
    26: [27],
    27: [28],
    28: [28],

    29: [30, 34, 38, 42, 46],
    30: [31],
    31: [32],
    32: [33],
    33: [33],
    34: [35],
    35: [36],
    36: [37],
    37: [37],
    38: [39],
    39: [40],
    40: [41],
    41: [41],
    42: [43],
    43: [44],
    44: [45],
    45: [45],
    46: [47],
    47: [48],
    48: [49],
    49: [49]
}

def ratio_for_passive_reconstruction():

    """
    Ratio of the length of the bones in the hand.
    Used in the passive_reconstruction function in feature_extraction/pose_estimation/zelinka/correct_2D_pose
    """

    return {
        0 : 9.336927783358426,
        1 : 4.0333925391168695,
        2 : 4.302876324841095,
        3 : 4.577051849693307,
        4 : 4.76569230665629,
        5 : 9.462966587238965,
        6 : 11.43761944985635,
        7 : 14.96564004703622,
        8 : 8.327557418526375,
        9 : 18.409455378599244,
        10 : 22.69549910185132,
        11 : 9.088796227360726,
        12 : 19.572031434700836,
        13 : 22.89950224204409,
        14 : 9.912244831497839,
        15 : 21.570911927317713,
        16 : 24.48530396213368,
        17 : 12.220458442813872,
        18 : 26.643719655108672,
        19 : 28.622799634014676
}

