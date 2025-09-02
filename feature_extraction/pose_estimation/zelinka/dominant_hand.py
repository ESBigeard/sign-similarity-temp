import numpy as np

"""
Uses the retrieved hand landmarks to determine the dominant hand.
To find said dominant hand, we will aim to find the hand with the highest average velocity.
If the video is flagged as having the left hand as the dominant hand, we will flip the coordinates
to make the left hand appear in the left part of the system. As if the video was mirrored.
"""

def avg_speed_btw_two_frames(x1, y1, x2, y2, delta_t):

    """
    Calculate the average speed between two frames.
    """

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / delta_t

def get_hand_wrist_coordinates(xs, ys): 

    """
    Get the x and y coordinates of the hand wrist.
    """
    
    x = []
    y = []

    for coord in xs:
        x.append(coord[0])
    
    for coord in ys:
        y.append(coord[0])

    return x, y

def get_hand_speeds(left_x, right_x, delta_t):

    """
    Get the average speed of the hand in the video.
    """

    speeds = []
    for idx, (x1, y1) in enumerate(zip(left_x, right_x)):
        if idx == 0:
            continue
    
        if x1 == 0 or y1 == 0 or left_x[idx - 1] == 0 or right_x[idx - 1] == 0:
            continue
        speed = avg_speed_btw_two_frames(left_x[idx - 1], right_x[idx - 1], x1, y1, delta_t)
        speeds.append(speed)

    return speeds


def get_delta_t(fps):

    """
    Get the time between each frame.
    """

    return 1 / fps

def check_highest_hand(Hx, Hy):

    """
    Check which hand is the highest in the video.
    """

    return min(Hy)

def get_skeleton_dominant_hand(left_hand_x, left_hand_y, right_hand_x, right_hand_y, fps):

    """
    Get the dominant hand in the video.
    """

    left_x, left_y = get_hand_wrist_coordinates(left_hand_x, left_hand_y)
    right_x, right_y = get_hand_wrist_coordinates(right_hand_x, right_hand_y)
  
    # If one hand is not visible, then the other hand is the dominant hand
    # So if the left hand is not visible, the right hand is the dominant hand
    # and we do not switch.
    # If the right hand is not visible, the left hand is the dominant hand
    # and we switch.
    if all(x == 0.0 for x in left_x) or all(y == 0.0 for y in left_y):
 
        return False
    elif all(x == 0.0 for x in right_x) or all(y == 0.0 for y in right_y):
    
        return True

    delta_t = get_delta_t(fps)

    left_hand_speeds = get_hand_speeds(left_x, left_y, delta_t)
    right_hand_speeds = get_hand_speeds(right_x, right_y, delta_t)
  
    avg_speed_left = np.mean(left_hand_speeds)
    avg_speed_right = np.mean(right_hand_speeds)



    # if the average speed of the left hand is greater than the average speed of the right hand
    # then the left hand is the dominant hand
    if avg_speed_left > avg_speed_right:
        print("Left hand is the dominant hand")
        return True
    
    y_left = check_highest_hand(left_x, left_y)
    y_right = check_highest_hand(right_x, right_y)
    if y_left < y_right: # < because y axis is inverted
        return True
    else:
        return False
   


