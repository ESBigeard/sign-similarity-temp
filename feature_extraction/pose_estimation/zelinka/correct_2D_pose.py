import numpy as np, math, logging
from feature_extraction.pose_estimation.zelinka.dominant_hand import get_skeleton_dominant_hand

"""
Corrects 2D pose estimation coordinates made by mediapipe.
This scripts allows to normalise and interpolate the coordinates if desired.
It also allows to switch the array if the dominant hand is the left hand and 
to reconstruct passive arms uniformly accross the dataset.
It is called in the extract_poses module with of the pose_estimation package.

It uses functions from the zelinka.dominant_hand script for dominant hand detection.

If you encounter any issues with the coordinates, please check the logs files and feel
free to add logs in the code to help debugging.
"""

######### GET X AND Y COORDINATES AND WEIGHTS #########

def get_coordinates_and_scores(input_sequence):

    """
    Get the X, Y coordinates and the weights from the input sequence.
    The input sequence is an array where each row corresponds to a frame
    and each column corresponds to the X, Y and weight of the joints in the skeleton.

    We retrieve the X, Y and weights from the input sequence.
    """

    Xx = input_sequence[0:input_sequence.shape[0], 0:(input_sequence.shape[1]):3] 
    # Retrieving all of the X coordinates from the input sequence. We get an array
    # where each row corresponds to a frame and each column corresponds to the X in 
    # the order of the joints in the skeleton. So [[xhead, xmidshoulder...][xhead..][...]]
    Xy = input_sequence[0:input_sequence.shape[0], 1:(input_sequence.shape[1]):3]
    # Same as Xx but for the Y coordinates.
    Xw = input_sequence[0:input_sequence.shape[0], 2:(input_sequence.shape[1]):3]
    # Same as Xx but for the weights.

    return Xx, Xy, Xw

######## UTILITY FUNCTIONS #########


def get_right_and_left_hand(Xx, Xy):

    """
    Get the right and left hand coordinates.
    """

    return Xx[:, 8:29], Xy[:, 8:29], Xx[:, 29:], Xy[:, 29:]

def get_right_and_left_body(Xx, Xy):

    """
    Get the right and left body coordinates.
    """

    return Xx[:, 2:5], Xy[:, 2:5], Xx[:, 5:8], Xy[:, 5:8]

def get_body(Xx, Xy):

    """
    Get the nose and neck coordinates.
    """

    return Xx[:, 0:2], Xy[:, 0:2]

def flag_immobile_hand(xs, ys):

    """
    Check if the hand is immobile.
    """

    return np.all([x[0] == xs[0][0] for x in xs]) or np.all([y[0] == ys[0][0] for y in ys])

def nulify_hand(hand_x, hand_y):

    """
    Nulify the hand coordinates.
    """

    return np.zeros((hand_x.shape[0], hand_x.shape[1])), np.zeros((hand_y.shape[0], hand_y.shape[1]))


def euclidean_distance(x1, y1, x2, y2):

    """
    Compute the euclidean distance between two points.
    """

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


######### NORMALISATION #########

def normalisation_zscore(Xx, Xy):

    """
    Normalise the X and Y coordinates using Z-score normalisation 
    using the standard deviation and the mean of the X and Y coordinates.
    """

    # Z-score normalisation, mean = 0, std = 1
    # T = number of frames, n = number of coordinates for each frame
    T, n = Xx.shape
    # Total number of coordinates
    sum0 = T * n 
    sum1Xx = np.sum(np.sum(Xx)) # Sum of all the X coordinates
    sum2Xx = np.sum(np.sum(Xx * Xx)) # Sum of all the X coordinates squared

    sum1Xy = np.sum(np.sum(Xy)) # Then the same for the Y coordinates
    sum2Xy = np.sum(np.sum(Xy * Xy))

    mux = sum1Xx / sum0 # We compute the mean of the X coordinates
    muy = sum1Xy / sum0 # We compute the mean of the Y coordinates

    sum0 = 2 * sum0 # We double the sum0 value to account for the X and Y coordinates
    sum1 = sum1Xx + sum1Xy # We sum the X and Y coordinates
    sum2 = sum2Xx + sum2Xy # We sum the X and Y coordinates squared

    mu = sum1 / sum0 # We compute the mean of the X and Y coordinates
    # We compute the variance of the X and Y coordinates
    sigma2 = (sum2 / sum0) - mu * mu 

    # We make sure the standard deviation is not too small
    if sigma2 < 1e-10:
        sigma2 = 1e-10

    sigma = math.sqrt(sigma2)

    # Return the normalised X and Y coordinates, z = (x - mux) / sigma
    return (Xx - mux) / sigma, (Xy - muy) / sigma


def anchor_based_normalisation_formula(joints, origin, anchor_distance):

    """
    Apply the normalisation to the non null values of the joints.
    """

    return np.where(joints != 0.0, (joints - origin) / anchor_distance, 0.0)

def anchor_based_normalisation(Xx, Xy):

    """
    Normalise the X and Y coordinates using the anchor based normalisation method.
    The body and the hands are normalised separately.
    The body is normalised using the distance between the shoulders as the anchor distance 
    and the neck coordinates as the origin.
    The hands are normalised using the distance between the shoulders as the anchor distance
    and the wrist coordinates as the origin.
    """

    ### GET ANCHOR DISTANCE AND ORIGIN ###

    # Non dominant shoulder, dominant shoulder, neck, non dominant wrist and dominant wrist
    Xx_NDS, Xy_NDS, Xx_DS, Xy_DS = Xx[:, 2], Xy[:, 2], Xx[:, 5], Xy[:, 5]
    Xx_NDW, Xy_NDW, Xx_DW, Xy_DW = Xx[:, 8], Xy[:, 8], Xx[:, 29], Xy[:, 29]
    Xx_N, Xy_N = Xx[:, 1], Xy[:, 1]
    Xx_NDS, Xy_NDS, Xx_DS, Xy_DS = Xx_NDS[:, np.newaxis], Xy_NDS[:, np.newaxis], Xx_DS[:, np.newaxis], Xy_DS[:, np.newaxis]
    Xx_NDW, Xy_NDW, Xx_DW, Xy_DW = Xx_NDW[:, np.newaxis], Xy_NDW[:, np.newaxis], Xx_DW[:, np.newaxis], Xy_DW[:, np.newaxis]
    Xx_N, Xy_N = Xx_N[:, np.newaxis], Xy_N[:, np.newaxis]

    # anchor distance based on the distance between the shoulders
    # using the mean to prevent extreme values if wrong annotation
    SS = np.sqrt((Xx_NDS - Xx_DS) ** 2 + (Xy_NDS - Xy_DS) ** 2)
    SS = np.mean(SS)

    ### NORMALISATION ###
   
    # Body joints
    Xx_B, Xy_B = Xx[:, :8], Xy[:, :8]
    Xx_B, Xy_B = anchor_based_normalisation_formula(Xx_B, Xx_N, SS), anchor_based_normalisation_formula(Xy_B, Xy_N, SS)
    
    # Non dominant hand joints
    Xx_NDH, Xy_NDH = Xx[:, 8:29], Xy[:, 8:29]
    Xx_NDH, Xy_NDH = anchor_based_normalisation_formula(Xx_NDH, Xx_NDW, SS), anchor_based_normalisation_formula(Xy_NDH, Xy_NDW, SS)

    # Dominant hand joints
    Xx_DH, Xy_DH = Xx[:, 29:], Xy[:, 29:]
    Xx_DH, Xy_DH = anchor_based_normalisation_formula(Xx_DH, Xx_DW, SS), anchor_based_normalisation_formula(Xy_DH, Xy_DW, SS)

    return np.concatenate([Xx_B, Xx_NDH, Xx_DH], axis=1), np.concatenate([Xy_B, Xy_NDH, Xy_DH], axis=1)


######### INTERPOLATION #########
    
def interpolation(Xx, Xy, Xw, threshold=0.9, dtype=np.float32, only_missing=True):

    """
    Interpolate the missing X and Y coordinates.
    The interpolation is done using a threshold value.
    If the weight of the joint is below the threshold, we interpolate the missing
    X and Y coordinates by averaging the X and Y weights of the previous and next frames
    until the weight is above the threshold. We then return the interpolated X and Y coordinates
    by averaging the X and Y coordinates of the previous and next frames multiplied by the weights.
    """

    T = Xw.shape[0]
    N = Xw.shape[1]

    Yx = np.zeros((T, N), dtype=dtype)
    Yy = np.zeros((T, N), dtype=dtype)

    # I added the `only_missing` arg to not smoothen all but just skeletal poses with missing points
    if only_missing:
        for t in range(T):
            for i in range(N):

                a1, a2, p = Xx[t, i], Xy[t, i], Xw[t, i]

                if p == 0:
         
                    sumpa1, sumpa2, sump = p * a1, p * a2, p
                    delta = 0
        
                    while sump < threshold:
                        change = False
                        delta = delta + 1
                        t2 = t + delta
                        if t2 < T:

                            a1, a2, p = Xx[t2, i], Xy[t2, i], Xw[t2, i]                    
                            sumpa1, sumpa2, sump = sumpa1 + p * a1, sumpa2 + p * a2, sump + p
                            change = True

                        t2 = t - delta
                        if t2 >= 0:
                            a1, a2, p = Xx[t2, i], Xy[t2, i], Xw[t2, i]
                            sumpa1, sumpa2, sump = sumpa1 + p * a1, sumpa2 + p * a2, sump + p
                            change = True

                        if not change:
                            break

                    if sump <= 0.0:
                        sump = 1e-10

                    Yx[t, i] = sumpa1 / sump
                    Yy[t, i] = sumpa2 / sump

                else:
                    Yx[t, i], Yy[t, i] = a1, a2


    return Yx, Yy, Xw

######## DETECTING PARASITE HANDS ########

def detect_parasite_hand(Hx, Hy):

	"""
	Detects if the hand is a parasite hand.
	A parasite hand is a hand that is detected only in 10 % of the frames.
	"""

	num_frames = Hx.shape[0]
	thresold = int(num_frames * 0.9)
    

	if np.sum(Hx[:, 0] == 0) >= thresold:
		Hx, Hy = np.zeros(Hx.shape), np.zeros(Hy.shape)
	return Hx, Hy


def remove_immobile_points(Xx, Xy):

    """
    Nulify hands coordinates belonging to hands that do not move in a video.
    Those hands were wrongly annotated by Mediapipe and interpolated by the Zelinka
    interoplation method. 

    Important because the arm can be moving even though the hand is frozen and not be flagged
    as passive. An immobile hand will allow for the arm to be flagged as passive in the evaluating
    passive arm process.
    """

    LHx, LHy, RHx, RHy = get_right_and_left_hand(Xx, Xy)
    LBx, LBy, RBx, RBy = get_right_and_left_body(Xx, Xy)
    Bx, By = get_body(Xx, Xy)

    # If the hand is immobile, we nulify the coordinates
    if flag_immobile_hand(LHx, LHy): 
        LHx, LHy = nulify_hand(LHx, LHy)
    
    if flag_immobile_hand(RHx, RHy):
        RHx, RHy = nulify_hand(RHx, RHy)
    

    return np.concatenate([Bx, LBx, RBx, LHx, RHx], axis=1), np.concatenate([By, LBy, RBy, LHy, RHy], axis=1)

######## DETECTING AND SWITCHING THE DOMINANT HAND ########



def switch_array_if_dominant_hand(Xx, Xy, Xw, fps):

    """
    Switch the array if the dominant hand is the left hand.
    The hand, in the final array, that is the first one always corresponds
    to the non-dominant hand.
    In most cases, the dominant hand is the right hand.
    When the dominant hand is the left hand, we switch the array. 
    We then remove the first column that corresponds to the head.
    """

    LHx, LHy, RHx, RHy = get_right_and_left_hand(Xx, Xy)
    switch = get_skeleton_dominant_hand(LHx, LHy, RHx, RHy, fps)

    if switch:

        # Flipping the array if the dominant hand is the left hand.
        # Xx - 1 for all element that are not 0, else, stay 0
        Xxswitch = np.where(Xx != 0, 1 - Xx, Xx)
        # Now, for PCA and to be able to have the dominant hand and pose
        # in the same columns, we need to switch and put the left hand in the
        # right hand position and vice versa.  

    

        return np.concatenate([Xxswitch[:, 0:2], Xxswitch[:, 5:8], Xxswitch[:, 2:5], Xxswitch[:, 29:], Xxswitch[:, 8:29]], axis=1), \
               np.concatenate([Xy[:, 0:2], Xy[:, 5:8], Xy[:, 2:5], Xy[:, 29:], Xy[:, 8:29]], axis=1), Xw
        

    return Xx, Xy, Xw

######## CHECKING IF THE ARM IS PASSIVE ########

def check_passive_arm_angle_variance(E_angles):

    """
    Check the variance of the passive arm angles.
    If the variance is too high, we consider that the arm is not passive.
    """

    return np.var(E_angles) < 80.0


def check_low_variance_active_arm_angles(Xx, Xy):

    """ 
    Second verification for arms that have low variance.
    Example of : ngt_police, sometimes, the active arm is considered passive.
    To not loose any active arm, we check if the hand is at one point far away
    from its original position. If yes, we consider that the arm is active.
    The verification is made when it has been decided that the arm was not
    in a straight position on the side of the body.
    Its fine for this case, because we aonly check the non dominant arm anyway.
    But for a two hands sign, we might have to check if the non dominant arm is indeed
    passive of not.
    """

    # We start by retrieving the coordinates of the body wrists (first_frame and last)
    Xx_NDW, Xy_NDW = np.mean([Xx[0, 4], Xx[-1, 4]]), np.mean([Xy[0, 4], Xy[-1, 4]])
    # We then retrieve the elbow coordinates
    Xx_NDE, Xy_NDE = np.mean([Xx[0, 3], Xx[-1, 3]]), np.mean([Xy[0, 3], Xy[-1, 3]])
    # We then retrieve the shoulder coordinates
    Xy_NDS = np.mean([Xx[0, 2], Xx[-1, 2]])

    # Then we compute the non euclidean distance between the wrist and elbow on the x axis
    X = Xx_NDW - Xx_NDE

    # We then compute the non euclidean distance between the wrist and shoulder on the y axis
    Y = Xy_NDW - Xy_NDS

    # We then retrieve the x coordinates of the midpoint between the shoulder and elbow
    Xmidpoint = Xx_NDW + X / 2
    Ymidpoint = Xy_NDW + Y / 2

    # Get all of the coordinates of the non dominant hand
    Xx_NDH, Xy_NDH = Xx[:, 8:29], Xy[:, 8:29]
    # Checking if there is at least one frame where all of the hand coordinates
    # are in between the elbow position and the midpoint position
    # If yes, we consider that the arm is active and return True

    for i in range(Xx_NDH.shape[0]):
        if np.all(Xx_NDH[i] >= Xx_NDE) and np.all(Xx_NDH[i] <= Xmidpoint):
            return True
        # if np.all(non_dom_hand_y[i] >= non_dom_shoulder_y) and np.all(non_dom_hand_y[i] <= mid_point_y):
        #     return True
    # If no, we consider that the arm is passive and return False
    return False


def evaluate_passive_arm(Xx, Xy, video_name):

    """
    To find arms that are not involved in the sign,
    arms that are on the side and not moving.
    arms that are resting on the stomach.
    arms that are intially resting on the stomach and
    the moving downwards.
    """

    # Retrieving coordinates 
    LHx, LHy, RHx, RHy = get_right_and_left_hand(Xx, Xy)
    LBx, LBy, RBx, RBy = get_right_and_left_body(Xx, Xy)
    Bx, By = get_body(Xx, Xy)

    # If the hand is not present in the video, we consider the entire arm as passive
    if all([np.all(LHx == 0), np.all(LHy == 0)]):
        PAx, PAy = reconstructing_passive_arm(Xx, Xy, video_name)
        return np.concatenate([Bx, PAx[:, 0:3], RBx, PAx[:, 3:], RHx], axis=1), np.concatenate([By, PAy[:, 0:3], RBy, PAy[:, 3:], RHy], axis=1)
    
    # Only evaluating the non dominant arm since this function will be called after the
    # switching of the array.

    Sx, Ex, Wx, Sy, Ey, Wy = Xx[:, 2], Xx[:, 3], Xx[:, 4], Xy[:, 2], Xy[:, 3], Xy[:, 4]

    # Getting the distance between the points
    SE, EW, WS = euclidean_distance(Sx, Sy, Ex, Ey), euclidean_distance(Ex, Ey, Wx, Wy), euclidean_distance(Wx, Wy, Sx, Sy)

    # Getting the angles between the points using the law of cosines
    E_angles = np.degrees(np.arccos((SE ** 2 + EW ** 2 - WS ** 2) / (2 * SE * EW)))
    passive = check_passive_arm_angle_variance(E_angles)

    # If the variance of the angles is low, we consider that the arm is passive
    if passive:
        # We want to confirm this assessment by checking if the arm is straight
        # and then how much the hand of the passive arm is moving away from its
        # original position.
        if np.all(E_angles >= 170) and np.all(E_angles <= 180): # If arm is straight
            # We confirm that the arm is passive and we reconstruct the arm
            PAx, PAy = reconstructing_passive_arm(Xx, Xy, video_name)
            LBx, LBy, LHx, LHy = PAx[:, 0:3], PAy[:, 0:3], PAx[:, 3:], PAy[:, 3:]
        else:
            # If the arm is not straight, we check the hand movement
            active = check_low_variance_active_arm_angles(Xx, Xy)
            if not active:
                PAx, PAy = reconstructing_passive_arm(Xx, Xy, video_name)
                LBx, LBy, LHx, LHy = PAx[:, 0:3], PAy[:, 0:3], PAx[:, 3:], PAy[:, 3:]
             
    # If the arm is not passive, we keep the original coordinates
    return np.concatenate([Bx, LBx, RBx, LHx, RHx], axis=1), np.concatenate([By, LBy, RBy, LHy, RHy], axis=1)


######### RECONSTRUCTING THE PASSIVE ARM ########

def one_by_one_reconstruction(x0, y0, angle, distance):


    x1 = x0 + distance * np.negative(np.cos(angle))
    y1 = y0 + distance * np.sin(angle)

    # Retrieve the coordinates of the first frame
    x1_1, y1_1 = x1[0], y1[0]

    # Duplicate it to the rest of the frames
    x1, y1 = np.full((x1.shape[0],), x1_1), np.full((y1.shape[0],), y1_1)
    
    # back to shape (T, )
    return x1[:, np.newaxis], y1[:, np.newaxis]

def reconstructing_passive_arm(Xx, Xy, video_name):

    # Getting the coordinates of the shoulders, elbows and wrists
    Nx, Ny = Xx[:, 1], Xy[:, 1]
    LSx, LSy, RSx, RSy = Xx[:, 2], Xy[:, 2], Xx[:, 5], Xy[:, 5]
    LEx, LEy, LWx, LWy = Xx[:, 3], Xy[:, 3], Xx[:, 4], Xy[:, 4]

    # Getting the mean distances between the shoulders, elbows and wrists
    SS, SE, EW = np.max(euclidean_distance(LSx, LSy, RSx, RSy)), np.max(euclidean_distance(LSx, LSy, LEx, LEy)), np.max(euclidean_distance(LEx, LEy, LWx, LWy))
      
    reconstruction_steps_body = [
        (Nx[:, np.newaxis], Ny[:, np.newaxis], np.pi, SS / 2),
        (None, None, 1.65, SE),
        (None, None, 1.65, EW),
        (None, None, 1.65, EW + 0.01)
    ]

    # Reconstructing the body
    reconstructed_points = []
    for i, (x0, y0, angle, distance) in enumerate(reconstruction_steps_body):
        if i == 0:
            # for the shoulder, we take the neck as the origin
            x0, y0 = one_by_one_reconstruction(x0, y0, angle, distance)
        elif i == 3:
            # If we do the hand wrist, we take the elbow as the origin (so two points before)
            x0, y0 = one_by_one_reconstruction(reconstructed_points[1][0], reconstructed_points[1][1], angle, distance)
        else:
            # Else, we take the previous point as the origin
            x0, y0 = one_by_one_reconstruction(reconstructed_points[-1][0], reconstructed_points[-1][1], angle, distance)
        reconstructed_points.append((x0, y0))
    
    # Reconstructing the hand
    reconstruction_steps_hand = {
        "thumb" : [5, 7, 9, 11],
        "index" : [3, 6, 7, 8],
        "middle" : [3, 6,7 , 8 ],
        "ring" : [3, 6 ,7 ,8],
        "pinky" : [3,6 ,7 ,8]
    }

    angles = [1.4, 1.5, 1.6, 1.7, 1.8]

    Hx, Hy = [], []
    i = 0
    for finger, ratios in reconstruction_steps_hand.items():
        # We start by reconstructing the base of the finger

        x, y = one_by_one_reconstruction(reconstructed_points[3][0], reconstructed_points[3][1], angles[i], SS / ratios[0])
        Hx.append(x)
        Hy.append(y)

        # We then reconstruct the rest of the finger
        for ratio in ratios[1:]:
            x, y = one_by_one_reconstruction(x, y, angles[i], SS / ratio)
            Hx.append(x)
            Hy.append(y)
        
        i += 1

    # Concatenate the reconstructed points
    # So we return a vector of 24 elements (3 for the body and 21 for the hand)
    return np.concatenate([point[0] for point in reconstructed_points] + Hx, axis=1), np.concatenate([point[1] for point in reconstructed_points] + Hy, axis=1)

########## MAIN FUNCTION #########

def get_corrected_2D_pose(all_skeleton, norm, interp, fps, video_name):

    """
    Get the corrected 2D pose estimation coordinates.
    """
    
    input_sequence = np.array(all_skeleton)
    logging.info(f"Input sequence shape : {input_sequence.shape}")
    Xx, Xy, Xw = get_coordinates_and_scores(input_sequence)

    logging.info("### Checking for parasite hands ###")
    left_hand_x, left_hand_y, right_hand_x, right_hand_y = get_right_and_left_hand(Xx, Xy) 
    left_hand_x, left_hand_y = detect_parasite_hand(left_hand_x, left_hand_y)
    right_hand_x, right_hand_y = detect_parasite_hand(right_hand_x, right_hand_y)
  
    Xx = np.concatenate([Xx[:, 0:2], Xx[:, 2:5], Xx[:, 5:8], left_hand_x, right_hand_x], axis=1)
    Xy = np.concatenate([Xy[:, 0:2], Xy[:, 2:5], Xy[:, 5:8], left_hand_y, right_hand_y], axis=1)
 
    if interp:
        logging.info("### Interpolating the coordinates ###")
        Xx, Xy, Xw = interpolation(Xx, Xy, Xw)
    
    logging.info(f"### Removing immobile points ###")
    Xx, Xy = remove_immobile_points(Xx, Xy)
    logging.info("### Dominant hand evaluation ###")
    Xx, Xy, Xw = switch_array_if_dominant_hand(Xx, Xy, Xw, fps)
    logging.info("### Passive arm evaluation ###")
    Xx, Xy = evaluate_passive_arm(Xx, Xy, video_name)

    if norm:
        logging.info("### Normalising the coordinates ###")
        Xx, Xy = anchor_based_normalisation(Xx, Xy)

    return Xx, Xy, Xw

