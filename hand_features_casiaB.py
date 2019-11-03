"""
Author: Md Mahedi Hasan
Description: Preprocess pose keypoints to find more gait features
Steps to do
        1. find each body joint coordinates
        2. find limb length
        3. find motion features

pose keypoints got 25 points (total = 75 elements, 25x3 (x, y and accuracy))
body join point = {
Neck(1), 
RShoulder(2), RElbow(3), RWrist(4),
LShoulder(5), LElbow(6), LWrist(7),
MHip(8), 
RHip(9), RKnee(10), RAnkle(11), RHeel(24), RBigToe(22), RSmallToe(23)
LHip(12), LKnee(13), LAnkle(14), LHeel(21), LBigToe(19), LSmallToe(20), 
Nose(0),
REye(15), LEye(16), 
REar(17), LEar(18), 
Back(25)}
"""

import numpy as np
import math

# nose
x_cor_nose = (0 * 3); y_cor_nose = (x_cor_nose + 1)

# neck
x_cor_neck = (1 * 3); y_cor_neck = (x_cor_neck + 1)

# hip
x_cor_r_hip = (9 * 3); y_cor_r_hip = (x_cor_r_hip + 1)
x_cor_mid_hip = (8 * 3); y_cor_mid_hip = (x_cor_mid_hip + 1)
x_cor_l_hip = (12 * 3); y_cor_l_hip = (x_cor_l_hip + 1)

# knee
x_cor_r_knee = (10 * 3); y_cor_r_knee = (x_cor_r_knee + 1)
x_cor_l_knee = (13 * 3); y_cor_l_knee = (x_cor_l_knee + 1)

# ankle
x_cor_r_ankle = (11 * 3); y_cor_r_ankle = (x_cor_r_ankle + 1)
x_cor_l_ankle = (14 * 3); y_cor_l_ankle = (x_cor_l_ankle + 1)

# BigToe
x_cor_r_btoe = (22 * 3); y_cor_r_btoe = (x_cor_r_btoe + 1)
x_cor_l_btoe = (19 * 3); y_cor_l_btoe = (x_cor_l_btoe + 1)

# Wrist
x_cor_r_wrist = (4 * 3); y_cor_r_wrist = (x_cor_r_wrist + 1)
x_cor_l_wrist = (7 * 3); y_cor_l_wrist = (x_cor_l_wrist + 1)

# Elbow
x_cor_r_elbow = (3 * 3); y_cor_r_elbow = (x_cor_r_elbow + 1)
x_cor_l_elbow = (6 * 3); y_cor_l_elbow = (x_cor_l_elbow + 1)

# Shoulder
x_cor_r_shoulder = (2 * 3); y_cor_r_shoulder = (x_cor_r_shoulder + 1)
x_cor_l_shoulder = (5 * 3); y_cor_l_shoulder = (x_cor_l_shoulder + 1)



#Trick to find partial body
def is_partial_body(body_kps):
    partial_body = False

    right_leg = body_kps[y_cor_r_ankle] - body_kps[y_cor_r_hip]
    left_leg =  body_kps[y_cor_l_ankle] - body_kps[y_cor_l_hip]

    # for partial body pose
    # print("right leg: ", right_leg); print("left leg: ", left_leg)
    if(right_leg <= 0 or left_leg <= 0): partial_body = True
    return partial_body



# normalize body keypoints according to PTSN paper         
def normalize_keypoints(body_kps):

    body_joint = [9, 10, 11, 12, 13, 14] 
    frame_kps = []
    
    # calculating distance between right_ankle and center of the hip
    unit_length =  body_kps[y_cor_mid_hip] - body_kps[y_cor_neck]
    
    # for complete body pose select joints
    for b_j in  body_joint:
        x_cor = b_j * 3
        y_cor = x_cor + 1
        
        # subtract join from the neck
        norm_x =   (body_kps[x_cor] - body_kps[x_cor_neck]) 
        norm_y =   (body_kps[y_cor] - body_kps[y_cor_neck]) 

        # without normalize
        frame_kps.append(norm_x)
        frame_kps.append(norm_y)

        # normalize
        #frame_kps.append(norm_x / unit_length)
        #frame_kps.append(norm_y / unit_length)

    return frame_kps


def get_distance(bkps, x1, y1, x2, y2):
    dist = np.sqrt((bkps[x1] - bkps[x2]) ** 2 + (bkps[y1] - bkps[y2]) ** 2)
    return dist


def get_body_limb(bkps):

    # feet
    r_feet = get_distance(bkps, x_cor_r_ankle, y_cor_r_ankle, 
                                x_cor_r_btoe, y_cor_r_btoe)

    l_feet = get_distance(bkps, x_cor_l_ankle, y_cor_l_ankle, 
                                x_cor_l_btoe, y_cor_l_btoe)

    # foot
    r_foot = get_distance(bkps, x_cor_r_ankle, y_cor_r_ankle, 
                                x_cor_r_knee, y_cor_r_knee)


    l_foot = get_distance(bkps, x_cor_l_ankle, y_cor_l_ankle, 
                                x_cor_l_knee, y_cor_l_knee)

    # run
    r_run = get_distance(bkps, x_cor_r_knee, y_cor_r_knee, 
                               x_cor_r_hip, y_cor_r_hip)

    l_run = get_distance(bkps, x_cor_l_knee, y_cor_l_knee, 
                               x_cor_l_hip, y_cor_l_hip)

    # body
    r_body = get_distance(bkps, x_cor_r_hip, y_cor_r_hip, 
                                x_cor_neck, y_cor_neck)

    l_body = get_distance(bkps, x_cor_l_hip, y_cor_l_hip, 
                                x_cor_neck, y_cor_neck)


    # hand
    r_hand = get_distance(bkps, x_cor_r_wrist, y_cor_r_wrist, 
                                x_cor_r_elbow, y_cor_r_elbow)

    l_hand = get_distance(bkps, x_cor_l_wrist, y_cor_l_wrist, 
                                x_cor_l_elbow, y_cor_l_elbow)

    # arm
    r_arm = get_distance(bkps, x_cor_r_elbow, y_cor_r_elbow, 
                               x_cor_r_shoulder, y_cor_r_shoulder)

    l_arm = get_distance(bkps, x_cor_l_elbow, y_cor_l_elbow, 
                               x_cor_l_shoulder, y_cor_l_shoulder)


    nose_to_neck = get_distance(bkps, x_cor_neck, y_cor_neck, 
                                      x_cor_nose, y_cor_nose)

    hip = get_distance(bkps, x_cor_r_hip, y_cor_r_hip,
                             x_cor_l_hip, y_cor_l_hip)

    l_shoulder = get_distance(bkps, x_cor_neck, y_cor_neck, 
                                    x_cor_l_shoulder, y_cor_l_shoulder)

    r_shoulder = get_distance(bkps, x_cor_neck, y_cor_neck, 
                                    x_cor_r_shoulder, y_cor_r_shoulder)

    pose_limb = [r_foot, l_foot,
                  r_run, l_run, r_body, l_body,
                  r_hand, l_hand, r_arm, l_arm,
                  l_shoulder, r_shoulder, nose_to_neck, hip]
    

    return pose_limb



def get_motion_featurs(bkps_2, bkps_1):
    body_joint = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14] 
    motion_features = []

    # for complete body pose of selected joints
    for b_j in  body_joint:
        x_cor = b_j * 3
        y_cor = x_cor + 1
        
        motion_x =  (bkps_2[x_cor] - bkps_1[x_cor]) 
        motion_y =  (bkps_2[y_cor] - bkps_1[y_cor]) 

        # motion features
        motion_features.append(motion_x)
        motion_features.append(motion_y)

    return motion_features


def get_joint_angle(bkps):
    # first point lower, second point higher values
    joint_pair = [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), 
                  (5, 6), (6, 7), (1, 9), (1, 12), (9, 10),
                  (10, 11), (12, 13), (13, 14)]
    
    angle_features = []
    for pair in joint_pair:
        del_x = bkps[(pair[1] * 3)] - bkps[(pair[0] * 3)] 
        del_y = bkps[(pair[1] * 3) + 1] - bkps[(pair[0] * 3) + 1] 

        if (del_x == 0):
            angle_features.append(math.pi / 2)
        else:
            angle_features.append(math.atan(del_y / del_x))

    return angle_features