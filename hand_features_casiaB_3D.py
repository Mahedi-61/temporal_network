"""
Author: Md Mahedi Hasan
Description: Preprocess pose keypoints to find more gait features
Steps to do
        1. find each body joint coordinates
        2. find limb length
        3. find motion features

pose keypoints got 32 joints (total = 96 elements, 31x3 (x, y and z))
body join point = {
Neck(14, 13, 16, 24), 
RShoulder(25), RElbow(26), RWrist(27),
LShoulder(17), LElbow(18), LWrist(19),
MHip(0, 11), 
RHip(1), RKnee(2), RAnkle(3),
LHip(6), LKnee(7), LAnkle(8)
Nose(15),
heart (20, 21, 22, 23, 28,  29,30, 31))}
"""

import numpy as np
import math

# neck
x_cor_neck = (14 * 3) 
y_cor_neck = (14 * 3) + 1
z_cor_neck = (14 * 3) + 2


# right hip
x_cor_r_hip = (1 * 3) 
y_cor_r_hip = (1 * 3) + 1
z_cor_r_hip = (1 * 3) + 2


# mid hip
x_cor_mid_hip = (0 * 3)
y_cor_mid_hip = (0 * 3) + 1
z_cor_mid_hip = (0 * 3) + 2


# left hip
x_cor_l_hip = (6 * 3)
y_cor_l_hip = (6 * 3) + 1
z_cor_l_hip = (6 * 3) + 2

# right ankle
x_cor_r_ankle = (3 * 3) 
y_cor_r_ankle = (3 * 3) + 1
z_cor_r_ankle = (3 * 3) + 2

# left ankle
x_cor_l_ankle = (8 * 3)
y_cor_l_ankle = (8 * 3) + 1
z_cor_l_ankle = (8 * 3) + 2



# Trick to find partial body
def is_partial_body(body_kps):
    partial_body = False

    right_leg = body_kps[y_cor_r_ankle] - body_kps[y_cor_r_hip]
    left_leg =  body_kps[y_cor_l_ankle] - body_kps[y_cor_l_hip]

    # for partial body pose
    # print("right leg: ", right_leg); print("left leg: ", left_leg)
    if(right_leg <= 0.0 or left_leg <= 0.0): partial_body = True
    return partial_body



# normalize body keypoints according to PTSN paper         
def normalize_keypoints(body_kps):

    body_joint = [1, 2, 3, 6, 7, 8, 23, 29, 13, 14] 
    frame_kps = []
    
    # calculating distance between right_ankle and center of the hip
    unit_length =  body_kps[y_cor_mid_hip] - body_kps[y_cor_neck]
    
    # for complete body pose select joints
    for b_j in  body_joint:
        x_cor = b_j * 3
        y_cor = (b_j * 3) + 1
        z_cor = (b_j * 3) + 2
        
        # subtract join from the neck
        norm_x =  (body_kps[x_cor])  #- body_kps[x_cor_mid_hip]) 
        norm_y =  (body_kps[y_cor]) #- body_kps[y_cor_mid_hip])
        norm_z =  (body_kps[z_cor]) #- body_kps[z_cor_mid_hip])

        # without normalize
        frame_kps.append(norm_x)
        frame_kps.append(norm_y)
        frame_kps.append(norm_z)

        # normalize
        #frame_kps.append(norm_x / unit_length)
        #frame_kps.append(norm_y / unit_length)
        #frame_kps.append(norm_z / unit_length)

    return frame_kps

"""
def get_distance(bkps, x1, y1, x2, y2):
    dist = np.sqrt((bkps[x1] - bkps[x2]) ** 2 + (bkps[y1] - bkps[y2]) ** 2)
    return dist


def get_body_limb(bkps):

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

    unit_length = get_distance(bkps, x_cor_neck, y_cor_neck,
                                     x_cor_mid_hip, y_cor_mid_hip)


    pose_limb = [r_feet, l_feet, r_foot, l_foot,
                 nose_to_neck, r_run, l_run]
    
    return pose_limb



def get_motion_featurs(bkps_2, bkps_1):
    body_joint = [0, 1, 8, 10, 11, 13, 14, 22, 19] 
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
    joint_pair = [(0, 1), (10, 11), (13, 14), (11, 22), (1, 2), (1, 5),
                (9, 10), (12, 13), (14, 19)]
    
    angle_features = []
    for pair in joint_pair:
        del_x = bkps[(pair[1] * 3)] - bkps[(pair[0] * 3)] 
        del_y = bkps[(pair[1] * 3) + 1] - bkps[(pair[0] * 3) + 1] 

        if (del_x == 0):
            angle_features.append(math.pi / 2)
        else:
            angle_features.append(math.atan(del_y / del_x))

    return angle_features
"""