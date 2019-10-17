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

    body_joint = [9, 10, 11, 12, 13, 14, 19, 21, 22, 24] 
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

        # normalize
        frame_kps.append(norm_x)
        frame_kps.append(norm_y)

        # without normalize
        #frame_kps.append(body_kps[x_cor])
        #frame_kps.append(body_kps[y_cor])

    return frame_kps



def get_body_limb(bkps):

    r_feet = np.sqrt((bkps[x_cor_r_ankle] - bkps[x_cor_r_btoe]) ** 2 + 
                     (bkps[y_cor_r_ankle] - bkps[y_cor_r_btoe]) ** 2)

    l_feet = np.sqrt((bkps[x_cor_l_ankle] - bkps[x_cor_l_btoe]) ** 2 + 
                     (bkps[y_cor_l_ankle] - bkps[y_cor_l_btoe]) ** 2)


    r_foot = np.sqrt((bkps[x_cor_r_ankle] - bkps[x_cor_r_knee]) ** 2 + 
                     (bkps[y_cor_r_ankle] - bkps[y_cor_r_knee]) ** 2)


    l_foot = np.sqrt((bkps[x_cor_l_ankle] - bkps[x_cor_l_knee]) ** 2 + 
                     (bkps[y_cor_l_ankle] - bkps[y_cor_l_knee]) ** 2)


    r_run = np.sqrt((bkps[x_cor_r_knee] - bkps[x_cor_r_hip]) ** 2 + 
                     (bkps[y_cor_r_knee] - bkps[y_cor_r_hip]) ** 2)


    l_run = np.sqrt((bkps[x_cor_l_knee] - bkps[x_cor_l_hip]) ** 2 + 
                     (bkps[y_cor_l_knee] - bkps[y_cor_l_hip]) ** 2)


    r_body = np.sqrt((bkps[x_cor_r_hip] - bkps[x_cor_neck]) ** 2 + 
                     (bkps[y_cor_r_hip] - bkps[y_cor_neck]) ** 2)


    l_body = np.sqrt((bkps[x_cor_l_hip] - bkps[x_cor_neck]) ** 2 + 
                     (bkps[y_cor_l_hip] - bkps[y_cor_neck]) ** 2)

    frame_limb = [r_feet, l_feet, r_foot, l_foot, r_run, l_run, r_body, l_body]
    
    return frame_limb



def get_motion_featurs(bkps_2, bkps_1):
    body_joint = [9, 10, 11, 12, 13, 14, 19, 21, 22, 24] 
    motion_featurs = []

    # for complete body pose of selected joints
    for b_j in  body_joint:
        x_cor = b_j * 3
        y_cor = x_cor + 1
        
        motion_x =  (bkps_2[x_cor] - bkps_1[x_cor]) 
        motion_y =  (bkps_2[y_cor] - bkps_1[y_cor]) 

        # motion features
        motion_featurs.append(motion_x)
        motion_featurs.append(motion_y)

    return motion_featurs