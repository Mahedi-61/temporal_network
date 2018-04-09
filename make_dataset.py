"""this file make pose sequence dataset to feed rnn model"""

# python packages
import os
import json, glob
import numpy as np
from keras.utils import to_categorical


# project modules
from . import config


# path variables and constant
nb_steps = config.nb_steps
actual_fps = config.actual_fps
nb_features = config.nb_features



# normalize body keypoints according to PTSN paper         
def normalize_keypoints(body_kps, is_aug = False):
    
    """pose keypoints got 18 points (total = 54 elements, 18x3 (x, y and accuracy))
    body join point = {Nose(0), Neck(1), RShoulder(2), RElbow(3), RWrist(4),
    LShoulder(5), LElbow(6), LWrist(7), RHip(8), RKnee(9), RAnkle(10),
    LHip(11), LKnee(12), LAnkle(13), REye(14), LEye(15), REar(16), LEar(17), Back(18)}
    """
    
    frame_kps = []
    partial_body = False
    
    # finding neck(1)
    x_cor_neck = body_kps[(1 * 3)]
    y_cor_neck = body_kps[(1 * 3) + 1]

    # finding center of the hip from (8, 11)
    x_center_hip = (body_kps[(8 * 3)] + body_kps[(11 * 3)]) / 2
    y_center_hip = (body_kps[(8 * 3) + 1] + body_kps[(11 * 3) + 1]) / 2

    # calculating distance between neck and center of the hip
    h = y_center_hip - y_cor_neck

    # for partial body pose
    if(h == 0):
        partial_body = True

    # for complete body pose
    elif(h != 0):
        body_joint = [2, 5, 9, 10,  12, 13]  # selected body joints

        for b_j in  body_joint:
            x_cor = b_j * 3
            y_cor = x_cor + 1
            
            #frame_kps.append((body_kps[x_cor] - x_cor_neck) / h)
            #frame_kps.append((body_kps[y_cor] - y_cor_neck) / h)

            frame_kps.append(body_kps[x_cor])
            frame_kps.append(body_kps[y_cor])


        # aumentation
        if(is_aug == True):
            
            # right
            frame_kps[0] = frame_kps[0] - 1
            frame_kps[1] = frame_kps[1] - 1

            # left
            frame_kps[2] = frame_kps[2] + 1
            frame_kps[3] = frame_kps[3] - 1
        
    return frame_kps, partial_body





# formating json file
def handling_json_data_file(data, is_aug):
    
    frame_kps = []
    is_multiple_people = False
    is_no_people = False
    is_partial_body = False
    
    # no people detected
    if len(data["people"]) == 0:
        is_no_people = True
     

    # more thant one people detected
    elif len(data["people"]) > 1: 
            is_multiple_people = True
            # print (file)

            
    # one people detected 
    else:
        pose_keypoints = data["people"][0]["pose_keypoints_2d"]
        frame_kps, is_partial_body = normalize_keypoints(pose_keypoints, is_aug)
        
    return frame_kps, is_no_people, is_multiple_people, is_partial_body





# dataset formatted for rnn input
def get_format_data(subject_id,
                    seq_kps,
                    seq,
                    start_id):


    seq_data = []
    seq_label = []
    
    # check how many image frame of length 32 we can get
    nb_images = len(seq_kps)

    # trick for this faulty dataset sub: 109 has miss some angle
    # for larger than 15 image sequene creating one timestep
    if(nb_images < nb_steps):
        if ((nb_steps - nb_images) > 16):
            nb_image_set = 0

        else:
            nb_image_set = 1
            seq_kps = seq_kps * 2
        
    else:
        nb_image_set = int((nb_images - nb_steps) / actual_fps) + 1


    # finding lable of from subject data file
    sub_label = int(subject_id[1:]) - start_id

    print(seq, "has total image:", nb_images)
    print("         total number of image_set:", nb_image_set)

    # for some value of image_set
    if(nb_image_set > 0):
        
        for i in range(0, nb_image_set):
            start_id = i * actual_fps
            end_id = start_id + nb_steps

            # saving each keypoints
            for line in range(start_id, end_id):
                seq_data.append(seq_kps[line])
                seq_label.append([sub_label])
        
        blocks = int(len(seq_data) / nb_steps)

        seq_data = np.array(seq_data)
        seq_label = np.array(seq_label)
        
        seq_data = np.array(np.split(seq_data, blocks))
        seq_label = np.array(np.split(seq_label, blocks))

    return seq_data, seq_label





def get_keypoints_for_all_subject(subject_id_list,
                                  walking_seq,
                                  data_type,
                                  start_id,
                                  angle_list):

    print("\n\n*********** Generating %s data ***********" % data_type)    
    total_detected_frame = 0
    total_dataset = []
    total_dataset_label = []

    for subject_id in subject_id_list:
        
        print("\n\n\n\n############# subject id: %s #############" % subject_id)

        # variable for each subject
        sub_data = []
        sub_label = []
        
        sub_total_frame = 0
        sub_single_people = 0
        sub_total_multiple_people = 0
        sub_total_no_people = 0
        sub_total_partial_body = 0

        # getting angle
        subject_dir = os.path.join(config.pose_data_dir, subject_id)
        num_angle =  len(angle_list)
        print("%s subject have: %d angle gait vidoes" % (subject_id, num_angle))
        
        # considering each angle
        for angle in angle_list:
            subject_angle_dir = os.path.join(subject_dir, angle)
            angle_data_list = []
            angle_label_list = []
            print("\n********** angle:", angle, "********** ")
            
            # considering each gait sequence
            for seq in walking_seq:                      
                if(seq == "aug_nm04"):
                    is_aug = True
                    seq_dir = os.path.join(subject_angle_dir, "nm04")
                    
                else:
                    seq_dir = os.path.join(subject_angle_dir, seq)
                    is_aug = False
                    
                seq_kps = []

                # setting directory
                os.chdir(seq_dir)

                # getting all json files
                json_files = sorted(glob.glob("*.json"))
                sub_total_frame += len(json_files)
                                   
                for file in json_files:
                    with open(file) as data_file:
                        data = json.load(data_file)
                        
                        frame_kps, no_people, multiple_people, partial_body = handling_json_data_file(data, is_aug)

                        # counting no, multiple people and partial body detected
                        if (no_people == True):
                            sub_total_no_people += 1
                            
                        elif (multiple_people == True):
                            sub_total_multiple_people += 1
                  
                        elif (partial_body == True):
                            sub_total_partial_body += 1
                            
                        # for single people save the frame key points
                        else:
                             # add all frame kps of all seq in an angle_kps
                            seq_kps.append(frame_kps)                            

                # saving each seq walking data
                seq_data, seq_label = get_format_data(subject_id,
                                                    seq_kps,
                                                    seq,
                                                    start_id)
                
                # adding each angle all seq data and label except empty list
                if(seq_data != []):
                    angle_data_list.append(seq_data)
                    angle_label_list.append(seq_label)
                
            # saving each angle walking data except empy list
            if (len(angle_data_list) != 0):
                angle_data = np.vstack(angle_data_list)
                angle_label = np.vstack(angle_label_list)

                print("angle data shape:", angle_data.shape)
                print("angle label shape:", angle_label.shape)

                sub_data.append(angle_data)
                
                # convert it to categorical value
                sub_label.append(to_categorical(angle_label, config.nb_classes))

            
        # collecting all subject data
        total_dataset.append(sub_data)
        total_dataset_label.append(sub_label)
        
        # per subject display info
        print("\nsubject id:", subject_id, "has total image set:",
              sum(len(i) for i in sub_data))
                
        print("total frame:", sub_total_frame)
        sub_single_people = sub_total_frame - (sub_total_no_people +
                                               sub_total_multiple_people)
        
        total_detected_frame += sub_single_people

        print("one people detected:", sub_single_people)
        print("no people detected:", sub_total_no_people)
        print("multiple people detected:", sub_total_multiple_people)
        print("partial people detected:", sub_total_partial_body)
        #### end of each subject work

    return total_dataset, total_dataset_label







