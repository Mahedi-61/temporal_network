"""
Author: Md Mahedi Hasan
Description: Preprocess pose sequence dataset to feed rnn model
Steps to do
        1. find out and sort partial body
        2. normalize keypoints
        3. handle no person, multiple person
        4. make train & validation dataset
"""

# python packages
import os
import json, glob
import numpy as np
from keras.utils import to_categorical

# project modules
from . import config
from . import hand_features as hf

# for motion features
first_frame_bkps = []

# formating json file
def handling_json_data_file(data):
    global first_frame_bkps
    frame_kps = []
    is_no_people = False
    is_partial_body = False
    
    # no people detected
    if len(data["people"]) == 0:
        is_no_people = True

    # one people detected 
    else:
        pose_keypoints = data["people"][0]["pose_keypoints_2d"]
        is_partial_body = hf.is_partial_body(pose_keypoints)

        # for complete pose
        if(not is_partial_body):
            #frame_kps = hf.normalize_keypoints(pose_keypoints)
            #frame_kps = hf.get_body_limb(pose_keypoints)
            frame_kps = hf.get_joint_angle(pose_keypoints)

            """
            # for first frame, store the  bpks and skip the motion feat.
            if(len(first_frame_bkps) == 0):
                first_frame_bkps = pose_keypoints
                is_no_people = True

            else:
                second_frame_bpks = pose_keypoints
                frame_kps = hf.get_motion_featurs(second_frame_bpks, 
                                                  first_frame_bkps)
                first_frame_bkps = second_frame_bpks
            """
    return frame_kps, is_no_people, is_partial_body



# dataset formatted for rnn input
def get_format_data(subject_id,
                    seq_kps,
                    start_id):

    seq_data = []
    seq_label = []
    
    # check how many image frame of length 28 we can get
    nb_images = len(seq_kps)

    # for larger than 15 image sequene creating one timestep
    if(nb_images < config.casiaA_nb_steps):
        if ((config.casiaA_nb_steps - nb_images) > (config.casiaA_nb_steps / 2)):
            nb_image_set = 0

        else:
            nb_image_set = 1
            seq_kps = seq_kps * 2
        
    else:
        nb_image_set = int((nb_images - config.casiaA_nb_steps) / config.actual_fps) + 1

    # finding lable of from subject data file
    sub_label = int(subject_id[1:]) - start_id
    print("has total image:", nb_images)
    print("         total number of image_set:", nb_image_set)

    # for some value of image_set
    if(nb_image_set > 0):

        for i in range(0, nb_image_set):
            start_frame_id = i * config.actual_fps
            end_frame_id = start_frame_id + config.casiaA_nb_steps

            # saving each keypoints
            for line in range(start_frame_id, end_frame_id):
                seq_data.append(seq_kps[line])
                seq_label.append([sub_label])

        seq_data = np.array(seq_data)
        seq_label = np.array(seq_label)

        seq_data = np.array(np.split(seq_data, nb_image_set))
        seq_label = np.array(np.split(seq_label, nb_image_set))

    return seq_data, seq_label



def get_keypoints_for_all_subject(subject_id_list,
                                  walking_seq,
                                  data_type,
                                  start_id):

    global first_frame_bkps
    print("\n\n*********** Generating %s data ***********" % data_type)    
    total_dataset = []
    total_dataset_label = []

    for subject_id in subject_id_list:
        print("\n\n\n\n############# subject id: %s #############" % subject_id)

        # variable for each subject
        sub_data = []
        sub_label = []

        angle_data = []
        angle_label = []
        
        sub_total_frame = 0
        sub_single_people = 0
        sub_total_no_people = 0
        sub_total_partial_body = 0

        # getting angle
        subject_dir = os.path.join(config.casiaA_pose_data_dir , subject_id)
        num_walking_seq=  len(walking_seq)
        print("%s subject have: %d walking gait vidoes" % (subject_id, num_walking_seq))
            
        # considering each gait sequence
        for seq in walking_seq:
            first_frame_bkps = []
            seq_dir = os.path.join(subject_dir, seq)

            # setting directory
            seq_kps = []
            os.chdir(seq_dir)

            # getting all json files
            json_files = sorted(glob.glob("*.json"))
            sub_total_frame += len(json_files)
                                
            for f, file in enumerate(json_files):
                with open(file) as data_file:
                    data = json.load(data_file)         
                    frame_kps, no_people, partial_body = handling_json_data_file(data)

                    #print("frame no: ", f+1); print(frame_kps)
                    # counting no, multiple people and partial body detected
                    if (no_people == True):  sub_total_no_people += 1
                    elif (partial_body == True): sub_total_partial_body += 1
                        
                    # for single people save the frame key points
                    else:
                        seq_kps.append(frame_kps)
                      
            # saving each seq walking data
            seq_data, seq_label = get_format_data(subject_id,
                                                seq_kps,
                                                start_id)
            
            # adding each angle all seq data and label except empty list
            if(seq_data != []):
                angle_data.append(seq_data)
                angle_label.append(seq_label)
        #### end of each gait sequence loop


        if (len(angle_data) != 0):
            sub_data = np.vstack(angle_data)
            sub_label = np.vstack(angle_label)

            print("subject data shape:", sub_data.shape)
            print("subject label shape:", sub_label.shape)

        # convert it to categorical value
        sub_label = to_categorical(sub_label, config.casiaA_nb_classes)
        #print(sub_label[0])

        # collecting all subject data
        total_dataset.append(sub_data)
        total_dataset_label.append(sub_label)
        
        # per subject display info
        print("\nsubject id", subject_id, "has") 
        
        print("total frame:", sub_total_frame)
        sub_single_people = sub_total_frame - (sub_total_no_people +
                                               sub_total_partial_body)
        
        print("suitable frame for detection:", sub_single_people)
        print("no people detected:", sub_total_no_people)
        print("partial people detected:", sub_total_partial_body)

        del sub_data, sub_label, angle_data, angle_label
        #### end of each subject work

    return total_dataset, total_dataset_label



if __name__ == "__main__":
    get_keypoints_for_all_subject(["p001"],
                                  ["90_3"],
                                  "train",
                                  1)