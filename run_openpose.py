"""
Author: Md Mahedi Hasan
Description: this file collect each subject's pose information using openpose library
"""

# python packages
import os
import numpy as np

# project modules
from .. import root_dir
from . import config


# declaring path variable and constants
#input_dir = os.path.join(root_dir.data_path(), "CasiaA_frames")

input_dir = os.path.join(root_dir.data_path(), "CasiaB_frames")


# running openpose for casiaA dataset
def run_openpose_casiaA(subject_id_list):

    # considering each subject
    for subject_id in subject_id_list:

        subject_dir = os.path.join(input_dir, subject_id)
        seq_list = sorted(os.listdir(subject_dir), key = lambda x: int(x[0:2]))

        num_seq =  len(seq_list)
        print("\n\n%s subject have: %d gait sequence vidoes" % (subject_id, num_seq))

        # considering each gait sequence
        for seq in seq_list:
            seq_dir = os.path.join(subject_dir, seq)
            
            # save_dir for saving pose keypoints data
            save_dir = os.path.join(config.casiaA_pose_data_dir, subject_id, seq)
            os.makedirs(save_dir, exist_ok = True)

            # setting openpose directory
            os.chdir(config.openpose_dir)

            print("\ncalculationg pose...")
            os.system("./build/examples/openpose/openpose.bin --image_dir " +  
                        seq_dir + " --number_people_max 1 " + " --write_json " +  
                        save_dir + " --display 0 --render_pose 0")



# running openpose for casiaB dataset
def run_openpose_casiaB(subject_id_list):

    # considering each subject
    for subject_id in subject_id_list:

        subject_dir = os.path.join(input_dir, subject_id)
        angle_list = sorted(os.listdir(subject_dir), key = lambda x: int(x[-3:]))

        num_angle =  len(angle_list)
        print("\n\n%s subject have: %d angle gait vidoes" % (subject_id, num_angle))

        # considering each angle
        for angle in angle_list:
            subject_angle_dir = os.path.join(subject_dir, angle)
            seq_list = sorted(os.listdir(subject_angle_dir))

            num_seq = len(seq_list)
            print("%s angle have %d gait sequence" % (angle, num_seq))

            # considering each gait sequence
            for seq in seq_list:
                seq_dir = os.path.join(subject_angle_dir, seq)

                # save_dir for saving pose keypoints data
                save_dir = os.path.join(config.casiaB_pose_data_dir, subject_id, angle, seq)
                os.makedirs(save_dir, exist_ok = True)

                # setting openpose directory
                os.chdir(config.openpose_dir)

                print("\ncalculationg pose...")

                os.system("./build/examples/openpose/openpose.bin --image_dir " +  
                        seq_dir + " --number_people_max 1 " + " --write_json " +  
                        save_dir + " --display 0 --render_pose 0")


# getting pose data for all available subject using openpose library
def get_pose_data():
    
    # calculating total number of person have gait videos
    num_subject = len(os.listdir(input_dir))
    print("\ntotal number subjects: ", num_subject)

    total_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))
    print(total_id_list)

    print("\ngallery subject id list: 25 to 124")
    gallery_subject_id_list = total_id_list[1:4]
    print(gallery_subject_id_list)

    run_openpose_casiaB(gallery_subject_id_list)


# run here
if __name__ == "__main__":
    get_pose_data()