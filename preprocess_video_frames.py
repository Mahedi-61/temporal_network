"""
# Project: gait_recognition
# Author : Md. Mahedi Hasan
# File : process_video_frames.py
# Description : detecting and cropping bounding boxes on the input training images using darknet
"""

# python packages
import os
import subprocess
import numpy


# project modules
from .. import root_dir


# declaring path variable and files
input_dir =  os.path.join(root_dir.data_path(), "video_frames")
output_dir = os.path.join(root_dir.data_path(), "crop_img")

cache_dir = os.path.join(root_dir.data_path(), "cache_files")
darknet_dir = os.path.join(root_dir.libs_path(), "darknet")



img_box = os.path.join(cache_dir, "bound_box_cor.txt")
img_crop = os.path.join(cache_dir, "crop_image_cor.txt")
ls_detected_img = os.path.join(cache_dir, "ls_detected_img.txt")
ls_input_img = os.path.join(cache_dir, "ls_input_img.txt")




# deleteing previous existing file
def remove_previous_cache_files():
    if (os.path.isfile(img_box)): os.remove(img_box)
    if (os.path.isfile(img_crop)): os.remove(img_crop)
    if (os.path.isfile(ls_detected_img)): os.remove(ls_detected_img)
    if (os.path.isfile(ls_input_img)): os.remove(ls_input_img)




# write list of available input image in a file for detecting in darknet
def get_list_input_img_for_all_subjejects(subject_id_list):
    
    # open file to write all input images for darkent to detect
    file_img_dir = open(ls_input_img, 'w')

    # considering each subject
    for subject_id in subject_id_list:
        
        subject_dir = os.path.join(input_dir, subject_id)
        angle_list = sorted(os.listdir(subject_dir), key = lambda x: int(x[-3:]))
        num_angle =  len(angle_list)
        print("\n%s subject have: %d angle gait vidoes" % (subject_id, num_angle))


        # considering each angle
        for angle in angle_list:
            subject_angle_dir = os.path.join(subject_dir, angle)
            seq_list = sorted(os.listdir(subject_angle_dir))
            
            num_seq = len(seq_list)
            print("%s angle have %d gait sequence" % (angle, num_seq))


            # considering each gait sequence
            for seq in seq_list:
                seq_dir = os.path.join(subject_angle_dir, seq)
                input_img_list = sorted(os.listdir(seq_dir), key = lambda x: int(x.split(".")[0]))

                for input_img in input_img_list:
                    input_img_dir = os.path.join(seq_dir, input_img)
                    file_img_dir.write(input_img_dir + "\n")

    file_img_dir.close()





def detect_people_using_darknet():

    # setting darkent
    os.chdir(darknet_dir)

    os.system("./darknet detect cfg/yolo.cfg weights/yolo.weights -thresh 0.70 <" + ls_input_img)
    print("Detection complete for all images ")





def crop_detected_images():
    
    # detected subject image list
    detected_image_list = []
    with open(ls_detected_img, "r") as fp:
        for line in fp:
            line = line.strip('\n')
            detected_image_list.append(line) 



    # collecing cropping co-ordinates from img_crop file
    cropping_cor_list = []
    with open(img_crop, "r") as fp:
        for line in fp:
            line = line.strip('\n')
            cropping_cor_list.append(line) 



    # checking where one person per image is detected or not
    if (len(detected_image_list) == len(cropping_cor_list)):
        print("cropping images using imagemagick tool")

        for i, input_img in enumerate(detected_image_list):
            img_no = input_img.split("/")[-1]
            seq_no = input_img.split("/")[-2]
            angle_no = input_img.split("/")[-3]
            subject_id = input_img.split("/")[-4]

            # builing output image directory
            out_img_dir = os.path.join(output_dir, subject_id, angle_no, seq_no)
            os.makedirs(out_img_dir, exist_ok = True)
            output_img = os.path.join(out_img_dir, img_no)
            
            # using imagemagic tool to cropping and centering
            print("converting: ", input_img) 
            os.system("convert " + input_img + " -crop " + cropping_cor_list[i] + " "  + output_img)

    else:
        print("something is wrong !!")







########################### main work here #############################

# calculating total number of person have gait videos
num_subject = len(os.listdir(input_dir))
print("total number subjects: ", num_subject)

subject_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))
print(subject_id_list)


remove_previous_cache_files()
get_list_input_img_for_all_subjejects(subject_id_list)
detect_people_using_darknet()
crop_detected_images()


