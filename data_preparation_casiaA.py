""" 
Author: Md Mahedi Hasan
Description: Preprocess subject pose information to gait descriptors
"""

# python packages
import numpy as np
import os

# project files
from . import config
from . import make_dataset

# path variables and constant
actual_fps = config.actual_fps



# making training and validation dataset
def set_dataset(data_type, angle = None):

    # calculating total number of person having gait videos
    num_subject = len(os.listdir(config.pose_data_dir))
    print("\ntotal number subjects: ", num_subject)

    total_id_list = sorted(os.listdir(config.pose_data_dir), key = lambda x: int(x[1:]))

    print("subject id list: 63 to 124")
    subject_id_list = total_id_list[:62]
    print(subject_id_list)

    # for label synchronization
    start_id = 63

    # train dataset
    if (data_type == "train"):
        data, label =  make_dataset.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.ls_gallery_train_seq,
                                        data_type,
                                        start_id,
                                        [angle])

    # validation dataset
    elif (data_type == "valid"):
        data, label =  make_dataset.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.ls_gallery_valid_seq,
                                        data_type,
                                        start_id,
                                        [angle])

    # probe-normal test set
    elif (data_type == "nm"):
        data, label  = make_dataset.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.ls_probe_nm_seq,
                                        data_type,
                                        start_id,
                                        config.angle_list)

    # probe-bag test set
    elif (data_type == "bg"):
        data, label  = make_dataset.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.ls_probe_bg_seq,
                                        data_type,
                                        start_id,
                                        config.angle_list)

    # probe-coat test set
    elif (data_type == "cl"):
        data, label  = make_dataset.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.ls_probe_cl_seq,
                                        data_type,
                                        start_id,
                                        config.angle_list)
        
    return data, label



# methods for returning unstateful training and validation data
def load_train_data_per_angle(angle):
    print("\nstart preprocessing training data")

    data, label = set_dataset("train", angle)
    # finding angle which contains maximum timesteps
    l = []
    for k in range(config.nb_classes):
        l.append(data[k][0].shape[0]) 

    #print(l)
    max_ts = max(l)
    print("\nmaximum timesteps is:", max_ts)

    X_train = np.ndarray(((max_ts * config.nb_classes),
                          config.nb_steps,
                          config.nb_features), dtype = np.float32)

    y_train = np.ndarray(((max_ts * config.nb_classes),
                          config.nb_steps,
                          config.nb_classes), dtype = np.float32)


   
    for i in range(config.nb_classes):
        for ts in range(max_ts):
            
            index = (i * max_ts) +  ts
            sub_angle_data =  data[i][0]

            # prepare label
            y_train[index] = label[i][0][0]
        
            # prepare data
            if(ts < sub_angle_data.shape[0]):
                X_train[index] = sub_angle_data[ts]

            # repeat to to make stateful for angle which got ts lower than max_ts
            # % for multiple repeat
            elif(ts >= sub_angle_data.shape[0]):
                j = ts % sub_angle_data.shape[0]
                
                #print(ts, "not ok, j:", j)
                X_train[index] = sub_angle_data[j]
    
    return X_train, y_train



def load_valid_data_per_angle(angle):
    
    print("\nstart preprocessing validation data")
    data, label = set_dataset("valid", angle)

    # logic for converting list to numpy array
    X_data = data[0][0]
    y_label = label[0][0]

    # adding all subjects sequence for that angle
    for s in range(config.nb_classes):
        if(not (s == 0)):
            X_data = np.append(X_data, data[s][0], axis = 0)
            y_label = np.append(y_label, label[s][0], axis = 0)

    return X_data, y_label
    


# methods for returning unstateful training and validation data
def load_probe_data(data_type):
    print("\nstart preprocessing %s data" % data_type)
    
    data, label = set_dataset(data_type)
    
    # storing all probeset angle numpy array into a list
    X_probe = []
    y_probe = []

    # logic for converting list to numpy array
    for i in range(len(config.angle_list)):
        for s in range(config.nb_classes):
            if(s == 0):
                X_data = data[0][i]
                y_label = label[0][i]
        
            elif(not (s == 0)):
                X_data  = np.append(X_data, data[s][i], axis = 0)
                y_label = np.append(y_label, label[s][i], axis = 0)
        
        X_probe.append(X_data)
        y_probe.append(y_label)

        del X_data, y_label

    return X_probe, y_probe



if __name__ == "__main__":
    data, l = load_train_data_per_angle("angle_036")
    print(data[0][0])