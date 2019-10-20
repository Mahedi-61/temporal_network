""" 
Author: Md Mahedi Hasan
Description: Preprocess subject pose information to gait descriptors
"""

# python packages
import numpy as np
import os

# project files
from . import config
from . import make_dataset_casiaA

# path variables and constant
actual_fps = config.actual_fps
probe_angle = ["0", "45", "90"]


# making training and validation dataset
def set_dataset(data_type, angle):

    # calculating total number of person having gait videos
    num_subject = len(os.listdir(config.casiaA_pose_data_dir))
    print("\ntotal number subjects: ", num_subject)

    total_id_list = sorted(os.listdir(config.casiaA_pose_data_dir), 
                            key = lambda x: int(x[1:]))
    print(total_id_list)

    # for label synchronization
    start_id = 1
    train_seq, valid_seq = config.get_casiaA_train_valid_seq(angle)
    
    # train dataset
    if (data_type == "train"):
        data, label =  make_dataset_casiaA.get_keypoints_for_all_subject(
                                        total_id_list,
                                        train_seq,
                                        data_type,
                                        start_id)

    # validation & test dataset
    elif (data_type == "valid" or data_type == "test"):
        data, label =  make_dataset_casiaA.get_keypoints_for_all_subject(
                                        total_id_list,
                                        valid_seq,
                                        data_type,
                                        start_id)        
    return data, label



# methods for returning unstateful training and validation data
def load_train_data_with_equal_ts(angle):
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



def load_data(data_type, angle):
    
    print("\nstart preprocessing %s data" % data_type)
    data, label = set_dataset(data_type, angle)

    # logic for converting list to numpy array
    X_data = data[0]
    y_label = label[0]

    # adding all subjects sequence for that angle
    for s in range(config.casiaA_nb_classes):
        if(not (s == 0)):
            X_data = np.append(X_data, data[s], axis = 0)
            y_label = np.append(y_label, label[s], axis = 0)

    return X_data, y_label



def load_probe_data():
    X_test = []
    y_test = []

    for angle in probe_angle:
        data, label = load_data("test", angle)
        X_test.append(data)
        y_test.append(label)

    return X_test, y_test


if __name__ == "__main__":
    X_valid, y_valid = load_data("train", "90")