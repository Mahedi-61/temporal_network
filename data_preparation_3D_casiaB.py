""" this file prepare subject pose data for temporal network"""

# python packages
import numpy as np
import os

# project files
from . import config
from . import make_dataset_casiaB

# calculating total number of person having gait videos
num_subject = len(os.listdir(config.casiaB_pose_data_dir))
print("\ntotal number subjects: ", num_subject)

total_id_list = sorted(os.listdir(config.casiaB_pose_data_dir), 
                key = lambda x: int(x[1:]))

print("subject id list: 25 to 124")
subject_id_list = total_id_list
print(subject_id_list)

# for label synchronization
start_id = 25


# methods for returning unstateful training and validation data
def load_train_data_per_angle(angle):
    print("\nstart preprocessing training data")

    data, label = make_dataset_casiaB.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.casiaB_ls_gallery_train_seq,
                                        "train",
                                        start_id,
                                        [angle])

    # finding angle which contains maximum timesteps
    l = []
    for k in range(config.casiaB_nb_classes):
        l.append(data[k][0].shape[0]) 

    #print(l)
    max_ts = max(l)
    print("\nmaximum timesteps is:", max_ts)

    X_train = np.ndarray(((max_ts * config.casiaB_nb_classes),
                          config.casiaB_nb_steps,
                          config.casiaB_nb_features), dtype = np.float32)

    y_train = np.ndarray(((max_ts * config.casiaB_nb_classes),
                          config.casiaB_nb_steps,
                          config.casiaB_nb_classes), dtype = np.float32)

    for i in range(config.casiaB_nb_classes):
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
    data, label = make_dataset_casiaB.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.casiaB_ls_gallery_valid_seq,
                                        "valid",
                                        start_id,
                                        [angle])

    # logic for converting list to numpy array
    # data[subject][angle] -> 3D numpy array (# of ts, image_no, features)
    X_data = data[0][0]
    y_label = label[0][0]

    # adding all subjects sequence for that angle
    for s in range(config.casiaB_nb_classes):
        if(not (s == 0)):
            X_data = np.append(X_data, data[s][0], axis = 0)
            y_label = np.append(y_label, label[s][0], axis = 0)

    return X_data, y_label
    


# methods for returning unstateful probe data
def load_probe_data(data_type):
    print("\nstart preprocessing %s data" % data_type)
    
    # probe-normal test set
    if (data_type == "nm"):
        data, label  = make_dataset_casiaB.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.casiaB_ls_probe_nm_seq,
                                        data_type,
                                        start_id,
                                        config.casiaB_angle_list)

    # probe-bag test set
    elif (data_type == "bg"):
        data, label  = make_dataset_casiaB.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.casiaB_ls_probe_bg_seq,
                                        data_type,
                                        start_id,
                                        config.casiaB_angle_list)

    # probe-coat test set
    elif (data_type == "cl"):
        data, label  = make_dataset_casiaB.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.casiaB_ls_probe_cl_seq,
                                        data_type,
                                        start_id,
                                        config.casiaB_angle_list)

    # storing all probeset angle numpy array into a list
    X_probe = []
    y_probe = []

    # logic for converting list to numpy array
    for i in range(len(config.casiaB_angle_list)):
        for s in range(config.casiaB_nb_classes):
            if(s == 0):
                X_data = data[0][i]
                y_label = label[0][i]
        
            elif(not (s == 0)):
                X_data  = np.append(X_data, data[s][i], axis = 0)
                y_label = np.append(y_label, label[s][i], axis = 0)
        
        X_probe.append(X_data)
        y_probe.append(y_label)

        del X_data, y_label

    #X_probe --> (# of angles, # of ts, image_no, features)
    return X_probe, y_probe



if __name__ == "__main__":
    d, l = load_probe_data('cl')
    print(l[0][0][0])
