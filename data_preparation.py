""" this file prepare subject pose data for temporal network"""

# python packages
import numpy as np
import os


# project files
from . import config
from . import make_dataset
from . import make_dataset

# path variables and constant
actual_fps = config.actual_fps



# making training and validation dataset
def set_dataset(data_type):

    # calculating total number of person have gait videos
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
                                        config.angle_list)


    # validation dataset
    elif (data_type == "valid"):
        data, label =  make_dataset.get_keypoints_for_all_subject(
                                        subject_id_list,
                                        config.ls_gallery_valid_seq,
                                        data_type,
                                        start_id,
                                        config.angle_list)


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
def load_data(data_type, load_previous):
    print("\nstart preprocessing %s data" % data_type)

    # laoding array
    if(load_previous == True):
        if(data_type == "train"):
            X_data = np.load(os.path.join(config.pose_dataset_dir, "X_train.npy"))
            y_label = np.load(os.path.join(config.pose_dataset_dir, "y_train.npy"))

        if(data_type == "valid"):
            X_data = np.load(os.path.join(config.pose_dataset_dir, "X_valid.npy"))
            y_label = np.load(os.path.join(config.pose_dataset_dir, "y_valid.npy"))


    if(load_previous == False):
        data, label = set_dataset(data_type)

        # logic for converting list to numpy array
        X_data = data[0][0]
        y_label = label[0][0]

        for s in range(config.nb_classes):
            # trick for this faulty dataset sub: 109 has miss some angle
            for a in range(len(data[s])):
            
                if(not (s == 0 and a == 0)):
                    X_data = np.append(X_data, data[s][a], axis = 0)
                    y_label = np.append(y_label, label[s][a], axis = 0)

        del data, label

        # saving array
        if(data_type == "train"):
            np.save(os.path.join(config.pose_dataset_dir, "X_train.npy"), X_data)
            np.save(os.path.join(config.pose_dataset_dir, "y_train.npy") ,y_label)

        if(data_type == "valid"):
            np.save(os.path.join(config.pose_dataset_dir, "X_valid.npy"), X_data)
            np.save(os.path.join(config.pose_dataset_dir, "y_valid.npy"), y_label)

    return X_data, y_label




# methods for returning unstateful training and validation data
def load_probe_data(probe_type):
    print("\nstart preprocessing probe data")

    probe_is = set_probe_set(probe_type)
    probe_data = load_X(config.X_probe_file)
    probe_label = load_y(config.y_probe_file)

    return probe_data, probe_label, probe_is





if __name__ == "__main__":
    d, l = load_data("valid", False)

    print(l.shape)







