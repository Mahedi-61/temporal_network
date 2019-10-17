"""
Author: Md Mahedi Hasan
Description: this file contains configuration info for temporal network
"""

# python packages
import os

# project modules
from ... import root_dir


# path vairables and constant
openpose_dir = os.path.join(root_dir.libs_path(), "openpose")
model_dir = os.path.join(root_dir.tn_path(), "model")
checkpoint_dir = os.path.join(root_dir.tn_path(), "checkpoint")

casiaA_pose_data_dir = os.path.join(root_dir.tn_path(), "cache", "casiaA_pose_data")
casiaB_pose_data_dir = os.path.join(root_dir.tn_path(), "cache", "casiaB_pose_data")



# train and validation sequence for gallery set
# CASIA A
def get_casiaA_train_valid_seq(angle):
    if angle == "0":
        casiaA_gallery_train_seq =  ["00_1", "00_2", "00_3"]
        casiaA_gallery_valid_seq =  ["00_4"]

    elif angle == "45":
        casiaA_gallery_train_seq =  ["45_1", "45_2", "45_3"]
        casiaA_gallery_valid_seq =  ["45_4"]

    else:
        casiaA_gallery_train_seq =  ["90_1", "90_2", "90_3"]
        casiaA_gallery_valid_seq =  ["90_4"]

    return casiaA_gallery_train_seq, casiaA_gallery_valid_seq

# CASIA B
casiaB_ls_gallery_train_seq =  ["nm01", "nm02", "nm03", "nm04"]
casiaB_ls_gallery_valid_seq =  ["cl01", "bg02"]


# test sequence for probe set
casiaB_ls_probe_nm_seq = ["nm05", "nm06"]
casiaB_ls_probe_bg_seq = ["bg01", "bg02"]
casiaB_ls_probe_cl_seq = ["cl01", "cl02"]


# angle
angle_list = ["angle_000", "angle_018", "angle_036", "angle_054",
              "angle_072", "angle_090", "angle_108", "angle_126",
              "angle_144", "angle_162", "angle_180"]


train_angle_nb = 9

# model testing configuration
# for CASIA A dataset
casiaA_nb_features = 20
casiaA_nb_classes = 20
casiaA_nb_angles = 3
casiaA_nb_steps = 28

# for CASIA B dataset
#casiaB_nb_features = 20
#casiaB_nb_classes = 100
casiaB_nb_angles = 11
casiaB_nb_steps = 28


# network architecture
actual_fps = 4  #frame per step
nb_layers = 2
nb_cells = 100


# model and their weights name
casiaA_rnn_model = "casiaA_rnn_model.json"
casiaA_rnn_model_path = os.path.join(model_dir, casiaA_rnn_model)
casiaA_rnn_model_weight = "casiaA_rnn_model_weight.h5"


# network training parameter
learning_rate = 1e-3 
lr_1 = 5e-4
lr_2 = 1e-4

# model utilites
early_stopping = 200