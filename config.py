"""this file contains temporal network configuration info"""

# python packages
import os

# project modules
from ... import root_dir


# path vairables and constant
openpose_dir = os.path.join(root_dir.libs_path(), "openpose")
model_dir = os.path.join(root_dir.tn_path(), "model")
checkpoint_dir = os.path.join(root_dir.tn_path(), "checkpoint")
pose_data_dir = os.path.join(root_dir.tn_path(), "cache", "pose_data")


# sequence parameter
actual_fps = 4  #frame per step


# train and validation sequence for gallery set
ls_gallery_train_seq =  ["nm01", "nm02", "nm03", "nm04"]
ls_gallery_valid_seq =  ["cl01", "bg02"]


# test sequence for probe set
ls_probe_nm_seq = ["nm05", "nm06"]
ls_probe_bg_seq = ["bg01", "bg02"]
ls_probe_cl_seq = ["cl01", "cl02"]


# angle
angle_list = ["angle_000", "angle_018", "angle_036", "angle_054",
              "angle_072", "angle_090", "angle_108", "angle_126",
              "angle_144", "angle_162", "angle_180"]


train_angle_nb = 9

# model testing configuration
nb_features = 20
nb_classes = 62
nb_angles = 11
nb_steps = 28

nb_layers = 2
nb_cells = 80


# model and their weights name
rnn_model = "rnn_model.json"
rnn_model_path = os.path.join(model_dir, rnn_model)

"""
rnn_model_stateful = "rnn_model_stateful.json"
rnn_model_stateful_path = os.path.join(model_dir, rnn_model_stateful)
rnn_model_stateful_weight = "rnn_model_stateful_weight.h5"
rnn_model_stateful_weight_path = os.path.join(model_dir, rnn_model_stateful_weight)
"""

# network training parameter
learning_rate = 5e-4
lr_1 = 1e-4
lr_2 = 5e-5
lr_3 = 2.5e-5


training_batch_size = 256
testing_batch_size = 1024
stateful_batch_size = nb_classes
training_epochs = 450


# model utilites
early_stopping = 50