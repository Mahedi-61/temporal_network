"""setting root directory for the main project"""

import os

def root_path():
    return os.path.dirname(__file__)


def database_path():
    return os.path.join(root_path(), "database")


def data_path():
    return os.path.join(root_path(), "data")


def libs_path():
    return os.path.join(root_path(), "libs")


def pose_path():
    return os.path.join(root_path(), "pose")


def checkpoint_path():
    return os.path.join(root_path(), "checkpoint")


def model_path():
    return os.path.join(root_path(), "model")


def stn_path():
    return os.path.join(root_path(), "spatio_temporal_network")



# CASIA-GAIT DATASET B
def casia_gait_dataset_B():
    return os.path.join(database_path(), "casia_gait", "DatasetB", "videos")



