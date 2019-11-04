"""
Author: Md Mahedi Hasan
Description: this file contains code for handling rnn model utilities
Notes:
# all models are palced and stored in model directory
# model checkpoint are saved in checkpoint directory with epoch number
# so final model should be replaced from checkpoint to model dir
"""

# python modules
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.callbacks import (EarlyStopping,
                             Callback,
                             ModelCheckpoint,
                             ReduceLROnPlateau)

# project modules
from ... import root_dir
from . import config
from .my_models import CenterLossLayer
 
if (config.working_dataset == "casiaA"):
    rnn_model_path = config.casiaA_rnn_model_path
    rnn_model_weight = config.casiaA_rnn_model_weight

elif(config.working_dataset == "casiaB"):
    rnn_model_path = config.casiaB_rnn_model_path
    rnn_model_weight = config.casiaB_rnn_model_weight


# reading model
def read_rnn_model(angle):
    print("\nreading stored rnn model architecture and weight ...")
    
    json_string = open(rnn_model_path).read()
    model = model_from_json(json_string, 
                    custom_objects={'CenterLossLayer': CenterLossLayer})

    rnn_model_weight_path = os.path.join(config.model_dir,
                            angle + "_" + rnn_model_weight)
    
    model.load_weights(rnn_model_weight_path)
    
    print("loaded model directory: ", angle + "_" + rnn_model_weight)
    return model



# saving checkpoint
def save_rnn_model_checkpoint(angle):

    rnn_model_weight_path = os.path.join(config.checkpoint_dir,
                            angle + "_" + rnn_model_weight)
    
    return ModelCheckpoint(rnn_model_weight_path,
                monitor = 'val_activation_1_acc',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 1)



# saving model
def save_rnn_model_weight(model, angle):

    rnn_model_weight_path = os.path.join(config.checkpoint_dir, 
                            angle + "_" + rnn_model_weight)

    return model.save_weights(rnn_model_weight_path)



# utilities funciton
class LossHistory(Callback):
    
    def on_train_begin(self, batch, logs = {}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs = {}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))



def set_early_stopping():
    return EarlyStopping(monitor = "val_loss",
                               patience = config.early_stopping,
                               mode = "auto",
                               verbose = 2)

if __name__ == "__main__":
    read_rnn_model("0")