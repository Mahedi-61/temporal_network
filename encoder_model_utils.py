"""
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
from .. import root_dir
from . import config
from .encoder_model  import CenterLossLayer


# reading encoder model
def read_encoder_model(group):
    print("\nreading stored rnn model architecture and weight ...")
    
    json_string = open(config.casiaB_encoder_model_path).read()
    model = model_from_json(json_string, 
                    custom_objects={'CenterLossLayer': CenterLossLayer})

    encoder_model_weight_path = os.path.join(root_dir.model_path(), "encoder",
                                group + "_" + config.casiaB_encoder_model_weight)
    
    model.load_weights(encoder_model_weight_path)
    
    print("loaded model directory: ",  encoder_model_weight_path)
    return model


# saving checkpoint
def save_encoder_model_checkpoint(group):

    encoder_model_weight_path = os.path.join(root_dir.checkpoint_path(),
                                group + "_" + config.casiaB_encoder_model_weight)
    
    return ModelCheckpoint(encoder_model_weight_path,
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 1)


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
    pass
    #read_encoder_model()