"""this file contains code for handling rnn model utilities"""
"""
Notes:
# all models are palced and stored in model directory
# model checkpoint are saved in checkpoint directory with epoch number
# so final model should be replaced from checkpoint to model dir without epoch number.
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

rnn_model_weight = "rnn_model_weight.h5"


# reading model
def read_rnn_model(angle):
    print("\nreading stored rnn model architecture and weight ...")
    
    json_string = open(config.rnn_model_path).read()
    model = model_from_json(json_string)

    rnn_model_weight_path = os.path.join(config.model_dir,
                            angle + "_" + rnn_model_weight)
    
    model.load_weights(rnn_model_weight_path)

    return model




# reading model
def read_rnn_model_stateful():
    print("\nreading stored rnn stateful model architecture and weight ...")
    
    json_string = open(config.rnn_model_stateful_path).read()
    model = model_from_json(json_string)
    
    model.load_weights(config.rnn_model_stateful_weight_path)

    return model



# saving checkpoint
def save_rnn_model_checkpoint(angle):

    rnn_model_weight_path = os.path.join(config.checkpoint_dir,
                                angle + "_" + rnn_model_weight)
    
    return ModelCheckpoint(rnn_model_weight_path,
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 1)




def save_rnn_model_stateful_checkpoint():

    rnn_model_stateful_weight_path = os.path.join(config.checkpoint_dir,
                            config.rnn_model_stateful_weight)
    
    return ModelCheckpoint(rnn_model_stateful_weight_path,
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 1)



# saving model
def save_rnn_model_stateful_weight(model, ep_nb = ""):

    rnn_model_stateful_weight = os.path.join(config.checkpoint_dir,
                            ep_nb + config.rnn_model_stateful_weight)


    return model.save_weights(rnn_model_stateful_weight)




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
                               patience = config.early_stopping_patience,
                               mode = "auto",
                               verbose = 2)




def set_reduce_lr():
    return ReduceLROnPlateau(monitor='loss',
                             factor = config.lr_reduce_factor,
                             patience = config.lr_reduce_patience,
                             min_lr = 1e-5)




def show_loss_function(loss, val_loss, nb_epochs):
    plt.xlabel("Epochs ------>")
    plt.ylabel("Loss -------->")
    plt.title("Loss function")
    plt.plot(loss, "blue", label = "Training Loss")
    plt.plot(val_loss, "green", label = "Validation Loss")
    plt.xticks(range(0, nb_epochs)[0::2])
    plt.legend()
    plt.show()



