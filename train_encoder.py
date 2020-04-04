""" train a rnn network using pose sequence of Casia dataset"""

# python packages
import numpy as np
import sys, os
"""
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.callbacks import LearningRateScheduler
"""
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import LearningRateScheduler


# project modules
from . import encoder_model
from . import encoder_model_utils
from . import make_dataset_encoder
from . import config
from .. import root_dir

def scheduler_casiaB(epoch):
    if (epoch == 80):
        K.set_value(model.optimizer.lr, config.lr_1)

    elif (epoch == 200):
        K.set_value(model.optimizer.lr, config.lr_2)

    elif (epoch == 270):
        K.set_value(model.optimizer.lr, config.lr_3)
        
    print("learning rate: ", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)


### custom loss
def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis = 0)


# path variables and constant
batch_size = 128
nb_epochs = 300
lr = config.learning_rate

# loading traing and validation data
X_train, y_train = make_dataset_encoder.load_encoder_train_data_per_group("train")
X_valid, y_valid = make_dataset_encoder.load_encoder_train_data_per_group("valid")

change_lr = LearningRateScheduler(scheduler_casiaB)



print("\ntrian data shape: ", X_train.shape)
print("train label shape: ", y_train.shape)

print("\nvalid data shape: ", X_valid.shape)
print("valid label shape: ", y_valid.shape)


# constructing model
model = encoder_model.get_autoencoder()

# train model once again
#model = encoder_model_utils.read_encoder_model(angle)


### run model
lambda_centerloss = 0.008

optimizer = Adam(lr = lr)
model.compile(optimizer = optimizer, loss='mse')

group = "g2"
# training and evaluating model
model_cp = encoder_model_utils.save_encoder_model_checkpoint(group)
early_stop = encoder_model_utils.set_early_stopping()


model.fit(X_train, y_train, 
            batch_size = batch_size,
            shuffle = True,
            epochs = nb_epochs,
            callbacks = [change_lr, model_cp],
            verbose = 2,
            validation_data=(X_valid, y_valid))

best_weight_path = os.path.join(root_dir.casiaB_encoder_model_path(),
                                group + "_" + "best_model.h5")
model.save(best_weight_path)