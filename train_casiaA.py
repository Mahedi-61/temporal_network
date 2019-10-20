""" 
Author: Md Mahedi Hasan
Description: train a rnn network using pose sequence of Casia-A dataset
"""

# python packages
import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import LearningRateScheduler


# project modules
from . import my_models_casiaA
from . import model_utils
from . import data_preparation_casiaA
from . import config


def scheduler(epoch):
    if (epoch == 50):
        K.set_value(model.optimizer.lr, config.lr_1)

    elif (epoch == 320):
        K.set_value(model.optimizer.lr, config.lr_2)

        
    print("learning rate: ", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)


### custom loss
def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis = 0)


# path variables and constant
batch_size = 128
nb_epochs = 400
lr = config.learning_rate
angle = "90"


# loading traing and validation data
X_train, y_train = data_preparation_casiaA.load_data("train", angle)
print("\ntrian data shape: ", X_train.shape)
print("train label shape: ", y_train.shape)


X_valid, y_valid = data_preparation_casiaA.load_data("valid", angle)
print("\nvalid data shape: ", X_valid.shape)
print("valid label shape: ", y_valid.shape)


# constructing model
model = my_models_casiaA.get_temporal_model()

# train model once again
#model = model_utils.read_rnn_model(angle)


### run model
lambda_centerloss = 0.008

optimizer = Adam(lr = lr)
model.compile(optimizer = optimizer,
                loss=[losses.categorical_crossentropy, zero_loss],
                loss_weights=[1, lambda_centerloss],
                metrics=['accuracy'])


# training and evaluating model
model_cp = model_utils.save_rnn_model_checkpoint(angle)
change_lr = LearningRateScheduler(scheduler)
early_stop = model_utils.set_early_stopping()


# fit
y_train_value = np.argmax(y_train, axis = 2)
y_valid_value = np.argmax(y_valid, axis = 2)

random_y_train = np.random.rand(X_train.shape[0], 1)
random_y_valid = np.random.rand(X_valid.shape[0], 1)


model.fit([X_train, y_train], [y_train, random_y_train], 
            batch_size = batch_size,
            shuffle = True,
            epochs = nb_epochs,
            callbacks = [change_lr, model_cp],
            verbose = 2,
            validation_data=([X_valid, y_valid], [y_valid, random_y_valid]))