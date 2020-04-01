""" train a rnn network using pose sequence of Casia dataset"""

# python packages
import numpy as np
import sys
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import LearningRateScheduler


# project modules
from . import encoder_model
from . import encoder_model_utils
from . import data_preparation_casiaB
from . import config


def scheduler_casiaB(epoch):
    if (epoch == 40):
        K.set_value(model.optimizer.lr, config.lr_1)

    elif (epoch == 90):
        K.set_value(model.optimizer.lr, config.lr_2)

    elif (epoch == 150):
        K.set_value(model.optimizer.lr, config.lr_3)
        
    print("learning rate: ", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)


### custom loss
def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis = 0)


# path variables and constant
batch_size = 128
nb_epochs = 200
lr = config.learning_rate

# loading traing and validation data
angle = "angle_036"
X_train, y_train = data_preparation_casiaB.load_data_per_angle(angle, "train")
X_valid, y_valid = data_preparation_casiaB.load_data_per_angle(angle, "valid")

change_lr = LearningRateScheduler(scheduler_casiaB)



print("\ntrian data shape: ", X_train.shape)
print("train label shape: ", y_train.shape)

print("\nvalid data shape: ", X_valid.shape)
print("valid label shape: ", y_valid.shape)


# constructing model
model = encoder_model.get_temporal_model()

# train model once again
#model = encoder_model_utils.read_encoder_model(angle)


### run model
lambda_centerloss = 0.008

optimizer = Adam(lr = lr)
model.compile(optimizer = optimizer,
                loss=[losses.categorical_crossentropy, zero_loss],
                loss_weights=[1, lambda_centerloss],
                metrics=['accuracy'])


# training and evaluating model
model_cp = encoder_model_utils.save_encoder_model_checkpoint(angle)
early_stop = encoder_model_utils.set_early_stopping()


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
