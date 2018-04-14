""" train a rnn network using pose sequence of CasiaB dataset"""

# python packages
import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler


# project modules
from . import my_models
from . import model_utils
from . import data_preparation
from . import config



def scheduler(epoch):
    if (epoch == 10):
        K.set_value(model.optimizer.lr, config.lr_fc)

    elif (epoch == 50):
        K.set_value(model.optimizer.lr, config.lr_sc)

    elif (epoch == 100):
        K.set_value(model.optimizer.lr, config.lr_tc)
        
    print("learning rate: ", K.get_value(model.optimizer.lr))
    
    return K.get_value(model.optimizer.lr)



# path variables and constant
batch_size = config.training_batch_size
nb_epochs = config.training_epochs
lr = config.learning_rate
angle = config.angle_list[config.train_angle_nb]



# loading traing and validation data
X_train, y_train = data_preparation.load_train_data_per_angle(angle)
print("\ntrian data shape: ", X_train.shape)
print("train label shape: ", y_train.shape)



X_valid, y_valid = data_preparation.load_valid_data_per_angle(angle)
print("\nvalid data shape: ", X_valid.shape)
print("valid label shape: ", y_valid.shape)



# constructing model
#model = my_models.model_rnn(stateful = False)

# train model once again
model = model_utils.read_rnn_model(angle)



# compiling model
optimizer = Adam(lr = lr)
objective = "categorical_crossentropy"

model.compile(loss = objective,
              optimizer = optimizer,
              metrics = ["accuracy"])

print("\nmodel compiled ...")



# training and evaluating model
model_cp = model_utils.save_rnn_model_checkpoint(angle)
reduce_lr = model_utils.set_reduce_lr()
change_lr = LearningRateScheduler(scheduler)


model.fit(X_train,
          y_train,
          batch_size = batch_size,
          shuffle = True,
          epochs = nb_epochs,
          callbacks = [reduce_lr, change_lr, model_cp],
          verbose = 2,
          validation_data = (X_valid, y_valid))



