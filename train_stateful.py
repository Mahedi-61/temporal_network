""" train a stateful rnn network using pose sequence of CasiaB dataset"""

# python packages
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from prettytable import PrettyTable

from keras.optimizers import Adam


# project modules
from . import my_models
from . import model_utils
from . import data_preparation
from . import config


# path variables and constant
batch_size = config.stateful_batch_size
lr = config.learning_rate
probe_type = "bg"

# display options
table = PrettyTable(["angle", "accuracy"])


# utilites function
def get_prediction_all_ts(predictions):
    pred_angle = []
    
    for sample in predictions:
        pred_ts = []
        
        for img_class in sample:
            pred_ts.append(np.argmax(img_class))

        pred_angle.append(Counter(pred_ts).most_common()[0][0])

    #print("predicted value: ")
    #print(pred_angle)
    return pred_angle




def get_reduce_dimension(y_true):
    y = []
    for sample in y_true:
        y.append(np.argmax(sample[0]))
        
    return y




# compiling model
def compile_model():

    # constructing stateful rnn model
    model = my_models.model_rnn(stateful = True)

    # train once again
    #model = model_utils.read_rnn_model_stateful()


    # compiling model
    optimizer = Adam(lr = lr)
    objective = "categorical_crossentropy"

    model.compile(loss = objective,
                  optimizer = optimizer,
                  metrics = ["accuracy"])


    print("\nmodel compiled ...")
    return model




def generate_stateful_data_per_angle(train_data, train_label, a):
    
    # finding angle which contains maximum timesteps
    l = []
    for s in range(config.nb_classes):
        l.append(train_data[s][a].shape[0]) 

    #print(l)
    max_ts = max(l)
    print("\nmaximum timesteps for angle:", a, " is:", max_ts)

    X_train = np.ndarray(((max_ts * config.nb_classes),
                          config.nb_steps,
                          config.nb_features), dtype = np.float32)

    y_train = np.ndarray(((max_ts * config.nb_classes),
                          config.nb_steps,
                          config.nb_classes), dtype = np.float32)

    for ts in range(max_ts):
        for i, s in enumerate(range(config.nb_classes)):

            index = (ts * config.nb_classes) +  i
            sub_angle_data = train_data[s][a]

            # prepare label
            y_train[index] = train_label[s][a][0]
            
            # prepare data
            if(ts < sub_angle_data.shape[0]):
                X_train[index] = sub_angle_data[ts]

            # repeat to to make stateful for angle which got ts lower than max_ts
            # % for multiple repeat
            elif(ts >= sub_angle_data.shape[0]):
                j = ts % sub_angle_data.shape[0]
                
                #print(s, "not ok, j:", j)
                X_train[index] = sub_angle_data[j]

    return X_train, y_train





# fitting data into model
def fit_train(train_data, train_label, valid_data, valid_label, model):

    print("\n\npreparing stateful training data ...")
    X_train = []
    y_train = []
    
    for a in range(config.nb_angles):
        X, y = generate_stateful_data_per_angle(train_data, train_label, a)

        print("total stateful data per angle shape: ", X.shape)
        print("total label per angle shape: ", y.shape)

        X_train.append(X)
        y_train.append(y)
        del X, y


    print("\n\npreparing stateful validation data ...")
    X_valid = []
    y_valid = []
    
    for a in range(config.nb_angles):
        X, y = generate_stateful_data_per_angle(valid_data, valid_label, a)

        print("total stateful data per angle shape: ", X.shape)
        print("total label per angle shape: ", y.shape)

        X_valid.append(X)
        y_valid.append(y)
        del X, y

        
    # fitting model statefully
    for k, epoch in enumerate(range(config.training_epochs)):        
        print("\n\n****************** Epoch:", k, "******************")

        # for each angle
        for a in range(config.nb_angles):
            
            model.fit(X_train[a],
                      y_train[a],
                      epochs = 1,
                      batch_size = batch_size,
                      verbose = 2,
                      validation_data = (X_valid[a], y_valid[a]),
                      shuffle = False)

            model.reset_states()

        # saving model
        if ((k % 5) == 0 and k != 0):
            
            # calculation for each angle
            for p_angle in range(config.nb_angles):

                print("\n\n************** angle:", p_angle, "**************")
                row = [p_angle]

                # true label
                y_true = get_reduce_dimension(y_valid[p_angle])
                print("true label length: ", len(y_true))

                # predicting at an angle each...
                print("\npredicting ...")
                predictions = model.predict(X_valid[p_angle],
                                            batch_size,
                                            verbose = 2)

                model.reset_states()
                
                print("predictions shape: ", predictions.shape)
                y_pred = get_prediction_all_ts(predictions)

                print("all timesteps prediction lenght: ", len(y_pred))

                acc_score = accuracy_score(y_true, y_pred)
                print("angle", p_angle, "accuracy: ", acc_score * 100)

                row.append(acc_score * 100)
                table.add_row(row)

            print("\n\n############## Summary of my rnn algorithm ############## ")
            print("Probe set type:", probe_type)
            print(table)
            
            print('\nsaving model snapshot...')
            ep_nb = "ep" + str(k) + "_"
            model_utils.save_rnn_model_stateful_weight(model, ep_nb)
            



        
########################### main work here ###########################    
# loading traing and validation data
train_data, train_label = data_preparation.set_stateful_dataset("train")

valid_data, valid_label = data_preparation.set_stateful_dataset("valid")

# compiling model
model = compile_model()


# fitting model to data
fit_train(train_data, train_label, valid_data, valid_label, model)














