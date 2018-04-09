""" test my rnn network using pose sequence of CasiaB dataset"""

# python packages
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable


# project modules
from . import model_utils
from . import data_preparation
from . import config


# path variables and constant
batch_size = config.stateful_batch_size
probe_type = "cl"

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

    print("predicted value: ")
    #print(pred_angle)
    return pred_angle




def get_reduce_dimension(y_true):
    y = []
    for sample in y_true:
        y.append(np.argmax(sample[0]))
        
    return y




def generate_stateful_data_per_angle(probe_data, probe_label, a):
    
    # finding angle which contains maximum timesteps
    l = []
    for s in range(config.nb_classes):
        l.append(probe_data[s][a].shape[0]) 

    #print(l)
    max_ts = max(l)
    print("\nmaximum timesteps for angle:", a, " is:", max_ts)

    X_probe = np.ndarray(((max_ts * config.nb_classes),
                          config.nb_steps,
                          config.nb_features), dtype = np.float32)

    y_probe = np.ndarray(((max_ts * config.nb_classes),
                          config.nb_steps,
                          config.nb_classes), dtype = np.float32)

    for ts in range(max_ts):
        for i, s in enumerate(range(config.nb_classes)):

            index = (ts * config.nb_classes) +  i
            sub_angle_data = probe_data[s][a]

            # prepare label
            y_probe[index] = probe_label[s][a][0]
            
            # prepare data
            if(ts < sub_angle_data.shape[0]):
                X_probe[index] = sub_angle_data[ts]

            # repeat to to make stateful for angle which got ts lower than max_ts
            # % for multiple repeat
            elif(ts >= sub_angle_data.shape[0]):
                j = ts % sub_angle_data.shape[0]
                
                #print(s, "not ok, j:", j)
                X_probe[index] = sub_angle_data[j]
                
    return X_probe, y_probe





def get_total_prediction(X_probe, model):
    
    print("\n\npreparing stateful probe data for :", probe_type)
    X_probe = []
    y_probe = []
    
    for a in range(config.nb_angles):
        X, y = generate_stateful_data_per_angle(probe_data, probe_label, a)

        print("total stateful probe data: ", X.shape)
        print("total label shape: ", y.shape)

        X_probe.append(X)
        y_probe.append(y)
        del X, y


    # calculation for each angle
    for p_angle in range(1):

        print("\n\n************** angle:", p_angle, "**************")
        row = [p_angle]

        # true label
        y_true = get_reduce_dimension(y_probe[p_angle])
        print("true label length: ", len(y_true))

        # predicting at an angle each...
        print("\npredicting ...")
        predictions = model.predict(X_probe[p_angle],
                                    batch_size,
                                    verbose = 2)

        
        print("predictions shape: ", predictions.shape)
        y_pred = get_prediction_all_ts(predictions)

        print("all timesteps prediction lenght: ", len(y_pred))

        acc_score = accuracy_score(y_true, y_pred)
        print("angle", p_angle, "accuracy: ", acc_score * 100)

        row.append(acc_score * 100)
        table.add_row(row)




    
############################ main work here ############################
# loading probe data
probe_data, probe_label = data_preparation.set_stateful_probeset(probe_type)


# loading trained model
model = model_utils.read_rnn_model_stateful()


print("\n### test result of my rnn algorithm on CASIA Dataset-B ###")
all_label = get_total_prediction(probe_data, model)



# display into taple
print("\n\n############## Summary of my rnn algorithm ############## ")
print("Probe set type:", probe_type)
print(table) 





