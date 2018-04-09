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
batch_size = config.testing_batch_size
probe_type = "bg"

# display options
table = PrettyTable(["data_angle", "probe_000", "probe_018", "probe_036",
                     "probe_054", "probe_072", "probe_090", "probe_108",
                     "probe_126", "probe_144", "probe_162", "probe_180"])


# utilites function
def get_prediction_all_ts(predictions):
    pred_angle = []
    
    for sample in predictions:
        pred_ts = []

        for img_class in sample:
            pred_ts.append(np.argmax(img_class))

        pred_angle.append(Counter(pred_ts).most_common()[0][0])

    #print(pred_angle)
    
    return pred_angle




def get_reduce_dimension(y_true):
    y = []
    for sample in y_true:
        y.append(np.argmax(sample[0]))
        
    return y
    
    
    

def get_total_prediction(X_probe, y_probe, model):
    
    print("\n\npreparing probe data for :", probe_type)
    row = []

    # calculation for each angle
    for p_angle in range(config.nb_angles):

        print("\n\n************** angle:", config.angle_list[p_angle], "**************")

        print("\n" + config.angle_list[p_angle], "probe data shape: ",
                                                  X_probe[p_angle].shape)
        
        print(config.angle_list[p_angle], "probe label shape: ",
                                                  y_probe[p_angle].shape)

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

        print("all timesteps prediction length: ", len(y_pred))

        acc_score = accuracy_score(y_true, y_pred)
        print(config.angle_list[p_angle], "accuracy: ", acc_score * 100)

        row.append("{0:.4f}".format(acc_score * 100))
        
    return row
        
 



    
############################ main work here ############################
# loading probe data
probe_data, probe_label = data_preparation.load_probe_data(probe_type)

print("\n### test result of my rnn algorithm on CASIA Dataset-B ###")

for data_angle in range(1):

    # loading trained model
    angle = config.angle_list[data_angle]
    model = model_utils.read_rnn_model(angle)

    all_label = get_total_prediction (probe_data, probe_label, model)

    # adding in pretty table one row
    table.add_row([angle] + all_label)





# display into taple
print("\n\n############## Summary of my rnn algorithm ############## ")
print("Probe set type:", probe_type)
print(table) 





