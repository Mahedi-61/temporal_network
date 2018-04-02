""" test my rnn network using pose sequence of CasiaB dataset"""

# python packages
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from prettytable import PrettyTable


# project modules
from . import model_utils
from . import data_preparation
from . import config


# path variables and constant
batch_size = config.testing_batch_size
probe_type = "bg"

# display options
table = PrettyTable(["angle", "accuracy"])


# utilites function
def get_prediction_per_angle(predictions):
    pred_angle = []
    
    for sample in predictions:
        pred_ts = []
        
        for ts in sample:
            pred_ts.append(np.argmax(ts))
            
        pred_angle.append(Counter(pred_ts).most_common()[0][0])

    print("predicted value: ")
    print(pred_angle)
    
    result_angle = Counter(pred_angle).most_common()[0][0]
    return result_angle




def get_reduce_dimension(y_true):
    y = []
    for sample in y_true:
        y.append(sample[0][0])
            
    return y



def get_total_prediction(X_probe, model):
    start = 0
    end = 0
    all_label = []

    # calculating for each subject
    for i in range(config.nb_classes):
        all_angle_label = []
        
        # calculation for each angle
        for j in range(config.nb_angles):

            # first empty labels
            y_true = []
            pred_angle = []
            

            # calculate angle sequence 
            start = end
            end += probe_is[i][j]

            # preprage angle data and label
            angle_data = X_probe[start : end]
            y_true = y_raw[start : end]
            y_true = get_reduce_dimension(y_true)
            
            print("\n\ndata shape: ", angle_data.shape)

            # predicting two videos each...
            print("predicting ...")
            predictions = model.predict(angle_data,  batch_size, verbose = 2)

            print("for sub: ", i, "angle: ", j)
            print("true value: ")
            print(y_true)

            # get per subject per angle result
            result_angle = get_prediction_per_angle(predictions)

            # gather all angles per subject result
            all_angle_label.append(result_angle)

        all_label.append(all_angle_label)


    print("\n\npredicted all label ...")
    print(all_label)
    
    return all_label




def compare(all_label):
    y_true = [i for i in range(config.nb_classes)]
    
    # calculation for each angle
    for p_angle in range(config.nb_angles):
        y_pred = []
        row = [p_angle]

        # for each subject
        for sub in range(config.nb_classes):
            y_pred.append(all_label[sub][p_angle])

    
        acc_score = accuracy_score(y_true, y_pred)
        print("angle", p_angle, "accuracy: ", acc_score * 100)

        row.append(acc_score * 100)
        table.add_row(row)
   
    

    
############################ main work here ############################
# loading probe data
X_probe, y_raw, probe_is = data_preparation.load_probe_data(probe_type)
y_probe = to_categorical(y_raw, config.nb_classes)

print("\nprobe data shape: ", X_probe.shape)
print("probe label shape: ", y_probe.shape)


# loading trained model
model = model_utils.read_rnn_model()


print("\n### test result of my rnn algorithm on CASIA Dataset-B ###")
all_label = get_total_prediction(X_probe, model)

# get accuracy
compare(all_label)

# display into taple
print("\n\n############## Summary of my rnn algorithm ############## ")
print("Probe set type:", probe_type)
print(table) 





