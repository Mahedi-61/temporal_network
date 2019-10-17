"""
Author: A cup of Tea
Description: test my rnn network using pose sequence of CasiaA dataset
"""

# python packages
import numpy as np
import sys
from collections import Counter
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable


# project modules
from . import model_utils
from . import data_preparation_casiaA
from . import config


# path variables and constant
probe_angle = ["0", "45", "90"]
batch_size = 256
#probe_type = str(sys.argv[1]) 

# display options
table = PrettyTable(["data_angle", "probe_00", "probe_45", "probe_90"])


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
    

def get_predicted_each_subject_with_trickery(y_true, y_pred):
    sub_prediction = []
    present_pointer = 0
    list_subject = []
    c = Counter(y_true)
    sub_ts_length = [c.get(i) for i in range(20)]
    
    for ts_len in sub_ts_length:
        
        next_pointer = present_pointer + ts_len

        each_sub_predicted_ts = y_pred[present_pointer : next_pointer]
        first_candiate = Counter(each_sub_predicted_ts).most_common()[0][0]

        if first_candiate not in list_subject:
            sub_prediction.append(first_candiate)
            list_subject.append(first_candiate)

        else:
            if(len(Counter(each_sub_predicted_ts).most_common()) == 1):
                sub_prediction.append(first_candiate)
                list_subject.append(first_candiate)

            else:
                second_candiate = Counter(each_sub_predicted_ts).most_common()[1][0]
                sub_prediction.append(second_candiate)
                #list_subject.append(second_candiate)

        present_pointer += ts_len
    return sub_prediction
        


def get_predicted_each_subject(y_true, y_pred):
    sub_prediction = []
    present_pointer = 0

    c = Counter(y_true)
    sub_ts_length = [c.get(i) for i in range(20)]
    
    for ts_len in sub_ts_length:
        
        next_pointer = present_pointer + ts_len

        each_sub_predicted_ts = y_pred[present_pointer : next_pointer]
        sub_prediction.append(Counter(each_sub_predicted_ts).most_common()[0][0])

        present_pointer += ts_len
    return sub_prediction
        
  

    
def get_total_prediction(X_probe, y_probe, model):
    
    print("\n\npreparing probe data for :")
    row = []

    # calculation for each angle
    for i, p_angle in enumerate(probe_angle):

        print("\n\n**************", p_angle, "**************")

        print("\nprobe data shape: ", X_probe[i].shape)
        print("probe label shape: ", y_probe[i].shape)

        # true label
        y_true = get_reduce_dimension(y_probe[i])
        #print("true label length: ", y_true)

        
        # predicting at an angle each...
        print("\npredicting ...")
        predictions = model.predict([X_probe[i], 
                                    y_probe[i]],
                                    batch_size,
                                    verbose = 2)

        predictions = predictions[0]
        print("predictions shape: ", predictions.shape)
        
        y_pred = get_prediction_all_ts(predictions)

        # get subject wise actual label and prediction
        sub_wise_pred = get_predicted_each_subject(y_true, y_pred)
        acutal_label = [i for i in range(20)]

        print("actual label:\n", acutal_label)
        print("\npredicted label\n", sub_wise_pred)

       
        acc_score = accuracy_score(acutal_label, sub_wise_pred)
        print("\n", p_angle, " accuracy: ", acc_score * 100)

        row.append("{0:.4f}".format(acc_score * 100))
    return row
        
 

############################ main work here ##########################
print("\n### test result of my rnn algorithm on CASIA Dataset-A ###")
# loading probe data
X_test, y_test = data_preparation_casiaA.load_probe_data()

for angle in probe_angle:

    # loading trained model
    model = model_utils.read_rnn_model(angle)
    all_label = get_total_prediction (X_test, y_test, model)

    # adding in pretty table one row
    table.add_row([angle] + all_label)


# display into table
print("\n\n############## Summary of my rnn algorithm ############## ")
print("Probe set type:")
print(table) 