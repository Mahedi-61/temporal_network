""" test my rnn network using pose sequence of CasiaB dataset"""

# python packages
import numpy as np
import sys
from collections import Counter
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable


# project modules
from . import model_utils
from . import data_preparation_casiaB
from . import config


# path variables and constant
batch_size = 128
probe_type = str(sys.argv[1]) 

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
    
    

def get_predicted_each_subject_with_trickery(y_true, y_pred):
    sub_prediction = []
    present_pointer = 0
    list_subject = []
    c = Counter(y_true)
    sub_ts_length = [c.get(i) for i in range(config.casiaB_nb_classes)]
    
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
    sub_ts_length = [c.get(i) for i in range(config.casiaB_nb_classes)]
    
    for ts_len in sub_ts_length:
        
        next_pointer = present_pointer + ts_len

        each_sub_predicted_ts = y_pred[present_pointer : next_pointer]
        sub_prediction.append(Counter(each_sub_predicted_ts).most_common()[0][0])

        present_pointer += ts_len
    return sub_prediction
        
  

    

def get_total_prediction(X_probe, y_probe, model):
    
    print("\n\npreparing probe data for :", probe_type)
    row = []

    # calculation for each angle
    for p_angle in range(config.casiaB_nb_angles):

        print("\n\n**************", config.casiaB_angle_list[p_angle], "**************")

        print("\n" + config.casiaB_angle_list[p_angle], "probe data shape: ",
                            X_probe[p_angle].shape)
        
        print(config.casiaB_angle_list[p_angle], "probe label shape: ",
                            y_probe[p_angle].shape)

        # true label
        y_true = get_reduce_dimension(y_probe[p_angle])
        #print("true label length: ", len(y_true))

        
        # predicting at an angle each...
        print("\npredicting ...")
        predictions = model.predict([X_probe[p_angle], y_probe[p_angle]],
                                    batch_size,
                                    verbose = 2)

        
        predictions = predictions[0]
        print("predictions shape: ", predictions.shape)
        y_pred = get_prediction_all_ts(predictions)

        # get subject wise actual label and prediction
        sub_wise_pred = get_predicted_each_subject(y_true, y_pred)
        acutal_label = [i for i in range(config.casiaB_nb_classes)]

        print("actual label:\n", acutal_label)
        print("\npredicted label\n", sub_wise_pred)

        acc_score = accuracy_score(acutal_label, sub_wise_pred)
        print("\n", config.casiaB_angle_list[p_angle], "accuracy: ", acc_score * 100)

        row.append("{0:.4f}".format(acc_score * 100))
        
    return row
        
 

############################ main work here ############################
# loading probe data
probe_data, probe_label = data_preparation_casiaB.load_probe_data(probe_type)

print("\n### test result of my rnn algorithm on CASIA Dataset-B ###")

for data_angle in range(2,4): #config.casiaB_nb_angles):

    # loading trained model
    angle = config.casiaB_angle_list[data_angle]
    model = model_utils.read_encoder_model(angle)

    all_label = get_total_prediction (probe_data, probe_label, model)

    # adding in pretty table one row
    table.add_row([angle] + all_label)



# display into table
print("\n\n############## Summary of my rnn algorithm ############## ")
print("Probe set type:", probe_type)
print(table) 

"""
if __name__ == "__main__":
    probe_data, probe_label = data_preparation_casiaB.load_probe_data("nm")
    model = model_utils.read_rnn_model("angle_162")
    get_total_prediction (probe_data, probe_label, model)
"""