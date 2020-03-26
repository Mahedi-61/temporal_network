""" test my two_stage_network for experimental format on Caisa-B Dataset"""

# python packages
import numpy as np
import os
from collections import Counter
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable


# project modules
from ..spatio_temporal_network import src
stn_model_utils = src.model_utils
stn_config = src.config
stn_test_preprocessing = src.test_preprocessing



from ..temporal_network.src import data_preparation
from ..temporal_network.src import model_utils
from ..temporal_network.src import config

from .. import root_dir


# path variable and constant 
input_dir = stn_config.crop_img_dir


# display options
table = PrettyTable(["angle", "accuracy"])


# utilites function
# using majority voting scheme
def get_predicted_subject(predictions):
    pred_angle = []
    
    for sample in predictions:
        pred_ts = []

        for img_class in sample:
            pred_ts.append(np.argmax(img_class))

        pred_angle.append(Counter(pred_ts).most_common()[0][0])

    pred_subject_id = Counter(pred_angle).most_common()[0][0]
    return pred_subject_id

        

    
def predict(stn_model, subject_id_list, p_angle, probe_seq):
    pred_subject_id = []
    row = [p_angle]
    
    for nb, subject_id in enumerate(subject_id_list):
        X_test, y_test = stn_test_preprocessing.load_test_data(subject_id,
                                                           p_angle,
                                                           probe_seq)

       
        # for angle classification
        # predicting two videos each...
        predictions = stn_model.predict(X_test,
                                    stn_config.testing_batch_size,
                                    verbose = 2)


        # getting total probabilty score for each probe set
        total_prob_score = np.sum(predictions, axis = 0)
        pred_angle_id = np.argmax(total_prob_score)

        print("predicted angle: ", pred_angle_id)
        print("loading temporal network model for", config.angle_list[pred_angle_id])
        tn_model = model_utils.read_rnn_model(config.angle_list[pred_angle_id])

        # predicting each subject suing temporal network
        predictions = tn_model.predict(data[nb][(config.angle_list).index(p_angle)],
                                    config.testing_batch_size,
                                    verbose = 2)

        pred_id = get_predicted_subject(predictions)
        print("predicted id:", pred_id)
        pred_subject_id.append(pred_id)
        

    # get subject wise actual label and prediction
    acutal_label = [i for i in range(62)]

    print("actual label:\n", acutal_label)
    print("\npredicted label\n", pred_subject_id)
    
    acc_score = accuracy_score(acutal_label, pred_subject_id)
    print("\n", p_angle, "accuracy: ", acc_score * 100)

    row.append("{0:.4f}".format(acc_score * 100))
        
    return row
    
        
        
	





################# main work here #################
print("\n#### test result of my tow_stage gait recognition algorithm on CASIA Dataset-B ####")
print("\nstart preprocessing test data ...")

### test configuration
# calculating total number of person having gait videos
num_subject = len(os.listdir(input_dir))
print("total number subjects:", num_subject)
total_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))


print("\ntest subject id list: 63 to 124")
subject_id_list = total_id_list[62:124]
print(subject_id_list)


# configure probe data
probe_type = "nm"
probe_angle = stn_config.angle_list

if(probe_type == "nm"): probe_seq = stn_config.ls_probe_nm_seq
elif(probe_type == "bg"): probe_seq = stn_config.ls_probe_bg_seq
elif(probe_type == "cl"): probe_seq = stn_config.ls_probe_cl_seq



# first loading my angle classification network (conv_modle)
stn_model = stn_model_utils.read_conv_model()


# loading temporal data and label
data, label = data_preparation.set_dataset(probe_type)



# looping for all probe view angle
for angle_nb, p_angle in enumerate(probe_angle):
    print("\nprobe set type:", probe_type,
          "     angle:", p_angle)

    
    # predicting
    all_label = predict(stn_model, subject_id_list, p_angle, probe_seq)
    table.add_row(all_label)



# display into table
print("\n\n############## Summary of my rnn algorithm ############## ")
print("Probe set type:", probe_type)
print(table) 





