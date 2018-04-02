"""my rnn architecture for gait detection"""

# python packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import (Input, Dense, Dropout,
                          TimeDistributed, GRU,
                          LSTM, BatchNormalization, Bidirectional,
                          Activation)

# project modules
from . import config



def model_rnn(stateful = False):
    print("\nconstructing rnn model ... ")
    model = Sequential()

    # for stateless model
    if (stateful == False):
        
        # input layer
        model.add(BatchNormalization(momentum = 0.99,
                        epsilon = 1e-5,
                        input_shape =(config.nb_steps,
                                      config.nb_features)))

        
        # hidden lstm layers
        for i in range(config.nb_layers):

            model.add(Bidirectional(
                GRU(
                    config.nb_cells,
                    return_sequences = True,
                    stateful = False),
                merge_mode = "concat"))


        model.add(Dropout(rate = 0.4))

        # output softmax layer
        model.add(Dense(config.nb_classes))
        
        model.add(BatchNormalization(momentum = 0.99,
                                     epsilon = 1e-5))
        
        model.add(Activation('softmax'))
        
        # saving as json file in model directory
        open(config.rnn_model_path, 'w').write(model.to_json())



    
    # for stateful model only predict
    elif (stateful == True):

        # input layer
        model.add(BatchNormalization(momentum = 0.99,
                        epsilon = 1e-5,
                        batch_input_shape =(config.stateful_batch_size,
                                            config.nb_steps,
                                            config.nb_features)))

        # hidden lstm layers
        for i in range(config.nb_layers):

            model.add(Bidirectional(
                GRU(
                    config.nb_cells,
                    return_sequences = True,
                    stateful = True),
                merge_mode = "concat"))

        model.add(Dropout(rate = 0.4))
        
        # output softmax layer
        model.add(TimeDistributed(Dense(config.nb_classes,
                                        kernel_initializer = "uniform")))
        
        model.add(BatchNormalization(momentum = 0.99,
                                     epsilon = 1e-5))
        
        model.add(Activation('softmax'))

        # saving as json file in model directory
        open(config.rnn_model_stateful_path, 'w').write(model.to_json())


    return model



if __name__ == "__main__":
    model_rnn(stateful = False).summary()






