#!/usr/bin/env python3

import fire
import bisect
import numpy as np
import pickle 
import cnn_utils
import tensorflow as tf
import keras
from keras import backend
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, \
        AveragePooling1D, Concatenate, concatenate
from tensorflow.keras.utils import plot_model
import time
from memory_profiler import profile

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def alignBranch(input):
    m = Conv1D(64, kernel_size=2, activation="relu")(input)
    m = Conv1D(64, kernel_size=2, activation="relu")(m)
    m = AveragePooling1D(pool_size=2)(m)
    m = Dropout(0.25)(m)
    m = Flatten()(m) 
    return m

def positionsBranch(input):
    m = Dense(64, activation="relu")(input)
    m = Dropout(0.1)(m)
    return m

def outputBranch(input):
    m = Dense(64, activation="relu")(input)
    m = Dropout(0.5)(m)
    m = Dense(64, activation="relu")(m)
    m = Dropout(0.5)(m)
    m = Dense(1)(m)
    return m

def split(data, train, test):
    training = data[:train] 
    validation = data[test:]
    testing = data[train:test]
    return training, validation, testing

def split_list(data, train, test):
    training_list = []
    validation_list = []
    testing_list = []
    for i in data:
        training, validation, testing = split(i, train, test) 
        training_list.append(training) 
        validation_list.append(validation)
        testing_list.append(testing)
    return training_list, validation_list, testing_list

@profile
def cnn(model, predictors, response, outname):
    print("*" * 80 + "\n" + outname + "\n" + "*" * 80)

    # Split data into training, validation, and testing
    total = response.shape[0]
    train = int(total - (total * .2))
    test =  int(total - (total * .1))
    train_pred, val_pred, test_pred = split_list(predictors, train, test) 
    train_resp, val_resp, test_resp = split(response, train, test) 

    # Generate network architecture diagram
    plot_model(model, to_file=f"{outname}-diagram.png", 
            show_shapes=False, show_layer_names=False, rankdir="LR")

    # Performing CNN training 
    # logs = f"logs/{outname}" #+ datetime.now().strftime("%Y%m%d-%H%M%S")
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                #  profile_batch = '10,15')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
    time_callback = TimeHistory()
    history = model.fit(train_pred, train_resp, 
        validation_data=(val_pred, val_resp),
        epochs=10, batch_size=32, 
        callbacks=[time_callback]#, tb_callback]
        )

    # Perform CNN testing
    pred = model.predict(test_pred)

    # Output results
    output = dict(
       train_loss=history.history["loss"], 
       val_loss=history.history["val_loss"],
       train_rmse=history.history["rmse"],
       val_rmse=history.history["val_rmse"],
       predicted=pred,
       simulated=test_resp,
       times=time_callback.times)
    pickle.dump(output, open(f"{outname}.p", "wb"))

def run_cnns(input, reduced_size):
    # TODO: Write description of these inputs, what does reduced size do?
    data = np.load(input, allow_pickle=True)
    
    popSizes = data["popSizes"]
    positions = data["positions"]
    varChars = data["varChars"]
    invarChars = data["invarChars"]

    # Reduced data sets to reduced_size
    redInvarChars = invarChars[:, :reduced_size]
    redPositions = np.empty(positions.shape[0], dtype=object)
    redVarChars = np.empty(varChars.shape[0], dtype=object)
    
    for i in range(positions.shape[0]):
        pos = positions[i]
        var = varChars[i] 
        ix = np.searchsorted(pos, reduced_size) - 1
        redPositions[i] = pos[:ix] 
        redVarChars[i] = var[:ix] 

    # Pad ragged arrays and matrices
    positions = cnn_utils.padArrays(positions, value=-1)
    varChars = cnn_utils.padMatrices(varChars)
    redPositions = cnn_utils.padArrays(positions, value=-1)
    redVarChars = cnn_utils.padMatrices(redVarChars)

    # Create inputs
    posInput = Input(shape=positions.shape[1:])
    redPosInput = Input(shape=redPositions.shape[1:])
    invarInput = Input(shape=invarChars.shape[1:])
    varInput = Input(shape=varChars.shape[1:])
    redVarInput = Input(shape=redVarChars.shape[1:])
    redInvarInput = Input(shape=redInvarChars.shape[1:])

    # Create alignment and position branches  
    posBranch = positionsBranch(posInput) 
    redPosBranch = positionsBranch(redPosInput) 
    varBranch = alignBranch(varInput)
    redVarBranch = alignBranch(redVarInput)
    invarBranch = alignBranch(invarInput)
    redInvarBranch = alignBranch(redInvarInput)

    # Create output branches, concatenating if necessary
    pos_var_network = outputBranch(concatenate([posBranch, varBranch]))  
    var_network = outputBranch(varBranch)
    red_pos_var_network = outputBranch(concatenate([redPosBranch, redVarBranch]))  
    red_var_network = outputBranch(redVarBranch)
    invar_network = outputBranch(invarBranch)
    red_invar_network = outputBranch(redInvarBranch)

    # Construct models 
    pos_var_model = Model(inputs=[posInput, varInput], outputs=[pos_var_network]) 
    var_model = Model(inputs=[varInput], outputs=[var_network]) 
    red_pos_var_model = Model(inputs=[redPosInput, redVarInput], outputs=[red_pos_var_network]) 
    red_var_model = Model(inputs=[redVarInput], outputs=[red_var_network]) 
    red_invar_model = Model(inputs=[redInvarInput], outputs=[red_invar_network]) 
    invar_model = Model(inputs=[invarInput], outputs=[invar_network]) 

    # Run CNN training and testing
    # cnn(var_model, [varChars], popSizes, "var-model")
    # cnn(pos_var_model, [positions, varChars], popSizes, "pos-var-model")
    # cnn(red_var_model, [redVarChars], popSizes, "red-var-model")
    # cnn(red_pos_var_model, [redPositions, redVarChars], popSizes, "red-pos-var-model")
    # cnn(red_invar_model, [redInvarChars], popSizes, "red-invar-model")
    cnn(invar_model, [invarChars], popSizes, "invar-model")

if __name__ == "__main__":
    fire.Fire(run_cnns)
