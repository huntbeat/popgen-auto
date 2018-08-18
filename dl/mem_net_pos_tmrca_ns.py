""" create neural network for constant, bottleneck, and natural selection"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import pickle
from network_helper import *
from memDataGenerator import *

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling2D, MaxPooling1D, Dropout, AveragePooling2D, AveragePooling1D
from keras import optimizers

from keras import backend as K
K.set_image_dim_ordering('th') # samples, channels, rows, cols

def main():
    prefix = sys.argv[2] + '/' + sys.argv[2] # directory to store all created files in
    data_file = sys.argv[1]
    partition = mem_split(120000, 0.2, 0.2)

    try:
        with open(prefix + '_model.hdf5', 'r') as f:
            print("Model exists. Evaluating now...")
        model = load_model(prefix + '_model.hdf5')
    except (FileNotFoundError, OSError) as e:
        print("Model does not exist. Training now...")
        training_generator = DataGenerator(data_file, partition['train'], 3)
        validation_generator = DataGenerator(data_file, partition['validation'], 3)
        model, history = neural_network(training_generator, validation_generator, prefix)
        plot_learning_curve(history, prefix)
        print("Evaluating model now...")

    # MSMS #
    eval_generator = DataGenerator(data_file, partition['test'], 3)
    '''
    n_print = 20
    n_predict = int(n_print/32)*32 + 32
    predict_generator = DataGenerator(data_file, partition['test'][:n_predict], 3)
    predict_and_evaluate(eval_generator, predict_generator, model, n_print)
    '''
    confusion_matrix(prefix+'_confmat.txt', eval_generator, model)

    '''
    # Real #
    real_file = sys.argv[3]
    real_indices = list(range(1050)) #TODO fill this in 
    real_generator = DataGenerator(real_file, real_indices, 3, real_data=True) 
    real_prediction_distribution(real_generator, model, prefix)
    '''

######################################################

def neural_network(training_generator, validation_generator, prefix):
    # Schrider's network, heavily modified

    ksize = (2,2)
    l2_lambda = 0.0001

    dims = (3,20,1500)
    model = Sequential()
    model.add(Conv2D(128*2, kernel_size=ksize,
                     activation='relu',
                     input_shape=(dims),
                     kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Conv2D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(MaxPooling2D(pool_size=ksize))
    model.add(Dropout(0.2))

    model.add(Conv2D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(MaxPooling2D(pool_size=ksize))
    model.add(Dropout(0.2))

    model.add(Conv2D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(AveragePooling2D(pool_size=ksize))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_initializer='normal',kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='softmax'))

    #model.compile(loss=weighted_categorical_crossentropy,
    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
    print(model.summary())

    history = model.fit_generator(training_generator, validation_data=validation_generator, epochs=10)

    model.save(prefix + '_model.hdf5')
    model.save_weights(prefix + '_weights.hdf5')
    with open(prefix + '_trainhist', 'wb') as f:
        pickle.dump(history.history, f)

    return model, history

######################################################

def weighted_categorical_crossentropy(target, output):
    import pdb; pdb.set_trace()
    target_max = K.max(target, axis=-1)
    output_max = K.max(output, axis=-1)
    abs_diff = K.abs(target_max-output_max)
    losses = abs_diff * keras.losses.categorical_crossentropy(target, output)
    return losses

main()
