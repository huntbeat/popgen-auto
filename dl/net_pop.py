""" create neural network for constant, bottleneck, and natural selection"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
from network_helper import *

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D, Dropout, AveragePooling1D
from keras import optimizers

def main():
    prefix = sys.argv[2] + '/' + sys.argv[2] # directory to store all created files in
    xdatasets = ['constant','bottleneck','naturalselection']
    ydatasets = ['pop_output']
    dset_dict = fetch_data(xdatasets+ydatasets)

    # for old h5py formatting
    x = np.concatenate((dset_dict['constant'],dset_dict['bottleneck'],dset_dict['naturalselection']))
    xdatasets = ['SNPs']
    dset_dict['SNPs'] = x
    del dset_dict['constant']
    del dset_dict['bottleneck']
    del dset_dict['naturalselection']

    shuffle_data(dset_dict)
    train_sets, test_sets = split_data(dset_dict, 0.8)
    model, history = neural_network(train_sets, test_sets, prefix)
    pred_file = make_predictions(model, test_sets, xdatasets, ydatasets, prefix)

    conf_file = prefix + '_confmat.txt'
    confusion_matrix(pred_file, conf_file)

    training_accuracy_plot(history, prefix)

######################################################

def neural_network(train_sets, test_sets, prefix):
    # Schrider's network

    ksize = 2
    l2_lambda = 0.0001

    n, L = train_sets['SNPs'].shape[1:]
    model = Sequential()
    model.add(Conv1D(128*2, kernel_size=ksize,
                     activation='relu',
                     input_shape=(n,L),
                     kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Conv1D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(MaxPooling1D(pool_size=ksize))
    model.add(Dropout(0.2))

    model.add(Conv1D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(MaxPooling1D(pool_size=ksize))
    model.add(Dropout(0.2))

    model.add(Conv1D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(AveragePooling1D(pool_size=ksize))
    model.add(Dropout(0.2))

    model.add(Conv1D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    #model.add(AveragePooling1D(pool_size=ksize))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_initializer='normal',kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Dropout(0.25))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
    print(model.summary())

    history = model.fit(train_sets['SNPs'], train_sets['pop_output'], batch_size=64,
            epochs=10,
            verbose=1,
            validation_split=0.2)

    model.save(prefix + '_model.hdf5')
    model.save_weights(prefix + '_weights.hdf5')

    loss, acc = model.evaluate(test_sets['SNPs'], test_sets['pop_output'])
    print('test accuracy:', acc)
    return model, history

main()
