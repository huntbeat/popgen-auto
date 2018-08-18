"""
functions and methods for dealing with memory issues for sim natural selection data

inspired by : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import keras

from keras import backend as K
K.set_image_dim_ordering('th') # samples, channels, rows, cols

'''
FOR NOW ASSUME ONE INPUT
@param test_percent is x% of all data
@param val_precent is x% of all data excluding test data
'''
def mem_split(num_samples, test_percent, val_percent):
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    test_end = int(num_samples*test_percent)
    test_indices = indices[:test_end]
    num_remain = num_samples - test_end
    val_end = test_end+int(num_remain*val_percent)
    val_indices = indices[test_end:val_end]
    train_indices = indices[val_end:]

    partition = {}
    partition['train'] = train_indices
    partition['validation'] = val_indices
    partition['test'] = test_indices
    return partition

def find_datasets(ID):
    strength_ds = ['NS_strength_[0 0 0 1]', 'NS_strength_[0 0 1 0]', 'NS_strength_[0 1 0 0]', 'NS_strength_[1 0 0 0]']
    input_ds = ['SNP_TMRCA_pos_[0 0 0 1]', 'SNP_TMRCA_pos_[0 0 1 0]', 'SNP_TMRCA_pos_[0 1 0 0]', 'SNP_TMRCA_pos_[1 0 0 0]']
    ds_index = int(ID/30000)
    ds_remainder = int(ID%30000)
    return strength_ds[ds_index], input_ds[ds_index], ds_remainder

def confusion_matrix(cm_file, prediction_generator, model, print_matrix=True):
    stars = '\n' + ('*' * 25) + '\n' # for printing

    # get predictions
    prediction_probs = model.predict_generator(prediction_generator)
    predictions = [p.argmax(axis=-1) for p in prediction_probs]

    # get truths
    correct = 0 #man acc
    truths = []

    with h5py.File(prediction_generator.data_file,'r') as f:
        for n in range(len(predictions)):
            strength_ds, _, index = find_datasets(prediction_generator.list_IDs[n])
            y = f.get(strength_ds)[index]
            truth = y.argmax(axis=-1)
            #import pdb; pdb.set_trace()
            truths.append(truth)
            if truth==predictions[n]: correct += 1 #man acc
    print("man acc:", correct/len(predictions)) #man acc

    if print_matrix:
        print('\nCONFUSION MATRIX:\nPredictions across, \nGround Truths down')
        print(stars)

    # set up confusion matrix dict
    conf_mat = {} # key = class, val = vector of predictions
    #import pdb; pdb.set_trace()
    classes = list(set(truths))
    for c in classes: conf_mat[c] = np.zeros(len(classes), dtype=int)
    for truth,pred in zip(truths,predictions):
        pred_vector = conf_mat[truth]
        pred_vector[pred] += 1
        conf_mat[truth] = pred_vector

    # print format
    pred_id_str = (' '*6)
    for c in range(len(classes)):
        pred_id_str += (str(c) + ' '*6)
    pred_id_str += '\n'
    vals_str = '\n'
    for c in range(len(classes)):
        vals_str += (str(c) + ' ')
        for n in conf_mat[c]:
            val = '{:>6}'.format(n)
            vals_str += (val + ' ')
        vals_str += '\n'
    if print_matrix:
        print(pred_id_str,vals_str,stars)

def real_prediction_distribution(generator, model, prefix, print_distrib=True):
    distribution = {}
    prediction_probs = model.predict_generator(generator)
    predictions = [p.argmax(axis=-1) for p in prediction_probs]
    for p in predictions:
        if p not in distribution.keys():
            distribution[p] = 1
        else:
            distribution[p] += 1
    with open(prefix+'_real_preds.txt','w') as f:
        f.write("label : num_times\n")
        for label in distribution.keys():
            line = str(label) + ' : ' + str(distribution[label]) + '\n'
            f.write(line)
    if print_distrib:
        print("label : num_times")
        for label in distribution.keys():
            print(label, ':', distribution[label])


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_file, list_IDs, n_channels, batch_size=32, dim=(20,1500), n_classes=4, shuffle=True, real_data=False):
        self.data_file = data_file
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.real_data = real_data
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size, self.n_classes))

        if self.real_data:
            with h5py.File(self.data_file,'r') as f:
                for i, ID in enumerate(list_IDs_temp):
                    if self.n_channels == 3:
                        X[i,] = f.get('SNP_TMRCA_pos')[ID]
                    elif self.n_channels == 2:
                        rm_tmrca = np.delete(f.get('SNP_TMRCA_pos')[ID],1,axis=0)
                        X[i,] = rm_tmrca 
                    y[i] = np.ones(4, dtype=int) * (-1)

        else:
            with h5py.File(self.data_file,'r') as f:
                for i, ID in enumerate(list_IDs_temp):
                    strength_ds, input_ds, index = find_datasets(ID)
                    if self.n_channels == 3:
                        X[i,] = f.get(input_ds)[index]
                    elif self.n_channels == 2:
                        rm_tmrca = np.delete(f.get(input_ds)[index],1,axis=0)
                        X[i,] = rm_tmrca
                    y[i] = f.get(strength_ds)[index]
        return X, y

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
