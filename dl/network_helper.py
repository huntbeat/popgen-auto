"""
helper functions for network scripts
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import keras
from keras.models import load_model

'''
@param datasets - list of string names for datasets stored in the h5py file
@return dictionary where k = dataset name, v = np array of the dataset
'''
def fetch_data(datasets):
    np_sets = {}
    data_file = sys.argv[1]
    path = '/scratch/saralab/'+data_file
    with h5py.File(path,'r') as f:
        for ds in datasets:
            np_sets[ds] = np.array(f.get(ds))
    return np_sets

'''
@param dset_dict
'''
def shuffle_data(dset_dict):
    num_samples = dset_dict[list(dset_dict.keys())[0]].shape[0]
    s = np.arange(num_samples)
    np.random.shuffle(s)
    for ds in dset_dict.keys():
        old_dset = dset_dict[ds]
        dset_dict[ds] = old_dset[s]

'''
@param dset_dict
@param train_percent
'''
def split_data(dset_dict, train_percent):
    split_index = int(dset_dict[list(dset_dict.keys())[0]].shape[0]*train_percent)
    train_sets = {}
    test_sets = {}
    for ds in dset_dict.keys():
        data = dset_dict[ds]
        train_sets[ds] = data[:split_index]
        test_sets[ds] = data[split_index:]
    return train_sets, test_sets


'''
@param model
@param test_sets
@param xnames
@param ynames
@param prefix
@return filename of truth:pred values file
'''
def make_predictions(model, test_sets, xnames, ynames, prefix):
    num_samples = test_sets[ynames[0]].shape[0]
    pred_file = prefix + '_preds.txt'
    xtest_sets = [test_sets[x] for x in xnames]
    prediction_probs = model.predict(xtest_sets)
    predictions = [[p.argmax(axis=-1) for p in prediction_probs]]
    with open(pred_file, "w") as f:
        for n in range(num_samples):
            truths = ''
            preds = ''
            for y in range(len(ynames)):
                ytrue = test_sets[ynames[y]][n]
                truths += (str(ytrue) + ',')
                ypred = predictions[y][n]
                preds += (str(ypred) + ',')
            line = truths[:-1] + ':' + preds[:-1] + '\n'
            f.write(line)
    print("wrote truth:prediction values to file:",pred_file,'\n')
    return pred_file

'''
compute the confusion matrix for a set of ground truths and respective predictions
also writes confusion matrix to a file
@param pred_file- string file name containing the truths and preds
@param cm_file - file to write confusion matrix to
@param print_matrix - boolean for printing matrix
'''
def confusion_matrix(pred_file, cm_file, print_matrix=True):
    stars = '\n' + ('*' * 25) + '\n' # for printing
    with open(pred_file,'r') as f:
        lines = f.readlines()
        num_output_layers = len(lines[0].split(':')[0].split(','))
    if print_matrix:
        print('\nCONFUSION MATRIX:\nPredictions across, \nGround Truths down')
        print(stars)

    # parse pred_file
    for x in range(num_output_layers):
        title = "Output Layer " + str(x) + '\n\n'
        classes = {} # key = vector, val = index of '1' (example: [0,0,1,0]:2)
        truths = []
        preds = []
        conf_mat = {} # key = class, val = vector of predictions
        for line in lines:
            output = line.split(':')
            pred = output[1].strip().split(',')[x]
            preds.append(int(pred))

            str_truth = output[0].split(',')[x][1:-1]
            truth = np.array([int(t) for t in str_truth.split()], dtype='int64')
            truths.append(str_truth)
            if str_truth not in classes.keys():
                classes[str_truth] = np.where(truth==1)[0][0]

        num_classes = len(classes.keys())
        for c in range(num_classes):
            conf_mat[c] = np.zeros((num_classes),dtype='int64')

        for i in range(len(preds)):
            class_id = classes[truths[i]]
            pred_vector = conf_mat[class_id]
            pred_vector[preds[i]] += 1
            conf_mat[class_id] = pred_vector

        # print format
        pred_id_str = (' '*6)
        for c in range(num_classes):
            pred_id_str += (str(c) + ' '*6)
        pred_id_str += '\n'
        vals_str = '\n'
        for c in range(num_classes):
            vals_str += (str(c) + ' ')
            for n in conf_mat[c]:
                val = '{:>6}'.format(n)
                vals_str += (val + ' ')
            vals_str += '\n'
        if print_matrix:
            print(title,pred_id_str,vals_str,stars)

        # write matrix to file
        if x == 0:
            with open(cm_file,'w') as cm:
                cm.write('\nCONFUSION MATRIX:\nPredictions across, \nGround Truths down')
                cm.write(stars)
                cm.write(title)
                cm.write(pred_id_str)
                cm.write(vals_str)
        else:
            with open(cm_file,'a') as cm:
                cm.write(stars)
                cm.write(title)
                cm.write(pred_id_str)
                cm.write(vals_str)
    print("wrote confusion matri(x/ces) to file:",cm_file,'\n')

'''
'''
def training_accuracy_plot(history, prefix):
    fig_file = prefix + '_accplot.png'
    plt.title('Model Training Accuracy')
    plt.ylabel('accuracy')
    plt.ylim(0,1)
    plt.xlabel('epoch')
    plt.xlim(0, 10)
    plt.xticks(range(10))
    possible_markers = ['*','s','D','p','o','v'] # add more if > 6 outputs (but why??)
    markers = {}
    legend = []
    for acc in history.history.keys():
        label = acc.replace("_acc", "").replace("val_", "")
        if label not in markers.keys():
            markers[label] = possible_markers.pop()
        marker1 = markers[label]
        plt.plot(history.history[acc], marker=marker1)
        legend.append(acc[:-4]) # discard '_acc'
    plt.legend(legend, loc='lower right')
    plt.savefig(fig_file)
    print("saved accuracy plot as:",fig_file,'\n')
    plt.show()

'''
returns a dictionary where key = layer name, and value = numpy array of weights
@param model_file - string file name of saved keras model
'''
def get_weights(model_file):
    model = load_model(model_file)
    layer_names = [weight.name for layer in model.layers for weight in layer.weights]
    layer_weights = model.get_weights()
    layers = {name:weight for name,weight in zip(layer_names,layer_weights)}

    kernels = {}
    for k in layers.keys():
        if 'kernel' in k:
            key = k.split('/')[0]
            kernels[key] = layers[k]

    print("layer names:")
    for k in kernels.keys():
        print(k)
    return kernels
