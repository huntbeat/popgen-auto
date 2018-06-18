'''
helper methods for networks
'''

import h5py 
import numpy as np
import sys

import keras
from keras.models import load_model

#################################################
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

  return kernels

#################################################
'''
compute the confusion matrix for a set of ground truths and respective predictions
also writes confusion matrix to a file 
@param output_file - string file name containing the truths and preds
@param print_matrix - boolean for printing matrix 
'''
def confusion_matrix(output_file, print_matrix=True):
  cm_file = 'confusion_matrix.txt'

  classes = {} # key = vector, val = index of '1' (example: [0,0,1,0]:2)
  truths = []
  preds = []
  conf_mat = {} # key = class, val = vector of predictions
  with open(output_file,'r') as f:
    lines = f.readlines()
  for line in lines:
    output = line.split(':')
    pred = output[1].strip()
    preds.append(int(pred))

    str_truth = output[0][1:-1]
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
  
  # format 
  stars = '\n' + '*' * 25
  note = '\nCONFUSION MATRIX:\nPredictions across, \nGround Truths down\n\n'
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

  with open(cm_file,'w') as cm:
    cm.write(note)
    cm.write(pred_id_str)
    cm.write(vals_str)

  if print_matrix:
    print(stars,note,pred_id_str,vals_str,stars)

#################################################
