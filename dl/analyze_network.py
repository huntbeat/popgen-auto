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
@param model_file - string file name
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

def confusion_matrix(output_file, print_matrix=True):
  classes = {}
  truths = []
  preds = []
  with open(output_file,'r') as f:
    lines = f.readlines()
  for line in lines:
    output = line.split(':')
    pred = output[1].strip()
    preds.append(int(pred))

    truth = output[0][1:-1]
    int_truth = [int(t) for t in truth.split()]

#################################################
