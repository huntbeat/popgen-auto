""" create simple neural network """

import h5py
import numpy as np

import keras
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Flatten
from keras import optimizers

with h5py.File('data.hdf5','r') as f:
  constant_h5 = f.get('constant')
  bottleneck_h5 = f.get('bottleneck')
  output_h5 = f.get('output')
  constant = np.array(constant_h5)
  bottleneck = np.array(bottleneck_h5)
  output = np.array(output_h5)
data = np.concatenate((constant,bottleneck))

n, L = data.shape[1:]
data = np.reshape(data,(-1,n,L,1))
model = Sequential()
model.add(Conv2D(3, (5, 5), strides=(1,1), activation='relu', input_shape=(n, L, 1)))
model.add(Conv2D(3, (4, 4), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

SGD = optimizers.SGD()
model.compile(optimizer=SGD, loss="binary_crossentropy", metrics=['accuracy'])
model.fit(data, output, verbose=1, validation_split=0.2, epochs=50)

