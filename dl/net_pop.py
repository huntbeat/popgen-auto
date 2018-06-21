""" create neural network for constant, bottleneck, and natural selection"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
from analyze_network import confusion_matrix

import keras
from keras.models import Sequential 
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D, Dropout, AveragePooling1D
from keras import optimizers

# fetch data 
network_name = sys.argv[2] # filename prefix for all files created for this NN (i.e. model, weights, etc.)
prefix = '/' + network_name + '/' + network_name
data_file = sys.argv[1]
path = '/scratch/nhoang1/'+data_file
with h5py.File(path,'r') as f:
  constant_h5 = f.get('constant')
  bottleneck_h5 = f.get('bottleneck')
  natselect_h5 = f.get('naturalselection')
  output_h5 = f.get('pop_output')
  constant = np.array(constant_h5)
  bottleneck = np.array(bottleneck_h5)
  natselect = np.array(natselect_h5)
  output = np.array(output_h5)

data = np.concatenate((constant,bottleneck,natselect))

# shuffle data before training
data_shape_before = data.shape
output_shape_before = output.shape
s = np.arange(output.shape[0])
np.random.shuffle(s)
data = data[s]
output = output[s]
#assert(data_shape_before == data.shape)
assert(output_shape_before == output.shape)

# split into train/validation and test
si = int(output.shape[0]*0.8)
trainX = data[:si]
trainY = output[:si]
testX = data[si:]
testY = output[si:]

# Schrider's network

ksize = 2
l2_lambda = 0.0001

n, L = trainX.shape[1:]
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
    
model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
print(model.summary())

history = model.fit(trainX, trainY, batch_size=64,
        epochs=10,
        verbose=1,
        validation_split=0.2)

model.save(prefix + '_model.hdf5')
model.save_weights(prefix + '_weights.hdf5')

num_samples, n2, L2 = testX.shape
testX = np.reshape(testX,(-1,n2,L2))
predictions = model.predict_classes(testX)
true_labels = testY
orig_labels = np.unique(testY,axis=0)
loss, acc = model.evaluate(testX,testY)
print('test accuracy:',acc)

# save pred vs true to file
pred_file = "/"+network_name+"/"+network_name+"_preds.txt"
with open(pred_file, "w") as f:
  for n in range(num_samples):
    line = str(true_labels[n]) + ":" + str(predictions[n]) + "\n"
    f.write(line)
print("wrote truth:prediction values to file:",pred_file,'\n')

# confusion matrix on the test set
conf_file = "/"+network_name+"/"+network_name+"_confmat.txt"
confusion_matrix(pred_file, conf_file)

# accuracy plot
fig_file = "/"+network_name+"/"+network_name+"_accplot.png"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Training Accuracy')
plt.ylabel('accuracy')
plt.ylim(0,1)
plt.xlabel('epoch')
plt.xlim(0,10)
plt.xticks(range(10))
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(fig_name)
print("saved accuracy plot as:",fig_file,'\n')
plt.show()
plt.xlim(0,10)
