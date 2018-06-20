""" create neural network for constant, bottleneck, and natural selection"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D, Dropout, AveragePooling1D, Input
from keras import optimizers

# fetch data
file_ = sys.argv[1]
path = '/scratch/nhoang1/'+file_
with h5py.File(path,'r') as f:
  constant_h5 = f.get('constant')
  bottleneck_h5 = f.get('bottleneck')
  natselect_h5 = f.get('naturalselection')
  pop_h5 = f.get('pop_output')
  TD_h5 = f.get('TD_output')
  constant = np.array(constant_h5)
  bottleneck = np.array(bottleneck_h5)
  natselect = np.array(natselect_h5)
  pops = np.array(pop_h5)
  TDs = np.array(TD_h5)

seqs = np.concatenate((constant,bottleneck,natselect)) 

# shuffle data before training
seqs_shape_before = seqs.shape
pop_shape_before = pops.shape
TD_shape_before = TDs.shape
s = np.arange(TDs.shape[0])
np.random.shuffle(s)
seqs = seqs[s]
pops = pops[s]
TDs = TDs[s]
#assert(seqs_shape_before == seqs.shape)
assert(pop_shape_before == pops.shape)
assert(TD_shape_before == TDs.shape)

# split into train/validation and test
si = int(TDs.shape[0]*0.8)
xtrain = seqs[:si]
ytrain_pop = pops[:si]
ytrain_TD = TDs[:si]
xtest = seqs[si:]
ytest_pop = pops[si:]
ytest_TD = TDs[si:]

# stats
print("data shape:",seqs.shape)
print("single num type:",type(seqs[0][0][0]))
print("pop output shape:",pops.shape)
print("TD output shape:",TDs.shape)
print("pop output type:",type(pops[0]))
print("TD output type:",type(TDs[0]))
print("num train:",xtrain.shape[0],",num test:",xtest.shape[0])

# Schrider's network, modified for multi output

ksize = 2
l2_lambda = 0.0001
n, L = xtrain.shape[1:]

seq_input = Input(shape=(n,L))
conv1 = Conv1D(128*2, kernel_size=ksize, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_lambda))(seq_input)
conv2 = Conv1D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda))(conv1)
maxpool1 = MaxPooling1D(pool_size=ksize)(conv2)
dropout1 = Dropout(0.2)(maxpool1)
conv3 = Conv1D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda))(dropout1)
maxpool2 = MaxPooling1D(pool_size=ksize)(conv3)
dropout2 = Dropout(0.2)(maxpool2)
conv4 = Conv1D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda))(dropout2)
avgpool1 = AveragePooling1D(pool_size=ksize)(conv4)
dropout3 = Dropout(0.2)(avgpool1)
conv5 = Conv1D(128*2, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda))(dropout3)
dropout4 = Dropout(0.2)(conv5)
flatten1 = Flatten()(dropout4)
dense1 = Dense(256, activation='relu', kernel_initializer='normal',kernel_regularizer=keras.regularizers.l2(l2_lambda))(flatten1)
dropout5 = Dropout(0.25)(dense1)
pop_output = Dense(3, activation='softmax', name='pop')(dropout5)
TD_output = Dense(3, activation='softmax', name='TD')(dropout5)

model = Model(inputs=[seq_input], outputs=[pop_output,TD_output])
model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
print(model.summary())

history = model.fit([xtrain], [ytrain_pop, ytrain_TD], batch_size=64,
        epochs=10,
        verbose=1,
        validation_split=0.2)

model.save('TD_pop_model.hdf5')
model.save_weights('TD_pop_weights.hdf5')

loss, pop_loss, TD_loss, pop_acc, TD_acc = model.evaluate(x=xtest,y=[ytest_pop,ytest_TD])
print("pop accuracy:", pop_acc)
print("TD accuracy:", TD_acc)

# confusion matrix on the test set
'''
print("\n\n**********************************")
print("\n**********************************")
'''
# save pred vs true to file
num_samples, n2, L2 = xtest.shape
predictions = model.predict(xtest)
pop_preds = predictions[0].argmax(axis=-1)
TD_preds = predictions[1].argmax(axis=-1)
with open("preds_TD_pop.txt", "w") as f:
  for n in range(num_samples):
    line = str(ytest_pop[n]) + "," + str(ytest_TD[n]) + ":" + str(pop_preds[n]) + "," + str(TD_preds[n]) + "\n"
    f.write(line)

# accuracy plot
plt.plot(history.history['pop_acc'], marker='v')
plt.plot(history.history['TD_acc'], marker='o')
plt.plot(history.history['val_pop_acc'], marker='v')
plt.plot(history.history['val_TD_acc'], marler='o')
plt.title('Model Training Accuracy')
plt.ylabel('accuracy')
plt.ylim(0,1)
plt.xlabel('epoch')
plt.xlim(0, 10)
plt.legend(['train pop', 'train TD', 'val pop', 'val TD'], loc='lower right')
plt.show()
