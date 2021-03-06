""" create simple neural network """

import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

import keras
from keras.models import Sequential 
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D, Dropout, AveragePooling1D
from keras import optimizers

# fetch data 
file_ = sys.argv[1]
path = '/scratch/nhoang1/'+file_
with h5py.File(path,'r') as f:
  constant_h5 = f.get('constant')
  bottleneck_h5 = f.get('bottleneck')
  output_h5 = f.get('output')
  constant = np.array(constant_h5)
  bottleneck = np.array(bottleneck_h5)
  #output = np.array(output_h5, dtype='int32')
  output = np.array(output_h5)

#constant = np.zeros_like(constant, dtype='int32')
#bottleneck = np.ones_like(bottleneck, dtype='int32')

data = np.concatenate((constant,bottleneck))
#output = np.reshape(output,(output.shape[0]))

# shuffle data before training
data_shape_before = data.shape
output_shape_before = output.shape
s = np.arange(output.shape[0])
np.random.shuffle(s)
data = data[s]
output = output[s]
assert(data_shape_before == data.shape)
assert(output_shape_before == output.shape)

# split into train/validation and test
si = int(output.shape[0]*0.8)
trainX = data[:si]
trainY = output[:si]
testX = data[si:]
testY = output[si:]

# stats
print("data shape:",data.shape)
print("single num type:",type(data[0][0][0]))
print("output shape:",output.shape)
print("output type:",type(output[0]))
print("num train:",trainX.shape[0],",num test:",testX.shape[0])
c = 0 
b = 0
'''for i in trainY:
  if i == 0: c += 1
  elif i == 1: b+= 1
print("num train const:",c,",num train bottl:",b)'''
'''
# neural network 
n, L = trainX.shape[1:]
trainX = np.reshape(trainX,(-1,n,L,1))
model = Sequential()
model.add(Conv2D(128, (5,5), activation='relu', input_shape=(n,L,1)))
model.add(Conv2D(128, (4,4), activation='relu'))
#model.add(Conv1D(128, 5, activation='relu', input_shape=(n, L)))
#model.add(MaxPooling1D())
#model.add(Conv1D(128, 4, activation='relu'))
#model.add(MaxPooling1D())
#model.add(Conv1D(128, 5, activation='relu'))
#model.add(MaxPooling1D())
#model.add(Conv1D(128, 5, activation='relu'))
#model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='relu'))
model.summary()
sgd = optimizers.SGD()
adam = optimizers.Adam()
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
history = model.fit(trainX, trainY, verbose=1, validation_split=0.2, epochs=10)

'''
# paper's network

batch_size = 3
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
model.add(Dense(2, activation='softmax'))
    
model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
print(model.summary())

history = model.fit(trainX, trainY, batch_size=64,
        epochs=10,
        verbose=1,
        validation_split=0.2)

# confusion matrix on the test set
num_samples, n2, L2 = testX.shape
testX = np.reshape(testX,(-1,n2,L2))
predictions = model.predict_classes(testX)
#predictions = np.reshape(predictions,(num_samples))
#true_labels = np.reshape(testY,(num_samples)) 
true_labels = testY
orig_labels = np.unique(testY,axis=0)
print("pred type:",type(predictions[0]),"true type:",type(true_labels[0]))
'''
print("\n\n**********************************")
confuse = defaultdict(lambda: defaultdict(int))
for lamodel in orig_labels:
  for lab2 in orig_labels:
    confuse[lamodel][lab2] = 0
for i in range(len(predictions)):
  confuse[true_labels[i]][predictions[i]] += 1
mat_str = "\n\t Prediction \n%11s" % sorted(confuse.keys())[0]
for lab in sorted(confuse.keys())[1:]:
  mat_str += "%6s" % lab
mat_str += "\n"
for row in sorted(confuse.keys()):
  mat_str += "%5s " % row
  for col in sorted(confuse[row].keys()):
    mat_str += "%5d " % confuse[row][col]
  mat_str += "\n"
print(mat_str)

# print acccuracy
correct = 0
for t,p in zip(true_labels,predictions):
  if t == p:
    correct += 1'''
loss, acc = model.evaluate(testX,testY)
#print('Accuracy (manual):',correct/predictions.shape[0])
print('Accuracy (model):',acc)
print("\n**********************************")

# save pred vs true to file
with open("predVStrue8.txt", "w") as f:
  for n in range(num_samples):
    line = str(true_labels[n]) + ":" + str(predictions[n]) + "\n"
    f.write(line)

# accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
