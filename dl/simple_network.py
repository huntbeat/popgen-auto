""" create simple neural network """

import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import keras
from keras.models import Sequential 
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D
from keras import optimizers

# fetch data 
path = '/scratch/nhoang1/data.hdf5'
with h5py.File(path,'r') as f:
  constant_h5 = f.get('constant')
  bottleneck_h5 = f.get('bottleneck')
  output_h5 = f.get('output')
  constant = np.array(constant_h5)
  bottleneck = np.array(bottleneck_h5)
  output = np.array(output_h5).astype(int)
data = np.concatenate((constant,bottleneck))

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
'''
trainX = data[:100]
trainY = output[:100]
testX = data[100:200]
testY = output[100:200]
'''

# neural network 
n, L = trainX.shape[1:]
trainX = np.reshape(trainX,(-1,n,L,1))
model = Sequential()
model.add(Conv2D(128, (5,5), activation='relu', input_shape=(n,L,1)))
#model.add(Conv2D(128, (4,4), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
#model.add(Conv1D(128, 5, activation='relu', input_shape=(n, L)))
#model.add(MaxPooling1D())
#model.add(Conv1D(128, 4, activation='relu'))
#model.add(MaxPooling1D())
#model.add(Conv1D(128, 5, activation='relu'))
#model.add(MaxPooling1D())
#model.add(Conv1D(128, 5, activation='relu'))
#model.add(MaxPooling1D())
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

SGD = optimizers.SGD()
model.compile(optimizer=SGD, loss="binary_crossentropy", metrics=['accuracy'])
history = model.fit(trainX, trainY, verbose=1, validation_split=0.2, epochs=10)

# confusion matrix on the test set
num_samples, n2, L2 = testX.shape
testX = np.reshape(testX,(-1,n2,L2,1))
predictions = model.predict_classes(testX, verbose=1)
predictions = np.reshape(predictions,(num_samples))
true_labels = np.reshape(testY,(num_samples)) 
orig_labels = [0,1] 

print("\n\n**********************************")
confuse = defaultdict(lambda: defaultdict(int))
for lab1 in orig_labels:
  for lab2 in orig_labels:
    confuse[lab1][lab2] = 0
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
print("**********************************")

# save pred vs true to file
with open("predVStrue2.txt", "w") as f:
  for n in range(num_samples):
    line = str(true_labels[n]) + " " + str(predictions[n]) + "\n"
    f.write(line)

# accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
