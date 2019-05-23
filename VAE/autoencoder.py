from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers

import numpy as np

import h5py

# turns off plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# turns off plotting
plt.ioff()
import argparse
import os

INPUT_FILE = '/scratch/saralab/VAE/input/test_0.h5'

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(222,))

# # "encoded" is the encoded representation of the input
# encoded = Dense(encoding_dim, activation='relu')(input_img)
# with sparsity
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(222, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# (x_train, _), (x_test, _) = mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

f = h5py.File(INPUT_FILE, 'r')
x_all = f['VAE']
x_all = np.array(x_all)
x_all = (x_all - x_all.min(axis=0)) / (x_all.max(axis=0) - x_all.min(axis=0))
original_dim = x_all.shape[1]
cut = int(x_all.shape[0] * (8/10))
x_train = x_all[:cut].astype('float32')
y_train = np.ones_like(x_train[:,1])
x_test = x_all[cut:].astype('float32')
y_test = np.ones_like(x_train[:,1])

print (x_train.shape)
print (x_test.shape)

autoencoder.summary()
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoder.save('statsZI/encoder_m.hdf5')
encoder.save_weights('statsZI/encoder_w.hdf5')

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(37, 6))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(37, 6))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('autotest.png')
plt.show()

