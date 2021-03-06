from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

# turns off plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# turns off plotting
plt.ioff()

import numpy as np
from vcf2input import *
from random import uniform
from tqdm import tqdm
import h5py

DATASET = 80000

class INFOGAN():
    def __init__(self):
        self.filename = '/scratch/hlee6/vcf/ALL.chr21.vcf.gz'
        self.chrom = 21
        self.sample_size = 12
        self.length = 10e4
        self.START = find_SNP_start(filename=self.filename) - self.length
        self.END = 48119740 - self.length
        self.pop_stats = [0 for i in range(3)] # stats: S, pi T_D 

        self.channel_max = [] # first channel: num_individuals, second channel: SNP distance

        self.n = self.sample_size
        self.l = 400

        self.img_rows = self.n
        self.img_cols = self.l
        self.channels = 2
        self.num_classes = 10
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 72

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', self.mutual_info_loss]

        # Build and the discriminator and recognition network
        self.discriminator, self.auxilliary = self.build_disk_and_q_net()

        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the recognition network Q
        self.auxilliary.compile(loss=[self.mutual_info_loss],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        gen_input = Input(shape=(self.latent_dim,))
        img = self.generator(gen_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        valid = self.discriminator(img)
        # The recognition network produces the label
        target_label = self.auxilliary(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(gen_input, [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        new_rows = int(self.img_rows/4)
        new_cols = int(self.img_cols/4)

        model.add(Dense(128 * new_rows * new_cols, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((new_rows, new_cols, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        gen_input = Input(shape=(self.latent_dim,))
        img = model(gen_input)

        model.summary()

        return Model(gen_input, img)


    def build_disk_and_q_net(self):

        img = Input(shape=self.img_shape)

        # Shared layers between discriminator and recognition network
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())

        img_embedding = model(img)

        # Discriminator
        validity = Dense(1, activation='sigmoid')(img_embedding)

        # Recognition
        q_net = Dense(128, activation='relu')(img_embedding)
        label = Dense(self.num_classes, activation='softmax')(q_net)

        # Return discriminator and recognition network
        return Model(img, validity), Model(img, label)


    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy

    def sample_generator_input(self, batch_size):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, 62))
        sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)

        return sampled_noise, sampled_labels

    def train(self, epochs, batch_size=128, sample_interval=50):

        X_train = []
        y_train = []
          
        dataset = DATASET 

        random_start = int(uniform(self.START, self.END))
        random_start = 39988628

        sample_list = pick_population(csv_file='igsr_samples.tsv')

        for i in tqdm(range(dataset)):
            x_input, y_input = cyvcf2input(filename= self.filename,
            chrom=self.chrom, sample_size=self.sample_size,
            start=random_start, length=self.length, sample_list=sample_list)
            print(y_input)
            for idx, stat in enumerate(y_input):
               self.pop_stats[idx] += stat / dataset
            X_input = pad_and_tile(x_input, length=self.img_cols, rows=self.img_rows)
            X_train.append(X_input)
            y_train.append([y_input[-1]])

        y_train = np.array(y_train)
        transposed_y_train = place_bins(np.transpose(y_train), num_bins=self.num_classes)

        X_train = np.array(X_train, dtype='float32')
        y_train = np.transpose(transposed_y_train)

        # Rescale -1 to 1
        # TODO: find each channel then normalize (half the max, divide by max)
        # TODO: architecture is prob set for one channel datasets, ours has two

        for channel in range(X_train.shape[1]):
            self.channel_max.append(np.amax(X_train[:,channel]))
            X_train[:,channel] = (X_train[:,channel] - float(self.channel_max[-1]/2)) / self.channel_max[-1]
        # Move channel axis to last
        X_train = np.moveaxis(X_train, 1, -1)
        y_train = y_train.reshape(-1, 1)

        with h5py.File('/scratch/saralab/ganInput.h5','w') as hf:
            hf.create_dataset("X_train", data=X_train)

        with h5py.File('/scratch/saralab/ganOutput.h5','w') as hf:
            hf.create_dataset("y_train", data=y_train)

        with h5py.File('/scratch/saralab/ganInput.h5','r') as hread:
            X_train = hread['X_train'][:]

        with h5py.File('/scratch/saralab/ganOutput.h5','r') as hread:
            y_train = hread['y_train'][:]

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and categorical labels
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(gen_input)

            # Train on real and generated data
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            # Avg. loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator and Q-network
            # ---------------------

            g_loss = self.combined.train_on_batch(gen_input, [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 10, 10

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(c)
            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)
            gen_input = np.concatenate((sampled_noise, label), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            gen_imgs = 0.5 * gen_imgs + 0.5
            print("MSE between pop_stats and gan_stats: %f" % imgs_to_mse(gen_imgs, self.channel_max, self.pop_stats, self.length))
            for j in range(r):
                axs[j,i].imshow(gen_imgs[j,:,:,0], cmap='gray')
                axs[j,i].axis('off')
        plt.title("%.2f" % imgs_to_mse(gen_imgs, self.channel_max, self.pop_stats, self.length))
        # fig.savefig("images/%d.png" % epoch)
        plt.close()


    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    infogan = INFOGAN()
    infogan.train(epochs=50000, batch_size=128, sample_interval=20)
