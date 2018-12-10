import optparse
from keras.layers import Input, Dense
from keras.models import Model, load_model 
from keras.datasets import mnist
from keras import regularizers, optimizers

import tensorflow as tf

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
import random
import subprocess
import sys
import time
from tqdm import tqdm
import simulation_helpers

# load model
ENCODER = load_model('encoder_m.hdf5')
ENCODER.load_weights('encoder_w.hdf5')
AUTOENCODER = load_model('autoencoder_m.hdf5')
AUTOENCODER.load_weights('autoencoder_w.hdf5')

# simulation parameters

NUM_PER_S = 2

Ne = 10000 # baseline effective population size
L = 100000  # region length: this fits with how we are doing the selection stats
MU = 1.25e-8 # from "Direct estimation of per nucleotide and genomic deleterious mutation rates in Drosophila" (2007)
R =  1.25e-8 # set to same as theta, as inferred by PMSC
THETA = 4*Ne*MU
RHO   = 4*Ne*R
THETA_REGION = int(L * THETA)
RHO_REGION   = int(L * RHO)
SAMPLE_SIZE  = 20 # we have up to 197 ZI individuals

# HARDSWEEP PARAMETERS, fixed for now, except selection location

# time the selection starts
#SEL_START = 0.005
SEL_START = 0.05
#SEL_START = 0.5

# selection strength
S_LST = [0.01/2]

# hard sweep occurs in one individual
SEL_FREQ = float(1)/Ne

# selection location
MIN_SEL_LOC = 0.4 
MAX_SEL_LOC = 0.6

DEMO = '-eN 0.041256116515351784 3.175392 -eN 0.12925483585726238 1.3056576 -eN 0.2257511147212906 2.8159248 -eN 0.33256322012449757 2.0676696 -eN 0.45216371111740455 1.3860624 -eN 0.5880366961573283 0.8607815999999999 -eN 0.7453180528888548 0.2549952 -eN 0.9320390888422941 0.057753599999999995 -eN 1.161818364571902 0.03078 -eN 1.4606587764087526 0.0431688 -eN 1.8888211820711855 0.0015647999999999999 -eN 2.653325301412962 0.0002496' 

DEMO = '-eN 0.041256116515351784 3.175392 -eN 0.2257511147212906 2.8159248 -eN 0.7453180528888548 0.2549952 -eN 2.653325301412962 0.0002496' 

# example command: msms -N 100000 -ms 100 1 -t 336 -r 336 100000 -eN 0 4.754958 -eN 0.05 0.668148 -eN 0.2 3.163320 -Sp 0.5 -SI 0.003 1 0.0001 -SAA 10000 -SAa 5000 -Saa 0 -Smark > msms/hs/demo0/data0.msms

def msms_commandline(var_params):
    msms_str = 'java -jar msms3.2rc-b163.jar -N ' + str(Ne) + ' -ms ' + str(SAMPLE_SIZE) + ' 1 '
    msms_str += '-t ' + str(int(L * var_params[0])) + ' '
    msms_str += '-r ' + str(RHO_REGION) + ' ' + str(L) + ' '
    msms_str += DEMO

    S = S_LST[int(p/NUM_PER_S)]
    SI  = str(SEL_START) + ' 1 ' + str(SEL_FREQ)
    SAA_BAL = 2*Ne*S # homozygote with selected allele
    SAa_BAL =   Ne*S # heterozygote half double selection strength of homozygote
    Saa_BAL =      0 # homozygote with unselected allele
    # msms_str += ' -Sp ' + str(sel_loc) + ' -SAA ' + str(SAA_BAL) + ' -SAa ' + str(SAa_BAL) + ' -Saa ' + str(Saa_BAL) + ' -SI ' + SI + ' -Smark'

    return msms_str

def stat2list(stat_filename):
    stat_file = open(stat_filename, 'r')
    header = next(stat_file).split(" ")
    stats = []
    for line in stat_file:
        stats.append(list(map(float,line.split(" ")))) 
    stat_file.close()
    return stats

def pipeline(var_params, num_sims = 5):
    # 1. simulate MSMS sequences
    msms_command = msms_commandline(var_params)
    for i in range(num_sims):
        msms_command += ' > ' + 'example/data/demo7/data' + str(i) + '.msms'
        subprocess.call(msms_command, shell=True)

    # 2. calculate summary statistics
    # subprocess
    stat_command = 'java -jar -Xmx5G statsZI.jar --beginDemo=7 --endDemo=8 --numPerDemo=1 --msmsFolder=example/data/ --statsFolder=example/stats/'
    subprocess.call(stat_command, shell=True)
    
    # 3. translate summary statistics to autoencoder data
    # turn txt file to input file
    stats = stat2list('example/stats/stats_7.txt')
    stats = np.array(stats)
    stats -= np.amin(stats, axis=1).reshape(-1, 1)
    stats *= 1/np.amax(stats, axis=1).reshape(-1, 1)
    #f = h5py.File("/scratch/saralab/VAE/input/" +
    #      stat_filename[stat_filename.find('stats'):.replace('.txt','.h5'), "w")
    f = h5py.File("/scratch/saralab/VAE/input/" +
          "pipeline_0.h5", "w")
    dset = f.create_dataset("VAE", data=stats)
    # f.close()

    # f = h5py.File(INPUT_FILE, 'r')
    x_all = f['VAE']
    x_all = np.array(x_all)
    x_all = (x_all - x_all.min(axis=0)) / (x_all.max(axis=0) - x_all.min(axis=0))
    original_dim = x_all.shape[1]

    # run it, get output from function
    print(x_all)
    y_all = encoder.predict(x_all)

    print(y_all)

    y_all = autoencoder.predict(x_all)

    print(y_all)
    print(np.abs(y_all-x_all))
    print(np.sum(x_all))

    return y_all

def optimize():
    # 4. optimize
    # Adam optimizer
    # x: varying parameters (# of input * # of varying parameters)
    # f(x): cost (# of input * MSE)

    # 1. Get the autoencoded statistics of the subsequence of interest

    answer = tf.constant([1,2,3,4,5], dtype=tf.float64, shape=[1,5])

    # 2. Set variable parameters input vector

    # 3. Retrieve autoencoded statistics of input vector

    # 4. Calculate loss

    # 5. Create Adam optimizer with all the above

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    minimize = optimizer.minimize(cost)

    # 6. Create session
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(minimize, feed_dict={X:X_data, Y: Y_data})

if __name__ == '__main__':
    optimize()
