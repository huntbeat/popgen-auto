"""
hmm_parallel.py
Hunter Lee

Input:  -d  difference sequence in FASTA format, first line must include start
            and end loci
        -b  number of bins
        -p  path to initial probability parameters file
        -t  path to true values
        -i  number of iterations
        -o  path to output folder for figures
        -P  path to output folder for updated probability parameters
        -n  number of processes for task

Example command:
python3 hmm_parallel.py -d dif/sequences_2mu.txt -p initial_parameters_2mu.txt -b 4 -i 1
python3 hmm_parallel.py -d dif/chr12_aldh2.txt -b 20 -i 1
python3 hmm_parallel.py -d dif/msprime_100000_m1e-7_Ne10000_r1e-7_w1.txt -b 6 -i 2 \
                -t dif/TMRCA_msprime_100000_m1e-7_Ne10000_r1e-7_w1.txt
python3 hmm_parallel.py -d dif/msprime_1000000_m1e-7_Ne10000_r1e-7_w10.txt -b 6 -i 2 \
                -t dif/TMRCA_msprime_1000000_m1e-7_Ne10000_r1e-7_w10.txt
python3 hmm_parallel.py -d dif/msprime_1000000_m1e-7_Ne10000_r1e-7_w1.txt -b 6 -i 2 \
                -t dif/TMRCA_msprime_1000000_m1e-7_Ne10000_r1e-7_w1.txt -n 10
python3 hmm_parallel.py -d dif/msprime_1000000_m1e-7_Ne10000_r1e-7_w1popchange.txt -b 6 -i 2 \
                -t dif/TMRCA_msprime_1000000_m1e-7_Ne10000_r1e-7_w1popchange.txt -n 8
python3 hmm_parallel.py -d dif/msprime_1000000_m1e-7_Ne10000_r1e-9_w1.txt -b 6 -i 2 \
                -t dif/TMRCA_msprime_1000000_m1e-7_Ne10000_r1e-9_w1.txt -n 10

python3 hmm_parallel.py -d dif/ms_-N_10000_2_1_-t_4000_-r_4000_1000000_-T_-L.txt -b 30 -i 11 \
                -t dif/TMRCA_ms_-N_10000_2_1_-t_4000_-r_4000_1000000_-T_-L.txt -n 8


python3 hmm_parallel.py -d dif/ms_-N_10000_2_1_-t_4000_-r_4000_100000_-SAA_1000_-SAa_50_-Saa_0_-Sp_0.5_-SF_0_-T_-L.txt -b 24 -i 11 -t dif/TMRCA_ms_-N_10000_2_1_-t_4000_-r_4000_100000_-SAA_1000_-SAa_50_-Saa_0_-Sp_0.5_-SF_0_-T_-L.txt -n 8



python3 hmm_parallel.py -d dif/ms_-N_10000_2_1_-t_4000_-r_4000_1000000_-eN_0.5_1.0_-eN_0.85_0.25_-eN_0.95_0.05_-T_-L.txt -b 30 -i 11 \
                -t dif/ms_-N_10000_2_1_-t_4000_-r_4000_1000000_-eN_0.5_1.0_-eN_0.85_0.25_-eN_0.95_0.05_-T_-L.txt -n 8;
python3 hmm_parallel.py -d dif/ms_-N_10000_2_1_-t_4000_-r_4000_1000000_-SAA_1000_-SAa_50_-Saa_0_-Sp_0.5_-eN_0.5_1.0_-eN_0.85_0.25_-eN_0.95_0.05_-T_-L.txt -b 30 -i 11           -t dif/TMRCA_ms_-N_10000_2_1_-t_4000_-r_4000_1000000_-SAA_1000_-SAa_50_-Saa_0_-Sp_0.5_-eN_0.5_1.0_-eN_0.85_0.25_-eN_0.95_0.05_-T_-L -n 8
python3 hmm_parallel.py -d dif/ms_-N_10000_2_1_-t_4000_-r_4000_1000000_-SAA_1000_-SAa_50_-Saa_0_-Sp_0.5_-T_-L.txt -b 30 -i 11 \
                -t dif/ms_-N_10000_2_1_-t_4000_-r_4000_1000000_-SAA_1000_-SAa_50_-Saa_0_-Sp_0.5_-T_-L.txt -n 8


python3 hmm_parallel.py -d dif/simPSMC.txt -b 32 -i 10 -t dif/TMRCA_simPSMC.txt -n 8;
python3 hmm_parallel.py -d dif/simPSMCsplit.txt -b 32 -i 10 -t dif/TMRCA_simPSMCsplit.txt -n 8;
"""

import gc
import optparse
from math import log
import numpy as np
from hmmlearn import hmm
import sys
import os
import multiprocessing as mp
from tqdm import tqdm
# # turns off plotting
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# turns off plotting
# plt.ioff()
from scipy.stats import expon

"""
from viterbi import Viterbi
from viterbi import FB
from baum_welch_retrieve import BW
"""

from bw import BW

from random import uniform

import os.path

# For defining intervals for log-spaced bins in the exponential distribution
ALPHA = 0.97
# Mutation rate
MU = 1.25*10e-8
# Effective population size
N = 10000
# Theta
THETA = 4*N*MU

def parse_args():
    """Parse and return command-line arguments"""

    parser = optparse.OptionParser(description='HMM for Tmrca')
    parser.add_option('-d', '--dif_fasta_filename', type='string', help='path to input difference sequence')
    parser.add_option('-b', '--num_bins', type='string', help='number of bins desired')
    parser.add_option('-p', '--param_filename', type='string', help='path to input parameter file')
    parser.add_option('-t', '--truth_filename', type='string', help='path to file of true values')
    parser.add_option('-i', '--num_iter', type='int', default=15, help='number of Baum-Welch iterations')
    parser.add_option('-o', '--out_folder', type='string', help='path to folder for output figures')
    parser.add_option('-P', '--out_param', type='string', help='path to folder for updated parameters')
    parser.add_option('-n', '--num_processes', type='string', help='number of processes for task')
    (opts, args) = parser.parse_args()

    mandatories = ['dif_fasta_filename','num_bins','num_iter']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    if opts.out_param == None:
        opts.out_param = 'updated_param'

    if not os.path.exists(opts.out_param):
        os.makedirs(opts.out_param)

    if opts.out_folder == None:
        opts.out_folder = 'fig'

    if opts.num_processes == None:
        opts.num_processes = '1'

    if not os.path.exists(opts.out_folder):
        os.makedirs(opts.out_folder)

    return opts

def parse_params(param_filename, num_bins):
    """ Parse initial parameter file to extract initial, transition, and
    emission probabilities.
    Authors: Andrew H. Chan, Sara Mathieson
    """
    param_file = open(param_filename,'r')
    param_lines = param_file.readlines()
    param_file.close()
    K = num_bins

    # parse init state
    init = np.array([float(l) for l in param_lines[1:1+K]])
    # parse transition matrix
    tran = np.array([[float(x) for x in l.split()] for l in param_lines[K+3:K+3+K]])
    # parse in emit distribution
    emit = np.array([[float(ber) for ber in l.split()] for l in param_lines[2*K+5:2*K+5+K]])

    # convert to log-space
    log_initial = init
    log_transition = tran
    log_emission = emit

    # # convert to log-space
    # log_initial = np.array([log(x) for x in init])
    # log_transition = np.array([[log(x) for x in row] for row in tran])
    # log_emission = np.array([[log(x) for x in row] for row in emit])

    return log_initial, log_transition, log_emission

def display_params(params):
    """Create a parameter string to write to a file (for Part 2).
    Authors: Andrew H. Chan, Sara Mathieson
    """
    init, tran, emit = params

    param_str = '# Initial probabilities\n'
    for i in range(len(init)):
        param_str += str("%.6e" % init[i]) + '\n'

    param_str += '\n# Transition Probabilities\n'
    for i in range(len(tran)):
        row = ''
        for j in range(len(tran)):
            row += str("%.6e" % tran[i][j]) + ' '
        param_str += row + '\n'

    param_str += '\n# Emission Probabilities\n'
    for i in range(len(emit)):
        row = ''
        for j in range(2):
            row += str("%.6e" % emit[i][j]) + ' '
        param_str += row + '\n'

    return param_str

"""
Create a BW object based on arguments, for parallelization
"""

def create_BW(bw_arg):
    return BW(dif_seq=bw_arg[0], log_init=bw_arg[1],
            log_tran=bw_arg[2], log_emit=bw_arg[3], state=bw_arg[4])

"""
Creates log-spaced bins
"""
def create_bins(num_bins):
    bins = [0.0]
    times = []
    quantile = ALPHA / num_bins
    cur_pos = 0.0
    for i in range(num_bins):
        cur_pos += quantile
        right_side = expon.ppf(cur_pos)
        middle = expon.ppf(cur_pos-(quantile/2))
        bins.append(right_side)
        times.append(middle)
    bins = np.array(bins)
    times = np.array(times)
    return bins, times

"""
Slots numbers into appropriate bins
"""
def decoded_to_bins(decoded_array, bins):
    bars = np.zeros((bins.size-1,), dtype=np.int64)
    for i in decoded_array:
        for j in range(len(bins)-1):
            if i <= bins[j+1]:
                bars[j] += 1
                break
    return bars

def main():
    # parse commandline arguments
    opts = parse_args()
    param_filename = opts.param_filename
    if opts.param_filename != None:
        log_init, log_tran, log_emit = parse_params(opts.param_filename, int(opts.num_bins))
    # create bins and appropriate time intervals
    bins, times = create_bins(int(opts.num_bins))
    time_length = times.size
    dif_string = ""
    input_filename = opts.dif_fasta_filename
    input_filename = input_filename[input_filename.rfind('/'):]
    print("This file is %s" % input_filename)
    with open(opts.dif_fasta_filename, 'r') as inputFasta:
        input_string = next(inputFasta).replace("\n","")
        length = int(input_string.split(" ")[9])
        num_indivs = int(input_string.split(" ")[3])
        num_iter = int(input_string.split(" ")[4])
        print("The length of the sequence is %d." % length)
        print("The number of individuals  is %d." % num_indivs)
        print("The number of iterations   is %d." % num_iter)
        input_param = "_".join(input_string.split(" ")[:-3])
        if os.path.isfile('/scratch/saralab/first/TMRCA' + input_filename):
            start_iteration = 0
            output_TMRCA = open('/scratch/saralab/first/TMRCA' + input_filename,'r')
            for line in output_TMRCA:
                if line[-2] == ">":
                    start_iteration = int(line.split(" ")[0])
            start_iteration += 1
            print("File exists, will begin at %d" % start_iteration)
        else:
            output_TMRCA = open('/scratch/saralab/first/TMRCA' + input_filename,'w+')
            output_TMRCA.write(input_param + "\n")
            start_iteration = 1
        output_TMRCA.close()
        next(inputFasta) # rand number
        next(inputFasta) # blank
        for iteration in tqdm(range(start_iteration, 2)):
            output_TMRCA = open('/scratch/saralab/first/TMRCA' + input_filename,'a+')
            next(inputFasta) # "//"
            next(inputFasta) # segsites: __
            pos_string_list = next(inputFasta).split(" ")[1:-1]
            pos_set = set()
            for pos_string in pos_string_list:
                position = int(float(pos_string) * length)
                while position in pos_set:
                    position += 1
                if position >= length:
                    break
                else:
                    pos_set.add(position)
            pos_list = sorted(list(pos_set))
            num_samples = 0
            seg_sequence_list = [] # segregating sites sequence for each individual
            line = next(inputFasta)
            while line != "\n":
                num_samples += 1
                seg_sequence_list.append(line.replace("\n",""))
                line = next(inputFasta)
            pair_left = int(uniform(0,num_indivs))
            pair_right = pair_left
            while (pair_right == pair_left):
                pair_right = int(uniform(0,num_indivs))
            dif_string = ""
            pair_right = (pair_left + 1) % num_indivs
            prev_pos = 0
            for idx0, pos in enumerate(pos_list):
                for i in range(prev_pos, pos):
                    dif_string += "0"
                if (int(seg_sequence_list[pair_left][idx0]) - int(seg_sequence_list[pair_right][idx0])) % 2 == 1:
                    dif_string += '1'
                else:
                    dif_string += '0'
                prev_pos = pos+1
            # after all the SNP's
            for i in range(prev_pos, length):
                dif_string += "0"
            SNP_POS = pos_list

            X_p = []

            bw_posterior_decoding = []
            bw_posterior_decoding_bars = []
            bw_posterior_mean = []
            bw_posterior_mean_bars = []

            print(np.fromstring(dif_string, dtype=int, sep=''))
            import pdb; pdb.set_trace()

            bw_posterior_mean = np.array(bw_posterior_mean)
            # delete
            bw_posterior_mean_bars = decoded_to_bins(bw_posterior_mean, bins)

            posterior_mean_TMRCA = []

            for snp_pos in SNP_POS:
                posterior_mean_TMRCA.append(str(bw_posterior_mean[snp_pos]))

            output_TMRCA.write(str(iteration) + " >>\n")
            output = " ".join(posterior_mean_TMRCA)
            for i in range(num_indivs):
                output_TMRCA.write(output + "\n")
            output_TMRCA.close()

            if True:
                plt.figure(2, figsize=(14,8))
                plt.title('PSMC plot')
                if opts.truth_filename != None:
                    trueTMRCA_bars = decoded_to_bins(trueTMRCA, bins)
                    plt.step(np.insert(times,[0],[0.0]), np.insert(trueTMRCA_bars,[0],[0])*float(1/len(trueTMRCA)), color='black', label='truth')
                plt.step(np.insert(times,[0],[0.0]), np.insert(bw_posterior_mean_bars,[0],[0])*float(1/len(bw_posterior_mean)), color='seagreen', label='post mean')
                plt.xlabel('Years (in coalescent unit)')
                plt.xscale('log')
                plt.ylabel('Population')
                axes = plt.gca()
                axes.set_xlim([0.0,3.0])
                axes.set_ylim([0,0.35])
                xposition = [200/20000, 750/20000]
                for xc in xposition:
                    plt.axvline(x=xc, color='r', linestyle='--')
                plt.legend(loc='upper right')
                plt.show()

            gc.collect()

if __name__ == "__main__":
  main()
