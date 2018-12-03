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
"""
REAL_LENGTH = int(5e6)

import pyximport; pyximport.install()
import cython
import gc
import optparse
from math import log
import numpy as np
import sys
import os
import multiprocessing as mp
from tqdm import tqdm
# turns off plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# turns off plotting
plt.ioff()
from scipy.stats import expon

from bw_cnn import BW

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
            log_tran=bw_arg[2], log_emit=bw_arg[3], state=bw_arg[4], update=bw_arg[5])

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
    bins, times = create_bins(int(opts.num_bins))
    time_length = times.size
    dif_string = ""
    param_filename = opts.param_filename
    if opts.param_filename != None:
        log_init, log_tran, log_emit = parse_params(opts.param_filename, int(opts.num_bins))
    else:
        init = np.ones((time_length,)) / time_length
        tran = np.ones((time_length, time_length)) / time_length
        given = 0.001
        tran *= given
        for i in range(time_length):
            tran[i,i] = 1 - given
        emit = np.ones((time_length, 2))
        for i in range(time_length):
            exp_emit_prob = np.exp(-THETA*times[i])
            emit[i,0] = exp_emit_prob
            emit[i,1] = 1 - exp_emit_prob
        log_init, log_tran, log_emit = np.log(init), np.log(tran), np.log(emit)
    # create bins and appropriate time intervals
    input_filename = opts.dif_fasta_filename
    input_filename = input_filename[input_filename.rfind('/'):]
    print("This file is %s" % input_filename)
    with open(opts.dif_fasta_filename, 'r') as inputFasta:
        input_string = next(inputFasta).replace("\n","")
        input_param = "_".join(input_string.split(" ")[:-3])
        next(inputFasta) # rand number
        length = REAL_LENGTH
        num_indivs = 20
        num_iter = 1
        print("The length of the sequence is %d." % length)
        print("The number of individuals  is %d." % num_indivs)
        print("The number of iterations   is %d." % num_iter)
        next(inputFasta) # blank
        for iteration in (range(1)):
            next(inputFasta) # "//"
            next(inputFasta) # segsites: __
            pos_string_list = next(inputFasta).split(" ")[1:-1]
            pos_set = set()
            for pos_string in pos_string_list:
                position = int(float(pos_string) * length)
                while position in pos_set:
                    position += 1
                pos_set.add(position)
            pos_list = sorted(list(pos_set))
            num_samples = 0
            seg_sequence_list = [] # segregating sites sequence for each individual
            line = next(inputFasta)
            window = 10
            while line != "\n":
                num_samples += 1
                seg_sequence_list.append(line.replace("\n",""))
                line = next(inputFasta)
            for pair_left in tqdm(range(1, num_indivs+1)):
                dif_string_0 = ""
                dif_string_1 = ""
                pair_left -= 1
                pair_right = (pair_left + 1) % num_indivs
                prev_pos = 0

                for idx0, pos in enumerate(pos_list):
                    for i in range(prev_pos, pos):
                        dif_string_0 += "0"
                    if (int(seg_sequence_list[pair_left][idx0]) - int(seg_sequence_list[pair_right][idx0])) % 2 == 1:
                        dif_string_0 += '1'
                    else:
                        dif_string_0 += '0'
                    prev_pos = pos+1
                # after all the SNP's
                for i in range(prev_pos, length):
                    dif_string_0 += "0"

                # # for windows
                # for i in range(0, length, window):
                #     found = False
                #     for j in range(i, i+window):
                #         if j in pos_list:
                #             if seg_sequence_list[pair_left][pos_list.index(j)] != seg_sequence_list[pair_right][pos_list.index(j)]:
                #                 for i in range(1):
                #                     dif_string_1 += '1'
                #                 found = True
                #                 break
                #     if not found:          
                #         for i in range(1):
                #             dif_string_1 += '0'

                dif_string = dif_string_0
                SNP_POS = pos_list

                # Baum-Welch
                if int(opts.num_processes) <= 0:
                    print("ERROR: Number of processes is too low - must be one or higher.")
                    return -1
                num_processes = int(opts.num_processes)

                X_p = []


                bw_posterior_mean = []
                bw_posterior_mean_bars = []

                bw_arg = [(dif_string[i*(int(len(dif_string)/num_processes)):(i+1)*(int(len(dif_string)/num_processes))], log_init, log_tran, log_emit, times, True) for i in range(num_processes)]
                for num_iterations in tqdm(range(1,opts.num_iter+1)):
                    u_log_init = np.zeros(log_init.shape)
                    u_log_tran = np.zeros(log_tran.shape)
                    u_log_emit = np.zeros(log_emit.shape)
                    X_p_ele = 0.0
                    new_bw_arg = []
                    pool = mp.Pool(processes=num_processes)
                    all_bw = pool.map(create_BW, bw_arg)
                    for bw in all_bw:
                        u_log_init += bw.u_log_init / num_processes
                        u_log_tran += bw.u_log_tran / num_processes
                        u_log_emit += bw.u_log_emit / num_processes
                        X_p_ele += bw.fb.X_p / num_processes
                    for arg in bw_arg:
                        arg_list = list(arg)
                        arg_list[1] = u_log_init
                        arg_list[2] = u_log_tran
                        arg_list[3] = u_log_emit
                        new_bw_arg.append(tuple(arg_list))
                    bw_arg = new_bw_arg
                    X_p.append(X_p_ele)
                    if num_iterations == opts.num_iter:
                        for bw in all_bw:
                            bw_posterior_mean.extend(bw.fb.P_mean.tolist())
                    pool.close()

                for snp_pos in SNP_POS:
                    if snp_pos >= length:
                        print("SNP was bigger, let's check some time.")
                        snp_pos = length-1
                    neighbors = 4
                    TMRCA_avg = 0
                    sig = sum(list(range(neighbors+1)))
                    for neighbor in range(neighbors):
                        left = snp_pos - neighbor
                        right = snp_pos + neighbor
                        if left < 0:
                            left = 0
                        elif left >= length:
                            left = length - 1
                        if right < 0:
                            right = 0
                        elif right >= length:
                            right = length -1
                        ratio = (1/2) * (neighbors - neighbor) / sig 
                        TMRCA_avg += (bw_posterior_mean[left] + bw_posterior_mean[right]) * ratio

                bw_posterior_mean = np.array(bw_posterior_mean)
                bw_posterior_mean_bars = decoded_to_bins(bw_posterior_mean, bins)

                posterior_mean_TMRCA = []

                for snp_pos in SNP_POS:
                    posterior_mean_TMRCA.append(str(bw_posterior_mean[snp_pos]))

                # Plot the estimated hidden time sequences
                locus = np.array(list(range(length)))

                """
                Estimated Parameters
                """
                estimated_param = display_params([u_log_init, u_log_tran, u_log_emit])
                with open(opts.out_param + "/" + input_filename, 'w') as outputFile:
                     outputFile.write(estimated_param)

                bars = np.array(bw_posterior_mean_bars)
                bars = bars / np.average(bars)
                demography = []
                for i in range(len(bars)):
                    demography.append('-eN')
                    demography.append(str(bins[i]))
                    demography.append(str(bars[i]))
                demography_string = ' '.join(demography)
                demography_file = open('../simulationTest/MXL_demo.txt','w')
                demography_file.write(demography_string)
                demography_file.close()
                gc.collect()

                plt.figure(0, figsize=(14,8))
                plt.title('locus - TMRCA : BW')
                plt.plot(locus, bw_posterior_mean, color='seagreen', label='post mean')
                if opts.truth_filename != None:
                    plt.plot(np.arange(1,len(trueTMRCA)+1), trueTMRCA, color='black', label='truth')
                plt.xlabel('locus')
                plt.ylabel('TMRCA')
                plt.legend(loc='upper right')
                #plt.savefig(opts.dif_fasta_filename.replace('.msms','_lengthwise.png'))
                plt.savefig('check_lengthwise.png')
                plt.show()
                if True:
                    plt.close()
                    plt.figure(2, figsize=(14,8))
                    plt.title('PSMC plot')
                    if opts.truth_filename != None:
                        trueTMRCA_bars = decoded_to_bins(trueTMRCA, bins)
                        plt.step(np.insert(times,[0],[0.0]), np.insert(trueTMRCA_bars,[0],[0])*float(1/len(trueTMRCA)), color='black', label='truth')
                    plt.step(np.insert(times,[0],[0.0]), np.insert(bw_posterior_mean_bars,[0],[0])/np.average(np.array(bw_posterior_mean_bars)), color='seagreen', label='post mean')
                    plt.xlabel('Years (in coalescent unit)')
                    plt.xscale('log')
                    plt.ylabel('Population')
                    axes = plt.gca()
                    ##axes.set_xlim([0.0,3.0])
                    ## axes.set_ylim([0,0.35])
                    #xposition = [200/20000, 750/20000]
                    #for xc in xposition:
                    #    plt.axvline(x=xc, color='r', linestyle='--')
                    plt.legend(loc='upper right')
                    #plt.savefig(opts.dif_fasta_filename.replace('.msms','.png'))
                    plt.savefig('check_TMRCA.png')
                    plt.show()

                """
                Baum-Welch Iteration outputs
                """
                # for ind, ele in enumerate(X_p):
                #     print("Iteration %d" % (ind))
                #     print(ele)

                plt.close()
                plt.figure(3, figsize=(14,8))
                plt.plot(np.arange(1,len(X_p)+1), X_p)
                plt.title('Baum-Welch accuracy plot')
                plt.xlabel('Baum-Welch iteration')
                plt.ylabel('Log-likelihood, P(X)')
                plt.savefig('test_hmm/BW_training.png', format='png')
                plt.show()

if __name__ == "__main__":
  main()
