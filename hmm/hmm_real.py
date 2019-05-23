"""
hmm_cnn.py
Hunter Lee

Input:  -d  difference sequence in FASTA format, first line must include start
            and end loci
        -b  number of bins
        -p  path to initial probability parameters file
        -t  path to true values
        -i  number of iterations
        -o  path to output folder for figures
        -n  number of processes for task

Example command:
python3 hmm_real.py -d /scratch/saralab/first/chrom2_MXL.msms -n 8 -p /scratch/saralab/first/million.txt -i 1 -b 16
"""
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
    if opts.param_filename != None:
        log_init, log_tran, log_emit = parse_params(opts.param_filename, int(opts.num_bins))
    # if parameter path not given, automatically create initial probabilities
    else:
        init = np.ones((time_length,)) / time_length
        tran = np.ones((time_length, time_length)) / time_length
        GIVEN = 0.001
        tran *= GIVEN
        for i in range(time_length):
            tran[i,i] = 1 - GIVEN
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
        output_TMRCA = open('/scratch/saralab/first/TMRCA' + input_filename.replace('.msms','.txt'),'w+')
        output_TMRCA.write(input_param + "\n")
        input_string = next(inputFasta) # rand number
        length = 100000
        num_indivs = 20
        next(inputFasta) # blank
        num_times = 2391
        for iteration in tqdm(range(1, num_times+1)):
            output_TMRCA.write(">>\n")
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
            while line != "\n":
                num_samples += 1
                seg_sequence_list.append(line.replace("\n",""))
                line = next(inputFasta)
            # for pair_left in tqdm(range(1, num_indivs+1)):
            for pair_left in tqdm(range(1)):
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

                dif_string = dif_string_0
                SNP_POS = pos_list

                # Baum-Welch
                if int(opts.num_processes) <= 0:
                    print("ERROR: Number of processes is too low - must be one or higher.")
                    return -1
                num_processes = int(opts.num_processes)

                X_p = []

                bw_posterior_mean = []
                posterior_mean_TMRCA = []

                bw_arg = [(dif_string[i*(int(len(dif_string)/num_processes)):(i+1)*(int(len(dif_string)/num_processes))], log_init, log_tran, log_emit, times, False) for i in range(num_processes)]
                for num_iterations in tqdm(range(1,opts.num_iter+1)):
                    pool = mp.Pool(processes=num_processes)
                    all_bw = pool.map(create_BW, bw_arg)
                    if num_iterations == opts.num_iter:
                        for bw in all_bw:
                            bw_posterior_mean.extend(bw.fb.P_mean.tolist())
                    pool.close()

                bw_posterior_mean = np.array(bw_posterior_mean)

                for snp_pos in SNP_POS:
                    # posterior_mean_TMRCA.append(str(bw_posterior_mean[int(snp_pos/window)]))
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
                    posterior_mean_TMRCA.append("%.8f" % TMRCA_avg)


                for pair_left in range(num_indivs):
                    output = " ".join(posterior_mean_TMRCA)
                    output_TMRCA.write(output + "\n")

                gc.collect()

        output_TMRCA.close()

if __name__ == "__main__":
  main()
