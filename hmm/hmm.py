"""
hmm.py
Hunter Lee

Input:  -d  difference sequence in FASTA format, first line must include start
            and end loci
        -b  number of bins
        -p  path to initial probability parameters file
        -t  path to true values
        -i  number of iterations
        -o  path to output folder for figures
        -P  path to output folder for updated probability parameters

Example command:
python3 hmm.py -d dif/sequences_2mu.txt -p initial_parameters_2mu.txt -b 4 -i 1
python3 hmm.py -d dif/chr12_aldh2.txt -b 20 -i 1
python3 hmm.py -d dif/msprime_100000_m1e-7_Ne10000_r1e-7_w1.txt -b 6 -i 2 \
                -t dif/TMRCA_msprime_100000_m1e-7_Ne10000_r1e-7_w1.txt
python3 hmm.py -d dif/msprime_1000000_m1e-7_Ne10000_r1e-7_w10.txt -b 6 -i 2 \
                -t dif/TMRCA_msprime_1000000_m1e-7_Ne10000_r1e-7_w10.txt
"""

import optparse
from math import log
import numpy as np
import sys
import os
# # turns off plotting
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# # turns off plotting
# plt.ioff()
from scipy.stats import expon

from viterbi import Viterbi
from viterbi import FB
from baum_welch import BW

# For defining intervals for log-spaced bins in the exponential distribution
ALPHA = 0.999
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

    if not os.path.exists(opts.out_folder):
        os.makedirs(opts.out_folder)

    return opts

def parse_params(param_filename):
    """ Parse initial parameter file to extract initial, transition, and
    emission probabilities.
    Authors: Andrew H. Chan, Sara Mathieson
    """
    param_file = open(param_filename,'r')
    param_lines = param_file.readlines()
    param_file.close()
    K = 4

    # parse init state
    init = np.array([float(l.split()[1]) for l in param_lines[2:2+K]])
    # parse transition matrix
    tran = np.array([[float(x) for x in l.split()] for l in param_lines[11:11+K]])
    # parse in emit distribution
    emit = np.array([[float(l.split()[1]), float(l.split()[2])] for l in param_lines[19:19+K]])

    # convert to log-space
    log_initial = np.array([log(x) for x in init])
    log_transition = np.array([[log(x) for x in row] for row in tran])
    log_emission = np.array([[log(x) for x in row] for row in emit])

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
Creates log-spaced bins
"""
def create_bins(num_bins):
    bins = np.array([0.0])
    times = np.array([])
    quantile = ALPHA / num_bins
    cur_pos = 0.0
    for i in range(num_bins):
        cur_pos += quantile
        right_side = expon.ppf(cur_pos)
        left_side = bins[-1]
        bins = np.append(bins,right_side)
        times = np.append(times,(left_side+right_side)/2)
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
    dif_string = ""
    with open(opts.dif_fasta_filename, 'r') as inputFasta:
        next(inputFasta)
        for line in inputFasta:
            dif_string += line.replace("\n","")
    inputFasta = opts.dif_fasta_filename.replace("dif/","")

    # create bins and appropriate time intervals
    bins, times = create_bins(int(opts.num_bins))
    time_length = times.size

    # read truth file
    trueTMRCA = np.array([])
    if opts.truth_filename != None:
        with open(opts.truth_filename, 'r') as truthFile:
            next(truthFile)
            for line in truthFile:
                trueTMRCA = np.append(trueTMRCA, float(line))

    # read parameter file
    if opts.param_filename != None:
        log_init, log_tran, log_emit = parse_params(opts.param_filename)
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

    # Test whether time and bin making works
    # plt.figure(7)
    # plt.scatter(times, np.ones((times.shape)))
    # plt.scatter(BINS, 2*np.ones(BINS.shape))
    # plt.savefig("test.png")

    # # Forward-Backward
    # fb = FB(dif_seq=dif_string, log_init=log_init,
    #        log_tran=log_tran, log_emit=log_emit, state=times)

    # posterior_decoding = fb.P_decoded
    # posterior_decoding_bars = decoded_to_bins(posterior_decoding, BINS)
    # posterior_mean = fb.P_mean
    # posterior_mean_bars = decoded_to_bins(posterior_mean, BINS)

    # Baum-Welch
    bw = BW(dif_seq=dif_string, log_init=log_init,
            log_tran=log_tran, log_emit=log_emit, state=times, i=opts.num_iter)

    X_p = bw.X_p_list

    u_log_init = bw.u_log_init
    u_log_tran = bw.u_log_tran
    u_log_emit = bw.u_log_emit

    bw_posterior_decoding = bw.fb.P_decoded
    bw_posterior_decoding_bars = decoded_to_bins(bw_posterior_decoding, bins)
    bw_posterior_mean = bw.fb.P_mean
    bw_posterior_mean_bars = decoded_to_bins(bw_posterior_mean, bins)

   #  """
   #  Decodings Estimated
   #  """
   #  with open(opts.out_folder + 'decodings_estimated_' + opts.suffix + '.txt', 'w') as outputFile:
   #      outputFile.write("# Viterbi_decoding posterior_decoding posterior_mean\n")
   #      for i in range(len(u_viterbi)):
   #          outputFile.write("{} {} {}\n".format(u_viterbi[i], bw.fb.P_decoded[i], bw.fb.P_mean[i]))

   #  """
   #  Likelihoods
   #  """
   #  with open(opts.out_folder + 'likelihoods_' + opts.suffix + '.txt', 'w') as outputFile:
   #      outputFile.write("initial log-likelihood: %f\n" % (fb.X_p) )
   #      outputFile.write("final log-likelihood: %f\n" % (bw.fb.X_p) )

    """
    Plot Initial
    """
    # Plot the estimated hidden time sequences
    length = len(bw_posterior_decoding)
    locus = np.array(list(range(length)))

    plt.figure(0)
    plt.title('locus - TMRCA : BW')
    plt.plot(locus, bw_posterior_decoding, color='royalblue', label='post decoding')
    plt.plot(locus, bw_posterior_mean, color='seagreen', label='post mean')
    if opts.truth_filename != None:
        plt.plot(np.arange(1,len(trueTMRCA)+1), trueTMRCA, color='black', label='truth')
    plt.xlabel('locus')
    plt.ylabel('TMRCA')
    plt.legend(loc='upper right')
    plt.savefig(opts.out_folder + "/" + inputFasta.replace(".txt", "_bw_line.png"), format='png')
    plt.show()

    plt.figure(1)
    width = 0.30
    plt.title('Number of loci within TMRCA interval : BW')
    plt.bar(np.arange(1,int(opts.num_bins)+1) + width, bw_posterior_decoding_bars, width, color='royalblue', label='post decoding')
    plt.bar(np.arange(1,int(opts.num_bins)+1), bw_posterior_mean_bars, width, color='seagreen', label='post mean')
    if opts.truth_filename != None:
        trueTMRCA_bars = decoded_to_bins(trueTMRCA, bins)
        plt.bar(np.arange(1,int(opts.num_bins)+1) + width + width, trueTMRCA_bars, width, color='black', label='truth')
    plt.xlabel('TMRCA bins')
    plt.ylabel('number of loci')
    plt.legend(loc='upper left')
    plt.savefig(opts.out_folder + "/" + inputFasta.replace(".txt", "_bw_bar.png"), format='png')
    plt.show()
    """
    SANITY
    """
    print(len(bw_posterior_mean))
    print(len(bw_posterior_decoding))

    # """
    # Plot estimated
    # """
    # locus = np.array(range(len(u_viterbi)))
    # plt.figure(1)
    # plt.plot(locus, true_tmrca)
    # plt.plot(locus, u_viterbi)
    # plt.plot(locus, bw.fb.P_decoded)
    # plt.plot(locus, bw.fb.P_mean)
    #
    # plt.title('Estimated Decodings, ' + opts.suffix)
    # plt.xlabel('locus')
    # plt.ylabel('TMRCA')
    # plt.legend(['truth', 'Viterbi', 'Decoding', 'Mean'])
    # plt.savefig(opts.out_folder + "plot_estimated_" + opts.suffix + ".pdf")

    """
    Estimated Parameters
    """
    estimated_param = display_params([u_log_init, u_log_tran, u_log_emit])
    with open(opts.out_param + "/" + inputFasta.replace(".txt",'upd_param.txt'), 'w') as outputFile:
        outputFile.write(estimated_param)

    """
    Baum-Welch Iteration outputs
    """
    # for ind, ele in enumerate(X_p):
    #     print("Iteration %d" % (ind))
    #     print(ele)

    plt.close()
    plt.figure(2)
    plt.plot(np.arange(1,len(X_p)+1), X_p)
    plt.title('Baum-Welch accuracy plot')
    plt.xlabel('Baum-Welch iteration')
    plt.ylabel('Log-likelihood, P(X)')
    plt.savefig('fig/BW_training.png', format='png')
    plt.show()

if __name__ == "__main__":
  main()
