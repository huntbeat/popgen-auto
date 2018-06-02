"""
Top-level comment
Note: feel free to modify the starter code below
"""

import optparse
from math import log
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import expon

from viterbi import Viterbi
from viterbi import FB
from baum_welch import BW

NUMBER_BINS = 16
ALPHA = 0.999
LAMBDA = 500
TIMES = []

def parse_args():
    """Parse and return command-line arguments"""

    parser = optparse.OptionParser(description='HMM for Tmrca')
    parser.add_option('-f', '--fasta_filename', type='string', help='path to input fasta file')
    parser.add_option('-p', '--param_filename', type='string', help='path to input parameter file')
    parser.add_option('-t', '--truth_filename', type='string', help='path to file of true values')
    parser.add_option('-i', '--num_iter', type='int', default=15, help='number of Baum-Welch iterations')
    parser.add_option('-o', '--out_folder', type='string', help='path to folder for output files')
    parser.add_option('-s', '--suffix', type='string', help='suffix string to include in output files')
    (opts, args) = parser.parse_args()

    mandatories = ['fasta_filename','truth_filename','out_folder','suffix']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

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

def create_bins():
    left, right = expon.interval(ALPHA)
    left_bins = np.linspace(np.log(left),0,NUMBER_BINS/2)
    right_bins = np.linspace(0,np.log(right),NUMBER_BINS/2)
    left_bins = np.exp(left_bins)
    right_bins = np.exp(right_bins)
    bins = np.concatenate((left_bins,right_bins))
    return LAMBDA * np.insert(bins,[0],0)

def main():

    TIMES = create_bins()
    time_length = TIMES.size
    seq_file_types = 2

    # parse commandline arguments
    opts = parse_args()
    param_filename = opts.param_filename

    # read parameter file
    if opts.param_filename != None:
        log_init, log_tran, log_emit = parse_params(opts.param_filename)
    else:
        log_init = np.ones((time_length,)) / time_length
        log_tran = np.ones((time_length,time_length)) / time_length
        log_emit = np.ones((time_length,seq_file_types)) / seq_file_types

    plt.figure(7)
    plt.plot(TIMES, np.ones((TIMES.shape)))
    plt.savefig("test.png")
    
    import pdb; pdb.set_trace()

    bw = BW(seq_file=opts.fasta_filename, log_init=log_init,
            log_tran=log_tran, log_emit=log_emit, state=TIMES, i=opts.num_iter)

    X_p = bw.X_p_list

    u_log_init = bw.u_log_init
    u_log_tran = bw.u_log_tran
    u_log_emit = bw.u_log_emit

    u_viterbi = Viterbi(seq_file=opts.fasta_filename, log_init=u_log_init,
    log_tran=u_log_tran, log_emit=u_log_emit, state=TIMES).path

    """
    Decodings Estimated
    """
    with open(opts.out_folder + 'decodings_estimated_' + opts.suffix + '.txt', 'w') as outputFile:
        outputFile.write("# Viterbi_decoding posterior_decoding posterior_mean\n")
        for i in range(len(u_viterbi)):
            outputFile.write("{} {} {}\n".format(u_viterbi[i], bw.fb.P_decoded[i], bw.fb.P_mean[i]))

    """
    Likelihoods
    """
    with open(opts.out_folder + 'likelihoods_' + opts.suffix + '.txt', 'w') as outputFile:
        outputFile.write("initial log-likelihood: %f\n" % (fb.X_p) )
        outputFile.write("final log-likelihood: %f\n" % (bw.fb.X_p) )

    """
    Plot Initial
    """
    # Plot the estimated hidden time sequences
    locus = np.array(range(length))

    # Get the true values, read them into an array
    true_tmrca = []
    with open(opts.truth_filename) as f:
        for line in f:
            true_tmrca.append(line.strip("\n"))
    plt.figure(0)
    plt.plot(locus, true_tmrca)
    plt.plot(locus, viterbi)
    plt.plot(locus, posterior_decoding)
    plt.plot(locus, posterior_mean)

    plt.title('Initial Decodings, ' + opts.suffix)
    plt.xlabel('locus')
    plt.ylabel('TMRCA')
    plt.legend(['truth', 'Viterbi', 'Decoding', 'Mean'])
    plt.savefig(opts.out_folder + "plot_initial_" + opts.suffix + ".pdf")


    """
    Plot estimated
    """
    locus = np.array(range(len(u_viterbi)))
    plt.figure(1)
    plt.plot(locus, true_tmrca)
    plt.plot(locus, u_viterbi)
    plt.plot(locus, bw.fb.P_decoded)
    plt.plot(locus, bw.fb.P_mean)

    plt.title('Estimated Decodings, ' + opts.suffix)
    plt.xlabel('locus')
    plt.ylabel('TMRCA')
    plt.legend(['truth', 'Viterbi', 'Decoding', 'Mean'])
    plt.savefig(opts.out_folder + "plot_estimated_" + opts.suffix + ".pdf")

    """
    Estimated Parameters
    """
    estimated_param = display_params([u_log_init, u_log_tran, u_log_emit])
    with open(opts.out_folder + 'estimated_parameters_' + opts.suffix + '.txt', 'w') as outputFile:
        outputFile.write(estimated_param)

    # """
    # Baum-Welch Iteration outputs
    # """
    #
    # for ind, ele in enumerate(X_p):
    #     print("Iteration %d" % (ind))
    #     print(ele)
    #
    # iter_index = range(len(X_p))
    #
    # plt.figure(2)
    # plt.plot(iter_index, X_p)
    # plt.title('Baum-Welch on example dataset, ' + opts.suffix)
    # plt.xlabel('Baum-Welch iteration')
    # plt.ylabel('Log-likelihood, P(X)')
    # plt.show()

if __name__ == "__main__":
  main()
