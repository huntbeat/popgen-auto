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
# import cProfile
# import re
# cProfile.run('re.compile("main()")')
import optparse
from math import log
import numpy as np
import sys
import os
import multiprocessing as mp
from tqdm import tqdm
## turns off plotting
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
# # turns off plotting
# plt.ioff()
from scipy.stats import expon

from viterbi import Viterbi
from viterbi import FB
from baum_welch_parallel import BW
from bw import BW

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
    trueTMRCA = []
    if opts.truth_filename != None:
        with open(opts.truth_filename, 'r') as truthFile:
            next(truthFile)
            for line in truthFile:
                trueTMRCA.append(float(line))
    trueTMRCA = np.array(trueTMRCA)

    # read parameter file
    if opts.param_filename != None:
        log_init, log_tran, log_emit = parse_params(opts.param_filename, int(opts.num_bins))
    # if parameter path not given, automatically create initial probabilities
    else:
        init = np.ones((time_length,)) / time_length
        tran = np.ones((time_length, time_length)) / time_length
        emit = np.ones((time_length, 2))
        GIVEN = 0.001
        tran *= GIVEN
        for i in range(time_length):
            tran[i,i] = 1 - GIVEN
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
    if int(opts.num_processes) <= 0:
        print("ERROR: Number of processes is too low - must be one or higher.")
        return -1
    num_processes = int(opts.num_processes)

    X_p = []

    bw_posterior_decoding = []
    bw_posterior_decoding_bars = []
    bw_posterior_mean = []
    bw_posterior_mean_bars = []

    bw_arg = [(dif_string[i*(int(len(dif_string)/num_processes)+1):(i+1)*(int(len(dif_string)/num_processes)+1)], log_init, log_tran, log_emit, times, True) for i in range(num_processes)]
    for iteration in tqdm(range(1,opts.num_iter+1)):
        X_p_ele = 0.0
        new_bw_arg = []

        u_log_init = np.zeros(log_init.shape)
        u_log_tran = np.zeros(log_tran.shape)
        u_log_emit = np.zeros(log_emit.shape)
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
        pool.close()

    for bw in all_bw:
        u_log_init += bw.u_log_init / num_processes
        u_log_tran += bw.u_log_tran / num_processes
        u_log_emit += bw.u_log_emit / num_processes

        bw_posterior_decoding.extend(bw.fb.P_decoded.tolist())
        bw_posterior_mean.extend(bw.fb.P_mean.tolist())

    bw_posterior_decoding = np.array(bw_posterior_decoding)
    bw_posterior_mean = np.array(bw_posterior_mean)
    bw_posterior_decoding_bars = decoded_to_bins(bw_posterior_decoding, bins)
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

    plt.figure(0, figsize=(14,8))
    plt.title('locus - TMRCA : BW')
    plt.plot(locus, bw_posterior_decoding, color='royalblue', label='post decoding')
    plt.plot(locus, bw_posterior_mean, color='seagreen', label='post mean')
    if opts.truth_filename != None:
        plt.plot(np.arange(1,len(trueTMRCA)+1), trueTMRCA, color='black', label='truth')
    plt.xlabel('locus')
    plt.ylabel('TMRCA')
    plt.legend(loc='upper right')
    plt.savefig(opts.out_folder + "/" + inputFasta.replace(".txt", "_bw_PL_line_" + "b" + opts.num_bins + ".png"), format='png')
    plt.show()

    plt.figure(1, figsize=(14,8))
    width = 0.30
    plt.title('Number of loci within TMRCA interval : BW')
    if opts.truth_filename != None:
        trueTMRCA_bars = decoded_to_bins(trueTMRCA, bins)
        plt.bar(np.arange(1,int(opts.num_bins)+1) + width + width, trueTMRCA_bars, width, color='black', label='truth')
    plt.bar(np.arange(1,int(opts.num_bins)+1) + width, bw_posterior_decoding_bars, width, color='royalblue', label='post decoding')
    plt.bar(np.arange(1,int(opts.num_bins)+1), bw_posterior_mean_bars, width, color='seagreen', label='post mean')
    plt.xlabel('TMRCA bins')
    plt.ylabel('number of loci')
    plt.legend(loc='upper left')
    plt.savefig(opts.out_folder + "/" + inputFasta.replace(".txt", "_bw_PL_bar_" + "b" + opts.num_bins + ".png"), format='png')
    plt.show()

    plt.figure(2, figsize=(14,8))
    plt.title('PSMC plot')
    if opts.truth_filename != None:
        trueTMRCA_bars = decoded_to_bins(trueTMRCA, bins)
        plt.step(np.insert(times,[0],[0.0]), np.insert(trueTMRCA_bars,[0],[0])*float(1/len(trueTMRCA)), color='black', label='truth')
    plt.step(np.insert(times,[0],[0.0]), np.insert(bw_posterior_decoding_bars,[0],[0])*float(1/len(bw_posterior_decoding)), color='royalblue', label='post decoding')
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
    plt.savefig(opts.out_folder + "/" + inputFasta.replace(".txt", "_bw_PL_step_" + "b" + opts.num_bins + ".png"), format='png')
    plt.show()

    """
    Text timeline of population changes
    """
    if opts.truth_filename != None:
        TMRCA_text = ""
        first_row = "TIME\t\tBIN_START\t\tBIN_END\t\test. pop. size\tact. pop. size\n"
        TMRCA_text += first_row
        for idx, time in enumerate(times):
            new_row = "%.6f\t\t%.6f\t\t%.6f\t\t%.6f\t\t%.6f\n" % (times[idx], bins[idx], bins[idx+1], bw_posterior_mean_bars[idx]/len(bw_posterior_mean), trueTMRCA_bars[idx]/len(trueTMRCA))
            TMRCA_text += new_row
        print(TMRCA_text)
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
    plt.figure(3, figsize=(14,8))
    plt.plot(np.arange(1,len(X_p)+1), X_p)
    plt.title('Baum-Welch accuracy plot')
    plt.xlabel('Baum-Welch iteration')
    plt.ylabel('Log-likelihood, P(X)')
    plt.savefig('fig/BW_training.png', format='png')
    plt.show()

if __name__ == "__main__":
  main()
