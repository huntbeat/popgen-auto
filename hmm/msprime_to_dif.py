"""
msprime_to_dif.py
Hunter Lee

Input: -l length of sequence, -m mutation_rate, -n effective population size,
       -r recombination rate, -w window size,  -o output folder name
Output: 1. DIF_string, FASTA format file, first line includes above parameters
        2. True TRMCA, FASTA format file, first line includes above parameters

Example command:
python3 msprime_to_dif.py -l 1000000 -m 1e-7 -n 10000 -r 1e-9 -w 100
"""

import msprime
import numpy as np
import matplotlib.pyplot as plt
import os
import optparse

def parse_args():
    """Parse and return command-line arguments"""

    parser = optparse.OptionParser(description='MSprime simulator to difference sequence')
    parser.add_option('-l', '--length', type='string', help='desired length of sequence')
    parser.add_option('-m', '--mu', type='string', help='mutation rate')
    parser.add_option('-n', '--n_e', type='string', help='effective pop size')
    parser.add_option('-r', '--recomb', type='string', help='recombination rate')
    parser.add_option('-w', '--window', type='string', help='window size')
    parser.add_option('-s', '--win_stat', type='string', help='statistic choice for TMRCA')
    parser.add_option('-o', '--out_folder', type='string', help='path to output files folder')
    (opts, args) = parser.parse_args()

    mandatories = ['length','mu','n_e','recomb']

    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    if opts.out_folder == None:
        opts.out_folder = "dif"

    if opts.window == None:
        opts.window = "1"

    if opts.win_stat == None:
        opts.win_stat = "mean"

    if not os.path.exists(opts.out_folder):
        os.makedirs(opts.out_folder)

    return opts

def msprime_to_dif(length, mu, n_e, recomb, window, win_stat):
    TMRCA = np.array([])
    SEQ_1 = ""
    SEQ_2 = ""
    SEQ_D = ""

    # simulate with n=2, N=10000, and L=10000
    tree_sequence = msprime.simulate(sample_size=2, Ne=n_e, \
        length=length, mutation_rate = mu, recombination_rate=recomb)

    # see when the Tmrca changes
    interval = 0
    for tree in tree_sequence.trees():
        prev_interval = interval
        interval = int(tree.interval[1]) # (start, end)
        tmrca = tree.tmrca(0,1)     # in units of years
        for i in range(prev_interval, interval):
            TMRCA = np.append(TMRCA, tmrca / (2 * n_e))
    old_TMRCA = TMRCA
    TMRCA = np.array([])
    for i in range(0,length,window):
        if win_stat == 'mean':
            TMRCA = np.append(TMRCA, np.average(old_TMRCA[i:i+window]))

    # see when mutations occur
    pos = 0
    printed = False
    number_rep = 0
    for variant in tree_sequence.variants():
        prev_pos = pos
        pos = int(variant.site.position)  # locus
        geno = variant.genotypes          # [0,1]
        if (pos-prev_pos) < window and not printed:
            number_rep += 1
            printed = True
        for i in range(prev_pos, pos-1):
            SEQ_1 += '0'
            SEQ_2 += '0'
            SEQ_D += '0'
        if geno[0] == '1':
            SEQ_1 += '1'
            SEQ_2 += '0'
            SEQ_D += '1'
        else:
            SEQ_1 += '0'
            SEQ_2 += '1'
            SEQ_D += '1'
    if printed:
        print("WARNING: there are more than one mutation within a window. Maybe make the window smaller.")
        print(number_rep)
    for i in range(length - len(SEQ_D)):
        SEQ_1 += '0'
        SEQ_2 += '0'
        SEQ_D += '0'
    old_SEQ_D = SEQ_D
    SEQ_D = ""
    for i in range(0,length,window):
        SEQ_D += "1" if old_SEQ_D[i:i+window].find("1") != -1 else "0"
    return TMRCA, SEQ_D

def main():
    opts = parse_args()
    TMRCA, SEQ_D = msprime_to_dif(int(opts.length),
                            float(opts.mu), int(opts.n_e), float(opts.recomb), int(opts.window), opts.win_stat)
    out_filename = "msprime_" + opts.length + "_m" + opts.mu + "_Ne" + opts.n_e + "_r" + opts.recomb 
    with open(opts.out_folder + "/" + out_filename + ".txt", 'w') as outputFile:
        outputFile.write(">> " + out_filename.replace("_", " ") + "\n")
        for i in range(0, len(SEQ_D), 100):
            outputFile.write(SEQ_D[i:i+100] + "\n")

    with open(opts.out_folder + "/" + "TMRCA_" + out_filename + ".txt", 'w') as outputFile:
        outputFile.write(">> " + "TMRCA " + out_filename.replace("_", " ") + "\n")
        for i in TMRCA:
            outputFile.write(str(i) + "\n")

if __name__ == '__main__':
    main()
