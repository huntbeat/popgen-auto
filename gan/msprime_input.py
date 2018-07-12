"""
msprime_to_dif.py
Hunter Lee

Input: -l length of sequence, -m mutation_rate, -n effective population size,
       -r recombination rate, -w window size,  -o output folder name
Output: 1. DIF_string, FASTA format file, first line includes above parameters
        2. True TRMCA, FASTA format file, first line includes above parameters

Example command:
python3 msprime_to_dif.py -l 1000000 -m 1e-7 -n 10000 -r 1e-7 -w 1
"""

import msprime
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import os
import optparse

def msprime_to_input(sample_size, length, mu, n_e, recomb):
    
    # Step 1: declare the desired statistics

    # 1. sequential
    TMRCA = []
    SNP_loci = []
    SNP_num_individuals = []

    # 2. summary
    S = 0
    pi = 0
    T_D = 0

    # Step 2: simulate

    tree_sequence = msprime.simulate(sample_size=sample_size, Ne=n_e, \
        length=length, mutation_rate = mu, recombination_rate=recomb)

    # Step 3: calculate sequential statistics

    # TMRCA: separated into regions
    interval = 0
    all_TMRCA = []
    for tree in tree_sequence.trees():
        prev_interval = interval
        interval = int(tree.interval[1]) # (start, end)
        tmrca = tree.tmrca(0,1)     # in units of years
        for i in range(prev_interval, interval):
            all_TMRCA.append(tmrca / (2* n_e))

    # SNP_loci and SNP_num_individuals
    pos = 0
    for variant in tree_sequence.variants():
        num_individuals_with_SNP = 0
        prev_pos = pos
        pos = int(variant.site.position)  # locus
        geno = variant.genotypes          # [0,1]
        # if (pos-prev_pos) < window and not printed:
        #     number_rep += 1
        #     printed = True
        for i in range(prev_pos, pos-1):
            pass
        for index in range(sample_size):
            if geno[index] == 1:
                num_individuals_with_SNP += 1
        if num_individuals_with_SNP > sample_size / 2:
            num_individuals_with_SNP = sample_size - num_individuals_with_SNP
        SNP_num_individuals.append(num_individuals_with_SNP)
        SNP_loci.append(pos-prev_pos)
        TMRCA.append(all_TMRCA[pos if pos < length else length-1])
    for i in range(length - pos):
        pass

    # Step 4: calculate summary statistics
    n = sample_size

    # total number of segregating sites in the entire sequence
    S = len(SNP_loci)

    # pi, number of pairwise differences
    for freq in SNP_num_individuals:
        pi += freq*(n-freq)
    pi = pi / (n*(n-1)/2)

    # Tajima's D, with variance
    a_1 = sum([1/i for i in range(1,n)])
    b_1 = (n + 1) / (3*(n-1))
    c_1 = b_1 - (1 / a_1)
    e_1 = c_1 / a_1

    a_2 = sum([1/(i*i) for i in range(1,n)])
    b_2 = (2*(n*n + n + 3))/ (9*n *(n-1))
    c_2 = b_2 - (n+2)/(a_1 * n) + a_2 / (a_1*a_1)
    e_2 = c_2 / (a_1 * a_1 + a_2)

    d = pi - S / a_1
    var = sqrt(e_1 * S + e_2 * S * (S - 1))
    T_D = d / var
    
    # print(TMRCA, SNP_loci, SNP_num_individuals, S, pi, T_D)
    print(len(TMRCA))
    print(len(SNP_loci))
    print(len(SNP_num_individuals))
    print(S)
    print(pi)
    print(T_D)
    
    return None


def main():
    _ = msprime_to_input(sample_size=20, length=100000, mu=1e-7, n_e=10000, recomb=1e-7)

if __name__ == '__main__':
    main()
