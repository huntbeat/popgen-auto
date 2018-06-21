"""
Parses MSMS file, retrieves summary statistics

"""

import numpy as np
from math import sqrt
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import optparse
import sys

#---------------
# READ VCF FILE
#---------------

def parse_msms(filename, num_iterations):
    """Read msms file line by line, parsing out SNP information"""
    D_list = []
    S_list = []
    with open(filename, 'r') as msms_file:
        input_string = next(msms_file)
        input_param = input_string.replace("\n","").split(" ")[:-2]
        input_string = "".join(input_param)
        # ms -N 10000 25 50000 -t 40 -r 40 100000 -SAA 100 -SAa 50 -Saa 0 -Sp 0.5 -SF 0
        Ne = int(input_param[2])
        sample_size = int(input_param[3])
        total_length = int(input_param[9])
        # num_iterations = int(input_param[4])

        next(msms_file) # rand number
        next(msms_file) # ""
        # the start of a loop -----------------------------------------------
        for i in range(num_iterations):
            next(msms_file) # //
            # segsites: 888
            seg_string = next(msms_file)
            total_snps = int(seg_string.split(" ")[1])
            window = total_length
            genomic_locations = []

            # Start and end
            pos_start = 0
            pos_end = total_length

            # Random S
            S = 0

            # We want to look at base pair regions in the genome instead of grouping
            # together SNPS (which could be across a long region), because related genes
            # are next to each other
            # If a SNP falls into one of these regions, save that there.
            # bp_regions = list(range(112131266,112352266,1000))
            bp_regions = list(range(pos_start // window * window, pos_end // window * window + window, window))

            bp_buckets = dict((el,[]) for el in bp_regions)
            num_indivs = {} # {k: genomic_location, v: num_indivs}

            pos_string_list = next(msms_file).split(" ")[1:-1]
            # positions: 0.00141 0.00249 0.00277 0.00328 0.00450 0.00453
            # ["0.00141", "0.00249", ...]
            seq_string_all = []
            for person in range(sample_size):
                line = next(msms_file).replace("\n", "")
                if line != "":
                    seq_string_all.append(line)
            pos_set = set()
            idx_pos_list = []
            for idx, pos_string in enumerate(pos_string_list):
                pos = int(float(pos_string)*total_length)
                while pos in pos_set:
                    pos += 1
                pos_set.add(pos)
                idx_pos_list.append((idx,pos))

            for idx_pos in idx_pos_list:
                idx, pos = idx_pos
                genomic_locations.append(pos)
                bp_buckets[pos // window * window].append(pos)
                num = 0 # number of those that have the SNP
                for indiv_seq_string in seq_string_all:
                    if indiv_seq_string[idx] == "1":
                            num += 1
                            S += 1
                num_indivs[pos] = num

            D = calculate_D(bp_buckets, genomic_locations, num_indivs, sample_size, window, input_string, pos_start, pos_end)

            D_list.append(D)
            S_list.append(S)

            _ = next(msms_file) # should be "" empty line
        return D_list, S_list

def calculate_D(bp_buckets, genomic_locations, num_indivs, n, window, input_string, pos_start, pos_end):

    # See how many SNPS are in each bucket region, and write to file
    # for k,v in bp_buckets.items():
    #     print("Region",k, "-", k+window-1, ":", len(v))

    # Calculate Tajima's D with theoretical variance (Wikipedia)
    # a_1
    a_1 = sum([1/i for i in range(1,n)])
    b_1 = (n + 1) / (3*(n-1))
    c_1 = b_1 - (1 / a_1)
    e_1 = c_1 / a_1

    a_2 = sum([1/(i*i) for i in range(1,n)])
    b_2 = (2*(n*n + n + 3))/ (9*n *(n-1))
    c_2 = b_2 - (n+2)/(a_1 * n) + a_2 / (a_1*a_1)
    e_2 = c_2 / (a_1 * a_1 + a_2)

    D_map, d_map, pi_map, S_map, var_map = {}, {}, {}, {}, {}
    # <k: region, v: tajima's (big) D>

    for region, snp_locations in bp_buckets.items():
        S = len(snp_locations) # num snps in side region of size window
        if S != 0:
            pi = 0
            for snp in sorted(snp_locations):
                freq = num_indivs[snp]
                pi += freq*(n-freq)
            pi = pi / (n*(n-1)/2)

            # Tajima's d
            d = pi - S / a_1
            var = sqrt(e_1 * S + e_2 * S * (S - 1))
            D = d / var
            d_map[region] = d
            D_map[region] = D
            pi_map[region] = pi
            S_map[region] = S
            var_map[region] = var
        else:
            D_map[region] = 0
            d_map[region] = 0
            pi_map[region] = 0
            S_map[region] = 0
            var_map[region] = 0

    D_list, bp_list, d_list, pi_list, S_list, var_list = [], [], [], [], [], []
    return d
