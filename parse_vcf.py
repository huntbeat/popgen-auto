"""
Parses VCF file to

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

def parse_vcf(filename):
    """Read VCF file line by line, parsing out SNP information"""
    vcf_file = open(filename, 'r')
    super_pop = filename[filename.find('/')+1:filename.find('_')]
    total_snps = 0 # count snps
    window = 5000
    genomic_locations = []

    # Start and end for rs671
    # snp range: 112117221 - 112352228
    gene_start_loc = 111800000
    gene_end_loc = 112600000
    # LCT
    # gene_start_loc = int(1.350E8)
    # gene_end_loc = int(1.360E8)


    # We want to look at base pair regions in the genome instead of grouping
    # together SNPS (which could be across a long region), because related genes
    # are next to each other
    # If a SNP falls into one of these regions, save that there.
    # bp_regions = list(range(112131266,112352266,1000))
    bp_regions = list(range(gene_start_loc // window * window, gene_end_loc // window * window + window, window))

    bp_buckets = dict((el,[]) for el in bp_regions)
    num_indivs = {} # {k: genomic_location, v: num_indivs}

    for line in vcf_file:
        if not line.startswith("##"): # ignore headers
            tokens = line.split()

            # get the sample size n
            if line.startswith("#"):
                n = len(tokens[9:])*2 # each sample has two chroms

            # parse each SNP
            else:
                # SNP location
                pos = int(tokens[1])
                # count of "1s" (alternate allele)
                count1 = "".join(tokens[9:]).count("1")

                # make sure SNP is segregating
                if 0 < count1 < n:
                    # TODO do something with SNP (printing every 100th snp now)
                    total_snps += 1
                    genomic_locations.append(pos)
                    # Figure out which bucket it should go into
                    bp_buckets[pos // window * window].append(pos)
                    num_indivs[pos] = count1
    vcf_file.close()
    return bp_buckets, genomic_locations, num_indivs, n, window, super_pop

if __name__ == '__main__':
  parse_vcf():
