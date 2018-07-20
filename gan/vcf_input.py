import vcf
import csv
import numpy as np
import random
from math import sqrt
from tqdm import tqdm

'''
pick n individuals from 1000 genome file
@param n
'''
def pick_individuals(n, csv_file):
  indiv_names = []
  with open(csv_file,'r') as csvfile:
    reader = csv.DictReader(csvfile, dialect='excel-tab')
    count = 0
    for row in reader:
      if (count < n) and ('phase 3' in row['Data collections']):
        line = row['Sample name']
        indiv_names.append(line)
        count += 1
      else: break
    if count < n: print("Warning: less than",n,"samples in TSV file")
  return indiv_names

def find_SNP_start(filename, chrom):
    vcf_reader = vcf.Reader(filename=filename)
    records = vcf_reader.fetch(chrom=str(chrom))
    return next(records).POS

def vcf_to_input(filename, chrom, sample_size, start, length):

    # Step 1: declare the desired statistics

    # 1. sequential
    TMRCA = []
    SNP_loci = []
    SNP_num_individuals = []

    # 2. summary
    S = 0
    pi = 0
    T_D = 0

    vcf_reader = vcf.Reader(filename=filename)
    records = vcf_reader.fetch(chrom=str(chrom), start=start, end=start+length)
    # samples = vcf_reader.samples
    # sample_set = set()

    # # retrieve a sample from the total sample list
    # if sample_size < len(samples):
    #     for i in range(sample_size):
    #         chosen = random.choice(samples)
    #         while chosen in sample_set:
    #             chosen = random.choice(samples)
    #         sample_set.add(chosen)
    # else:
    #     print("Sample size bigger than total number of samples")
    #     return -1
    #
    # sample_list = list(sample_set)

    # 10 people from MXC, top
    samples = pick_individuals(n=sample_size, csv_file='igsr_samples.tsv')
    sample_list = samples
    sample_list = ['NA19658', 'NA19649', 'NA19651', 'NA19723', 'NA19719', 'NA19728', 'NA19657', 'NA19664', 'NA19735', 'NA19669']

    # find SNP positions and alleles
    pos = start
    for record in records:
        individuals_with_SNP = 0
        for indiv in sample_list:
            genotype = record.genotype(indiv)['GT'].split("|")
            left_genotype = int(genotype[0])
            # right_genotype = int(genotype[1])
            individuals_with_SNP += left_genotype
        if individuals_with_SNP == 0 or individuals_with_SNP == sample_size:
            pass
        else:
            S += 1
            prev_pos = pos
            pos = record.POS
            SNP_loci.append(pos - prev_pos)
            if sample_size - individuals_with_SNP < individuals_with_SNP:
                individuals_with_SNP = sample_size - individuals_with_SNP
            SNP_num_individuals.append(individuals_with_SNP)

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
    T_D = d / var if var != 0 else 0

    # print(SNP_loci, SNP_num_individuals, S, pi, T_D)
    print("Number of segregating sites: %d" % len(SNP_loci))
    assert(len(SNP_num_individuals) == len(SNP_loci))
    assert(S == len(SNP_loci))
    print("Average pairwise difference : %.3f" % pi)
    print("Tajima's D: %.3f" % T_D)

    x_input = [SNP_loci, SNP_num_individuals]
    y_input = [T_D]

    return x_input, y_input

def pad_and_tile(list_of_vectors, length, rows):
    lists = [[] for i in range(len(list_of_vectors))]
    for idx, input_vector in enumerate(list_of_vectors):
        cols = len(input_vector)
        if cols >= length:
            np_vector = np.array(input_vector[:length])
            for i in range(rows): lists[idx].append(np_vector)
        else: # less
            padding_width = length - cols
            zeros = np.zeros((padding_width,), dtype='int32')
            half = int(padding_width/2)
            np_vector = np.concatenate((zeros[:half],input_vector,zeros[half:]))
            for i in range(rows): lists[idx].append(np_vector)
    input_matrix = np.array(lists)
    # will return numpy array for input
    return input_matrix

def place_bins(list_output_vectors, num_bins):
    output_matrix = []
    length = len(list_output_vectors[0])
    for vector in list_output_vectors:
        # step 1 : create the bins
        sorted_vector = sorted(vector)
        bins = [sorted_vector[int(length / num_bins * (i+1)) - 1]
                        for i in range(num_bins)]
        # step 2 : place into bins
        output_vector = []
        for element in vector:
            for bin_number, interval in enumerate(bins):
                if element <= interval:
                    output_vector.append(bin_number)
                    break
        output_matrix.append(output_vector)
    return np.array(output_matrix)

def main():
    print("Minimum SNP start idx: %d" % find_SNP_start(filename='/scratch/hlee6/vcf/ALL.chr21.vcf.gz', chrom=21))
    vcf_to_input(filename='/scratch/hlee6/vcf/ALL.chr21.vcf.gz', chrom=21, sample_size=10, start=9500e3, length=10e3)

if __name__ == '__main__':
    main()
