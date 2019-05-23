import vcf
from cyvcf2 import VCF
import csv
import numpy as np
import random
from math import sqrt
from tqdm import tqdm

'''
pick n individuals from 1000 genome file
@param n
'''
def pick_population(csv_file):
  sample_list = []
  with open(csv_file,'r') as csvfile:
    reader = csv.DictReader(csvfile, dialect='excel-tab')
    for row in reader:
        line = row['Sample name']
        sample_list.append(line)
  return sample_list

def find_SNP_start(filename):
    vcf = VCF(filename)
    for record in vcf:
        return record.POS

def cyvcf2input(filename, chrom, sample_size, start, length, sample_list=None):

    # Step 1: declare the desired statistics

    # 1. sequential
    TMRCA = []
    SNP_loci = []
    SNP_num_individuals = []

    # 2. summary
    S = 0
    pi = 0
    T_D = 0

    vcf = VCF(filename)

    if sample_list == None:
        samples = vcf.samples
    else:
        samples = sample_list

    sample_set = set()

    # retrieve a sample from the total sample list
    if sample_size < len(samples):
        for i in range(sample_size):
            chosen = random.choice(samples)
            while chosen in sample_set or chosen not in vcf.samples:
                chosen = random.choice(samples)
            sample_set.add(chosen)
    else:
        print("Sample size bigger than total number of samples")
        return -1

    sample_list = list(sample_set)
    vcf.set_samples(sample_list)

    # find SNP positions and alleles
    pos = start
    for record in vcf(str(chrom) + ":" + str(start) + "-" + str(start+length)):
        individuals_with_SNP = 0
        for genotype in record.genotypes:
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
    assert(len(SNP_num_individuals) == len(SNP_loci))
    assert(S == len(SNP_loci))

    x_input = [SNP_num_individuals, SNP_loci]
    y_input = [S, pi, T_D]

    return x_input, y_input

# TODO think abt either cutting in middle or start
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

def imgs_to_mse(imgs, scalars, pop_stats, total_length):
    MSE = [0 for i in range(len(pop_stats))]
    pop_S = pop_stats[0]
    pop_pi = pop_stats[1]
    pop_T_D = pop_stats[2]
    for img in imgs:
        SNP_num_individuals = (img[0,:,0]*scalars[0]).astype(int).tolist()
        SNP_loci = (img[0,:,1]*scalars[1]).astype(int).tolist()

        zero_indices = [idx for idx, element in enumerate(SNP_num_individuals) if element == 0]

        for index in reversed(zero_indices):
            del SNP_num_individuals[index]
            SNP_loci[index-1] += SNP_loci[index]
            del SNP_loci[index]

        # Step 4: calculate summary statistics
        n = img.shape[0]

        # total number of segregating sites in the entire sequence
        S = len(SNP_loci)

        # pi, number of pairwise differences
        pi = 0
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

        MSE[0] += abs(pop_S - S) ** 2
        MSE[1] += abs(pop_pi - pi) ** 2
        MSE[2] += abs(pop_T_D - T_D) ** 2

        if sum(SNP_loci) < total_length:
            print('shorter')

    MSE = np.array(MSE, dtype='float32')
    MSE = MSE / len(imgs)
    MSE = np.sqrt(MSE)
    print(MSE)

    return np.sum(MSE)

def main():
    pass

if __name__ == '__main__':
    main()
