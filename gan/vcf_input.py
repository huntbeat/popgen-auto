import csv
import vcf
import numpy as np

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
    records = vcf_reader.fetch(chrom=chrom, start=start, end=start+length)
    samples = vcf_reader.samples
    sample_set = set()

    # retrieve a sample from the total sample list
    if sample_size < len(samples):
        for i in range(sample_size):
            chosen = random.choice(samples)
            while chosen in sample_set:
                chosen = random.choice(samples)
            sample_set.add(chosen)
    else:
        print("Sample size bigger than total number of samples")
        return -1

    sample_list = list(sample_set)

    # find SNP positions and alleles

    pos = start
    for record in records:
        S += 1 
        prev_pos = pos
        pos = record.POS
        SNP_loci.append(pos - prev_pos)
        for indiv in sample_list:
            genotype = record.genotype(indiv)['GT'].split("|")
            left_genotype = int(genotype[0])
            right_genotype = int(genotype[1])

    # do TMRCA calculations

    
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
    pass

if __name__ == '__main__':
    main()
