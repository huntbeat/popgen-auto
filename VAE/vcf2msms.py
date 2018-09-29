import optparse
import sys
from cyvcf2 import VCF
import numpy as np

import random
from vcf2input import pick_population
from tqdm import tqdm

def cyvcf2msms(vcf_filename, chrom, sample_size, sample_batch, length, sample_list=None):
    # set new sample list
    vcf = VCF(vcf_filename)
    chrom_start = 0
    chrom_end = 24188367

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

    start_set = set()

    for increment in tqdm(range(sample_batch)):
        while True:
            reformatted_hap_list = []
            reformatted_seg_site_idx_list = []

            sample_size = len(sample_list)
            start = int(random.uniform(chrom_start, chrom_end))
            while start in start_set:
                start = int(random.uniform(chrom_start, chrom_end))
            start_set.add(start)
            pos = start
            for record in vcf(str(chrom) + ":" + str(int(start)) + "-" + str(int(start+length))):
                hap = []
                for genotype in record.genotypes:
                    if int(genotype[0]) == 0:
                        hap.append(0)
                    elif int(genotype[0]) == 1:
                        hap.append(1)
                    elif int(genotype[0]) == 2:
                        hap.append(1)
                    else:
                        hap.append(0)
                if sum(hap) == 0 or sum(hap) == sample_size:
                    pass
                else:
                    pos = record.POS
                    if sample_size - sum(hap) < sum(hap):
                        for idx, snp in enumerate(hap):
                            hap[idx] = 1 - snp
                    reformatted_hap_list.append(hap)
                    reformatted_seg_site_idx_list.append((pos-start))
            if len(reformatted_hap_list) != 0:
                break

        reformatted_hap_list = np.transpose(np.array(reformatted_hap_list))
        reformatted_hap_list = reformatted_hap_list.tolist()

        assert(len(reformatted_hap_list) == sample_size)
        assert(len(reformatted_hap_list[0]) == len(reformatted_seg_site_idx_list))

        # write msms file
        cores = 8
        folder_number = (increment*cores) / sample_batch
        msms_filename =  '/scratch/saralab/VAE/statsZI/example/data/'
        msms_filename += 'demo' + str(int(folder_number)) + '/'
        msms_filename += 'data' + str(int(increment % (sample_batch/cores))) + '.msms'
        msms_file = open(msms_filename, 'w')
        msms_file.write('ms -N '+ '10000' + ' ' + str(sample_size) + ' ' +
                '1 ' + '-t ANYTHING -r ANYTHING '+ str(length) + '\n')
        msms_file.write(str(chrom) + ', block: ' + str(start) + '-' + str(start+length) + '\n')
        msms_file.write('\n//\nsegsites: ' + str(len(reformatted_seg_site_idx_list)) + '\npositions: ')
        msms_file.write(' '.join(['%.5f'%(float(s)/length) for s in reformatted_seg_site_idx_list]) + ' \n')
        for hap in reformatted_hap_list:
            hap = ''.join([str(x) for x in hap])
            msms_file.write(hap + '\n')
        msms_file.write('\n')
        
        msms_file.close()

def main():
    vcf_filename = '/scratch/hlee6/vcf/ALL.chr2.vcf.gz'
    MXL_pop = pick_population(csv_file='igsr_samples.tsv')
    cyvcf2msms(vcf_filename=vcf_filename, chrom=2, sample_size=20, sample_batch=80000, length=100000, sample_list=MXL_pop)

if __name__ == '__main__':
    main()
