import optparse
import sys
from cyvcf2 import VCF
import numpy as np

import random
from vcf2input import pick_population
from tqdm import tqdm

def cyvcf2msms(vcf_filename, chrom, sample_size, start, length, sample_list=None):

    # set new sample list
    vcf = VCF(vcf_filename)

    if sample_list == None:
        samples = vcf.samples
    else:
        samples = sample_list

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
    vcf.set_samples(sample_list)

    reformatted_hap_list = []
    reformatted_seg_site_idx_list = []

    sample_size = len(sample_list)
    pos = start
    for record in vcf(str(chrom) + ":" + str(start) + "-" + str(start+length)):
        hap = []
        for genotype in record.genotypes:
            hap.append(int(genotype[0]))
            # hap.append(int(genotype[1]))
        if sum(hap) == 0 or sum(hap) == sample_size:
            pass
        else:
            pos = record.POS
            if sample_size - sum(hap) < sum(hap):
                for idx, snp in enumerate(hap):
                    hap[idx] = 1 - snp
            reformatted_hap_list.append(hap)
            reformatted_seg_site_idx_list.append((pos-start))

    reformatted_hap_list = np.transpose(np.array(reformatted_hap_list))
    print(reformatted_hap_list.shape)
    reformatted_hap_list = reformatted_hap_list.tolist()

    assert(len(reformatted_hap_list) == sample_size)
    assert(len(reformatted_hap_list[0]) == len(reformatted_seg_site_idx_list))

    # write msms file
    msms_filename = '/scratch/saralab/first/real0.msms'
    msms_file = open(msms_filename, 'w')
    msms_file.write('fasta2msms.py Nov2014 -n ' + str(len(reformatted_hap_list)) + '\n')
    msms_file.write(str(chrom) + ', block: ' + str(start) + '-' + str(start+length))
    msms_file.write('\n\n//\nsegsites: ' + str(len(reformatted_seg_site_idx_list)) + '\npositions:')
    msms_file.write(' '.join(['%.5f'%(float(s)/length) for s in reformatted_seg_site_idx_list]) + '\n')
    for hap in reformatted_hap_list:
        hap = ''.join([str(x) for x in hap])
        msms_file.write(hap + '\n')
    
    msms_file.close()

def chrom2msms(vcf_filename, chrom, sample_size, length, increments, sample_list):
    # set new sample list
    vcf = VCF(vcf_filename)
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
    print(sample_list)

    for increment in range(increments):
        header_written = False
        msms_header = '/scratch/saralab/VAE/statsZI/example/data/demo0/'
        msms_filename = msms_header + 'data' + str(increment) + '.msms'
        print('Creating: ' + msms_filename)
        msms_file = open(msms_filename, 'w')
        #no_snp_filename = '/scratch/saralab/first/chrom2_none_long.txt'
        #no_snp_file = open(no_snp_filename, 'w')


        start = 0
        # find start and end
        if False:
            end = 0
            for record in tqdm(vcf):
                pass
            end = record.POS
            print('Endpos : %d' % end)
        end = 243188367 + length
        counter = 0

        start = random.uniform(start, end)

        # for each segment, write the snps
        while start < end:
            print('Segment ' + str(start) + '-' + str(start+length))
            reformatted_hap_list = []
            reformatted_seg_site_idx_list = []
            pos = start
            for record in vcf(str(chrom) + ":" + str(start) + "-" + str(start+length)):
                hap = []
                for genotype in record.genotypes:
                    left = 0 if genotype[0] == 0 else 1
                    right = 0 if genotype[1] == 0 else 1
                    hap.append(left)
                    #hap.append(right)
                if sum(hap) == 0 or sum(hap) == sample_size:
                    pass
                else:
                    pos = record.POS
                    if sample_size - sum(hap) < sum(hap):
                        for idx, snp in enumerate(hap):
                            hap[idx] = 1 - snp
                    reformatted_hap_list.append(hap)
                    reformatted_seg_site_idx_list.append((pos-start))

            start = start+length

            if len(reformatted_hap_list) == 0:
                no_snp_file.write('No SNP in segment '+str(start-length)+'-'+str(start)+', at '+str(counter)+'\n')
                continue
            reformatted_hap_list = np.transpose(np.array(reformatted_hap_list))
            print(reformatted_hap_list.shape)
            reformatted_hap_list = reformatted_hap_list.tolist()
            counter += 1

            print(len(reformatted_hap_list[0]))
            assert(len(reformatted_hap_list) == sample_size)
            assert(len(reformatted_hap_list[0]) == len(reformatted_seg_site_idx_list))

            # write msms file
            if not header_written:
                msms_file.write('chrom2msms.py Nov2014 -n ' + str(len(reformatted_hap_list)) + '\n')
                msms_file.write('chrom: '+str(chrom) + ', block: ' + str(start-length) + '-' + str(end) + ', length: '+ str(length)+'\n')
                header_written = True

            msms_file.write('\n//\nsegsites: ' + str(len(reformatted_seg_site_idx_list)) + '\npositions: ')
            msms_file.write(' '.join(['%.5f'%(float(s)/length) for s in reformatted_seg_site_idx_list]) + '\n')
            for hap in reformatted_hap_list:
                hap = ''.join([str(x) for x in hap])
                msms_file.write(hap + '\n')

            break
        
        msms_file.write('\n')
        msms_file.close()
        print('There are %d segments.' % counter)

def main():
    vcf_filename = '/scratch/hlee6/vcf/ALL.chr2.vcf.gz'
    MXL_pop = pick_population(csv_file='igsr_samples.tsv')
    dataset_size = 10
    chrom2msms(vcf_filename=vcf_filename, chrom=2, sample_size=10, increments=dataset_size, length=100000, sample_list=MXL_pop)

if __name__ == '__main__':
    main()
