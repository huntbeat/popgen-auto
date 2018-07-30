import optparse
import sys
from cyvcf2 import VCF
import numpy as np

import random

# sent from Yun, match up with ancestral file
chr2L_len = 23011544
chr2R_len = 21146708
chr3L_len = 24543557
chr3R_len = 27905053
chrX_len  = 22422827


N_ALLELE = 'N' # represents missing data
N_THRESH = 0.35 # if we have more than this fraction of Ns in a block, throw it out
SHIFT = 20000 # shift over this much each time
L = 100000 # block length (consistent with statistic parsing, etc)
PROB_THRESH = 0.8 # threshold for accepting an ancestral allele probability
BASES = ['A','C','G','T']
MAX_LEN = 4 # max length of a number (i.e. we have less than 9999 datasets)


#------------------------------
# PARSE COMMAND LINE ARGUMENTS
#------------------------------


parser = optparse.OptionParser(description='creates a plot of the likelihood over the EM iterations')

parser.add_option('-c', '--chrom', type='string', help='name of chromosome so we write informative output files')
parser.add_option('-f', '--fasta_filename', type='string', help='path to input fasta file')
parser.add_option('-a', '--anc_filename', type='string', help='path to input ancestral allele file')
parser.add_option('-m', '--msms_folder', type='string', help='path to output msms folder')
parser.add_option('-s', '--file_start', type='int', help='block number to start with, in the case we have a big fasta file')
parser.add_option('-e', '--file_end', type='int', help='block number to end with, in the case we have a big fasta file')

(opts, args) = parser.parse_args()

mandatories = ['chrom','fasta_filename','anc_filename','msms_folder']
for m in mandatories:
    if not opts.__dict__[m]:
        print('mandatory option ' + m + ' is missing\n')
        parser.print_help()
        sys.exit()


#---------
# HELPERS
#---------


# pad file name with 0's so sorting works properly
def pad_int(b):
    num_zeros = MAX_LEN - len(str(b))
    return '0'*num_zeros + str(b)


# determine whether a list of alleles is segregating
def seg_site(allele_lst):
    if len(allele_lst) == 1:
        return False
    if len(allele_lst) == 2 and N_ALLELE in allele_lst:
        return False
    return True


# reformat haps from A/C/G/T to 0/1
def reformat_haps(seg_hap_lst, seg_site_idx_lst, anc_dict, b): # b is the block number
    reformatted_hap_lst = ['']*len(seg_hap_lst)
    reformatted_seg_site_idx_lst = []
    n = len(seg_hap_lst)

    snps_in_anc_dict = 0
    for l in range(len(seg_hap_lst[0])):
        alleles = []
        for seg_hap in seg_hap_lst:
            if seg_hap[l] not in alleles and seg_hap[l] != N_ALLELE:
                alleles.append(seg_hap[l])

        # only take non-tri/quad-allelic sites
        if len(alleles) == 2:
            reformatted_seg_site_idx_lst.append(seg_site_idx_lst[l])

            # if we do know the ancestral allele, use 0/1
            if (seg_site_idx_lst[l] + b*SHIFT) in anc_dict:
                anc_allele = anc_dict[seg_site_idx_lst[l] + b*SHIFT]
                #snps_in_anc_dict += 1
                for i in range(n):
                    target = seg_hap_lst[i][l]
                    if target == N_ALLELE:
                        reformatted_hap_lst[i] += N_ALLELE # adding missing data, which means we'll need to change our real data stats
                    elif target == anc_allele:
                        reformatted_hap_lst[i] += '0'
                    else:
                        reformatted_hap_lst[i] += '1'

            # if we don't know the ancestral allele, encode string as M (major) and m (minor)
            else:
                counts = [[seg_hap_lst[i][l] for i in range(n)].count(x) for x in alleles]
                major = alleles[counts.index(max(counts))]
                for i in range(n):
                    target = seg_hap_lst[i][l]
                    if target == N_ALLELE:
                        reformatted_hap_lst[i] += N_ALLELE # adding missing data, which means we'll need to change our real data stats
                    elif target == major:
                        reformatted_hap_lst[i] += 'M'
                    else:
                        reformatted_hap_lst[i] += 'm'
                    

    for hap in (reformatted_hap_lst):
        assert len(hap) == len(reformatted_seg_site_idx_lst)

    #print('frac snps in anc file: ' + str(float(snps_in_anc_dict)/len(reformatted_seg_site_idx_lst)))
                    
    return reformatted_hap_lst, reformatted_seg_site_idx_lst


# from a fasta file, get the indices of seg sites
def fasta2seg_sites(fasta_filename):
    fasta_file = file(fasta_filename,'r')
    first_line = True
    allele_dict = {} # for each base, list of alleles
    #num_Ns = 0 # count number of N's and if this is above a threshold, ignore file
    all_Ns = 0 # number of samples that are all N's in this region
    line_count = 0
    
    for line in fasta_file:
        line = line.strip()
        if line[0] != '>':
            line_count += 1

            if line.count(N_ALLELE) == len(line):
                all_Ns += 1

            # the first hap will initialize the dictionary
            if first_line:
                for l in range(len(line)):
                    allele_dict[l] = [line[l]]
                    #if line[l] == N_ALLELE:
                    #    num_Ns += 1
                first_line = False
            else:
                for l in range(len(line)):
                    a = line[l]
                    if a not in allele_dict[l]:
                        allele_dict[l].append(a)
                    #if line[l] == N_ALLELE:
                    #    num_Ns += 1
    fasta_file.close()

    # do a check on the fraction of Ns
    #frac_Ns = float(num_Ns)/(line_count*L)
    #if frac_Ns > N_THRESH:
    #    print('throwing out: ' + fasta_filename)
    #    print('fraction unknown bases: ' + str(frac_Ns))
    #    return [] # return empty list so it's thrown out later

    # do a check on the number of haps that are all Ns
    if all_Ns >= 5:
        print('throwing out: ' + fasta_filename)
        print('num all Ns: ' + str(all_Ns))
        return []
    
    # from allele_dict, get list of seg sites
    seg_site_idx_lst = []
    for l in range(len(allele_dict.keys())):
        if seg_site(allele_dict[l]):
            seg_site_idx_lst.append(l)
    return seg_site_idx_lst


# from a fasta file and list of seg sites, write an msms file
def cyvcf2msms(vcf_filename, chrom, sample_list, sample_size, start, length):

    # set new sample list

    # create vcf with sample list

    vcf = VCF(vcf_filename)
    vcf.set_samples(sample_list)

    reformatted_hap_lst = []
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
            reformatted_seg_site_idx_list.append((pos-start)/length)

    reformatted_hap_list = np.transpose(np.array(reformatted_hap_list))
    reformatted_hap_list = reformatted_hap_list.to_list()

    assert(len(reformatted_hap_list) == sample_size)
    assert(len(reformatted_hap_list[0] == len(reformatted_seg_site_idx_list))

    # write msms file
    msms_filename = opts.msms_folder + opts.chrom + '_block' + pad_int(b) + '.msms'
    msms_file = file(msms_filename,'w')
    msms_file.write('fasta2msms.py Nov2014 -n ' + str(len(reformatted_hap_lst)) + '\n')
    msms_file.write(opts.chrom + ', block: ' + str(b*SHIFT) + '-' + str(b*SHIFT+L))
    msms_file.write('\n\n//\nsegsites: ' + str(len(reformatted_seg_site_idx_lst)) + '\npositions:')
    msms_file.write(' '.join(['%.5f'%(float(s)/L) for s in reformatted_seg_site_idx_lst]) + '\n')
    for hap in reformatted_hap_lst:
        msms_file.write(hap + '\n')
    
    msms_file.close()


#------
# MAIN
#------


# parse fasta file and write temp fasta files for each block
fasta_file = file(opts.fasta_filename,'r')

if opts.file_start == None:
    start = 0
else:
    start = opts.file_start
if opts.file_end == None:
    end = None
else:
    end = opts.file_end
    
first_line = True
temp_file_lst = []
for line in fasta_file:
    line = line.strip()
    if line[0] != '>':

        if first_line:
            if opts.file_end == None:
                end = len(line)/SHIFT # not adding plus one here so we truncate last bit
            for b in range(start, end):
                temp_file_lst.append(file(opts.msms_folder + opts.chrom + '_block' + pad_int(b) + '.fasta','w'))
            first_line = False
            
        for b in range(start, end):
            temp_file_lst[b-start].write('>block' + str(b) + '\n' + line[b*SHIFT:b*SHIFT+L] + '\n')
                
for b in range(start, end):
    temp_file_lst[b-start].close()

    
# parse ancestral allele file to get a dictionary of the segregating sites
anc_dict = {}
anc_file = file(opts.anc_filename,'r')
for line in anc_file:
    tokens = line.split()
    probs = [float(x) for x in tokens[1:]]
    max_prob = max(probs)
    if max_prob >= PROB_THRESH:
        anc_dict[int(tokens[0])-1] = BASES[probs.index(max_prob)] # subtract off 1 to be consistent with our labeling    


# from each block in fasta format, create an msms file
for b in range(start, end):
    block_filename = opts.msms_folder + opts.chrom + '_block' + pad_int(b) + '.fasta'
    seg_site_idx_lst = fasta2seg_sites(block_filename)

    # only create an msms file if we have some segregating sites
    if len(seg_site_idx_lst) > 0:
        fasta2msms(block_filename, seg_site_idx_lst, anc_dict, b, end)
