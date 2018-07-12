import csv
import vcf
import numpy as np

def main():

  #pick_individuals(20, 'MXL_samples.txt', 'igsr_samples.tsv') 


  data_file = '/scratch/nhoang1/saralab/smallMXL_164-165Mb.chr21.vcf.gz' 
  #data_file = '/scratch/nhoang1/saralab/ALL.chr21.vcf.gz' 
  vcf_reader = vcf.Reader(filename=data_file)
  max_num = 100

  positions = [] # list of ints
  snps = [] # list of int vectors 
  for i in range(max_num): #assumes max_num <= total num of records
    record = next(vcf_reader)
    positions.append(record.POS)
    #snp = snp_vector(record)
    #snps.append(snp)

  #pv = position_distance_vector(positions)
  #snp_matrix = np.array(snps).T # row is n, col is L

  print("finished gathering SNPs")
  window_size = 100000
  snps_per_window = count_snps_per_window(window_size, positions)

##############################################################
##############################################################

'''
@return a vector of the difference in adjacent seg site positions
'''
def position_distance_vector(positions):
  distances = []
  for j in range(len(positions)-1):
    diff = positions[j+1] - positions[j]
    distances.append(diff)
  distances.append(0) # last col zero padding 
  return np.array(distances)

##############################################################

'''
@return a vector of the binary SNPs at one site position
'''
def snp_vector(record):
  snps = []
  for sample in record.samples:
    gt = sample['GT'].split('|')
    snps.extend([int(x) for x in gt])
  return np.array(snps)

##############################################################

'''
counts how many SNPs are in each "window" of sequences
'''
def count_snps_per_window(window_size, snp_positions):
  counts = {} # k = region num, v = num snps
  count = 93
  remaining_pos = np.array(snp_positions)
  last_position_index = 0
  while remaining_pos.shape[0] != 0:
    window_end = ((count+1)*window_size)
    in_window = np.where(remaining_pos < window_end)
    num_snps = in_window[0].shape[0]
    counts[count] = num_snps
    count += 1
    if num_snps != 0:
      last_position_index = in_window[0][-1] + 1
    remaining_pos = remaining_pos[last_position_index:]
  regions = sorted(list(counts.keys()))
  print("region : num_snps")
  for r in regions:
    window = str(r*window_size+1)+'-'+str((r+1)*window_size)
    print(window,':',counts[r])
  return counts

##############################################################

'''
pick n individuals from 1000 genome file 
@param n
'''
def pick_individuals(n, txt_file, tsv_file):
  indiv_names = open(txt_file,'w')
  with open(tsv_file,'r') as tsv:
    reader = csv.DictReader(tsv, dialect='excel-tab')
    count = 0
    for row in reader:
      if (count < n) and ('phase 3' in row['Data collections']):
        line = row['Sample name'] + '\n'
        indiv_names.write(line)
        count += 1
      else: break
    if count < 10: print("Warning: less than",n,"samples in TSV file")
  indiv_names.close()

##############################################################
main()
