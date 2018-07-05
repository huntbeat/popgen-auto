import vcf
import numpy as np

def main():
  data_file = '/scratch/nhoang1/ALL.chr21.vcf.gz' 
  vcf_reader = vcf.Reader(filename=data_file)
  max_num = 100000

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
  snps_per_window = count_snps_per_window(100000, positions)
  regions = sorted(list(snps_per_window.keys()))
  print("region : num_snps")
  for r in regions:
    print(r,':',snps_per_window[r])

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
  return counts

##############################################################
main()
