import vcf
import numpy as np

def main():
  data_file = '/scratch/nhoang1/ALL.chr21.vcf.gz' 
  vcf_reader = vcf.Reader(filename=data_file)
  max_num = 500

  positions = [] # list of ints
  snps = [] # list of int vectors 
  for i in range(max_num): #assumes max_num <= total num of records
    record = next(vcf_reader)
    positions.append(record.POS)
    snp = snp_vector(record)
    snps.append(snp)

  pv = positions_vector(positions)
  snp_matrix = np.array(snps).T # row is n, col is L

##############################################################
##############################################################

'''
@return a vector of the difference in adjacent seg site positions
'''
def positions_vector(positions):
  distances = []
  for j in range(len(positions)-1):
    diff = positions[j+1] - positions[j]
    distances.append(diff)
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
main()
