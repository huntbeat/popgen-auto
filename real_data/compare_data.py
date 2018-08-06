'''
Comparing real data to simulated data
'''

from format_real import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

'''
@param sim_files - msms filenames
@param num_grid - number of x-points to use for kernel density smoothing
@param bandwidth - how much to smooth distribution
'''
def plot_snp_distribution(sim_files, num_grid=1000, bandwidth=0.3):
  labels = []

  # 1000 GENOME #
  labels.append("MXL chr2")
  print("Parsing real...")
  real = RealData('/scratch/nhoang1/saralab/MXL/MXL.0-10M.chr2.vcf.gz',(0,10000000))
  print("Done parsing real.")
  num_snps = []
  for w in real.windows:
    num_snps.append(w.absolute_positions.shape[0])
  x_grid = np.linspace(min(num_snps), max(num_snps), num_grid)
  dist = kde_scipy(num_snps, x_grid, bandwidth)
  plt.plot(dist)

  # MSMS #
  for sf in sim_files:
    label = sf.split('/')[4][:-4]
    labels.append(label)
    num_snps = read_snps(sf)
    x_grid = np.linspace(min(num_snps), max(num_snps), num_grid)
    dist = kde_scipy(num_snps, x_grid, bandwidth)
    plt.plot(dist)

  plt.legend(labels, loc='upper left')
  plt.show()

def read_snps(filename, n=20):
  print("Reading msms file...")
  snps_count = []
  with open(filename,'r') as f:
    lines = f.readlines()
    for i in range(3,len(lines),n+4): # excludes header, move through n SNP seqs
      assert(lines[i].strip()=='//')
      n_snps = int(lines[i+1].split(' ')[1])
      snps_count.append(n_snps)
  return snps_count

"""Kernel Density Estimation with Scipy"""
def kde_scipy(x, x_grid, bandwidth):
  kde = gaussian_kde(x, bw_method=bandwidth)
  return kde.evaluate(x_grid)

def main():
  prefix = '/scratch/saralab/first/strength'
  strengths = ['0', '10', '100', '1000']
  plot_snp_distribution([prefix+s+'.txt' for s in strengths]) 

main()
