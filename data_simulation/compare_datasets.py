"""
Compare and plot the variation across genomes for bottleneck and natural selection datasets
"""

import h5py
import numpy as np
import msprime
import matplotlib.pyplot as plt

def main():
  # fetch natural selection data (msms)
  ns_matrix = []
  with open("dif/natSelData.txt","r") as sn:
    header = sn.readline()
    for line in sn.readlines():
      int_line = [int(i) for i in list(line.strip())]
      ns_matrix.append(np.array(int_line))
  ns_matrix = np.array(ns_matrix[:-1]) # there's an 11th sequence that's not the full length

  # create bottleneck data (msprime)
  bottleneck = msprime.PopulationParametersChange(time=200,initial_size=1000)
  recovery = msprime.PopulationParametersChange(time=750,initial_size=10000)
  size_change_lst = [bottleneck, recovery]
  tree_sequence = msprime.simulate(sample_size=10, Ne=10000, \
      length=10000, mutation_rate = 1e-7, recombination_rate=1e-7, \
      demographic_events=size_change_lst)
  seg_genotype = tree_sequence.genotype_matrix()
  bn_matrix = impute_full_genotype_matrix(tree_sequence, 10, 10000)

  assert(ns_matrix.shape == bn_matrix.shape)

  # variance comparison and plot 
  ns_variance = []
  bn_variance = []
  for i in range(0,10000,100): #region len is 100
    #ns_var = ...
    ns_variance.append(ns_var)
    #bn_var = ...
    bn_variance.append(bn_var)

  plt.plot(ns_variance, marker='v')
  plt.plot(bn_variance, marker='o')
  plt.title('Variance Comparison')
  plt.ylabel('variance')
  plt.xlabel('region')
  plt.legend(['natural selection', 'bottleneck'], loc='upper left')
  plt.show()


########HELPER FUNCTIONS##########

def impute_full_genotype_matrix(tree_seq, n, L):
  vPositions = [] # list of ints
  vGenotypes = [] # list of numpy array of int variation
  for variant in tree_seq.variants():
    vPositions.append(int(variant.site.position))
    vGenotypes.append(list(variant.genotypes))

  zeros = [[0 for col in range(n)] for row in range(vPositions[0])]
  full_matrix = zeros + [vGenotypes[0]]

  for p in range(1,len(vPositions)):
    height = vPositions[p] - vPositions[p-1] - 1
    zeros = [[0 for col in range(n)] for row in range(height)]
    full_matrix += zeros
    full_matrix += [vGenotypes[p]]

  zeros = [[0 for col in range(n)] for row in range(L-vPositions[-1]-1)]
  full_matrix += zeros 

  for r in range(L):
    full_matrix[r] = np.array(full_matrix[r])
  full_matrix = np.array(full_matrix).T

  assert(full_matrix.shape[0] == n)
  assert(full_matrix.shape[1] >= L)

  full_matrix = full_matrix[:L] # account for rounding issues with positions
  return full_matrix

##################################
main()
