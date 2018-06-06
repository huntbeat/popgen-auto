"""
Simulate constant and bottleneck data with msprime
"""

import msprime
import h5py as h5
import numpy as np

CONSTANT_SIZE = 5
BOTTLENECK_SIZE = 5

data_file = h5.File('data.h5','w')
constant_matrices = [] 
bottleneck_matrices = []

##################################

def uniform_mutation_count(tree_sequence, length):
  genotypes = []
  # see when mutations occur
  for variant in tree_sequence.variants():
    genotypes.append(np.array(variant.genotypes))
  genotypes = np.array(genotypes)
  len_diff = genotypes.shape[0] - length
  if len_diff > 0: 
    genotypes = genotypes[:length]
  elif len_diff < 0:
    if abs(len_diff) % 2 == 0:
      empty_rows = abs(int(len_diff/2))
    else:
      empty_rows = abs(int(len_diff/2)) + 1
    padding = np.zeros((empty_rows,25))
    genotypes = np.concatenate((padding,genotypes,padding))
    genotypes = genotypes[:length]
  genotypes = genotypes.T.astype(int)
  return genotypes

############ CONSTANT ############

for i in range(CONSTANT_SIZE):
  tree_sequence = msprime.simulate(sample_size=25, Ne=10000, \
      length=3000, mutation_rate = 1e-7, recombination_rate=1e-7)
  genotypes = uniform_mutation_count(tree_sequence,45)
  constant_matrices.append(genotypes)

############ BOTTLENECK ############

for j in range(BOTTLENECK_SIZE):

  # change population size from Ne to 1/10 of Ne at 200 generations
  bottleneck = msprime.PopulationParametersChange(time=200,initial_size=1000)

  # change back to Ne at 750 generations
  recovery = msprime.PopulationParametersChange(time=750,initial_size=10000)

  # make a list of the changes
  size_change_lst = [bottleneck, recovery]

  tree_sequence = msprime.simulate(sample_size=25, Ne=10000, \
      length=3000, mutation_rate = 1e-7, recombination_rate=1e-7, \
      demographic_events=size_change_lst)
  genotypes = uniform_mutation_count(tree_sequence,30)
  bottleneck_matrices.append(genotypes)

####################################

data_file.create_dataset("constant",data=constant_matrices)
data_file.create_dataset("bottleneck",data=bottleneck_matrices)
data_file.close()
'''
with h5.File('metadata.h5','r') as f:
  lens = f.get('mutation_lengths')
  np_lens = np.array(lens)
  print(np_lens.shape)
  print(np_lens)
'''
