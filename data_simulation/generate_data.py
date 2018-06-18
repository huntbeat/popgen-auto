"""
Simulate constant and bottleneck data with msprime, and natural selection data with msms
"""

import msprime
import h5py as h5
import numpy as np

from natsel_fcns import parse_natsel, uniform_natsel

CONSTANT_SIZE = 50000
BOTTLENECK_SIZE = 50000
NATSELECT_SIZE = 50000
NUM_SITES = 100 # use count_data.py to find a number

##################################

''' pads or cuts MSPRIME matrix to uniform shape '''
def uniform_mutation_count(tree_sequence, length):
  genotypes = []
  # see when mutations occur
  for variant in tree_sequence.variants():
    genotypes.append(np.array(variant.genotypes))
  genotypes = np.array(genotypes)

  # account for when there was no variation
  if genotypes.shape == (0,):
      genotypes = np.reshape(genotypes,(0,25))

  len_diff = genotypes.shape[0] - length
  if len_diff >= 0: 
    genotypes = genotypes[:length]
  elif len_diff < 0:
    padding = np.zeros((abs(len_diff),25))
    half = int(abs(len_diff)/2)
    padded_gt = np.concatenate((padding[:half],genotypes,padding[half:]))
    assert(padded_gt.shape[0] == length)
    genotypes = padded_gt
  genotypes = genotypes.T.astype(int)
  return genotypes

############ CONSTANT ############

constant_matrices = [] 
for i in range(CONSTANT_SIZE):
  tree_sequence = msprime.simulate(sample_size=25, Ne=10000, \
      length=3000, mutation_rate = 1e-7, recombination_rate=1e-7)
  genotypes = uniform_mutation_count(tree_sequence,NUM_SITES)
  constant_matrices.append(genotypes)

  if i % 100 == 0:
    print(i)

print(constant_matrices[0][0])
print(constant_matrices[0][10])
print(constant_matrices[0][20],"\n")

############ BOTTLENECK ############

bottleneck_matrices = []
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
  genotypes = uniform_mutation_count(tree_sequence,NUM_SITES)
  bottleneck_matrices.append(genotypes)

  if j % 100 == 0:
    print(j+CONSTANT_SIZE)

print(bottleneck_matrices[0][0])
print(bottleneck_matrices[0][10])
print(bottleneck_matrices[0][20],"\n")

######## NATURAL SELECTION #########

unpadded_ns = parse_natsel('/scratch/nhoang1/simNatK.txt',25)
natselect_matrices = uniform_natsel(unpadded_ns, NUM_SITES)

####################################

path = '/scratch/nhoang1/data5.hdf5'
data_file = h5.File(path,'w')
data_file.create_dataset("constant",data=np.array(constant_matrices))
data_file.create_dataset("bottleneck",data=np.array(bottleneck_matrices))
data_file.create_dataset("naturalselection",data=np.array(natselect_matrices))

# create output array
constant_output = np.zeros((CONSTANT_SIZE,3), dtype='int32')
constant_output[:,0] = 1
bottleneck_output = np.zeros((BOTTLENECK_SIZE,3), dtype='int32')
bottleneck_output[:,1] = 1
natselect_output = np.zeros((NATSELECT_SIZE,3), dtype='int32')
natselect_output[:,2] = 1
assert(np.all(constant_output==[1,0,0]))
assert(np.all(bottleneck_output==[0,1,0]))
assert(np.all(natselect_output==[0,0,1]))
output = np.concatenate((constant_output,bottleneck_output,natselect_output))
print("output shape:",output.shape)
data_file.create_dataset("output",data=output)

data_file.close()
