"""
Simulate constant and bottleneck data with msprime, and natural selection data with msms
"""

import msprime
import h5py as h5
import numpy as np
from math import sqrt
from tajima import *
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from natsel_fcns import parse_natsel, uniform_natsel

CONSTANT_SIZE = 50000
BOTTLENECK_SIZE = 50000
NATSELECT_SIZE = 50000 # restricted by prior natsel sim
NUM_SITES = 150 # use count_data.py to find a number
D_list = []

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

##################################

'''Find D for msprime data'''
def find_D(num_mutations, pairwise_diversity, n):
    S = num_mutations
    if S != 0:
        pi = pairwise_diversity
        a_1 = sum([1/i for i in range(1,n)])
        b_1 = (n + 1) / (3*(n-1))
        c_1 = b_1 - (1 / a_1)
        e_1 = c_1 / a_1

        a_2 = sum([1/(i*i) for i in range(1,n)])
        b_2 = (2*(n*n + n + 3))/ (9*n *(n-1))
        c_2 = b_2 - (n+2)/(a_1 * n) + a_2 / (a_1*a_1)
        e_2 = c_2 / (a_1 * a_1 + a_2)

        d = pi - S / a_1
        var = sqrt(e_1 * S + e_2 * S * (S - 1))
        D = d / var
    else:
        D = 0
    return D

##################################

'''Find interval that subdivides D into thirds'''
def find_thirds(long_list):
    NUM_GRID   = len(long_list)    # number of x-points to use for kernel density smoothing
    BANDWIDTH  = 0.3     # how much to smooth distribution
    sim_values = np.array(long_list)
    x_grid = np.linspace(min(sim_values), max(sim_values), NUM_GRID)
    dist = kde_scipy(sim_values, x_grid, BANDWIDTH)
    plt.figure(1)
    plt.plot(x_grid, dist)
    plt.show()
    sorted_D = sorted(long_list)
    lower_third_idx = int(NUM_GRID / 3)
    upper_third_idx = 2 * lower_third_idx
    return sorted_D[lower_third_idx], sorted_D[upper_third_idx]

def kde_scipy(x, x_grid, bandwidth, **kwargs):
   """Kernel Density Estimation with Scipy"""
   kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)
   return kde.evaluate(x_grid)

############ CONSTANT ############

constant_matrices = []
for i in range(CONSTANT_SIZE):
  tree_sequence = msprime.simulate(sample_size=25, Ne=10000, \
      length=3000, mutation_rate = 1e-7, recombination_rate=1e-7)
  trees = tree_sequence.trees()
  big_S = 0
  for tree in trees:
    big_S += tree.num_mutations
  genotypes = uniform_mutation_count(tree_sequence,NUM_SITES)
  constant_matrices.append(genotypes)
  D_list.append(find_D(big_S, tree_sequence.pairwise_diversity(), 25))

  if i % 1000 == 0:
    print(i)
#
#
# print(constant_matrices[0][0])
# print(constant_matrices[0][10])
# print(constant_matrices[0][20],"\n")

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
  trees = tree_sequence.trees()
  big_S = 0
  for tree in trees:
    big_S += tree.num_mutations
  genotypes = uniform_mutation_count(tree_sequence,NUM_SITES)
  bottleneck_matrices.append(genotypes)
  D_list.append(find_D(big_S, tree_sequence.pairwise_diversity(), 25))

  if j % 1000 == 0:
    print(j+CONSTANT_SIZE)
#
# print(bottleneck_matrices[0][0])
# print(bottleneck_matrices[0][10])
# print(bottleneck_matrices[0][20],"\n")

######## NATURAL SELECTION #########

unpadded_ns = parse_natsel('/scratch/nhoang1/simNatK.txt',25)
natselect_matrices = uniform_natsel(unpadded_ns, NUM_SITES)
nat_D_list = parse_msms('/scratch/nhoang1/simNatK.txt', NATSELECT_SIZE)
D_list.extend(nat_D_list)

####################################

path = '/scratch/nhoang1/data9.hdf5'
data_file = h5.File(path,'w')
data_file.create_dataset("constant",data=np.array(constant_matrices))
data_file.create_dataset("bottleneck",data=np.array(bottleneck_matrices))
data_file.create_dataset("naturalselection",data=np.array(natselect_matrices))

lower_third, upper_third = find_thirds(D_list)
print(lower_third, upper_third)

# create output arrays
constant_output = np.zeros((CONSTANT_SIZE,3), dtype='int32')
constant_output[:,0] = 1
bottleneck_output = np.zeros((BOTTLENECK_SIZE,3), dtype='int32')
bottleneck_output[:,1] = 1
natselect_output = np.zeros((NATSELECT_SIZE,3), dtype='int32')
natselect_output[:,2] = 1
pop_output = np.concatenate((constant_output,bottleneck_output,natselect_output))
print("pop output shape:",pop_output.shape)
data_file.create_dataset("pop_output",data=pop_output)

TD_output = np.zeros((CONSTANT_SIZE + BOTTLENECK_SIZE + NATSELECT_SIZE,3), dtype='int32')
for i in range(CONSTANT_SIZE + BOTTLENECK_SIZE + NATSELECT_SIZE):
    if D_list[i] <= lower_third:
        TD_output[i,0] = 1
    elif D_list[i] <= upper_third:
        TD_output[i,1] = 1
    else:
        TD_output[i,2] = 1
print("TD output shape:",TD_output.shape)
data_file.create_dataset("TD_output",data=TD_output)

data_file.close()
