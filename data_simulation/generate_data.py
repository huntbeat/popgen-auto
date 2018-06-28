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
import sys

from natsel_fcns import parse_natsel, uniform_natsel

###################################################################
###################################################################

def main():
  data_name = sys.argv[1]
  path = '/scratch/nhoang1/'+data_name+'.hdf5'
  data_file = h5.File(path,'w')
  constant_size = 50000
  bottleneck_size = 50000
  natselect_size = 50000 # restricted by prior natsel sim
  num_sites = 150 # use count_data.py to find a number

  cSequences, cPositions, cD_list = generate_constant_samples(constant_size)
  bSequences, bPositions, bD_list = generate_bottleneck_samples(bottleneck_size)
  nsSequences, nsPositions, nsD_list = get_natural_selection_samples(natselect_size)

  D_list = cD_list + bD_list + nsD_list
  #S_list what is this for?

  # network inputs
  data_file.create_dataset("constant",data=np.array(cSequences))
  data_file.create_dataset("bottleneck",data=np.array(bSequences))
  data_file.create_dataset("naturalselection",data=nsSequences)
  data_file.create_dataset("constant_positions",data=np.array(cPositions))
  data_file.create_dataset("bottleneck_positions",data=np.array(bPositions))
  data_file.create_dataset("natsel_positions",data=nsPositions)

  # network pop outputs
  constant_output = generate_pop_output(constant_size,0)
  bottleneck_output = generate_pop_output(bottleneck_size,1)
  natselect_output = generate_pop_output(natselect_size,2)
  pop_output = np.concatenate((constant_output,bottleneck_output,natselect_output))
  data_file.create_dataset("pop_output",data=pop_output)

  # network second outputs
  dupl_output = np.copy(pop_output)
  data_file.create_dataset("pop_duplicate_output",data=dupl_output)
  tajimaD_output = generate_tajimaD_output(constant_size+bottleneck_size+natselect_size,D_list)
  data_file.create_dataset("TD_output",data=TD_output)
  rand_output = generate_random_output(constant_size+bottleneck_size+natselect_size)
  data_file.create_dataset("random_output",data=rand_output)

  data_file.close()

###################################################################
############################ SAMPLES ##############################

def generate_constant_samples(constant_size):
  constant_matrices = []
  cPositions = []
  D_list = []
  for i in range(constant_size):
    tree_sequence = msprime.simulate(sample_size=25, Ne=10000, \
        length=3000, mutation_rate = 1e-7, recombination_rate=1e-7)
    trees = tree_sequence.trees()
    big_S = 0
    for tree in trees:
      big_S += tree.num_mutations
    genotypes, positions = uniform_mutation_count(tree_sequence,NUM_SITES,3000)
    constant_matrices.append(genotypes)
    cPositions.append(positions)
    D_list.append(find_D(big_S, tree_sequence.pairwise_diversity(), 25))
    if i % 1000 == 0:
      print("constant:",i)
  return constant_matrices, cPositions, D_list

def generate_bottleneck_samples(bottleneck_size):
  bottleneck_matrices = []
  bPositions = []
  D_list
  for j in range(bottleneck_size):
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
    genotypes, positions = uniform_mutation_count(tree_sequence,NUM_SITES,3000)
    bottleneck_matrices.append(genotypes)
    bPositions.append(positions)
    D_list.append(find_D(big_S, tree_sequence.pairwise_diversity(), 25))
    if i % 1000 == 0:
        print("bottleneck:",i)
  return bottleneck_matrices, bPositions, D_list

def get_natural_selection_samples(natselect_size):
  D_list = []
  unpadded_ns, unpadded_pos_vecs = parse_natsel('/scratch/nhoang1/simNatK.txt',25,100000)
  unpadded_pos_vecs = check_positions(unpadded_pos_vecs)
  natselect_matrices, nsPositions = uniform_natsel(unpadded_ns, unpadded_pos_vecs, NUM_SITES)
  nat_D_list, nat_S_list = parse_msms('/scratch/nhoang1/simNatK.txt', NATSELECT_SIZE)
  D_list.extend(nat_D_list) # append instead?
  return natselect_matrices, nsPositions, D_list

def generate_pop_output(size,index):
  output = np.zeros((size,3), dtype='int32')
  output[:,index] = 1
  return output

def generate_tajimaD_output(size, D_list):
  lower_third, upper_third = find_thirds(D_list)
  TD_output = np.zeros((size,3), dtype='int32')
  for i in range(size):
    if D_list[i] <= lower_third:
      TD_output[i,0] = 1
    elif D_list[i] <= upper_third:
      TD_output[i,1] = 1
    else:
      TD_output[i,2] = 1
  return TD_output

def generate_random_output(size):
  rand_output = np.zeros((size,3), dtype='int32')
  for i in range(size):
    R = np.random.rand(1)[0]
    if R <= (1/3):
      rand_output[i,0] = 1
    elif R <= (2/3):
      rand_output[i,1] = 1
    else:
      rand_output[i,2] = 1
  return rand_output

###################################################################
######################### HELPER FUNCTIONS ########################

'''
pads or cuts MSPRIME matrixand positions to uniform shape
@param tree_sequence
@param length - length to extend seg site vectors to
@param L - length of original sequence
'''
def uniform_mutation_count(tree_sequence, length, L):
  genotypes = []
  float_positions = []
  # see when mutations occur
  for variant in tree_sequence.variants():
    genotypes.append(np.array(variant.genotypes))
    float_positions.append(variant.site.position)are we going off campus for lunch/
  genotypes = np.array(genotypes)
  positions = [int(fp*L) for fp in float_positions]

  # account for when there was no variation
  if genotypes.shape == (0,):
      genotypes = np.reshape(genotypes,(0,25))

  positions = check_positions(positions)

  len_diff = genotypes.shape[0] - length
  if len_diff >= 0:
    genotypes = genotypes[:length]
    positions = positions[:length]
  elif len_diff < 0:
    padding = np.zeros((abs(len_diff),25))
    half = int(abs(len_diff)/2)
    padded_gt = np.concatenate((padding[:half],genotypes,padding[half:]))
    assert(padded_gt.shape[0] == length)
    genotypes = padded_gt
    pos_padding = [0] * abs(len_diff)
    padded_pos = pos_padding[:half] + positions + pos_padding[half:]
    positions = padded_pos
  positions = np.array(positions, dtype=int)
  genotypes = genotypes.T.astype(int)
  return genotypes, positions

###################################################################

'''
rare, but if two mutations are really close to each other,
they could be settling into the same position number. check
for that and separate them by a value of 1
'''
def check_positions(position_vector):
  for p in range(len(position_vector)-1):
    if position_vector[p] == position_vector[p+1]:
      position_vector[p+1] += 1
  return position_vector

###################################################################

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

###################################################################

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

###################################################################

def kde_scipy(x, x_grid, bandwidth, **kwargs):
  """Kernel Density Estimation with Scipy"""
  kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)
  return kde.evaluate(x_grid)

###################################################################

main()


# # FOR S -- NOT DONE TODO
# for i in range(CONSTANT_SIZE + BOTTLENECK_SIZE + NATSELECT_SIZE):
#     if S_list[i] <= S_lower_third:
#         TD_output[i,0] = 1
#     elif S_list[i] <= S_upper_third:
#         TD_output[i,1] = 1
#     else:
#         TD_output[i,2] = 1
