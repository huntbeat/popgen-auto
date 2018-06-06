"""
Simulate data with msprime
"""

import msprime
import h5py as h5

data_file = h5.File('data.py','w')
metadata_file = h5.File('metadata.py','w')

# simulate with n=25, N=10000, and L=3000
tree_sequence = msprime.simulate(sample_size=25, Ne=10000, \
    length=3000, mutation_rate = 1e-7, recombination_rate=1e-7)

# see when the Tmrca changes
for tree in tree_sequence.trees():
    print(tree.interval, tree.tmrca(0,1)) # 0 and 1 here are the two sequences

# see when mutations occur
for variant in tree_sequence.variants():
    print(variant.site.position, variant.genotypes, sep="\t")
