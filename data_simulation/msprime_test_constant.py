"""
Simulate data with msprime
"""

import msprime

# simulate with n=25, N=10000, and L=3000
tree_sequence = msprime.simulate(sample_size=10, Ne=10000, \
    length=10000, mutation_rate = 1e-7, recombination_rate=1e-7)

# see when the Tmrca changes
#for tree in tree_sequence.trees():
#    print(tree.interval, tree.tmrca(0,1)) # 0 and 1 here are the two sequences

# see when mutations occur
S = 0
for variant in tree_sequence.variants():
#    print(variant.site.position, variant.genotypes, sep="\t")
    S += 1
print(S)