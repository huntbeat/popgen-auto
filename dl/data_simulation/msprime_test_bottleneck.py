"""
Simulate data with msprime
"""

import msprime

# change population size from Ne to 1/10 of Ne at 200 generations
bottleneck = msprime.PopulationParametersChange(time=200,initial_size=1000)
# change back to Ne at 750 generations
recovery = msprime.PopulationParametersChange(time=750,initial_size=10000)
# make a list of the changes
size_change_lst = [bottleneck, recovery]

# simulate with n=2, N=10000, and L=10000
tree_sequence = msprime.simulate(sample_size=2, Ne=10000, \
    length=10000, mutation_rate = 1e-7, recombination_rate=1e-7, \
    demographic_events=size_change_lst)

# see when the Tmrca changes
for tree in tree_sequence.trees():
    print(tree.interval, tree.tmrca(0,1)) # 0 and 1 here are the two sequences

# see when mutations occur
for variant in tree_sequence.variants():
    print(variant.site.position, variant.genotypes, sep="\t")
