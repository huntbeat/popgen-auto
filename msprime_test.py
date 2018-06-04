"""
Simulate data with msprime
"""

import msprime
import numpy as np
import matplotlib.pyplot as plt

LEN = 10000
MU = 1e-7
N_e = 10000
R_r = 1e-7

TMRCA = np.array([])
SEQ_1 = "" 
SEQ_2 = ""
SEQ_D = ""

# simulate with n=2, N=10000, and L=10000
tree_sequence = msprime.simulate(sample_size=2, Ne=N_e, \
    length=LEN, mutation_rate = MU, recombination_rate=R_r)

# see when the Tmrca changes
for tree in tree_sequence.trees():
    interval = tree.interval    # (start, end)
    tmrca = tree.tmrca(0,1)     # in units of years
    for i in range(int(tree.interval[1])): 
        TMRCA = np.append(TMRCA, tmrca)

plt.figure(1)
plt.plot(np.arange(1,len(TMRCA)+1),TMRCA)
plt.show()

# see when mutations occur
pos = 0
for variant in tree_sequence.variants():
    prev_pos = pos
    pos = variant.site.position # locus
    pos = int(pos)
    geno = variant.genotypes    # [0,1]
    for i in range(prev_pos, pos):
        SEQ_1 += '0'
        SEQ_2 += '0'
        SEQ_D += '0'
    if geno[0] == '1':
        SEQ_1 += '1'
        SEQ_2 += '0'
        SEQ_D += '1'
    else:
        SEQ_1 += '0'
        SEQ_2 += '1'
        SEQ_D += '1'
print(len(SEQ_1))
print(len(SEQ_2))
print(len(SEQ_D))
