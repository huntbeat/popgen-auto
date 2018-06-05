"""
Simulate data with msprime

Example simulation:
(0.0, 331.36482265111175) 9468.168251749572
612.8830808691417       [1 0]
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
interval = 0
for tree in tree_sequence.trees():
    prev_interval = interval
    interval = int(tree.interval[1]) # (start, end)
    tmrca = tree.tmrca(0,1)     # in units of years
    for i in range(prev_interval, interval): 
        TMRCA = np.append(TMRCA, tmrca)
print(len(TMRCA))

# see when mutations occur
pos = 0
for variant in tree_sequence.variants():
    prev_pos = pos
    pos = int(variant.site.position)  # locus
    geno = variant.genotypes          # [0,1]
    for i in range(prev_pos, pos-1):
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
for i in range(LEN - len(SEQ_D)):
    SEQ_1 += '0'
    SEQ_2 += '0'
    SEQ_D += '0'
print(len(SEQ_D))
