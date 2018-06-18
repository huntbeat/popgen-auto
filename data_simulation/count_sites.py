"""
Count number of mutation sites
"""

import msprime

CONSTANT_SIZE = 100
BOTTLENECK_SIZE= 100
NATSELECT_SIZE = 100

constant_counts = []
bottleneck_counts = []
natselect_counts = []

############ CONSTANT ############

for i in range(CONSTANT_SIZE):
  tree_sequence = msprime.simulate(sample_size=25, Ne=10000, \
      length=3000, mutation_rate = 1e-7, recombination_rate=1e-7)
  count = 0
  for v in tree_sequence.variants():
    count += 1
  constant_counts.append(count)

max_count = max(constant_counts)
min_count = min(constant_counts)
mean_count = float(sum(constant_counts)) / max(len(constant_counts), 1)
print("\n\nStats (Constant):")
print("max:",max_count,"\nmin:",min_count,"\nmean:",mean_count)

############ BOTTLENECK ############

for j in range(BOTTLENECK_SIZE):
  # change population size from Ne to 1/10 of Ne at 200 generations
  bottleneck = msprime.PopulationParametersChange(time=200,initial_size=1000)
  # change back to Ne at 750 generations
  recovery = msprime.PopulationParametersChange(time=750,initial_size=10000)
  # make a list of the changes
  size_change_lst = [bottleneck, recovery]

  # simulate with n=25, N=10000, and L=3000
  tree_sequence = msprime.simulate(sample_size=25, Ne=10000, \
      length=3000, mutation_rate = 1e-7, recombination_rate=1e-7, \
      demographic_events=size_change_lst)

  count = 0
  # see when mutations occur
  for variant in tree_sequence.variants():
    count += 1
  bottleneck_counts.append(count)  

max_count = max(bottleneck_counts)
min_count = min(bottleneck_counts)
mean_count = float(sum(bottleneck_counts)) / max(len(bottleneck_counts), 1)
print("\nStats (Bottleneck):")
print("max:",max_count,"\nmin:",min_count,"\nmean:",mean_count,"\n\n")

######### NATURAL SELECTION #########

for k in range(NATSELECT_SIZE):
  # read from natsim file

max_count = max(natselect_counts)
min_count = min(natselect_counts)
mean_count = float(sum(natselect_counts)) / max(len(natselect_counts), 1)
print("\nStats (Natural Selection):")
print("max:",max_count,"\nmin:",min_count,"\nmean:",mean_count,"\n\n")

