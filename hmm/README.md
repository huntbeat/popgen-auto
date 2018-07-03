# popgen-hmm-dl

Population genetics using Hidden Markov Models

Hyong Hark Lee, under Professor Sara Mathieson's guidance

---

Example entry

Sam and Hunter: 05-03-18 (1.5hrs)
- downloading 1000genomes data on chromosome 12, use bcftools to parse VCF file. But running into a lot of trouble trying to find rs671. VCF Data is now on `/scratch/cs68`
```
bcftools view -r 12:111766887-111817529 -Oz -o ALL_1117-1118MB.chr12.vcf.gz ALL.chr12.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz
```
- https://www.snpedia.com/index.php/Rs671 Found what the variants of rs61 mean. Found out that [Disulfiram](https://www.snpedia.com/index.php/Disulfiram) can be used to treat alcoholism only for people with the (G;G) variant, not for (A;G) and (A;A)

---

05-31-18

- First goal: validate that the exponential distribution assumption derived from ideas of coalescence using HMM, to ensure the accuracy of HMM.

- Building from there: analyzing population size outputted by HMM analysis for causal factors like natural selection, population structure, etc., testing random mating assumption using differnt chromosome pairs

- Why might HMM not work: logic flow from exponential distribution for a SNP to HMM sequential type of thinking

- Segmentation in genes caused by recombination, natural selection, regions of higher mutation rates (compared to others)

- Work to do: 1. retrieved log-time buckets for histogram, 2. initialized init, tran, emit probabilities based on new buckets, 3. VCF -> disparities sequence file

---
06-02-18

- Times bucket is used for the algorithm, bins bucket is used for creating bar graphs

- Fixed init, tran, emit initial probabilities to be closer to starting assumptions

- Added decodings to buckets method to assign results to bars, can graph bar graphs

- Created VCF to difference sequence translator

- Need to do: Window translation for VCF -> difference is not working properly, come up with datasets to test now, clean up code, particularly BW and global variables with bar graph

---
06-04-18

- Fixed HMM to work with data (separated dif_seq translator, changed initial probabilities, fixed details) (took some time to debug...)

- Works well! We get a bar graph nearly equal to one another, which is what we expect

- Things to do: 1. understand why it may not an exact equal line; 2. create a statistic that measures how much it deviates (is variance good enough?), 3. clean up \\
  more code, 4. use simulated data to test whether the exp. distribution holds 5. read papers

---
06.05.18

- Checking which dimension or parameter makes the model more accurate (closer to ground truth)? L, mu, window, bins, iterations, etc.

- Use average for TMRCA windowing operation

- Think about C++, parallelization (python multiprocessing, thread)

- Initial probabilities Parameters depend on mutation, recombination rate

---
06.06.18

- Either discretization or msprime is not working

- Check whether log bin making works

- Varying pop. size for msprime

- In the end, the logic for log buckets was not correct - made it too complicated than it should be - and the bar method had a bug. All fixed, now results follow hypotheses

- Goal: make the HMM model follow the truth as closely as it can

- It seems like the recombination rate and length both help to converge the truth to the exponential distribution

---
06.07.18

- Fix bins to be exactly half-size the intervals

- Work with different params and changed bins on same data, add iterations too

- Combine 100000 or parallelize HMM.py

- In the end: simulator works nicely, parallelization works as well. Need to think about parameter updates with different processes, how to make the model more accurate.

---
06.11.18

- Changed plot lines to resemble TMRCA - pop size step graphs

- Created translating script from msms sim to sequences

- Need to look at psmc, diCal for fixing accuracy issues, possibly create GOF plot

- Implement natural selection into msms
---
06.12.18

- Configured msms to include and output TMRCA

- Presented on papers

- Trying to figure out how the figures change with natural selection, pop. size change, and both

---
06.13.18

- Fixed msms sim translator to work with hmm model

- Looked at constant pop, selected region, pop change, and both

---
06.14.18

- Finding S, pi, SFS, etc. summary statistics for msms simulator to plot

- Discuss hmm output results

- Add a bottleneck in the middle, shorter time interval, bottleneck to a place where the resolution is high

- -SAA 1000 add strength to natural selection
---
06.19.18

- Fixed data generator for neural network to output correct Tajima's D statistics

- Turned HMM outputs into text outputs

- Need to consider how to change HMM outputs to fit into CNN inputs
---
06.20.18

- Translate outputs from HMM to a demographic event for training the neural net

- Check how well Tajima's D predicts nat. sel., bottleneck, constant pop.size

- Training-wise: Have multiple natural selection features, work with both nat. sel. and pop. change

---
06.21.18

1. Sort node weights, see distribution easier
2. Perfectly matched, perfectly mismatched, in the middle - (number of 1's in the sequence)

---
06.22.18

1. Created data resembling simulations from PSMC -- short and medium length versions with bins 32 and 10 iterations, for accuracy comparison and figures for presentation

---
06.27.18

1. RNU - recurrent neural networks ; VAE - variational auto encoder
2. Exchangeable NN to make row order unconditional
3. Feed in real data to current pipelines

---
07.02.18

1. Creating data to feed to neural networks for NS
2. See how much windows impact accuracy of HMM
3. Look into optimization for model
