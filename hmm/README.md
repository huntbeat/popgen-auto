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

-
