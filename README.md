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

