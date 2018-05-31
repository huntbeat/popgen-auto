# popgen-hmm-dl

Sam and Hunter: 05-03-18 (1.5hrs)
- downloading 1000genomes data on chromosome 12, use bcftools to parse VCF file. But running into a lot of trouble trying to find rs671. VCF Data is now on `/scratch/cs68`
```
bcftools view -r 12:111766887-111817529 -Oz -o ALL_1117-1118MB.chr12.vcf.gz ALL.chr12.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz
```
- https://www.snpedia.com/index.php/Rs671 Found what the variants of rs61 mean. Found out that [Disulfiram](https://www.snpedia.com/index.php/Disulfiram) can be used to treat alcoholism only for people with the (G;G) variant, not for (A;G) and (A;A)
