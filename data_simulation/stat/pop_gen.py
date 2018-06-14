"""
This program will run the scripts needed to run the analysis of the rs671 SNP
across different super-populations across the 1000 genome database based on Tajima's D.
"""

from tajima import parse_vcf, calculate_D
import matplotlib.pyplot as plt

def main():
    super_pop_list = ['AFR','AMR','EAS','EUR','SAS']
    for pop in super_pop_list:
        pop = 'data/' + pop + '_final.chr12.vcf'
        bp_buckets, genomic_locations, num_indivs, n, window, super_pop = parse_vcf(pop)
        calculate_D(bp_buckets, genomic_locations, num_indivs, n, window, super_pop)
        import pdb; pdb.set_trace()

    ## LCT Gene
    # pop = '/home/smathieson/public/cs68/1000g/EAS_135-136Mb.chr2.vcf'
    # bp_buckets, genomic_locations, num_indivs, n, window, super_pop = parse_vcf(pop)
    # calculate_D(bp_buckets, genomic_locations, num_indivs, n, window, super_pop)
    # plt.legend(super_pop_list)
    # plt.savefig('figs/tajimas_d_allpop_1000_humans.png')

main()
